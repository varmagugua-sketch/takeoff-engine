"""
main.py - Multi-page MEP takeoff engine (FastAPI).

Endpoints
  GET  /health            -> {"status": "ok"}   (keep-alive target, no auth)
  POST /process-takeoff   -> 202 Accepted immediately; pipeline runs in a background task
                             multipart form: pdf_file (PDF), legend_file (PNG),
                                             finished_folder_id (Drive folder ID, text)
  GET  /status            -> progress of the running / last job (auth if API_KEY set)

Pipeline (one job at a time, guarded by asyncio.BoundedSemaphore(1)):
  validate -> for every page: 4x4 tiles @300 DPI -> GeminiTakeoffClient.analyze_tile_with_legend
  -> translate to global PDF coords (overlap-deduped) -> Visual Audit PDF (red OCG circles)
  + BOM xlsx -> upload both to Drive folder -> delete every temp file.

Environment variables
  GOOGLE_SERVICE_ACCOUNT_JSON   service-account key as a JSON string (required for upload)
  GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET / GOOGLE_OAUTH_REFRESH_TOKEN
                                optional fallback if no service account is set
  API_KEY                       if set, callers must send header  X-API-Key: <value>
  MAX_PDF_MB                    default 60
  JOB_MAX_MINUTES               default 90 (remaining pages are skipped, partial results delivered)
  TILE_RETRIES                  extra attempts per tile on API errors, default 2
  GEMINI_MIN_INTERVAL_SEC       minimum seconds between Gemini calls (rate-limit pacing), default 0
  DRIVE_PREFLIGHT               "0" disables the folder-access check done before returning 202
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import excel_builder
from slicer import (
    PageAccumulator,
    SliceConfig,
    SliceError,
    build_audit_pdf,
    guess_sheet_label,
    iter_page_tiles,
    iter_pages,
    open_pdf,
    page_workspace,
    plan_page,
    release_memory,
    rss_mb,
    translate_tile_result,
    validate_legend_image,
)

try:
    from gemini_client import GeminiTakeoffClient
    _GEMINI_IMPORT_ERROR = ""
except Exception as _exc:  # keep the server up so /health works and the cause is visible
    GeminiTakeoffClient = None  # type: ignore[assignment,misc]
    _GEMINI_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


# ------------------------------- settings ----------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


API_KEY = os.getenv("API_KEY", "").strip()
MAX_PDF_BYTES = int(_env_float("MAX_PDF_MB", 60.0) * 1024 * 1024)
JOB_MAX_SECONDS = _env_int("JOB_MAX_MINUTES", 90) * 60
TILE_RETRIES = max(0, _env_int("TILE_RETRIES", 2))
MIN_CALL_INTERVAL = max(0.0, _env_float("GEMINI_MIN_INTERVAL_SEC", 0.0))
DRIVE_PREFLIGHT = os.getenv("DRIVE_PREFLIGHT", "1").strip() != "0"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,128}$")

app = FastAPI(title="Takeoff Engine", version="2.0.0")

# One job at a time (512 MB RAM). Acquired in the endpoint, released when the job ends.
_job_slot = asyncio.BoundedSemaphore(1)

_state_lock = threading.Lock()
_state: Dict[str, Any] = {"state": "idle"}


def _set_state(**kw: Any) -> None:
    with _state_lock:
        _state.update(kw)


def _snapshot() -> Dict[str, Any]:
    with _state_lock:
        snap = dict(_state)
    if snap.get("state") == "running" and "started_at" in snap:
        snap["elapsed_sec"] = round(time.time() - snap["started_at"])
    snap["busy"] = _job_slot.locked()
    snap["rss_mb"] = round(rss_mb(), 1)
    return snap


# --------------------------------- auth -------------------------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


# ------------------------------ Google Drive --------------------------------
class DriveError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _load_service_account_info() -> Optional[Dict[str, Any]]:
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        return None
    info: Optional[Dict[str, Any]] = None
    try:
        info = json.loads(raw)
    except ValueError:
        try:  # tolerate a base64-encoded key
            info = json.loads(base64.b64decode(raw).decode("utf-8"))
        except Exception:
            info = None
    if not isinstance(info, dict) or not info.get("client_email") or not info.get("private_key"):
        raise DriveError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is not a valid service-account key "
            "(needs client_email and private_key).", 500,
        )
    pk = str(info["private_key"])
    if "\\n" in pk and "\n" not in pk:  # newlines escaped by copy/paste
        info["private_key"] = pk.replace("\\n", "\n")
    return info


def _credentials():
    info = _load_service_account_info()
    if info:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    cid = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    sec = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    rtk = os.getenv("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip()
    if cid and sec and rtk:
        from google.oauth2.credentials import Credentials
        return Credentials(
            token=None, refresh_token=rtk, token_uri="https://oauth2.googleapis.com/token",
            client_id=cid, client_secret=sec, scopes=DRIVE_SCOPES,
        )
    raise DriveError("No Google credentials configured. Set GOOGLE_SERVICE_ACCOUNT_JSON.", 500)


def _drive_service():
    try:
        import google_auth_httplib2
        import httplib2
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise DriveError(f"Google client libraries are not installed: {exc}", 500)
    creds = _credentials()
    http = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=120))
    return build("drive", "v3", http=http, cache_discovery=False)


def _http_status(exc: Any) -> int:
    try:
        return int(getattr(exc.resp, "status", 0) or 0)
    except Exception:
        return 0


def _explain_drive_http(status: int, exc: Any) -> str:
    body = ""
    try:
        body = (exc.content or b"").decode("utf-8", "ignore")
    except Exception:
        pass
    if "storageQuota" in body or "storage quota" in body.lower():
        return ("Google refused the upload: service accounts have no storage quota in a normal "
                "My Drive folder. Use a Shared Drive folder, or set the GOOGLE_OAUTH_* variables.")
    if status == 404:
        return "Drive folder not found, or it is not shared with the service account (Editor)."
    if status in (401, 403):
        return f"Google rejected the credentials or folder permissions (HTTP {status})."
    return f"Drive API error (HTTP {status})."


def _drive_preflight(folder_id: str) -> str:
    """Fail fast (before a 15-minute job) if the folder is unreachable or read-only."""
    from googleapiclient.errors import HttpError
    svc = _drive_service()
    try:
        meta = svc.files().get(
            fileId=folder_id,
            fields="id,name,mimeType,capabilities(canAddChildren)",
            supportsAllDrives=True,
        ).execute(num_retries=2)
    except HttpError as exc:
        st = _http_status(exc)
        raise DriveError(_explain_drive_http(st, exc), 400 if st == 404 else (403 if st in (401, 403) else 503))
    except Exception as exc:
        raise DriveError(f"Drive check failed: {type(exc).__name__}", 503)
    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        raise DriveError("finished_folder_id is not a folder.", 400)
    if (meta.get("capabilities") or {}).get("canAddChildren") is False:
        raise DriveError("The service account can view this folder but cannot add files (needs Editor).", 403)
    return str(meta.get("name", ""))


def _upload_file(service: Any, path: str, name: str, mime: str, folder_id: str) -> Dict[str, Any]:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    last: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            media = MediaFileUpload(path, mimetype=mime, resumable=True, chunksize=8 * 1024 * 1024)
            req = service.files().create(
                body={"name": name, "parents": [folder_id]},
                media_body=media,
                fields="id,name,webViewLink,size",
                supportsAllDrives=True,
            )
            resp = None
            while resp is None:
                _status, resp = req.next_chunk(num_retries=3)
            return {"name": resp.get("name", name), "id": resp.get("id"), "link": resp.get("webViewLink")}
        except HttpError as exc:
            st = _http_status(exc)
            if 400 <= st < 500 and st != 429:
                raise DriveError(f"Upload of '{name}' failed: {_explain_drive_http(st, exc)}", 502)
            last = exc
        except Exception as exc:
            last = exc
        log.warning("Upload attempt %d/3 for %s failed: %s", attempt, name, last)
        time.sleep(3 * attempt)
    raise DriveError(f"Upload of '{name}' failed after 3 attempts: {type(last).__name__}", 502)


def _upload_text(service: Any, text: str, name: str, folder_id: str) -> None:
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(io.BytesIO(text.encode("utf-8")), mimetype="text/plain", resumable=False)
    service.files().create(
        body={"name": name, "parents": [folder_id]}, media_body=media,
        fields="id", supportsAllDrives=True,
    ).execute(num_retries=3)


# ------------------------------- job model ----------------------------------
@dataclass
class Job:
    job_id: str
    workdir: str
    pdf_path: str
    legend_path: str
    pdf_name: str
    stem: str
    folder_id: str
    page_count: int

    @property
    def audit_name(self) -> str:
        return f"{self.stem}_Visual_Audit.pdf"

    @property
    def bom_name(self) -> str:
        return f"{self.stem}_BOM_Ledger.xlsx"


def _safe_stem(filename: Optional[str]) -> str:
    base = os.path.splitext(os.path.basename(filename or ""))[0]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" ._")[:60]
    return base or "blueprint"


def _public_error(exc: BaseException) -> str:
    if isinstance(exc, (SliceError, DriveError, excel_builder.BomError)):
        return str(exc)
    return f"Internal error ({type(exc).__name__}). Check the server logs."


# --------------------------- Gemini call wrapper ----------------------------
_throttle_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    if MIN_CALL_INTERVAL <= 0:
        return
    with _throttle_lock:
        wait = MIN_CALL_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _call_gemini(client: Any, tile_path: str, legend_path: str, page_number: int, tile_index: int) -> Any:
    last: Optional[BaseException] = None
    for attempt in range(TILE_RETRIES + 1):
        _throttle()
        try:
            result = client.analyze_tile_with_legend(tile_path, legend_path, page_number, tile_index)
            if asyncio.iscoroutine(result):       # tolerate an async client implementation
                result = asyncio.run(result)
            if result is None:
                raise ValueError("empty result from vision client")
            return result
        except Exception as exc:
            last = exc
            if attempt < TILE_RETRIES:
                delay = min(30, 5 * (attempt + 1) ** 2)    # 5s, 20s, 30s...
                log.warning("p%d t%d attempt %d failed (%s); retry in %ss",
                            page_number, tile_index, attempt + 1, type(exc).__name__, delay)
                time.sleep(delay)
    assert last is not None
    raise last


# -------------------------------- pipeline ----------------------------------
def _process_page(page: Any, page_index: int, acc: PageAccumulator, client: Any,
                  job: Job, cfg: SliceConfig, tiles_root: str) -> None:
    plan = plan_page(page, page_index, cfg)
    acc.tiles_total = len(plan.tiles)
    acc.label = guess_sheet_label(page)
    with page_workspace(tiles_root, plan.page_number) as workdir:   # removed on exit
        for tile in iter_page_tiles(page, plan, cfg, workdir):      # PNG deleted after each tile
            if tile.is_blank:
                acc.mark_blank()
                continue
            try:
                result = _call_gemini(client, tile.path, job.legend_path, plan.page_number, tile.spec.index)
                acc.add_translation(translate_tile_result(result, tile, plan))
            except SliceError:
                raise
            except Exception as exc:
                log.warning("p%d t%d failed: %s", plan.page_number, tile.spec.index, exc)
                acc.mark_failed(tile.spec.index, f"{type(exc).__name__}: {exc}")
            with _state_lock:
                _state["tiles_sent"] = int(_state.get("tiles_sent", 0)) + 1


def _pipeline(job: Job) -> Dict[str, Any]:
    """Runs in a worker thread so the event loop (and /health) stay responsive."""
    started = time.monotonic()
    cfg = SliceConfig()
    if GeminiTakeoffClient is None:
        raise RuntimeError(f"gemini_client unavailable: {_GEMINI_IMPORT_ERROR}")
    client = GeminiTakeoffClient()
    tiles_root = os.path.join(job.workdir, "tiles")
    accs: List[PageAccumulator] = []

    # ---- analysis: for page_index in range(len(doc)) -----------------------
    doc = open_pdf(job.pdf_path, cfg)
    try:
        _set_state(stage="analyzing", pages_total=doc.page_count, page=0, tiles_sent=0)
        for page_index, page, err in iter_pages(doc, cfg):
            page_number = page_index + 1
            acc = PageAccumulator(page_number=page_number)
            acc.tiles_total = cfg.grid * cfg.grid
            accs.append(acc)

            if time.monotonic() - started > JOB_MAX_SECONDS:
                acc.tiles_failed = acc.tiles_total
                acc.notes.append("skipped: job time limit reached")
                continue
            if err is not None:
                acc.tiles_failed = acc.tiles_total
                acc.notes.append(f"page skipped: {err}")
                continue

            _set_state(page=page_number)
            try:
                _process_page(page, page_index, acc, client, job, cfg, tiles_root)
            except SliceError as exc:
                log.error("page %d aborted: %s", page_number, exc)
                acc.notes.append(f"page aborted: {exc}")
            except Exception as exc:
                log.exception("page %d aborted", page_number)
                acc.notes.append(f"page aborted: {type(exc).__name__}: {exc}")
            finally:
                accounted = acc.tiles_analyzed + acc.tiles_blank + acc.tiles_failed
                if accounted < acc.tiles_total:
                    acc.tiles_failed += acc.tiles_total - accounted
                page = None
                shutil.rmtree(os.path.join(tiles_root), ignore_errors=True)  # belt and braces
                release_memory()                                             # store_shrink + gc.collect
    finally:
        doc.close()
        release_memory()

    analyzed = sum(a.tiles_analyzed for a in accs)
    failed = sum(a.tiles_failed for a in accs)
    if analyzed == 0 and failed > 0:
        first = next((n for a in accs for n in a.notes), "no detail available")
        raise RuntimeError(f"All {failed} tiles failed; nothing to deliver. First error: {first}")

    # ---- deliverables -------------------------------------------------------
    _set_state(stage="building_outputs")
    dets_by_page = {a.page_number: a.detections for a in accs if a.detections}
    audit_path = os.path.join(job.workdir, job.audit_name)
    bom_path = os.path.join(job.workdir, job.bom_name)

    bom = excel_builder.build_bom_workbook(accs, bom_path, project_name=job.stem)
    stats = build_audit_pdf(job.pdf_path, dets_by_page, audit_path, cfg)   # opens a fresh doc
    release_memory()

    # ---- upload ---------------------------------------------------------------
    _set_state(stage="uploading")
    svc = _drive_service()
    uploaded = [
        _upload_file(svc, bom_path, job.bom_name, XLSX_MIME, job.folder_id),
        _upload_file(svc, audit_path, job.audit_name, PDF_MIME, job.folder_id),
    ]

    return {
        "pages": len(accs),
        "items_placed": bom.grand_total_qty,
        "items_unplaced": bom.grand_total_unplaced,
        "symbols": bom.symbols,
        "markers": stats.get("markers", 0),
        "tiles_analyzed": analyzed,
        "tiles_blank": sum(a.tiles_blank for a in accs),
        "tiles_failed": failed,
        "uploaded": uploaded,
        "elapsed_sec": round(time.monotonic() - started),
    }


def _report_failure(job: Job, message: str) -> None:
    """Drop a small text file in the output folder so a silent failure is never invisible."""
    svc = _drive_service()
    text = (
        f"Takeoff FAILED\nFile: {job.pdf_name}\nJob: {job.job_id}\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\nReason: {message}\n"
    )
    _upload_text(svc, text, f"TAKEOFF_FAILED_{job.stem}.txt", job.folder_id)


async def _run_job(job: Job) -> None:
    """Background task. Owns the semaphore permit acquired by the endpoint."""
    try:
        _set_state(state="running", job_id=job.job_id, pdf=job.pdf_name, stage="starting",
                   started_at=time.time(), pages_total=job.page_count, page=0, tiles_sent=0,
                   error=None, summary=None)
        summary = await run_in_threadpool(_pipeline, job)
        _set_state(state="done", stage="complete", finished_at=time.time(), summary=summary)
        log.info("Job %s done: %s", job.job_id, summary)
    except BaseException as exc:  # noqa: BLE001 - must always release the slot
        msg = _public_error(exc)
        log.exception("Job %s failed", job.job_id)
        _set_state(state="failed", stage="failed", finished_at=time.time(), error=msg)
        try:
            await run_in_threadpool(_report_failure, job, msg)
        except Exception:
            log.warning("Could not upload failure notice for job %s", job.job_id)
        if not isinstance(exc, Exception):
            raise
    finally:
        shutil.rmtree(job.workdir, ignore_errors=True)   # PDF, legend, tiles, outputs - everything
        release_memory()
        _job_slot.release()


# --------------------------------- uploads ----------------------------------
async def _stash_upload(upload: UploadFile, dest: str, max_bytes: int, kind: str) -> str:
    size, head = 0, b""
    with open(dest, "wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            if size == 0:
                head = chunk[:1024]
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, f"{kind.upper()} exceeds the {max_bytes // (1024 * 1024)} MB limit.")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, f"{kind.upper()} file is empty.")
    if kind == "pdf" and b"%PDF-" not in head:
        raise HTTPException(415, "pdf_file is not a valid PDF.")
    if kind == "png" and not head.startswith(PNG_MAGIC):
        raise HTTPException(415, "legend_file must be a PNG image.")
    return dest


def _preflight_inputs(pdf_path: str, legend_path: str, cfg: SliceConfig) -> int:
    validate_legend_image(legend_path, cfg)
    doc = open_pdf(pdf_path, cfg)
    try:
        return int(doc.page_count)
    finally:
        doc.close()
        release_memory()


# ---------------------------------- routes ----------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status", dependencies=[Depends(require_api_key)])
def status() -> Dict[str, Any]:
    return _snapshot()


@app.post("/process-takeoff", status_code=202, dependencies=[Depends(require_api_key)])
async def process_takeoff(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    legend_file: UploadFile = File(...),
    finished_folder_id: str = Form(...),
):
    folder_id = (finished_folder_id or "").strip()
    if not _FOLDER_ID_RE.match(folder_id):
        raise HTTPException(422, "finished_folder_id is not a valid Google Drive folder ID.")
    if GeminiTakeoffClient is None:
        raise HTTPException(503, f"Vision client unavailable: {_GEMINI_IMPORT_ERROR}")

    # Single-job guard: reject instead of queueing so uploads never pile up on disk.
    if _job_slot.locked():
        raise HTTPException(429, "A takeoff is already running. Retry in a few minutes.",
                            headers={"Retry-After": "120"})
    await _job_slot.acquire()          # does not suspend: the slot was just seen free

    workdir: Optional[str] = None
    scheduled = False
    try:
        cfg = SliceConfig()
        workdir = tempfile.mkdtemp(prefix="job_")
        pdf_path = await _stash_upload(pdf_file, os.path.join(workdir, "input.pdf"), MAX_PDF_BYTES, "pdf")
        legend_path = await _stash_upload(legend_file, os.path.join(workdir, "legend.png"), cfg.max_legend_bytes, "png")

        page_count = await run_in_threadpool(_preflight_inputs, pdf_path, legend_path, cfg)
        if DRIVE_PREFLIGHT:
            await run_in_threadpool(_drive_preflight, folder_id)

        job = Job(
            job_id=uuid.uuid4().hex[:12], workdir=workdir, pdf_path=pdf_path, legend_path=legend_path,
            pdf_name=os.path.basename(pdf_file.filename or "blueprint.pdf"),
            stem=_safe_stem(pdf_file.filename), folder_id=folder_id, page_count=page_count,
        )
        background_tasks.add_task(_run_job, job)
        scheduled = True
        log.info("Accepted job %s: %s (%d pages)", job.job_id, job.pdf_name, page_count)
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "job_id": job.job_id,
                "pdf": job.pdf_name,
                "pages": page_count,
                "outputs": [job.audit_name, job.bom_name],
                "message": "Processing started. Results will appear in the finished folder.",
            },
        )
    except HTTPException:
        raise
    except SliceError as exc:
        raise HTTPException(exc.status_code, str(exc))
    except DriveError as exc:
        raise HTTPException(exc.status, str(exc))
    except Exception:
        log.exception("Failed to accept job")
        raise HTTPException(500, "Could not start the takeoff job.")
    finally:
        if not scheduled:
            if workdir:
                shutil.rmtree(workdir, ignore_errors=True)
            _job_slot.release()
