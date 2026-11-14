"""
main.py - Multi-page MEP takeoff engine (FastAPI), Drive-to-Drive edition.

Endpoints
  GET  /health            -> {"status": "ok"}   (keep-alive target, no auth)
  POST /process-takeoff   -> 202 Accepted immediately; the job runs in a background task
                             JSON body (TakeoffJobRequest):
                               {"pdf_file_id": "...", "pdf_filename": "...",
                                "legend_file_id": "...", "finished_folder_id": "..."}
  GET  /status            -> progress of the running / last job (auth if API_KEY set)

No file ever travels in the HTTP request. The worker downloads the blueprint and the
legend from Google Drive (MediaIoBaseDownload, streamed to disk), so there is no
request-size ceiling; the practical limits are RAM and temp disk (see MAX_PDF_MB).

Pipeline (one job at a time, guarded by asyncio.BoundedSemaphore(1)):
  download -> for every page: 4x4 tiles @300 DPI -> GeminiTakeoffClient.analyze_tile_with_legend
  -> global PDF coordinates (overlap-deduped) -> Visual Audit PDF (red OCG circles) + BOM xlsx
  -> upload both to finished_folder_id -> delete every temp file.

Environment variables
  GOOGLE_SERVICE_ACCOUNT_JSON   service-account key as a JSON string (required)
                                scope used: https://www.googleapis.com/auth/drive
  GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET / GOOGLE_OAUTH_REFRESH_TOKEN
                                optional fallback if no service account is set
  API_KEY                       if set, callers must send header  X-API-Key: <value>
  MAX_PDF_MB                    default 150 (rejects larger Drive files before downloading)
  JOB_MAX_MINUTES               default 90 (remaining pages skipped, partial results delivered)
  TILE_RETRIES                  extra attempts per tile on API errors, default 2
  GEMINI_MIN_INTERVAL_SEC       minimum seconds between Gemini calls (rate-limit pacing), default 0
  DRIVE_PREFLIGHT               "0" disables the Drive access check done before returning 202
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

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
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


MB = 1024 * 1024
API_KEY = os.getenv("API_KEY", "").strip()
MAX_PDF_BYTES = int(_env_float("MAX_PDF_MB", 150.0) * MB)
JOB_MAX_SECONDS = _env_int("JOB_MAX_MINUTES", 90) * 60
TILE_RETRIES = max(0, _env_int("TILE_RETRIES", 2))
MIN_CALL_INTERVAL = max(0.0, _env_float("GEMINI_MIN_INTERVAL_SEC", 0.0))
DRIVE_PREFLIGHT = os.getenv("DRIVE_PREFLIGHT", "1").strip() != "0"

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
_DRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,128}$")

app = FastAPI(title="Takeoff Engine", version="3.0.0")

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


# ------------------------------ request model -------------------------------
class TakeoffJobRequest(BaseModel):
    pdf_file_id: str = Field(..., min_length=10, max_length=128)
    pdf_filename: str = Field(default="blueprint.pdf", max_length=255)
    legend_file_id: str = Field(..., min_length=10, max_length=128)
    finished_folder_id: str = Field(..., min_length=10, max_length=128)

    @field_validator("pdf_file_id", "legend_file_id", "finished_folder_id")
    @classmethod
    def _valid_drive_id(cls, v: str) -> str:
        v = (v or "").strip()
        if not _DRIVE_ID_RE.match(v):
            raise ValueError("not a valid Google Drive ID")
        return v

    @field_validator("pdf_filename", mode="before")
    @classmethod
    def _clean_filename(cls, v: Any) -> str:
        name = os.path.basename(str(v or "").strip())
        return name or "blueprint.pdf"


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
    http = google_auth_httplib2.AuthorizedHttp(creds, http=httplib2.Http(timeout=300))
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
    if "fileNotDownloadable" in body or "cannotDownloadFile" in body:
        return "Google does not allow this file to be downloaded."
    if status == 404:
        return "Drive item not found, or it is not shared with the service account."
    if status in (401, 403):
        return f"Google rejected the credentials or the sharing permissions (HTTP {status})."
    return f"Drive API error (HTTP {status})."


def _file_meta(svc: Any, file_id: str, label: str) -> Dict[str, Any]:
    from googleapiclient.errors import HttpError
    try:
        meta = svc.files().get(
            fileId=file_id,
            fields="id,name,mimeType,size,trashed,capabilities(canDownload,canAddChildren)",
            supportsAllDrives=True,
        ).execute(num_retries=2)
    except HttpError as exc:
        st = _http_status(exc)
        raise DriveError(f"{label}: {_explain_drive_http(st, exc)}",
                         400 if st == 404 else (403 if st in (401, 403) else 503))
    except Exception as exc:
        raise DriveError(f"{label}: Drive check failed ({type(exc).__name__}).", 503)
    if meta.get("trashed"):
        raise DriveError(f"{label}: the file is in the Drive trash.", 400)
    return meta


def _assert_downloadable(meta: Dict[str, Any], label: str, max_bytes: int, kind: str) -> int:
    """Validate type, permission and size from metadata alone. Returns size in bytes (0 if unknown)."""
    mime = str(meta.get("mimeType") or "")
    if mime.startswith("application/vnd.google-apps."):
        raise DriveError(f"{label}: this is a native Google file ({mime}); upload a real file.", 422)
    ok = (mime in (PDF_MIME, "application/octet-stream", "")) if kind == "pdf" else \
         (mime.startswith("image/") or mime in ("application/octet-stream", ""))
    if not ok:
        raise DriveError(f"{label}: unexpected file type '{mime}'.", 422)
    if (meta.get("capabilities") or {}).get("canDownload") is False:
        raise DriveError(f"{label}: downloading is disabled for this file.", 403)
    try:
        size = int(meta.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size and size > max_bytes:
        raise DriveError(
            f"{label}: file is {size / MB:.0f} MB; the limit is {max_bytes // MB} MB (MAX_PDF_MB).", 413)
    return size


def _check_folder(svc: Any, folder_id: str) -> None:
    meta = _file_meta(svc, folder_id, "finished_folder_id")
    if meta.get("mimeType") != DRIVE_FOLDER_MIME:
        raise DriveError("finished_folder_id: this is not a folder.", 400)
    if (meta.get("capabilities") or {}).get("canAddChildren") is False:
        raise DriveError("finished_folder_id: the service account can view this folder but cannot "
                         "add files (share it as Editor).", 403)


def _ensure_disk(path: str, needed_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    if free < needed_bytes:
        raise DriveError(
            f"Not enough temporary disk space (need ~{needed_bytes // MB} MB, "
            f"{free // MB} MB free).", 507)


def _download_file(svc: Any, file_id: str, dest: str, max_bytes: int, label: str) -> int:
    """Stream a Drive file to disk with MediaIoBaseDownload. Returns bytes written."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload

    last: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
            with open(dest, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=8 * MB)
                done = False
                while not done:
                    _status, done = downloader.next_chunk(num_retries=3)
                    written = fh.tell()
                    if written > max_bytes:
                        raise DriveError(f"{label}: download exceeded {max_bytes // MB} MB.", 413)
                    _set_state(download_mb=round(written / MB, 1))
            return os.path.getsize(dest)
        except DriveError:
            raise
        except HttpError as exc:
            st = _http_status(exc)
            if 400 <= st < 500 and st != 429:
                raise DriveError(f"{label}: {_explain_drive_http(st, exc)}", 502)
            last = exc
        except Exception as exc:
            last = exc
        log.warning("Download attempt %d/3 for %s failed: %s", attempt, label, last)
        try:
            os.remove(dest)
        except OSError:
            pass
        time.sleep(3 * attempt)
    raise DriveError(f"{label}: download failed after 3 attempts ({type(last).__name__}).", 502)


def _upload_file(service: Any, path: str, name: str, mime: str, folder_id: str) -> Dict[str, Any]:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    last: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            media = MediaFileUpload(path, mimetype=mime, resumable=True, chunksize=8 * MB)
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
    pdf_file_id: str
    legend_file_id: str
    folder_id: str
    pdf_name: str
    stem: str
    workdir: str = ""
    pdf_path: str = ""
    legend_path: str = ""
    page_count: int = 0

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


def _fetch_inputs(job: Job, cfg: SliceConfig) -> None:
    """Authenticate, validate metadata, and stream both inputs from Drive into job.workdir."""
    _set_state(stage="downloading", download_mb=0)
    svc = _drive_service()

    pdf_meta = _file_meta(svc, job.pdf_file_id, "pdf_file_id")
    pdf_size = _assert_downloadable(pdf_meta, "pdf_file_id", MAX_PDF_BYTES, "pdf")
    legend_meta = _file_meta(svc, job.legend_file_id, "legend_file_id")
    _assert_downloadable(legend_meta, "legend_file_id", cfg.max_legend_bytes, "image")

    # source PDF + the audit copy we write + tile PNGs + headroom
    _ensure_disk(job.workdir, int(max(pdf_size, 1) * 2.6) + 150 * MB)

    job.pdf_path = os.path.join(job.workdir, "input.pdf")
    job.legend_path = os.path.join(job.workdir, "legend.png")

    _download_file(svc, job.legend_file_id, job.legend_path, cfg.max_legend_bytes, "legend_file_id")
    validate_legend_image(job.legend_path, cfg)            # PNG signature + dimensions

    _download_file(svc, job.pdf_file_id, job.pdf_path, MAX_PDF_BYTES, "pdf_file_id")
    with open(job.pdf_path, "rb") as fh:
        if b"%PDF-" not in fh.read(1024):
            raise DriveError("pdf_file_id: the downloaded file is not a valid PDF.", 415)
    log.info("Job %s: downloaded %s (%.1f MB)", job.job_id, job.pdf_name,
             os.path.getsize(job.pdf_path) / MB)


def _pipeline(job: Job) -> Dict[str, Any]:
    """Runs in a worker thread so the event loop (and /health) stay responsive."""
    started = time.monotonic()
    cfg = SliceConfig()
    if GeminiTakeoffClient is None:
        raise RuntimeError(f"gemini_client unavailable: {_GEMINI_IMPORT_ERROR}")
    client = GeminiTakeoffClient()               # fail fast, before downloading anything

    _fetch_inputs(job, cfg)

    tiles_root = os.path.join(job.workdir, "tiles")
    accs: List[PageAccumulator] = []

    # ---- analysis: for page_index in range(len(doc)) -----------------------
    doc = open_pdf(job.pdf_path, cfg)
    try:
        job.page_count = int(doc.page_count)
        _set_state(stage="analyzing", pages_total=job.page_count, page=0, tiles_sent=0)
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
                shutil.rmtree(tiles_root, ignore_errors=True)   # belt and braces
                release_memory()                                 # store_shrink + gc.collect
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

    # ---- upload (fresh service: the old connection may be stale after a long run) ----
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
        job.workdir = tempfile.mkdtemp(prefix="job_")
        _set_state(state="running", job_id=job.job_id, pdf=job.pdf_name, stage="starting",
                   started_at=time.time(), pages_total=0, page=0, tiles_sent=0, download_mb=0,
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
        if job.workdir:
            shutil.rmtree(job.workdir, ignore_errors=True)   # PDF, legend, tiles, outputs - everything
        release_memory()
        _job_slot.release()


# ---------------------------------- routes ----------------------------------
def _preflight_request(req: TakeoffJobRequest) -> Dict[str, Any]:
    """Fast (~1-2 s) Drive access check so a bad ID or missing share fails NOW, not in 15 minutes."""
    cfg = SliceConfig()
    svc = _drive_service()
    _check_folder(svc, req.finished_folder_id)
    pdf_meta = _file_meta(svc, req.pdf_file_id, "pdf_file_id")
    pdf_size = _assert_downloadable(pdf_meta, "pdf_file_id", MAX_PDF_BYTES, "pdf")
    legend_meta = _file_meta(svc, req.legend_file_id, "legend_file_id")
    _assert_downloadable(legend_meta, "legend_file_id", cfg.max_legend_bytes, "image")
    return {"pdf_size_mb": round(pdf_size / MB, 1)}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status", dependencies=[Depends(require_api_key)])
def status() -> Dict[str, Any]:
    return _snapshot()


@app.post("/process-takeoff", status_code=202, dependencies=[Depends(require_api_key)])
async def process_takeoff(req: TakeoffJobRequest, background_tasks: BackgroundTasks):
    if GeminiTakeoffClient is None:
        raise HTTPException(503, f"Vision client unavailable: {_GEMINI_IMPORT_ERROR}")

    # Single-job guard: reject instead of queueing.
    if _job_slot.locked():
        raise HTTPException(429, "A takeoff is already running. Retry in a few minutes.",
                            headers={"Retry-After": "120"})
    await _job_slot.acquire()          # does not suspend: the slot was just seen free

    scheduled = False
    try:
        info: Dict[str, Any] = {}
        if DRIVE_PREFLIGHT:
            info = await run_in_threadpool(_preflight_request, req)

        job = Job(
            job_id=uuid.uuid4().hex[:12],
            pdf_file_id=req.pdf_file_id,
            legend_file_id=req.legend_file_id,
            folder_id=req.finished_folder_id,
            pdf_name=req.pdf_filename,
            stem=_safe_stem(req.pdf_filename),
        )
        background_tasks.add_task(_run_job, job)
        scheduled = True
        log.info("Accepted job %s: %s", job.job_id, job.pdf_name)
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "job_id": job.job_id,
                "pdf": job.pdf_name,
                **info,
                "outputs": [job.audit_name, job.bom_name],
                "message": "Processing started. Results will appear in the finished folder.",
            },
        )
    except HTTPException:
        raise
    except DriveError as exc:
        raise HTTPException(exc.status, str(exc))
    except SliceError as exc:
        raise HTTPException(exc.status_code, str(exc))
    except Exception:
        log.exception("Failed to accept job")
        raise HTTPException(500, "Could not start the takeoff job.")
    finally:
        if not scheduled:
            _job_slot.release()
