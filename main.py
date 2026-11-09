"""
main.py - FastAPI microservice for the Takeoff Pipeline (Day 1: slicing).

Endpoints
  GET  /health            liveness probe (no auth)
  POST /v1/inspect        PDF sanity check: pages, sizes, detected content area
  POST /v1/slice          returns JSON tile manifest (no images)
  POST /v1/slice/zip      returns ZIP of tile PNGs + manifest.json (visual QA)

Env vars
  API_KEY         if set, every /v1 call must send header  X-API-Key: <value>
  MAX_UPLOAD_MB   default 50
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import tempfile
import threading
import zipfile
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from slicer import (
    SliceConfig,
    SliceError,
    detect_content_rect,
    iter_tiles,
    load_page,
    open_pdf,
    plan_header,
    plan_tiles,
    render_legend,
    tile_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

API_KEY = os.getenv("API_KEY", "").strip()
try:
    MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024)
except ValueError:
    MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Free tier = 512 MB RAM -> exactly one heavy job at a time.
_busy = threading.BoundedSemaphore(1)

app = FastAPI(title="Takeoff Engine", version="0.1.0")


# ------------------------------- plumbing ----------------------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


def _acquire() -> None:
    if not _busy.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Engine is busy with another job. Retry shortly.",
            headers={"Retry-After": "30"},
        )


async def _stash_upload(file: UploadFile, workdir: str) -> str:
    """Stream upload to disk with a size cap and PDF magic-byte check."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload must be a .pdf file.")
    path = os.path.join(workdir, "input.pdf")
    size, head = 0, b""
    with open(path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            if size == 0:
                head = chunk[:1024]
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                )
            out.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if b"%PDF-" not in head:
        raise HTTPException(status_code=415, detail="File is not a valid PDF.")
    return path


def _cfg(page: int, dpi: int, grid: int, overlap: float, crop: bool) -> SliceConfig:
    cfg = SliceConfig(page_index=page, dpi=dpi, grid=grid, overlap=overlap, crop_to_content=crop)
    cfg.validate()
    return cfg


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, SliceError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, MemoryError):
        log.error("MemoryError during job")
        return HTTPException(status_code=503, detail="Out of memory. Lower dpi or raise grid.")
    log.exception("Unhandled error")
    return HTTPException(status_code=500, detail="Internal processing error.")


# ------------------------------- job bodies --------------------------------
def _inspect_job(path: str, page_index: int) -> dict:
    doc = open_pdf(path)
    try:
        pages = [
            {"index": i, "width_in": round(doc[i].rect.width / 72, 2),
             "height_in": round(doc[i].rect.height / 72, 2)}
            for i in range(min(doc.page_count, 100))
        ]
        page = load_page(doc, page_index)
        region = detect_content_rect(page)
        return {"page_count": doc.page_count, "pages": pages,
                "selected_page": page_index, "content_region_pt": tuple(round(v, 2) for v in region)}
    finally:
        doc.close()


def _manifest_job(path: str, cfg: SliceConfig) -> dict:
    doc = open_pdf(path)
    try:
        page = load_page(doc, cfg.page_index)
        plan = plan_tiles(page, cfg.page_index, cfg)
        metas = [tile_meta(t) for t in iter_tiles(page, plan, cfg)]  # PNG bytes discarded each loop
        return {**plan_header(plan), "tiles": metas,
                "non_blank_tiles": sum(1 for m in metas if not m["is_blank"])}
    finally:
        doc.close()


def _zip_job(path: str, cfg: SliceConfig, workdir: str) -> str:
    doc = open_pdf(path)
    zip_path = os.path.join(workdir, "tiles.zip")
    try:
        page = load_page(doc, cfg.page_index)
        plan = plan_tiles(page, cfg.page_index, cfg)
        metas = []
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for t in iter_tiles(page, plan, cfg):
                name = f"tile_r{t.spec.row}_c{t.spec.col}.png"
                zf.writestr(name, t.png)
                metas.append(tile_meta(t, name))
            manifest = {**plan_header(plan), "tiles": metas}
            if plan.legend is not None:
                try:
                    lg = render_legend(page, plan.legend, cfg)
                    zf.writestr("legend.png", lg.png)
                    manifest["legend"]["file"] = "legend.png"
                    manifest["legend"]["width_px"] = lg.width_px
                    manifest["legend"]["height_px"] = lg.height_px
                    manifest["legend"]["render_dpi"] = lg.dpi
                except Exception:
                    log.exception("Legend crop render failed; ZIP continues without it.")
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        return zip_path
    finally:
        doc.close()


# -------------------------------- routes -----------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/inspect", dependencies=[Depends(require_api_key)])
async def inspect(
    file: UploadFile = File(...),
    page: int = Query(0, ge=0, description="0-based page index"),
) -> dict:
    _acquire()
    workdir = tempfile.mkdtemp(prefix="takeoff_")
    try:
        path = await _stash_upload(file, workdir)
        return await run_in_threadpool(_inspect_job, path, page)
    except Exception as exc:
        raise _fail(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        _busy.release()


@app.post("/v1/slice", dependencies=[Depends(require_api_key)])
async def slice_manifest(
    file: UploadFile = File(...),
    page: int = Query(0, ge=0),
    dpi: int = Query(300, ge=72, le=300),
    grid: int = Query(4, ge=1, le=8),
    overlap: float = Query(0.05, ge=0.0, le=0.2),
    crop_to_content: bool = Query(True),
) -> dict:
    _acquire()
    workdir = tempfile.mkdtemp(prefix="takeoff_")
    try:
        cfg = _cfg(page, dpi, grid, overlap, crop_to_content)
        path = await _stash_upload(file, workdir)
        return await run_in_threadpool(_manifest_job, path, cfg)
    except Exception as exc:
        raise _fail(exc)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        _busy.release()


@app.post("/v1/slice/zip", dependencies=[Depends(require_api_key)])
async def slice_zip(
    file: UploadFile = File(...),
    page: int = Query(0, ge=0),
    dpi: int = Query(300, ge=72, le=300),
    grid: int = Query(4, ge=1, le=8),
    overlap: float = Query(0.05, ge=0.0, le=0.2),
    crop_to_content: bool = Query(True),
):
    _acquire()
    workdir = tempfile.mkdtemp(prefix="takeoff_")
    keep_workdir = False
    try:
        cfg = _cfg(page, dpi, grid, overlap, crop_to_content)
        path = await _stash_upload(file, workdir)
        zip_path = await run_in_threadpool(_zip_job, path, cfg, workdir)
        keep_workdir = True  # file must outlive this function; cleaned after send
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="tiles.zip",
            background=BackgroundTask(shutil.rmtree, workdir, True),
        )
    except Exception as exc:
        raise _fail(exc)
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        _busy.release()
