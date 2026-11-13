"""
slicer.py - Multi-page MEP takeoff slicer (PyMuPDF only; no OpenCV, no numpy).

Responsibilities
  1. Open / validate the blueprint PDF and the agency-supplied legend.png.
  2. For every page: plan a 4x4 grid (16 tiles) and render each tile at up to
     300 DPI straight from the vector PDF using a clip rectangle, so the full
     canvas (~233 MB for 36x24in) is NEVER held in RAM. Peak is ~1 tile.
  3. Translate tile-pixel detections into GLOBAL PDF page coordinates, dropping
     duplicates in overlap bands (a detection is kept only by the tile whose
     non-overlapping CORE contains it).
  4. Stamp translucent red vector circles on an Optional Content Group (OCG)
     layer into a Visual Audit PDF.
  5. Strict per-page cleanup: tile PNGs deleted after each tile, the per-page
     temp directory removed, MuPDF cache shrunk, gc.collect().

Coordinate contract
    page_x_pt = tile.origin_pt[0] + tile_pixel_x / plan.scale
    page_y_pt = tile.origin_pt[1] + tile_pixel_y / plan.scale
Points = 1/72 inch, origin top-left of the (rotated, as-viewed) page.

This module never imports gemini_client; it reads result objects by duck typing:
    result.detections[i].symbol_name / .count / .coordinates[j].x / .y
"""
from __future__ import annotations

import gc
import logging
import math
import os
import re
import shutil
import struct
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3
except ImportError:  # older installs
    import fitz

log = logging.getLogger("slicer")

Rect4 = Tuple[float, float, float, float]

# ----------------------------- hard limits ---------------------------------
MIN_DPI, MAX_DPI = 72, 300
MIN_GRID, MAX_GRID = 1, 8
MAX_OVERLAP = 0.20
MIN_PAGE_PT, MAX_PAGE_PT = 72.0, 14400.0   # 1in .. 200in per side
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
UNLABELED = "UNLABELED SYMBOL"


class SliceError(Exception):
    """Expected, user-facing failure carrying an HTTP-style status code."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SliceConfig:
    dpi: int = 300
    grid: int = 4
    overlap: float = 0.05                 # fraction of a tile cell added on each side
    max_tile_pixels: int = 30_000_000
    blank_white_fraction: float = 0.99999  # tile is "blank" if >= this share of bytes are 0xFF
    max_pages: int = 60
    # legend.png limits (header-checked only; the image is never decoded here)
    max_legend_bytes: int = 15 * 1024 * 1024
    max_legend_pixels: int = 20_000_000
    # memory guard (Render free tier = 512 MB)
    rss_soft_limit_mb: float = 430.0
    # audit overlay look
    ocg_name: str = "AI Takeoff Audit"
    marker_radius_pt: float = 12.0
    marker_stroke_width: float = 1.25
    marker_fill_opacity: float = 0.30
    marker_stroke_opacity: float = 0.90

    def validate(self) -> None:
        if not (MIN_DPI <= self.dpi <= MAX_DPI):
            raise SliceError(f"dpi must be {MIN_DPI}-{MAX_DPI}", 422)
        if not (MIN_GRID <= self.grid <= MAX_GRID):
            raise SliceError(f"grid must be {MIN_GRID}-{MAX_GRID}", 422)
        if not (0.0 <= self.overlap <= MAX_OVERLAP):
            raise SliceError(f"overlap must be 0-{MAX_OVERLAP}", 422)
        if self.max_pages < 1:
            raise SliceError("max_pages must be >= 1", 422)
        if not (2.0 <= self.marker_radius_pt <= 100.0):
            raise SliceError("marker_radius_pt must be 2-100", 422)
        for name in ("marker_fill_opacity", "marker_stroke_opacity"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise SliceError(f"{name} must be 0-1", 422)


# ------------------------------ data model ---------------------------------
@dataclass(frozen=True)
class TileSpec:
    index: int          # 0 .. grid*grid-1, row-major
    row: int
    col: int
    rect: Rect4         # rendered area incl. overlap (page points)
    core: Rect4         # non-overlapping cell that OWNS detections (page points)


@dataclass(frozen=True)
class PagePlan:
    page_index: int     # 0-based
    page_number: int    # 1-based
    page_rect: Rect4
    dpi: int
    scale: float        # pixels per point = dpi / 72
    grid: int
    overlap: float
    tiles: Tuple[TileSpec, ...]


@dataclass
class TileFile:
    spec: TileSpec
    path: str           # PNG on disk; "" when the tile is blank (no file written)
    width_px: int
    height_px: int
    origin_pt: Tuple[float, float]
    is_blank: bool
    white_fraction: float


@dataclass(frozen=True)
class GlobalDetection:
    page_number: int
    tile_index: int
    symbol_name: str
    x_pt: float
    y_pt: float


@dataclass
class TileTranslation:
    page_number: int
    tile_index: int
    detections: List[GlobalDetection] = field(default_factory=list)
    unplaced: Dict[str, int] = field(default_factory=dict)  # reported count > coordinates given
    reported_count: int = 0
    dropped_overlap: int = 0   # owned by a neighbouring tile's core
    clamped: int = 0           # coordinates outside the tile image and clamped back in


@dataclass
class PageAccumulator:
    """Per-page running totals. Attribute names match what excel_builder reads."""
    page_number: int
    label: str = ""
    quantities: Dict[str, int] = field(default_factory=dict)
    unplaced: Dict[str, int] = field(default_factory=dict)
    detections: List[GlobalDetection] = field(default_factory=list)
    tiles_total: int = 0
    tiles_analyzed: int = 0
    tiles_blank: int = 0
    tiles_failed: int = 0
    dropped_overlap: int = 0
    clamped: int = 0
    notes: List[str] = field(default_factory=list)

    def add_translation(self, tr: TileTranslation) -> None:
        for d in tr.detections:
            self.quantities[d.symbol_name] = self.quantities.get(d.symbol_name, 0) + 1
            self.detections.append(d)
        for name, n in tr.unplaced.items():
            self.unplaced[name] = self.unplaced.get(name, 0) + n
        self.dropped_overlap += tr.dropped_overlap
        self.clamped += tr.clamped
        self.tiles_analyzed += 1
        if tr.clamped:
            self.notes.append(f"tile {tr.tile_index}: {tr.clamped} coordinate(s) outside image, clamped")

    def mark_blank(self) -> None:
        self.tiles_blank += 1

    def mark_failed(self, tile_index: int, reason: str) -> None:
        self.tiles_failed += 1
        self.notes.append(f"tile {tile_index} FAILED: {reason[:160]}")

    @property
    def notes_text(self) -> str:
        return "; ".join(self.notes)


# ------------------------------ small helpers ------------------------------
def _r(values) -> Rect4:
    return tuple(round(float(v), 3) for v in values)  # type: ignore[return-value]


def rss_mb() -> float:
    """Resident memory of this process in MB (Linux). 0.0 if unavailable."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def release_memory() -> None:
    """Shrink MuPDF's internal cache and run the garbage collector."""
    try:
        fitz.TOOLS.store_shrink(100)
    except Exception:
        pass
    gc.collect()


def enforce_memory(cfg: SliceConfig, where: str = "") -> float:
    """Release memory, then abort cleanly (503) if still above the soft ceiling."""
    release_memory()
    rss = rss_mb()
    if cfg.rss_soft_limit_mb and rss > cfg.rss_soft_limit_mb:
        suffix = f" at {where}" if where else ""
        log.error("RSS %.0f MB > soft limit %.0f MB%s", rss, cfg.rss_soft_limit_mb, suffix)
        raise SliceError(
            f"Memory ceiling reached ({rss:.0f} MB){suffix}. Lower dpi or use a lighter PDF.", 503
        )
    return rss


def _safe_unlink(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        log.warning("Could not delete temp file %s", path)


def _pix_origin_px(pix: Any, fallback_pt: Tuple[float, float], scale: float) -> Tuple[int, int]:
    """Device-space pixel origin of a clipped pixmap: pt = (origin_px + px) / scale."""
    try:
        ir = pix.irect
        return int(ir.x0), int(ir.y0)
    except Exception:
        return int(math.floor(fallback_pt[0] * scale)), int(math.floor(fallback_pt[1] * scale))


def normalize_symbol_name(name: Any) -> str:
    """Canonical BOM key: collapse whitespace, uppercase, cap length."""
    s = re.sub(r"\s+", " ", str(name or "")).strip().upper()
    return s[:80] if s else UNLABELED


# --------------------------- input validation ------------------------------
def open_pdf(path: str, cfg: Optional[SliceConfig] = None) -> "fitz.Document":
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise SliceError(f"File could not be opened as a PDF: {exc}", 415)
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise SliceError("Encrypted/password-protected PDFs are not supported.", 422)
    if doc.page_count < 1:
        doc.close()
        raise SliceError("PDF has no pages.", 422)
    if cfg is not None and doc.page_count > cfg.max_pages:
        n = doc.page_count
        doc.close()
        raise SliceError(f"PDF has {n} pages; the limit is {cfg.max_pages} per run.", 413)
    return doc


def validate_legend_image(path: str, cfg: SliceConfig) -> Tuple[int, int]:
    """
    Cheap legend.png check: size cap, PNG signature, IHDR dimensions.
    The image is NOT decoded (a big screenshot would cost 60+ MB of RAM).
    Returns (width_px, height_px).
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        raise SliceError("Legend file is missing.", 400)
    if size == 0:
        raise SliceError("Legend file is empty.", 400)
    if size > cfg.max_legend_bytes:
        raise SliceError(f"Legend exceeds {cfg.max_legend_bytes // (1024 * 1024)} MB.", 413)
    with open(path, "rb") as fh:
        head = fh.read(24)
    if len(head) < 24 or head[:8] != PNG_MAGIC or head[12:16] != b"IHDR":
        raise SliceError("Legend must be a valid PNG image (legend.png).", 415)
    width, height = struct.unpack(">II", head[16:24])
    if width < 64 or height < 64:
        raise SliceError("Legend image is too small to read (min 64x64 px).", 422)
    if width * height > cfg.max_legend_pixels:
        raise SliceError(
            f"Legend is {width}x{height}px; limit is {cfg.max_legend_pixels:,} pixels. "
            "Crop or downscale the screenshot.", 413,
        )
    return width, height


def load_page(doc: "fitz.Document", page_index: int) -> "fitz.Page":
    if not (0 <= page_index < doc.page_count):
        raise SliceError(f"Page index {page_index} out of range (0-{doc.page_count - 1}).", 422)
    try:
        page = doc.load_page(page_index)
    except Exception as exc:
        raise SliceError(f"Page {page_index + 1} could not be loaded: {exc}", 422)
    r = page.rect
    if not (MIN_PAGE_PT <= r.width <= MAX_PAGE_PT and MIN_PAGE_PT <= r.height <= MAX_PAGE_PT):
        raise SliceError(
            f"Page {page_index + 1} size {r.width / 72:.1f}x{r.height / 72:.1f}in is outside the supported range.",
            422,
        )
    return page


def iter_pages(
    doc: "fitz.Document", cfg: SliceConfig
) -> Iterator[Tuple[int, Optional["fitz.Page"], Optional[SliceError]]]:
    """
    Multi-page loop:  for page_index in range(len(doc)).
    Yields (page_index, page, error). A page that fails validation yields
    (page_index, None, SliceError) instead of aborting the whole run, so one bad
    sheet never kills a 40-page set. MuPDF cache is shrunk after every page.
    """
    for page_index in range(len(doc)):
        page: Optional["fitz.Page"] = None
        err: Optional[SliceError] = None
        try:
            page = load_page(doc, page_index)
        except SliceError as exc:
            err = exc
        try:
            yield page_index, page, err
        finally:
            page = None
            release_memory()


_SHEET_RE = re.compile(r"^[A-Z]{1,3}-?\d{1,3}(?:\.\d{1,2})?[A-Z]?$")


def guess_sheet_label(page: "fitz.Page") -> str:
    """
    Best-effort sheet number (E-101, FP-2.01 ...) from the title block (bottom-right
    30% x 30% of the page). Picks the matching word with the tallest glyph box.
    Never raises; returns "" when nothing plausible is found.
    """
    try:
        r = page.rect
        clip = fitz.Rect(r.x0 + 0.70 * r.width, r.y0 + 0.70 * r.height, r.x1, r.y1)
        best, best_h = "", 0.0
        for w in page.get_text("words", clip=clip):
            token = str(w[4]).strip().upper().strip(".,:;()")
            if _SHEET_RE.match(token):
                h = float(w[3]) - float(w[1])
                if h > best_h:
                    best, best_h = token, h
        return best
    except Exception:
        return ""


# -------------------------------- planning ---------------------------------
def plan_page(page: "fitz.Page", page_index: int, cfg: SliceConfig) -> PagePlan:
    cfg.validate()
    prect = page.rect
    scale = cfg.dpi / 72.0
    tw, th = prect.width / cfg.grid, prect.height / cfg.grid
    ox, oy = tw * cfg.overlap, th * cfg.overlap

    specs: List[TileSpec] = []
    for row in range(cfg.grid):
        for col in range(cfg.grid):
            core = (
                prect.x0 + col * tw, prect.y0 + row * th,
                prect.x0 + (col + 1) * tw, prect.y0 + (row + 1) * th,
            )
            rect = (
                max(prect.x0, core[0] - ox), max(prect.y0, core[1] - oy),
                min(prect.x1, core[2] + ox), min(prect.y1, core[3] + oy),
            )
            w_px = math.ceil((rect[2] - rect[0]) * scale)
            h_px = math.ceil((rect[3] - rect[1]) * scale)
            if w_px * h_px > cfg.max_tile_pixels:
                raise SliceError(
                    f"Page {page_index + 1}: tile would be {w_px}x{h_px}px "
                    f"(> {cfg.max_tile_pixels:,}). Lower dpi or raise grid.", 413,
                )
            specs.append(TileSpec(len(specs), row, col, _r(rect), _r(core)))

    return PagePlan(
        page_index=page_index,
        page_number=page_index + 1,
        page_rect=_r((prect.x0, prect.y0, prect.x1, prect.y1)),
        dpi=cfg.dpi,
        scale=round(scale, 6),
        grid=cfg.grid,
        overlap=cfg.overlap,
        tiles=tuple(specs),
    )


# ------------------------------- rendering ---------------------------------
@contextmanager
def page_workspace(base_dir: str, page_number: int) -> Iterator[str]:
    """Per-page temp directory. Always removed on exit, followed by memory release."""
    os.makedirs(base_dir, exist_ok=True)
    path = tempfile.mkdtemp(prefix=f"p{page_number:04d}_", dir=base_dir)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        release_memory()


def render_tile_file(
    page: "fitz.Page", spec: TileSpec, plan: PagePlan, cfg: SliceConfig, workdir: str
) -> TileFile:
    clip = fitz.Rect(*spec.rect)
    pix = None
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(plan.scale, plan.scale),
            clip=clip,
            colorspace=fitz.csRGB,
            alpha=False,
        )
        width, height = int(pix.width), int(pix.height)
        ox_px, oy_px = _pix_origin_px(pix, (clip.x0, clip.y0), plan.scale)

        samples = pix.samples                      # one transient copy (~18 MB at 300 DPI)
        white = samples.count(b"\xff") / float(max(1, len(samples)))
        del samples

        is_blank = white >= cfg.blank_white_fraction
        path = ""
        if not is_blank:
            path = os.path.join(workdir, f"p{plan.page_number:04d}_t{spec.index:02d}.png")
            pix.save(path)
        return TileFile(
            spec=spec, path=path, width_px=width, height_px=height,
            origin_pt=(ox_px / plan.scale, oy_px / plan.scale),
            is_blank=is_blank, white_fraction=round(white, 7),
        )
    except SliceError:
        raise
    except Exception as exc:
        raise SliceError(
            f"Rendering failed (page {plan.page_number}, tile {spec.index}): {exc}", 500
        )
    finally:
        pix = None


def iter_page_tiles(
    page: "fitz.Page", plan: PagePlan, cfg: SliceConfig, workdir: str
) -> Iterator[TileFile]:
    """
    Yields TileFile objects one at a time. The PNG for a tile is deleted as soon as
    the consumer asks for the next tile (or stops iterating), and MuPDF's cache is
    shrunk. Blank tiles are yielded with path == "" so callers can skip the API call.
    """
    for spec in plan.tiles:
        enforce_memory(cfg, f"page {plan.page_number} tile {spec.index}")
        tile = render_tile_file(page, spec, plan, cfg, workdir)
        try:
            yield tile
        finally:
            _safe_unlink(tile.path)
            release_memory()


# ------------------------- coordinate translation --------------------------
def _in_core(x: float, y: float, spec: TileSpec, plan: PagePlan) -> bool:
    """Half-open ownership test so adjacent cores partition the page exactly."""
    x0, y0, x1, y1 = spec.core
    last_col = spec.col == plan.grid - 1
    last_row = spec.row == plan.grid - 1
    in_x = (x0 <= x < x1) or (last_col and x == x1)
    in_y = (y0 <= y < y1) or (last_row and y == y1)
    return in_x and in_y


def tile_px_to_page_pt(tile: TileFile, plan: PagePlan, x_px: float, y_px: float) -> Tuple[float, float]:
    """Tile pixel (x, y) -> global page point, clamped to the page rectangle."""
    px0, py0, px1, py1 = plan.page_rect
    gx = tile.origin_pt[0] + x_px / plan.scale
    gy = tile.origin_pt[1] + y_px / plan.scale
    return min(max(gx, px0), px1), min(max(gy, py0), py1)


def translate_tile_result(result: Any, tile: TileFile, plan: PagePlan) -> TileTranslation:
    """
    Convert one tile's vision result into global detections.
      * coordinates are the source of truth: each retained coordinate = 1 counted item
      * out-of-image pixels are clamped back in and counted in .clamped
      * detections whose centre lies outside this tile's CORE belong to a neighbour
        (overlap band) and are dropped here -> no double counting
      * if the model reported count > number of coordinates, the shortfall is recorded
        in .unplaced (not in quantities) so a human can review it
    """
    tr = TileTranslation(page_number=plan.page_number, tile_index=tile.spec.index)
    detections = getattr(result, "detections", None) or []
    for det in detections:
        name = normalize_symbol_name(getattr(det, "symbol_name", ""))
        coords = getattr(det, "coordinates", None) or []
        try:
            reported = max(0, int(getattr(det, "count", len(coords))))
        except (TypeError, ValueError):
            reported = len(coords)
        tr.reported_count += max(reported, len(coords))

        valid = 0
        for c in coords:
            try:
                x, y = float(c.x), float(c.y)
            except (AttributeError, TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            valid += 1
            cx = min(max(x, 0.0), float(tile.width_px))
            cy = min(max(y, 0.0), float(tile.height_px))
            if cx != x or cy != y:
                tr.clamped += 1
            gx, gy = tile_px_to_page_pt(tile, plan, cx, cy)
            if not _in_core(gx, gy, tile.spec, plan):
                tr.dropped_overlap += 1
                continue
            tr.detections.append(
                GlobalDetection(plan.page_number, tile.spec.index, name, round(gx, 3), round(gy, 3))
            )
        shortfall = reported - valid
        if shortfall > 0:
            tr.unplaced[name] = tr.unplaced.get(name, 0) + shortfall
    return tr


# ---------------------------- audit PDF stamper ----------------------------
def build_audit_pdf(
    src_pdf_path: str,
    detections_by_page: Mapping[int, Sequence[GlobalDetection]],
    out_path: str,
    cfg: SliceConfig,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    """
    Open the source PDF fresh (call AFTER the analysis document is closed so only one
    document is in RAM), stamp translucent red circles on a single OCG layer, and save.
    Keys of detections_by_page are 1-based page numbers. Returns stats.
    """
    cfg.validate()
    doc = open_pdf(src_pdf_path)
    total_pages = int(doc.page_count)
    stamped_pages = markers = 0
    try:
        ocg = doc.add_ocg(cfg.ocg_name, on=True)
        wanted = sorted(n for n, dets in detections_by_page.items() if dets)
        for i, page_number in enumerate(wanted, start=1):
            if not (1 <= page_number <= doc.page_count):
                log.warning("Skipping overlay for out-of-range page %s", page_number)
                continue
            page = doc.load_page(page_number - 1)
            try:
                shape = page.new_shape()
                derot = page.derotation_matrix
                for d in detections_by_page[page_number]:
                    pt = fitz.Point(d.x_pt, d.y_pt) * derot   # as-viewed -> unrotated page space
                    shape.draw_circle(pt, cfg.marker_radius_pt)
                    markers += 1
                shape.finish(
                    color=(1, 0, 0),
                    fill=(1, 0, 0),
                    width=cfg.marker_stroke_width,
                    stroke_opacity=cfg.marker_stroke_opacity,
                    fill_opacity=cfg.marker_fill_opacity,
                    oc=ocg,
                )
                shape.commit()
                stamped_pages += 1
            finally:
                shape = None
                page = None
                release_memory()
            if progress:
                progress(i, len(wanted))

        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        doc.save(out_path, garbage=1, deflate=True)
    except SliceError:
        raise
    except Exception as exc:
        raise SliceError(f"Could not build the audit PDF: {exc}", 500)
    finally:
        doc.close()
        release_memory()
    return {"pages_stamped": stamped_pages, "markers": markers, "pages_total": total_pages}
