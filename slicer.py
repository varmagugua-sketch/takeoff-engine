"""
slicer.py - Trade-blind canvas slicing engine (PyMuPDF + OpenCV).

Pipeline role (Day 1b):
  1. locate the sheet's native SYMBOL LEGEND box          -> find_legend / render_legend
  2. tile the drawing into an NxN grid at up to 300 DPI   -> plan_tiles / iter_tiles
  3. white-out the legend area inside tiles so legend symbols are never counted

Memory strategy (Render free tier = 512 MB):
  * Never render the full 300 DPI canvas (~233 MB for 36x24in). Every tile is
    rendered straight from the vector PDF via a clip rectangle (~15 MB peak).
  * Tiles are yielded one at a time; arrays are deleted and MuPDF's internal
    cache is shrunk after each tile; RSS is checked against a soft ceiling.

Coordinate contract (used by the Day 4 OCG overlay):
    page_x_pt = tile.origin_pt[0] + pixel_x / scale_px_per_pt
    page_y_pt = tile.origin_pt[1] + pixel_y / scale_px_per_pt
Points = 1/72 inch, origin top-left of the page.
"""
from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

log = logging.getLogger("slicer")

Rect4 = Tuple[float, float, float, float]

# ----------------------------- hard limits ---------------------------------
MIN_DPI, MAX_DPI = 72, 300
MIN_GRID, MAX_GRID = 1, 8
MAX_OVERLAP = 0.20
MIN_PAGE_PT, MAX_PAGE_PT = 72.0, 14400.0  # 1in .. 200in per side
PROBE_DPI = 36            # low-res pass for content detection
LEGEND_PROBE_DPI = 100    # window render used to find the legend border
LEGEND_MAX_HITS = 6       # keyword hits examined per keyword

# Most specific first. Matching is case-insensitive.
LEGEND_KEYWORDS = (
    "SYMBOL LEGEND", "SYMBOLS LEGEND", "ELECTRICAL LEGEND", "ELECTRICAL SYMBOLS",
    "LEGEND OF SYMBOLS", "SYMBOL LIST", "LEGEND", "SYMBOLS",
)
# Legend box must be 0.4% .. 35% of the page area to be believable.
LEGEND_MIN_AREA_FRAC, LEGEND_MAX_AREA_FRAC = 0.004, 0.35


class SliceError(Exception):
    """Expected, user-facing failure with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SliceConfig:
    page_index: int = 0
    dpi: int = 300
    grid: int = 4
    overlap: float = 0.05
    crop_to_content: bool = True
    blank_ink_threshold: float = 0.0003
    max_tile_pixels: int = 30_000_000
    # --- legend (Context Lock) ---
    detect_legend: bool = True
    mask_legend: bool = True                    # white-out legend inside tiles
    legend_rect_pt: Optional[Rect4] = None      # manual override (x0,y0,x1,y1) in points
    legend_dpi: int = 300
    max_legend_pixels: int = 20_000_000
    # --- memory guard ---
    rss_soft_limit_mb: float = 430.0            # abort cleanly before Render's 512 MB OOM kill

    def validate(self) -> None:
        if not (MIN_DPI <= self.dpi <= MAX_DPI):
            raise SliceError(f"dpi must be {MIN_DPI}-{MAX_DPI}", 422)
        if not (MIN_GRID <= self.grid <= MAX_GRID):
            raise SliceError(f"grid must be {MIN_GRID}-{MAX_GRID}", 422)
        if not (0.0 <= self.overlap <= MAX_OVERLAP):
            raise SliceError(f"overlap must be 0-{MAX_OVERLAP}", 422)
        if self.page_index < 0:
            raise SliceError("page must be >= 0", 422)
        if not (MIN_DPI <= self.legend_dpi <= MAX_DPI):
            raise SliceError(f"legend_dpi must be {MIN_DPI}-{MAX_DPI}", 422)


@dataclass(frozen=True)
class LegendInfo:
    rect: Rect4
    method: str        # manual | keyword_box | keyword_column_fallback
    confidence: str    # high | medium | low
    keyword: Optional[str] = None


@dataclass(frozen=True)
class TileSpec:
    index: int
    row: int
    col: int
    rect: Rect4        # rendered area incl. overlap (page points)
    core: Rect4        # non-overlapping core cell (page points)
    touches_legend: bool = False


@dataclass
class TileResult:
    spec: TileSpec
    width_px: int
    height_px: int
    ink_ratio: float
    is_blank: bool
    origin_pt: Tuple[float, float] = (0.0, 0.0)
    rss_mb: float = 0.0
    png: bytes = field(default=b"", repr=False)


@dataclass
class LegendResult:
    info: LegendInfo
    width_px: int
    height_px: int
    dpi: int
    png: bytes = field(repr=False)


@dataclass
class PagePlan:
    page_index: int
    page_rect: Rect4
    region: Rect4
    dpi: int
    scale: float
    grid: int
    overlap: float
    tiles: List[TileSpec]
    legend: Optional[LegendInfo] = None
    mask_legend: bool = True


# ------------------------------ helpers ------------------------------------
def _pix_to_ndarray(pix: fitz.Pixmap) -> np.ndarray:
    """Pixmap -> HxWxN uint8 array, honouring row stride. Returns a copy."""
    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = buf.reshape(pix.height, pix.stride)[:, : pix.width * pix.n]
    return arr.reshape(pix.height, pix.width, pix.n).copy()


def _pix_origin_px(pix: fitz.Pixmap, fallback_pt: Tuple[float, float], scale: float) -> Tuple[int, int]:
    """Device-space (pixel) origin of the pixmap, so pt = (origin + px) / scale."""
    try:
        ir = pix.irect
        return int(ir.x0), int(ir.y0)
    except Exception:
        return int(math.floor(fallback_pt[0] * scale)), int(math.floor(fallback_pt[1] * scale))


def _r(rect) -> Rect4:
    return tuple(round(float(v), 3) for v in rect)  # type: ignore[return-value]


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


def _release_memory() -> None:
    """Drop MuPDF's render cache and run the GC. Call after every heavy render."""
    try:
        fitz.TOOLS.store_shrink(100)
    except Exception:
        pass
    gc.collect()


def _intersects(a: Rect4, b: Rect4) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


# ------------------------------ PDF access ---------------------------------
def open_pdf(path: str) -> fitz.Document:
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
    return doc


def load_page(doc: fitz.Document, page_index: int) -> fitz.Page:
    if page_index >= doc.page_count:
        raise SliceError(
            f"page={page_index} out of range (PDF has {doc.page_count} pages, 0-based).", 422
        )
    page = doc.load_page(page_index)
    r = page.rect
    if not (MIN_PAGE_PT <= r.width <= MAX_PAGE_PT and MIN_PAGE_PT <= r.height <= MAX_PAGE_PT):
        raise SliceError(
            f"Page size {r.width / 72:.1f}x{r.height / 72:.1f}in is outside supported range.", 422
        )
    return page


# --------------------------- content detection -----------------------------
def detect_content_rect(page: fitz.Page, margin_pt: float = 18.0) -> Rect4:
    """
    Cheap 36 DPI grayscale pass + OpenCV to find where ink actually is, so tiles
    are not wasted on blank sheet margins. Falls back to the full page.
    """
    prect = page.rect
    full: Rect4 = (prect.x0, prect.y0, prect.x1, prect.y1)
    scale = PROBE_DPI / 72.0
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False
        )
        gray = _pix_to_ndarray(pix)[:, :, 0]
        del pix
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        pts = cv2.findNonZero(bw)
        if pts is None:
            return full
        x, y, w, h = cv2.boundingRect(pts)
        x0 = max(prect.x0, prect.x0 + x / scale - margin_pt)
        y0 = max(prect.y0, prect.y0 + y / scale - margin_pt)
        x1 = min(prect.x1, prect.x0 + (x + w) / scale + margin_pt)
        y1 = min(prect.y1, prect.y0 + (y + h) / scale + margin_pt)
        if (x1 - x0) * (y1 - y0) < 0.25 * prect.width * prect.height:
            log.warning("Content bbox <25%% of page; using full page.")
            return full
        return (x0, y0, x1, y1)
    except Exception:
        log.exception("Content detection failed; using full page.")
        return full
    finally:
        _release_memory()


# ---------------------------- legend isolation -----------------------------
def _edge_ink(bw: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Fraction of the rectangle's 4 edges (3px bands) covered by ink. 1.0 = solid box."""
    H, W = bw.shape
    x2, y2 = min(W, x + w), min(H, y + h)
    top = bw[y:min(H, y + 3), x:x2].any(axis=0).mean()
    bot = bw[max(0, y2 - 3):y2, x:x2].any(axis=0).mean()
    lef = bw[y:y2, x:min(W, x + 3)].any(axis=1).mean()
    rig = bw[y:y2, max(0, x2 - 3):x2].any(axis=1).mean()
    return float(min(top, bot, lef, rig))


def _box_around_hit(page: fitz.Page, hit: fitz.Rect) -> Optional[Rect4]:
    """
    Find the smallest rectangular border enclosing a keyword hit. Only a window
    around the hit is rendered (bounded RAM), at 100 DPI, then analysed with
    OpenCV contours. Window grows once if the border runs off the edge.
    """
    prect = page.rect
    page_area = prect.width * prect.height
    scale = LEGEND_PROBE_DPI / 72.0
    cx, cy = (hit.x0 + hit.x1) / 2.0, (hit.y0 + hit.y1) / 2.0

    for frac in (0.25, 0.45):
        ww, wh = prect.width * frac, prect.height * frac
        win = fitz.Rect(
            max(prect.x0, cx - ww / 2), max(prect.y0, cy - wh / 2),
            min(prect.x1, cx + ww / 2), min(prect.y1, cy + wh / 2),
        )
        pix = None
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale), clip=win, colorspace=fitz.csGRAY, alpha=False
            )
            ox, oy = _pix_origin_px(pix, (win.x0, win.y0), scale)
            gray = _pix_to_ndarray(pix)[:, :, 0]
        finally:
            pix = None
        H, W = gray.shape
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _h = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        hx, hy = cx * scale - ox, cy * scale - oy
        hw_px, hh_px = hit.width * scale, hit.height * scale
        best: Optional[Tuple[float, Rect4]] = None
        touched_edge = False

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if not (x <= hx <= x + w and y <= hy <= y + h):
                continue
            if w < 3 * hw_px or h < 5 * hh_px:
                continue                                  # too small: letters / sub-boxes
            area_frac = (w * h) / (scale * scale) / page_area
            if not (LEGEND_MIN_AREA_FRAC <= area_frac <= LEGEND_MAX_AREA_FRAC):
                continue
            if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                touched_edge = True                       # box may extend beyond window
                continue
            if _edge_ink(bw, x, y, w, h) < 0.85:
                continue                                  # not a closed rectangle outline
            rect = ((ox + x) / scale, (oy + y) / scale, (ox + x + w) / scale, (oy + y + h) / scale)
            if best is None or area_frac < best[0]:
                best = (area_frac, rect)

        del gray, bw, contours
        _release_memory()
        if best is not None:
            return _r(best[1])
        if not touched_edge:
            return None
    return None


def find_legend(page: fitz.Page, cfg: SliceConfig) -> Optional[LegendInfo]:
    """
    Locate the sheet's native symbol legend. Never raises; returns None if not found.
      1. manual override (cfg.legend_rect_pt)
      2. text search for LEGEND keywords + OpenCV rectangle isolation (vector PDFs)
      3. low-confidence fallback: column of space below/right of the keyword
    Scanned (image-only) PDFs have no text layer -> returns None; the caller should
    then require cfg.legend_rect_pt (see PLACEHOLDER notes at bottom of file).
    """
    prect = page.rect
    try:
        if cfg.legend_rect_pt is not None:
            x0, y0, x1, y1 = (float(v) for v in cfg.legend_rect_pt)
            x0, y0, x1, y1 = max(prect.x0, x0), max(prect.y0, y0), min(prect.x1, x1), min(prect.y1, y1)
            if x1 - x0 < 36 or y1 - y0 < 36:
                raise SliceError("legend_rect_pt is invalid or off-page.", 422)
            return LegendInfo(_r((x0, y0, x1, y1)), "manual", "high", None)

        if not cfg.detect_legend:
            return None

        first_hit: Optional[Tuple[str, fitz.Rect]] = None
        for kw in LEGEND_KEYWORDS:
            for hit in page.search_for(kw)[:LEGEND_MAX_HITS]:
                if first_hit is None:
                    first_hit = (kw, hit)
                box = _box_around_hit(page, hit)
                if box is not None:
                    return LegendInfo(box, "keyword_box", "high", kw)

        if first_hit is not None:
            kw, hit = first_hit
            rect = (
                max(prect.x0, hit.x0 - 12), max(prect.y0, hit.y0 - 12),
                min(prect.x1, hit.x0 + 0.22 * prect.width),
                min(prect.y1, hit.y0 + 0.35 * prect.height),
            )
            return LegendInfo(_r(rect), "keyword_column_fallback", "low", kw)
        return None
    except SliceError:
        raise
    except Exception:
        log.exception("Legend detection failed; continuing without legend.")
        return None
    finally:
        _release_memory()


def render_legend(page: fitz.Page, info: LegendInfo, cfg: SliceConfig) -> LegendResult:
    """
    Render the legend crop (PNG) for the Gemini 'Context Lock' call.
    DPI is lowered automatically if the crop would exceed cfg.max_legend_pixels.
    """
    clip = fitz.Rect(*info.rect)
    dpi = float(cfg.legend_dpi)
    scale = dpi / 72.0
    px = math.ceil(clip.width * scale) * math.ceil(clip.height * scale)
    if px > cfg.max_legend_pixels:
        scale *= math.sqrt(cfg.max_legend_pixels / px)
        dpi = scale * 72.0
    pix = None
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), clip=clip, colorspace=fitz.csRGB, alpha=False
        )
        rgb = _pix_to_ndarray(pix)
    finally:
        pix = None
    h, w = rgb.shape[:2]
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR, dst=rgb)
    ok, enc = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    del rgb
    _release_memory()
    if not ok:
        raise SliceError("PNG encoding failed for legend crop.", 500)
    return LegendResult(info, w, h, int(round(dpi)), enc.tobytes())


# ------------------------------- planning ----------------------------------
def plan_tiles(page: fitz.Page, page_index: int, cfg: SliceConfig) -> PagePlan:
    cfg.validate()
    prect = page.rect
    region = detect_content_rect(page) if cfg.crop_to_content else (
        prect.x0, prect.y0, prect.x1, prect.y1
    )
    legend = find_legend(page, cfg)
    legend_rect = legend.rect if legend else None

    rx0, ry0, rx1, ry1 = region
    tw, th = (rx1 - rx0) / cfg.grid, (ry1 - ry0) / cfg.grid
    ox, oy = tw * cfg.overlap, th * cfg.overlap
    scale = cfg.dpi / 72.0

    specs: List[TileSpec] = []
    for row in range(cfg.grid):
        for col in range(cfg.grid):
            core = (rx0 + col * tw, ry0 + row * th, rx0 + (col + 1) * tw, ry0 + (row + 1) * th)
            rect = (
                max(prect.x0, core[0] - ox), max(prect.y0, core[1] - oy),
                min(prect.x1, core[2] + ox), min(prect.y1, core[3] + oy),
            )
            w_px = math.ceil((rect[2] - rect[0]) * scale)
            h_px = math.ceil((rect[3] - rect[1]) * scale)
            if w_px * h_px > cfg.max_tile_pixels:
                raise SliceError(
                    f"Tile would be {w_px}x{h_px}px (> {cfg.max_tile_pixels:,} px cap). "
                    "Increase grid or lower dpi.", 413,
                )
            touches = bool(legend_rect and _intersects(rect, legend_rect))
            specs.append(TileSpec(len(specs), row, col, _r(rect), _r(core), touches))

    return PagePlan(
        page_index=page_index,
        page_rect=_r((prect.x0, prect.y0, prect.x1, prect.y1)),
        region=_r(region),
        dpi=cfg.dpi,
        scale=round(scale, 6),
        grid=cfg.grid,
        overlap=cfg.overlap,
        tiles=specs,
        legend=legend,
        mask_legend=cfg.mask_legend,
    )


# ------------------------------- rendering ---------------------------------
def render_tile(page: fitz.Page, spec: TileSpec, plan: PagePlan, cfg: SliceConfig) -> TileResult:
    clip = fitz.Rect(*spec.rect)
    pix = None
    try:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(plan.scale, plan.scale),
            clip=clip,
            colorspace=fitz.csRGB,
            alpha=False,
        )
        ox_px, oy_px = _pix_origin_px(pix, (clip.x0, clip.y0), plan.scale)
        rgb = _pix_to_ndarray(pix)
    finally:
        pix = None  # free MuPDF buffer immediately; only the numpy copy lives on

    h, w = rgb.shape[:2]

    # Legend white-out: legend symbols must never be counted as plan instances.
    if plan.mask_legend and plan.legend and spec.touches_legend:
        lx0, ly0, lx1, ly1 = plan.legend.rect
        x0 = max(0, int(math.floor(lx0 * plan.scale)) - ox_px)
        y0 = max(0, int(math.floor(ly0 * plan.scale)) - oy_px)
        x1 = min(w, int(math.ceil(lx1 * plan.scale)) - ox_px)
        y1 = min(h, int(math.ceil(ly1 * plan.scale)) - oy_px)
        if x1 > x0 and y1 > y0:
            rgb[y0:y1, x0:x1] = 255

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = float(np.count_nonzero(gray < 200)) / float(gray.size)
    del gray
    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR, dst=rgb)  # in-place: no second 15 MB buffer
    ok, enc = cv2.imencode(".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    del rgb
    if not ok:
        raise SliceError(f"PNG encoding failed for tile {spec.index}.", 500)

    return TileResult(
        spec=spec, width_px=w, height_px=h, ink_ratio=round(ink, 6),
        is_blank=ink < cfg.blank_ink_threshold,
        origin_pt=(round(ox_px / plan.scale, 4), round(oy_px / plan.scale, 4)),
        png=enc.tobytes(),
    )


def iter_tiles(page: fitz.Page, plan: PagePlan, cfg: SliceConfig) -> Iterator[TileResult]:
    """
    Yields one tile at a time. After each tile: MuPDF cache shrunk, GC run, RSS
    checked. If RSS still exceeds the soft ceiling the job aborts with a clean 503
    instead of letting Render's OOM-killer take the whole container down.
    """
    for spec in plan.tiles:
        result = render_tile(page, spec, plan, cfg)
        _release_memory()
        result.rss_mb = round(rss_mb(), 1)
        if cfg.rss_soft_limit_mb and result.rss_mb > cfg.rss_soft_limit_mb:
            log.error("RSS %.0f MB > soft limit %.0f MB at tile %d",
                      result.rss_mb, cfg.rss_soft_limit_mb, spec.index)
            raise SliceError(
                f"Memory ceiling reached ({result.rss_mb:.0f} MB). "
                "Lower dpi or raise grid for this sheet.", 503,
            )
        yield result


# ------------------------------- manifest ----------------------------------
def plan_header(plan: PagePlan) -> dict:
    legend = None
    if plan.legend:
        legend = {
            "rect_pt": plan.legend.rect,
            "method": plan.legend.method,
            "confidence": plan.legend.confidence,
            "keyword": plan.legend.keyword,
            "masked_in_tiles": plan.mask_legend,
        }
    return {
        "page_index": plan.page_index,
        "page_rect_pt": plan.page_rect,
        "content_region_pt": plan.region,
        "dpi": plan.dpi,
        "scale_px_per_pt": plan.scale,
        "grid": f"{plan.grid}x{plan.grid}",
        "overlap": plan.overlap,
        "legend": legend,
        "coordinate_rule": "page_pt = tile.origin_pt + pixel_xy / scale_px_per_pt",
    }


def tile_meta(t: TileResult, filename: str | None = None) -> dict:
    d = {
        "index": t.spec.index,
        "row": t.spec.row,
        "col": t.spec.col,
        "rect_pt": t.spec.rect,
        "core_pt": t.spec.core,
        "origin_pt": t.origin_pt,
        "width_px": t.width_px,
        "height_px": t.height_px,
        "ink_ratio": t.ink_ratio,
        "is_blank": t.is_blank,
        "touches_legend": t.spec.touches_legend,
        "png_bytes": len(t.png),
        "rss_mb": t.rss_mb,
    }
    if filename:
        d["file"] = filename
    return d


# ===========================================================================
# PLACEHOLDERS (wired in later days; intentionally not implemented here)
# ---------------------------------------------------------------------------
# Day 2  gemini_client.py : learn_legend(legend_png) -> JSON array of
#        {symbol_id, description, visual_features}; then per tile
#        count_tile(tile_png, legend_array) with a strict response_schema.
# Day 3  de-duplication of detections inside overlap bands using
#        TileSpec.core (keep a hit only if it falls inside its own tile's core).
# Day 4  overlay.py : convert (tile.origin_pt, pixel_xy, scale) -> page points and
#        draw translucent red circles on an OCG layer with PyMuPDF.
# Scanned PDFs: no text layer -> find_legend returns None; expose legend_rect_pt
#        as a query param (or n8n field) so the freelancer can supply the box.
# ===========================================================================
