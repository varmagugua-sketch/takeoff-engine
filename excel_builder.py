"""
excel_builder.py - Floor-by-floor Bill of Materials workbook (openpyxl).

Input : a sequence of per-page accumulators (slicer.PageAccumulator) - read by duck
        typing, so this module does not import slicer:
            .page_number  int (1-based)
            .label        str  (sheet number such as "E-101"; may be "")
            .quantities   {SYMBOL NAME: placed count}
            .unplaced     {SYMBOL NAME: count reported without a map location}
            .notes_text   str  (falls back to .notes list/str if absent)
            .tiles_analyzed / .tiles_blank / .tiles_failed   (optional ints)

Output: BOM_Ledger.xlsx with
    "Grand Total"  one row per symbol across the whole set (+ grand total row)
    "Matrix"       symbol x page grid (floor-by-floor at a glance)
    "Page Index"   per-page status, item counts, tile stats, review notes
    "P001 E-101"   one detail tab per page that has items (qty + unplaced columns)

Quantities are PLACED detections only (each one has a red marker in the audit PDF).
Counts the model reported without coordinates are shown separately as
"Unplaced (verify)" and are NOT added to Qty.

With live_formulas=True (default) totals are real Excel formulas, so when the
estimator edits a quantity on a page tab, Matrix and Grand Total update.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

log = logging.getLogger("excel_builder")

DEFAULT_FILENAME = "BOM_Ledger.xlsx"
DEFAULT_DISCLAIMER = (
    "Quantities are machine-generated from the supplied drawings and symbol key. Verify every "
    "count against the Visual Audit PDF before use. 'Unplaced' counts were reported without a "
    "map location and are NOT included in Qty."
)

HEADER_ROW = 5
FIRST_DATA_ROW = 6
SUMIF_LAST_ROW = 2000          # SUMIF range end on page tabs (leaves room for manual edits)
MAX_SYMBOL_LEN = 80

# ------------------------------- styling -----------------------------------
_NAVY = "1F3864"
_THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
HEADER_FILL = PatternFill("solid", fgColor=_NAVY)
BAND_FILL = PatternFill("solid", fgColor="F2F5FA")
TOTAL_FILL = PatternFill("solid", fgColor="D9E2F3")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
BAD_FILL = PatternFill("solid", fgColor="F8CBAD")
TITLE_FONT = Font(name="Calibri", size=16, bold=True, color=_NAVY)
SUB_FONT = Font(name="Calibri", size=10, color="595959")
NOTE_FONT = Font(name="Calibri", size=9, italic=True, color="7F7F7F")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Calibri", size=11)
BOLD_FONT = Font(name="Calibri", size=11, bold=True)
LINK_FONT = Font(name="Calibri", size=11, color="0563C1", underline="single")
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
RIGHT = Alignment(horizontal="right", vertical="center")


class BomError(Exception):
    """Raised for unusable input or a failed save."""


@dataclass
class BomSummary:
    path: str
    sheets: List[str] = field(default_factory=list)
    pages_total: int = 0
    pages_with_items: int = 0
    symbols: int = 0
    grand_total_qty: int = 0
    grand_total_unplaced: int = 0


@dataclass
class _Page:
    number: int
    label: str
    qty: Dict[str, int]
    unp: Dict[str, int]
    tiles_analyzed: int
    tiles_blank: int
    tiles_failed: int
    notes: str
    tab: str = ""
    total_row: int = 0

    @property
    def has_items(self) -> bool:
        return bool(self.qty or self.unp)

    @property
    def symbols(self) -> List[str]:
        return sorted(set(self.qty) | set(self.unp))


# ------------------------------- cleaning ----------------------------------
def _clean_symbol(name: Any) -> str:
    """
    Safe, canonical BOM key. Removes characters that break Excel criteria matching
    (* ? ~), control characters, and leading =,+,-,@,<,> that Excel could read as
    formulas or operators.
    """
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(name or ""))
    s = s.replace("*", "x").replace("~", "-").replace("?", "")
    s = re.sub(r"\s+", " ", s).strip().lstrip("=+-@<>").strip().upper()
    return s[:MAX_SYMBOL_LEN] if s else "UNLABELED SYMBOL"


def _clean_label(label: Any) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(label or ""))
    s = re.sub(r"\s+", " ", s).strip().lstrip("=+-@").strip()
    return s[:40]


def _clean_counts(src: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not src:
        return out
    try:
        items = src.items()
    except AttributeError:
        return out
    for k, v in items:
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        key = _clean_symbol(k)
        out[key] = out.get(key, 0) + n
    return out


def _normalize_pages(pages: Sequence[Any]) -> List[_Page]:
    out: Dict[int, _Page] = {}
    for p in pages or []:
        if p is None:
            continue
        try:
            number = int(getattr(p, "page_number"))
        except (AttributeError, TypeError, ValueError):
            raise BomError("Each page record needs an integer .page_number.")
        if number < 1:
            raise BomError(f"Invalid page number {number}.")
        if number in out:
            raise BomError(f"Duplicate page number {number}.")
        notes = getattr(p, "notes_text", None)
        if notes is None:
            raw = getattr(p, "notes", "")
            notes = "; ".join(map(str, raw)) if isinstance(raw, (list, tuple)) else str(raw or "")

        def _int(attr: str) -> int:
            try:
                return max(0, int(getattr(p, attr, 0) or 0))
            except (TypeError, ValueError):
                return 0

        out[number] = _Page(
            number=number,
            label=_clean_label(getattr(p, "label", "")),
            qty=_clean_counts(getattr(p, "quantities", None)),
            unp=_clean_counts(getattr(p, "unplaced", None)),
            tiles_analyzed=_int("tiles_analyzed"),
            tiles_blank=_int("tiles_blank"),
            tiles_failed=_int("tiles_failed"),
            notes=str(notes or "")[:500],
        )
    return [out[k] for k in sorted(out)]


def _tab_name(page: _Page, used: Set[str]) -> str:
    base = f"P{page.number:03d}" + (f" {page.label}" if page.label else "")
    base = re.sub(r"[\[\]:*?/\\']", "", base).strip()[:31] or f"P{page.number:03d}"
    name, n = base, 2
    while name.lower() in used:
        suffix = f"_{n}"
        name = base[: 31 - len(suffix)] + suffix
        n += 1
    used.add(name.lower())
    return name


# ------------------------------ cell helpers -------------------------------
def _put(ws, row: int, col: int, value: Any, *, font=BODY_FONT, fill=None,
         align=LEFT, fmt: Optional[str] = None, border: bool = True):
    cell = ws.cell(row=row, column=col, value=value)
    if isinstance(value, str) and not value.startswith("="):
        cell.data_type = "s"          # never let text be interpreted as a formula
    cell.font = font
    if fill is not None:
        cell.fill = fill
    cell.alignment = align
    if fmt:
        cell.number_format = fmt
    if border:
        cell.border = BORDER
    return cell


def _title_block(ws, ncols: int, title: str, project: str, prepared_by: str, disclaimer: str) -> None:
    ncols = max(ncols, 2)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    c = ws.cell(row=1, column=1, value=title)
    c.font, c.alignment = TITLE_FONT, Alignment(vertical="center")
    ws.row_dimensions[1].height = 26

    bits = []
    if project:
        bits.append(f"Project: {project}")
    if prepared_by:
        bits.append(f"Prepared by: {prepared_by}")
    bits.append("Generated: " + _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    c = ws.cell(row=2, column=1, value="   |   ".join(bits))
    c.font, c.alignment = SUB_FONT, Alignment(vertical="center")

    ws.merge_cells(start_row=3, start_column=1, end_row=3, end_column=ncols)
    c = ws.cell(row=3, column=1, value=disclaimer)
    c.font, c.alignment = NOTE_FONT, Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[3].height = 40


def _header(ws, row: int, labels: Sequence[str]) -> None:
    for i, text in enumerate(labels, start=1):
        _put(ws, row, i, text, font=HEADER_FONT, fill=HEADER_FILL, align=CENTER)
    ws.row_dimensions[row].height = 32


def _autosize(ws, first_row: int, min_w: int = 8, max_w: int = 60) -> None:
    widths: Dict[int, int] = {}
    for row in ws.iter_rows(min_row=first_row):
        for cell in row:
            v = cell.value
            if v is None:
                continue
            text = "12345678" if isinstance(v, str) and v.startswith("=") else str(v)
            longest = max((len(part) for part in text.split("\n")), default=0)
            widths[cell.column] = max(widths.get(cell.column, 0), longest)
    for col, w in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = max(min_w, min(max_w, w + 3))


def _print_setup(ws) -> None:
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.print_title_rows = f"{HEADER_ROW}:{HEADER_ROW}"


def _q(sheet: str) -> str:
    return "'" + sheet.replace("'", "") + "'"


# -------------------------------- builder ----------------------------------
def build_bom_workbook(
    pages: Sequence[Any],
    output_path: str,
    *,
    project_name: str = "",
    prepared_by: str = "",
    disclaimer: str = DEFAULT_DISCLAIMER,
    live_formulas: bool = True,
) -> BomSummary:
    data = _normalize_pages(pages)
    if not data:
        raise BomError("No pages supplied to the BOM builder.")

    item_pages = [p for p in data if p.has_items]
    used: Set[str] = {"grand total", "matrix", "page index"}
    for p in item_pages:
        p.tab = _tab_name(p, used)
        p.total_row = FIRST_DATA_ROW + len(p.symbols)

    symbols = sorted({s for p in item_pages for s in p.symbols})
    sym_total = {s: sum(p.qty.get(s, 0) for p in item_pages) for s in symbols}
    sym_unp = {s: sum(p.unp.get(s, 0) for p in item_pages) for s in symbols}
    grand_qty, grand_unp = sum(sym_total.values()), sum(sym_unp.values())

    n_pg = len(item_pages)
    first_pg_col = 2
    last_pg_col = first_pg_col + n_pg - 1           # == 1 when there are no pages
    matrix_total_col = first_pg_col + n_pg
    last_pg_letter = get_column_letter(max(last_pg_col, first_pg_col))
    total_letter = get_column_letter(matrix_total_col)

    wb = Workbook()
    ws_gt = wb.active
    ws_gt.title = "Grand Total"
    ws_mx = wb.create_sheet("Matrix")
    ws_ix = wb.create_sheet("Page Index")
    page_ws = {p.number: wb.create_sheet(p.tab) for p in item_pages}

    ws_gt.sheet_properties.tabColor = _NAVY
    ws_mx.sheet_properties.tabColor = "2F5597"
    ws_ix.sheet_properties.tabColor = "7F7F7F"
    for ws in page_ws.values():
        ws.sheet_properties.tabColor = "8EA9DB"

    # ---------------------------- Grand Total ------------------------------
    _title_block(ws_gt, 5, "BILL OF MATERIALS - GRAND TOTAL", project_name, prepared_by, disclaimer)
    _header(ws_gt, HEADER_ROW, ["Item", "Symbol", "Total Qty", "Unplaced (verify)", "Pages Found"])
    if symbols:
        for i, s in enumerate(symbols):
            r = FIRST_DATA_ROW + i
            fill = BAND_FILL if i % 2 else None
            _put(ws_gt, r, 1, i + 1, fill=fill, align=CENTER)
            _put(ws_gt, r, 2, s, fill=fill)
            if live_formulas:
                _put(ws_gt, r, 3, f"=Matrix!{total_letter}{r}", fill=fill, align=CENTER, fmt="#,##0")
                pages_found = (f'=COUNTIF(Matrix!B{r}:{last_pg_letter}{r},">0")' if n_pg else 0)
            else:
                _put(ws_gt, r, 3, sym_total[s], fill=fill, align=CENTER, fmt="#,##0")
                pages_found = sum(1 for p in item_pages if p.qty.get(s, 0) > 0)
            _put(ws_gt, r, 4, sym_unp[s], fill=WARN_FILL if sym_unp[s] else fill, align=CENTER, fmt="#,##0")
            _put(ws_gt, r, 5, pages_found, fill=fill, align=CENTER)
        last = FIRST_DATA_ROW + len(symbols) - 1
        tr = last + 1
        _put(ws_gt, tr, 1, "", fill=TOTAL_FILL)
        _put(ws_gt, tr, 2, "GRAND TOTAL", font=BOLD_FONT, fill=TOTAL_FILL)
        _put(ws_gt, tr, 3, f"=SUM(C{FIRST_DATA_ROW}:C{last})" if live_formulas else grand_qty,
             font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
        _put(ws_gt, tr, 4, f"=SUM(D{FIRST_DATA_ROW}:D{last})" if live_formulas else grand_unp,
             font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
        _put(ws_gt, tr, 5, "", fill=TOTAL_FILL)
        ws_gt.auto_filter.ref = f"A{HEADER_ROW}:E{last}"
    else:
        ws_gt.merge_cells(start_row=FIRST_DATA_ROW, start_column=1, end_row=FIRST_DATA_ROW, end_column=5)
        _put(ws_gt, FIRST_DATA_ROW, 1,
             "No symbols were detected on any page. See 'Page Index' for tile status and notes.",
             font=BOLD_FONT, fill=WARN_FILL)
    ws_gt.freeze_panes = f"A{FIRST_DATA_ROW}"

    # ------------------------------- Matrix --------------------------------
    _title_block(ws_mx, max(matrix_total_col, 3), "FLOOR-BY-FLOOR MATRIX (SYMBOL x PAGE)",
                 project_name, prepared_by, "Each page column reads its detail tab; edit quantities there.")
    _header(ws_mx, HEADER_ROW, ["Symbol"]
            + [f"P{p.number}" + (f"\n{p.label}" if p.label else "") for p in item_pages] + ["TOTAL"])
    ws_mx.row_dimensions[HEADER_ROW].height = 36
    for i, s in enumerate(symbols):
        r = FIRST_DATA_ROW + i
        fill = BAND_FILL if i % 2 else None
        _put(ws_mx, r, 1, s, fill=fill)
        for j, p in enumerate(item_pages):
            col = first_pg_col + j
            if live_formulas:
                q = _q(p.tab)
                val: Any = (f"=SUMIF({q}!$B${FIRST_DATA_ROW}:$B${SUMIF_LAST_ROW},$A{r},"
                            f"{q}!$C${FIRST_DATA_ROW}:$C${SUMIF_LAST_ROW})")
            else:
                val = p.qty.get(s, 0)
            _put(ws_mx, r, col, val, fill=fill, align=CENTER, fmt="#,##0;-#,##0;;@")
        _put(ws_mx, r, matrix_total_col,
             (f"=SUM(B{r}:{last_pg_letter}{r})" if (live_formulas and n_pg) else sym_total[s]),
             font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
    if symbols:
        last = FIRST_DATA_ROW + len(symbols) - 1
        tr = last + 1
        _put(ws_mx, tr, 1, "TOTAL", font=BOLD_FONT, fill=TOTAL_FILL)
        for j, p in enumerate(item_pages):
            col = first_pg_col + j
            L = get_column_letter(col)
            _put(ws_mx, tr, col,
                 f"=SUM({L}{FIRST_DATA_ROW}:{L}{last})" if live_formulas else sum(p.qty.values()),
                 font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
        _put(ws_mx, tr, matrix_total_col,
             f"=SUM({total_letter}{FIRST_DATA_ROW}:{total_letter}{last})" if live_formulas else grand_qty,
             font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
    ws_mx.freeze_panes = f"B{FIRST_DATA_ROW}"

    # ---------------------------- Page detail tabs -------------------------
    for p in item_pages:
        ws = page_ws[p.number]
        title = f"PAGE {p.number}" + (f" - {p.label}" if p.label else "")
        _title_block(ws, 4, title, project_name, prepared_by, disclaimer)
        _header(ws, HEADER_ROW, ["Item", "Symbol", "Qty", "Unplaced (verify)"])
        syms = p.symbols
        for i, s in enumerate(syms):
            r = FIRST_DATA_ROW + i
            fill = BAND_FILL if i % 2 else None
            _put(ws, r, 1, i + 1, fill=fill, align=CENTER)
            _put(ws, r, 2, s, fill=fill)
            _put(ws, r, 3, p.qty.get(s, 0), fill=fill, align=CENTER, fmt="#,##0")
            u = p.unp.get(s, 0)
            _put(ws, r, 4, u, fill=WARN_FILL if u else fill, align=CENTER, fmt="#,##0")
        last = FIRST_DATA_ROW + len(syms) - 1
        tr = p.total_row
        _put(ws, tr, 1, "", fill=TOTAL_FILL)
        _put(ws, tr, 2, "PAGE TOTAL", font=BOLD_FONT, fill=TOTAL_FILL)
        _put(ws, tr, 3, f"=SUM(C{FIRST_DATA_ROW}:C{last})" if live_formulas else sum(p.qty.values()),
             font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
        _put(ws, tr, 4, f"=SUM(D{FIRST_DATA_ROW}:D{last})" if live_formulas else sum(p.unp.values()),
             font=BOLD_FONT, fill=TOTAL_FILL, align=CENTER, fmt="#,##0")
        ws.auto_filter.ref = f"A{HEADER_ROW}:D{last}"
        ws.freeze_panes = f"A{FIRST_DATA_ROW}"

    # ------------------------------ Page Index -----------------------------
    _title_block(ws_ix, 9, "PAGE INDEX & PROCESSING STATUS", project_name, prepared_by,
                 "Amber = items need verification (unplaced counts). Red = one or more tiles failed to process.")
    _header(ws_ix, HEADER_ROW, ["Page", "Sheet", "Detail Tab", "Items", "Unplaced",
                                "Tiles Analyzed", "Tiles Blank", "Tiles Failed", "Review Notes"])
    for i, p in enumerate(data):
        r = FIRST_DATA_ROW + i
        row_fill = BAD_FILL if p.tiles_failed else (WARN_FILL if sum(p.unp.values()) else (BAND_FILL if i % 2 else None))
        _put(ws_ix, r, 1, p.number, fill=row_fill, align=CENTER)
        _put(ws_ix, r, 2, p.label or "-", fill=row_fill, align=CENTER)
        if p.tab:
            link = _put(ws_ix, r, 3, p.tab, font=LINK_FONT, fill=row_fill)
            link.hyperlink = f"#{_q(p.tab)}!A1"
            items: Any = f"={_q(p.tab)}!C{p.total_row}" if live_formulas else sum(p.qty.values())
        else:
            _put(ws_ix, r, 3, "- (no items)", fill=row_fill)
            items = 0
        _put(ws_ix, r, 4, items, fill=row_fill, align=CENTER, fmt="#,##0")
        _put(ws_ix, r, 5, sum(p.unp.values()), fill=row_fill, align=CENTER, fmt="#,##0")
        _put(ws_ix, r, 6, p.tiles_analyzed, fill=row_fill, align=CENTER)
        _put(ws_ix, r, 7, p.tiles_blank, fill=row_fill, align=CENTER)
        _put(ws_ix, r, 8, p.tiles_failed, fill=row_fill, align=CENTER)
        _put(ws_ix, r, 9, p.notes, fill=row_fill)
    ws_ix.freeze_panes = f"A{FIRST_DATA_ROW}"
    ws_ix.auto_filter.ref = f"A{HEADER_ROW}:I{FIRST_DATA_ROW + len(data) - 1}"

    # -------------------------- sizing, print, save -------------------------
    for ws in [ws_gt, ws_mx, ws_ix, *page_ws.values()]:
        _autosize(ws, HEADER_ROW)
        _print_setup(ws)
    ws_ix.column_dimensions["I"].width = 70
    ws_gt.column_dimensions["B"].width = max(ws_gt.column_dimensions["B"].width or 0, 34)
    ws_mx.column_dimensions["A"].width = max(ws_mx.column_dimensions["A"].width or 0, 34)
    for ws in page_ws.values():
        ws.column_dimensions["B"].width = max(ws.column_dimensions["B"].width or 0, 34)

    wb.properties.title = "Bill of Materials"
    wb.properties.creator = prepared_by or "Takeoff"

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        wb.save(output_path)
    except Exception as exc:
        raise BomError(f"Could not save workbook: {exc}")
    finally:
        wb.close()

    return BomSummary(
        path=output_path,
        sheets=["Grand Total", "Matrix", "Page Index"] + [p.tab for p in item_pages],
        pages_total=len(data),
        pages_with_items=len(item_pages),
        symbols=len(symbols),
        grand_total_qty=grand_qty,
        grand_total_unplaced=grand_unp,
    )
