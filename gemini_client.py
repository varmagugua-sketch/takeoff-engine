import logging
import os
import struct
import time
from typing import List, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

log = logging.getLogger("gemini_client")

PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
]


class SymbolCoordinate(BaseModel):
    """Normalized 0-1000 center coordinate of a detected symbol inside Image 2."""
    x: int = Field(description="Horizontal center on a 0 to 1000 scale (0 = left edge of Image 2, 1000 = right edge).")
    y: int = Field(description="Vertical center on a 0 to 1000 scale (0 = top edge of Image 2, 1000 = bottom edge).")


class SymbolDetection(BaseModel):
    """Count and locations for a single symbol type matching the agency's legend."""
    symbol_name: str = Field(description="Exact label or symbol description read from the provided Legend image.")
    count: int = Field(description="Total instances of this exact symbol found on this grid tile.")
    coordinates: List[SymbolCoordinate] = Field(
        default_factory=list,
        description="List of center (x, y) coordinates on a 0-1000 normalized scale for every detected instance."
    )


class TileAnalysisResult(BaseModel):
    """Master structured output returned for one 4x4 matrix tile."""
    page_number: int = Field(description="1-based page number from the blueprint PDF.")
    tile_index: int = Field(description="Grid tile index (0 to 15).")
    detections: List[SymbolDetection] = Field(
        default_factory=list,
        description="All symbols from the Legend image detected inside this specific tile."
    )


def _read_png_dimensions(png_bytes: bytes) -> Tuple[int, int]:
    """Read exact (width_px, height_px) from PNG IHDR header."""
    if len(png_bytes) >= 24 and png_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", png_bytes[16:24])
    raise ValueError("Tile image is not a valid PNG.")


class GeminiTakeoffClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=120_000),
        )
        self._candidates = list(PREFERRED_MODELS)
        self.model_id = self._candidates[0]
        log.info("Initialized Gemini client (0-1000 precision scaling) with pool: %s", self._candidates)

    def analyze_tile_with_legend(
        self,
        tile_path: str,
        legend_path: str,
        page_number: int,
        tile_index: int
    ) -> TileAnalysisResult:
        if not os.path.exists(tile_path):
            raise FileNotFoundError(f"Grid tile not found: {tile_path}")
        if not os.path.exists(legend_path):
            raise FileNotFoundError(f"Legend image not found: {legend_path}")

        with open(legend_path, "rb") as f:
            legend_bytes = f.read()

        with open(tile_path, "rb") as f:
            tile_bytes = f.read()

        width_px, height_px = _read_png_dimensions(tile_bytes)

        system_instruction = (
            "You are a senior commercial MEP construction estimator specializing in Division 21 (Fire Sprinkler), "
            "Division 26 (Electrical), and Division 28 (Fire Alarm) quantity takeoffs. "
            "You are strictly trade-blind: rely ONLY on the visual symbols shown in Image 1 (Legend Key). "
            "Never count symbols inside a title block or legend table printed on the sheet itself—only count symbols placed on the architectural floorplan."
        )

        prompt = f"""
        Image 1 is the official SYMBOL LEGEND KEY for this project.
        Image 2 is Blueprint Page {page_number}, Matrix Grid Tile {tile_index} (out of 16).

        EXECUTION RULES:
        1. Study every symbol and its text label shown in Image 1 (Legend Key).
        2. Scan Image 2 (Blueprint Tile) carefully for every matching symbol instance on the floorplan (ignore any legend box or title block if visible in Image 2).
        3. Record the exact center (x, y) of each detected symbol using a NORMALIZED 0 to 1000 coordinate scale:
           - x = 0 is the left edge of Image 2; x = 1000 is the right edge of Image 2.
           - y = 0 is the top edge of Image 2; y = 1000 is the bottom edge of Image 2.
        4. Make sure `count` equals the exact number of coordinates returned for that symbol.
        5. If a legend symbol does not appear in Image 2, omit it.
        """

        legend_mime = "image/jpeg" if legend_bytes[:3] == b"\xff\xd8\xff" else "image/png"
        legend_part = types.Part.from_bytes(data=legend_bytes, mime_type=legend_mime)
        tile_part = types.Part.from_bytes(data=tile_bytes, mime_type="image/png")

        log.info("Analyzing Page %d Tile %d/15 (%dx%d px)...", page_number, tile_index, width_px, height_px)

        last_err = None
        for _ in range(2):
            for candidate in list(self._candidates):
                try:
                    cfg_kwargs = dict(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=TileAnalysisResult,
                    )
                    # Disable slow thinking delay on Gemini 2.5 / 3 Flash models
                    if "2.5" in candidate or "3-" in candidate or "latest" in candidate:
                        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

                    response = self.client.models.generate_content(
                        model=candidate,
                        contents=[legend_part, tile_part, prompt],
                        config=types.GenerateContentConfig(**cfg_kwargs),
                    )
                    if candidate != self.model_id:
                        self.model_id = candidate
                        self._candidates.remove(candidate)
                        self._candidates.insert(0, candidate)

                    parsed = response.parsed
                    if parsed is None and response.text:
                        parsed = TileAnalysisResult.model_validate_json(response.text)
                    if parsed is None:
                        return TileAnalysisResult(page_number=page_number, tile_index=tile_index, detections=[])

                    # Convert Gemini's native 0-1000 coordinates into exact tile pixels for slicer.py!
                    pixel_detections: List[SymbolDetection] = []
                    for det in parsed.detections:
                        px_coords: List[SymbolCoordinate] = []
                        for c in det.coordinates:
                            nx = max(0, min(1000, c.x))
                            ny = max(0, min(1000, c.y))
                            px_x = int(round((nx / 1000.0) * (width_px - 1)))
                            px_y = int(round((ny / 1000.0) * (height_px - 1)))
                            px_coords.append(SymbolCoordinate(x=px_x, y=px_y))
                        if px_coords:
                            pixel_detections.append(
                                SymbolDetection(
                                    symbol_name=det.symbol_name,
                                    count=len(px_coords),
                                    coordinates=px_coords,
                                )
                            )

                    return TileAnalysisResult(
                        page_number=page_number,
                        tile_index=tile_index,
                        detections=pixel_detections,
                    )

                except Exception as exc:
                    msg = str(exc).lower()
                    last_err = exc
                    if "404" in msg or "not found" in msg or "not supported" in msg:
                        if len(self._candidates) > 1:
                            self._candidates.remove(candidate)
                        continue
                    log.warning("Page %d Tile %d on %s error (%s); trying next...", page_number, tile_index, candidate, type(exc).__name__)
                    time.sleep(2.0)
                    continue

        raise last_err
