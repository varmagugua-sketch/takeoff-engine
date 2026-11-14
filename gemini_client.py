import logging
import os
import time
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

log = logging.getLogger("gemini_client")

# Strictly fast multimodal Flash models (no slow Pro models that hang background workers)
PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]


class SymbolCoordinate(BaseModel):
    """Exact center pixel location of a detected symbol inside a grid tile."""
    x: int = Field(description="Center X pixel coordinate of the symbol on this grid tile.")
    y: int = Field(description="Center Y pixel coordinate of the symbol on this grid tile.")


class SymbolDetection(BaseModel):
    """Count and locations for a single symbol type matching the agency's legend."""
    symbol_name: str = Field(description="Exact label or symbol description read from the provided Legend image.")
    count: int = Field(description="Total instances of this exact symbol found on this grid tile.")
    coordinates: List[SymbolCoordinate] = Field(
        default_factory=list,
        description="List of center X/Y pixel coordinates for every detected instance."
    )


class TileAnalysisResult(BaseModel):
    """Master structured output returned for one 4x4 matrix tile."""
    page_number: int = Field(description="1-based page number from the blueprint PDF.")
    tile_index: int = Field(description="Grid tile index (0 to 15).")
    detections: List[SymbolDetection] = Field(
        default_factory=list,
        description="All symbols from the Legend image detected inside this specific tile."
    )


class GeminiTakeoffClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY environment variable.")

        # 60,000 ms (60s) hard HTTP timeout so a stalled connection never freezes the job!
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta", timeout=60_000),
        )
        self._candidates = list(PREFERRED_MODELS)
        self.model_id = self._candidates[0]
        log.info("Initialized Gemini client (60s timeout) with pool: %s", self._candidates)

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

        system_instruction = (
            "You are a senior commercial MEP construction estimator specializing in Division 21 (Fire Sprinkler), "
            "Division 26 (Electrical), and Division 28 (Fire Alarm) quantity takeoffs. "
            "You are strictly trade-blind: rely ONLY on the visual symbols shown in Image 1 (Legend Key). "
            "Never guess or count symbols that do not appear in Image 1."
        )

        prompt = f"""
        Image 1 is the official SYMBOL LEGEND KEY for this project.
        Image 2 is Blueprint Page {page_number}, Matrix Grid Tile {tile_index} (out of 16).

        EXECUTION RULES:
        1. Study every symbol and its text label shown in Image 1 (Legend Key).
        2. Scan Image 2 (Blueprint Tile) carefully for every matching symbol instance, even if rotated or near piping/wiring lines.
        3. For each matching symbol type, count the total occurrences in Image 2 and record the exact center (x, y) pixel coordinate of each symbol relative to Image 2's dimensions.
        4. If a legend symbol does not appear in Image 2, omit it or return count 0.
        5. Return strictly valid JSON matching the required schema for Page {page_number}, Tile {tile_index}.
        """

        legend_mime = "image/jpeg" if legend_bytes[:3] == b"\xff\xd8\xff" else "image/png"
        legend_part = types.Part.from_bytes(data=legend_bytes, mime_type=legend_mime)
        tile_part = types.Part.from_bytes(data=tile_bytes, mime_type="image/png")

        log.info("Analyzing Page %d Tile %d/15...", page_number, tile_index)

        last_err = None
        # Two full passes across our fast Flash pool so brief 503 spikes never stall
        for attempt_round in range(2):
            for candidate in list(self._candidates):
                try:
                    response = self.client.models.generate_content(
                        model=candidate,
                        contents=[legend_part, tile_part, prompt],
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.1,
                            response_mime_type="application/json",
                            response_schema=TileAnalysisResult,
                        ),
                    )
                    if candidate != self.model_id:
                        self.model_id = candidate
                        self._candidates.remove(candidate)
                        self._candidates.insert(0, candidate)

                    if response.parsed is not None:
                        return response.parsed
                    if response.text:
                        return TileAnalysisResult.model_validate_json(response.text)
                    return TileAnalysisResult(page_number=page_number, tile_index=tile_index, detections=[])

                except Exception as exc:
                    msg = str(exc).lower()
                    last_err = exc
                    if "404" in msg or "not found" in msg or "not supported" in msg:
                        log.warning("Model %s returned 404; dropping from pool.", candidate)
                        if len(self._candidates) > 1:
                            self._candidates.remove(candidate)
                        continue
                    log.warning("Page %d Tile %d on %s temporary error (%s); trying next...", page_number, tile_index, candidate, type(exc).__name__)
                    time.sleep(2.0)
                    continue

        raise last_err
