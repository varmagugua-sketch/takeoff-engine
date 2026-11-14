import os
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


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

        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-pro"

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

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[legend_part, tile_part, prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=TileAnalysisResult,
            ),
        )

        return response.parsed
