import json
import os
import time
from typing import Any

import httpx

from google import genai
from google.genai import types
from google.genai.errors import APIError

# Must match the roles listed in pipeline/gemini/gemini_prompt.py
# (ARTICLE_GROUP_PROMPT) and pipeline/article/article_grouper.py
# (IGNORE_ROLES) -- constraining the schema to a different taxonomy
# than the prompt documents silently corrupts page-level grouping.
_BLOCK_ROLES = [
    "article_title",
    "article_text",
    "article_image",
    "caption",
    "byline",
    "advertisement",
    "comic",
    "weather",
    "teaser_box",
    "utility_box",
    "masthead",
    "section_header",
    "page_header",
    "page_footer",
    "page_number",
    "logo",
    "decoration",
    "unknown",
]

class GeminiService:
    MAX_RETRIES = 5
    RETRY_DELAY = 10.0

    def __init__(self, model: str | None = None):
        self.api_key = os.getenv("GEMINI_API_KEY")

        timeout_seconds = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,
            ),
        )
        self.model = model or os.getenv("GEMINI_BOUNDARY_MODEL", "gemini-3.6-flash")

    def _image_content(self, image_path: str) -> types.Part:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

    def _call(
        self,
        contents: list[Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        
        last_exc: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                
                response_text = (response.text or "").strip()
                if not response_text:
                    raise RuntimeError("Gemini returned an empty response.")
                
                return json.loads(response_text)

            except (APIError, httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                retryable = (
                    isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))
                    or getattr(exc, "code", None) in (429, 500, 502, 503, 504)
                )

                if not retryable or attempt == self.MAX_RETRIES:
                    raise

                print(f"WARNING: Gemini unavailable (attempt {attempt}/{self.MAX_RETRIES}): {exc}")
                print(f"  Retrying in {self.RETRY_DELAY} seconds...")
                time.sleep(self.RETRY_DELAY)

        raise last_exc  # type: ignore[misc]

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
    ) -> dict[str, Any]:

        schema = {
            "type": "object",
            "properties": {
                "newspaper_name": {"type": "string"},
                "publish_date": {"type": "string"},
                "edition": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["newspaper_name", "publish_date", "edition", "language"],
        }

        contents = [
            types.Part.from_text(text=prompt),
            self._image_content(image_path),
        ]

        return self._call(
            contents,
            schema_name="newspaper_metadata",
            schema=schema,
        )

    # ========================================================
    # COLUMN-NEIGHBOUR GAP MEASUREMENT
    #
    # Ported from pipeline/openai/openai_service.py -- kept
    # byte-for-byte identical so both engines give the grouping
    # prompt the same separation signal. Without this, a boxed
    # sidebar separated from its neighbour by less blank space than
    # an ordinary paragraph gap (confirmed on a real page: as little
    # as 17px, below that page's own 23px median) has no way to be
    # recognized as a new article, and its body text gets silently
    # absorbed into whichever story is above it.
    # ========================================================

    @staticmethod
    def _has_intervening_block(
        boxes: list[dict[str, Any] | None],
        self_index: int,
        y_start: float,
        y_end: float,
        bbox: dict[str, Any],
    ) -> bool:
        """
        True when some OTHER block's y-range falls inside (y_start,
        y_end) and it has ANY horizontal overlap with `bbox` at all --
        not just the stricter 50% same-column overlap used to find a
        gap neighbour. Such a block sits visually inside the measured
        gap, so the gap is not clean whitespace.
        """

        lo, hi = min(y_start, y_end), max(y_start, y_end)

        for other_index, other in enumerate(boxes):

            if other is None or other_index == self_index:
                continue

            if other["y1"] >= hi or other["y2"] <= lo:
                continue

            overlap = min(bbox["x2"], other["x2"]) - max(
                bbox["x1"], other["x1"]
            )

            if overlap > 0:
                return True

        return False

    @staticmethod
    def _annotate_gaps(compact_blocks: list[dict[str, Any]]) -> float | None:
        """
        Annotate each block with the whitespace gap to its nearest
        neighbour above and below within the same print column, plus
        return the page's median gap for scale.

        Newspapers separate stories with visibly more whitespace than
        they put between paragraphs of the same story, so the size of
        a gap relative to the page's normal gap is one of the
        strongest available separation signals. The model previously
        received only raw bounding boxes and had to re-derive this
        itself for every block, which it did inconsistently on dense
        pages.

        Column neighbours are found by horizontal overlap rather than
        by the 6-slot column index, because a wide block (a spanning
        headline, a photo) belongs to several index slots at once.
        """

        boxes = []

        for block in compact_blocks:

            bbox = block.get("bbox") or {}

            if None in (
                bbox.get("x1"),
                bbox.get("y1"),
                bbox.get("x2"),
                bbox.get("y2"),
            ):
                boxes.append(None)
                continue

            boxes.append(bbox)

        gaps: list[float] = []

        for index, bbox in enumerate(boxes):

            if bbox is None:
                continue

            width = max(1.0, float(bbox["x2"] - bbox["x1"]))

            gap_above = None
            gap_below = None
            above_neighbor_y = None
            below_neighbor_y = None

            for other_index, other in enumerate(boxes):

                if other is None or other_index == index:
                    continue

                other_width = max(1.0, float(other["x2"] - other["x1"]))

                overlap = min(bbox["x2"], other["x2"]) - max(
                    bbox["x1"], other["x1"]
                )

                # Same print column only.
                if max(0.0, overlap) / min(width, other_width) < 0.5:
                    continue

                if other["y2"] <= bbox["y1"]:
                    distance = float(bbox["y1"] - other["y2"])
                    if gap_above is None or distance < gap_above:
                        gap_above = distance
                        above_neighbor_y = other["y2"]

                elif bbox["y2"] <= other["y1"]:
                    distance = float(other["y1"] - bbox["y2"])
                    if gap_below is None or distance < gap_below:
                        gap_below = distance
                        below_neighbor_y = other["y1"]

            # The same-column search above can skip straight past some
            # OTHER block sitting visually in the gap (e.g. an inset
            # photo of a different width, which fails the 50%
            # column-overlap test and gets silently passed over) and
            # report the distance across it as if it were plain
            # whitespace. That number then gets handed to the model as
            # a "this looks like a new article" signal even though the
            # gap is actually occupied by unrelated content, not empty.
            # When any block -- regardless of column overlap -- sits
            # inside the measured span, the measurement isn't trustworthy
            # as a whitespace gap, so drop it rather than report a
            # number that can be silently wrong.
            if gap_above is not None and GeminiService._has_intervening_block(
                boxes, index, above_neighbor_y, bbox["y1"], bbox
            ):
                gap_above = None

            if gap_below is not None and GeminiService._has_intervening_block(
                boxes, index, bbox["y2"], below_neighbor_y, bbox
            ):
                gap_below = None

            compact_blocks[index]["gap_above"] = (
                None if gap_above is None else round(gap_above)
            )

            compact_blocks[index]["gap_below"] = (
                None if gap_below is None else round(gap_below)
            )

            for value in (gap_above, gap_below):
                if value is not None and value > 0:
                    gaps.append(value)

        if not gaps:
            return None

        gaps.sort()

        middle = len(gaps) // 2

        if len(gaps) % 2:
            median = gaps[middle]
        else:
            median = (gaps[middle - 1] + gaps[middle]) / 2.0

        return round(median, 1)

    @staticmethod
    def _compact_blocks_payload(json_path: str) -> str:
        with open(json_path, "r", encoding="utf-8") as f:
            page = json.load(f)

        compact_blocks = []
        for block in page.get("blocks", []):
            knowledge = block.get("knowledge") or {}
            compact_blocks.append({
                "id": block.get("id"),
                "class": block.get("class"),
                "type": block.get("type"),
                "bbox": block.get("bbox"),
                "reading_order": block.get("reading_order"),
                "column": block.get("column"),
                "text": block.get("text", ""),
                "confidence": block.get("ocr_confidence", 0.0),
                "category": knowledge.get("category") or None,
            })

        median_gap = GeminiService._annotate_gaps(compact_blocks)

        compact_page = {
            "page": page.get("page"),
            "page_width": page.get("page_width"),
            "page_height": page.get("page_height"),
            # Scale reference for gap_above/gap_below: the typical
            # spacing between adjacent blocks on THIS page, so the
            # model can tell an article break from a paragraph break
            # without hardcoding pixel thresholds that vary by
            # publication, page size and render DPI.
            "median_block_gap": median_gap,
            "blocks": compact_blocks,
        }

        return json.dumps(compact_page, ensure_ascii=False, separators=(",", ":"))

    def analyze_page(
        self,
        image_path: str,
        json_path: str,
        prompt: str,
    ) -> dict[str, Any]:

        page_json_text = self._compact_blocks_payload(json_path)

        schema = {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "role": {
                                "type": "string",
                                "enum": _BLOCK_ROLES,
                            },
                        },
                        "required": ["id", "role"],
                    },
                },
                "articles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "article_id": {"type": "integer"},
                            "blocks": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["article_id", "blocks"],
                    },
                },
            },
            "required": ["blocks", "articles"],
        }

        contents = [
            types.Part.from_text(text=prompt),
            types.Part.from_text(text="DETECTED LAYOUT BLOCKS (JSON):\n" + page_json_text),
            self._image_content(image_path),
        ]

        return self._call(
            contents,
            schema_name="article_group_extraction",
            schema=schema,
        )
