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

        compact_page = {
            "page": page.get("page"),
            "page_width": page.get("page_width"),
            "page_height": page.get("page_height"),
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
