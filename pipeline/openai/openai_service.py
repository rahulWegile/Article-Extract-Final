from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIStatusError

from pipeline.article.visual_separator import annotate_gaps

load_dotenv()


# ============================================================
# ROLE ENUM
#
# Must match the roles listed in the grouping prompts
# (each language's pipeline/languages/<language>/grouping_prompt.py) and
# pipeline/article/article_grouper.py (IGNORE_ROLES).
# ============================================================

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


class OpenAIService:
    """
    Vision + JSON extraction service backed by the OpenAI API.

    Used for:

    - Newspaper metadata extraction (first page).
    - Page-level article block classification / grouping.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    MAX_RETRIES = 5
    RETRY_DELAY = 5

    # Some models (e.g. the gpt-5.x reasoning family) reject any
    # temperature other than their default (1) with a 400. This is a
    # property of the MODEL, not of any one page/instance -- a new
    # OpenAIService is constructed per page (see run_openai), so
    # without this class-level cache every page would independently
    # re-probe and fail once before self-correcting. Shared across all
    # instances, keyed by model name, so the fact is learned once per
    # process and reused by every instance (current or future) for
    # that same model.
    _TEMPERATURE_SUPPORT_CACHE: dict[str, bool] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)

        # Article-batch/page-grouping requests attach many crop images
        # at once (dozens of MB of base64 data); a 120s timeout was
        # observed failing with a connection error on a 47-crop batch,
        # so the fallback default is raised to give large requests
        # enough time to actually upload.
        timeout_seconds = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "300"))

        self.client = OpenAI(
            api_key=self.api_key,
            timeout=timeout_seconds,
        )

        is_reasoning_model = any(
            x in (self.model or "").lower() for x in ("luna", "o1", "o3", "o4", "gpt-5")
        )
        self._temperature_supported = (
            False if is_reasoning_model else self._TEMPERATURE_SUPPORT_CACHE.get(self.model, True)
        )

        # Real token usage from the most recent successful _call(),
        # for callers that want to record actual cost (e.g.
        # page_processor_gemini.run_openai saving it alongside the
        # page's boundary-grouping response). None until a call
        # succeeds at least once.
        self.last_usage: dict[str, int] | None = None

    # ========================================================
    # IMAGE CONTENT PART
    # ========================================================

    @staticmethod
    def _image_content(image_path: str) -> dict[str, Any]:

        image_bytes = Path(image_path).read_bytes()

        suffix = Path(image_path).suffix.lower()

        mime_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        encoded = base64.b64encode(image_bytes).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            },
        }

    # ========================================================
    # JSON PARSE (defensive against stray code fences)
    # ========================================================

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:

        cleaned = text.strip()

        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        return json.loads(cleaned)

    # ========================================================
    # RETRY-AFTER (respect the API's own suggested wait, e.g. on a
    # 429 tokens-per-minute rate limit, instead of a blind fixed
    # delay that's often too short to actually clear the window)
    # ========================================================

    @staticmethod
    def _retry_after_seconds(exc: APIStatusError) -> float | None:

        try:
            header_value = exc.response.headers.get("retry-after")
        except Exception:
            return None

        if not header_value:
            return None

        try:
            return max(0.0, float(header_value)) + 1.0
        except (TypeError, ValueError):
            return None

    # ========================================================
    # CALL OPENAI (shared retry logic)
    # ========================================================

    def _call(
        self,
        contents: list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:

        last_exc: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": contents}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": schema,
                            "strict": True,
                        },
                    },
                }

                if self._temperature_supported:
                    kwargs["temperature"] = 0

                response = self.client.chat.completions.create(**kwargs)

                if response.usage is not None:
                    self.last_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                response_text = (
                    response.choices[0].message.content or ""
                ).strip()

                if not response_text:
                    raise RuntimeError("OpenAI returned an empty response.")

                return self._parse_json(response_text)

            except (APIStatusError, APIConnectionError) as exc:

                last_exc = exc

                if (
                    self._temperature_supported
                    and isinstance(exc, APIStatusError)
                    and exc.status_code == 400
                    and (
                        "temperature" in str(exc).lower()
                        or (
                            isinstance(exc.body, dict)
                            and (
                                exc.body.get("param") == "temperature"
                                or (
                                    isinstance(exc.body.get("error"), dict)
                                    and exc.body["error"].get("param") == "temperature"
                                )
                            )
                        )
                    )
                ):
                    # This model doesn't support a custom temperature at
                    # all (e.g. reasoning-family models) -- remember
                    # that (for every instance of this model, not just
                    # this one) and retry without it instead of
                    # repeating the same doomed request on every
                    # attempt/page.
                    self._temperature_supported = False
                    self._TEMPERATURE_SUPPORT_CACHE[self.model] = False
                    continue

                retryable = isinstance(exc, APIConnectionError) or getattr(
                    exc, "status_code", None
                ) in (429, 500, 502, 503, 504)

                if not retryable or attempt == self.MAX_RETRIES:
                    print(
                        f"ERROR: OpenAI call failed permanently "
                        f"(attempt {attempt}/{self.MAX_RETRIES}, "
                        f"model={self.model}): {type(exc).__name__}: {exc}"
                    )
                    print(
                        f"  status_code={getattr(exc, 'status_code', None)} "
                        f"body={getattr(exc, 'body', None)}"
                    )
                    raise

                delay = self._retry_after_seconds(exc) or self.RETRY_DELAY

                print(
                    f"WARNING: OpenAI unavailable (attempt {attempt}/{self.MAX_RETRIES}): {exc}"
                )
                print(f"  Retrying in {delay} seconds...")

                time.sleep(delay)

        # Unreachable, but keeps type-checkers happy.
        raise last_exc  # type: ignore[misc]

    # ========================================================
    # ANALYZE IMAGE (newspaper metadata)
    # ========================================================

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
    ) -> dict[str, Any]:

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "newspaper_name": {"type": "string"},
                "publish_date": {"type": "string"},
                "edition": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": [
                "newspaper_name",
                "publish_date",
                "edition",
                "language",
            ],
        }

        contents = [
            {"type": "text", "text": prompt},
            self._image_content(image_path),
        ]

        return self._call(
            contents,
            schema_name="newspaper_metadata",
            schema=schema,
        )

    # ========================================================
    # COMPACT BLOCKS PAYLOAD
    # ========================================================

    @staticmethod
    def _compact_blocks_payload(
        json_path: str,
        image_path: str | None = None,
    ) -> str:
        """
        The exported page JSON carries a lot of fields the grouping
        prompt never asks for (width/height/center derived from bbox,
        is_global/is_masthead/is_page_header, ocr_confidence,
        always-null article_id/article_confidence/gemini_notes,
        verbose knowledge dict) and is pretty-printed with 4-space
        indentation. For a ~100-block page that bloats the request to
        150-200k+ characters, which both risks hitting per-minute
        token limits and appears to degrade the model's ability to
        keep block-to-article assignment exclusive. Keep only what
        the prompt documents as input ("id, class, type, bbox,
        reading_order, OCR text, column, confidence, optional
        knowledge") and serialize it compactly.

        Also annotates each block with gap_above/gap_below (blank
        vertical distance to its nearest same-column neighbour) and,
        when `image_path` is given, has_rule_line_above/
        has_rule_line_below (a printed rule line or colored box edge
        probed directly from the page's own pixels) plus the page's
        median_block_gap for scale -- see
        pipeline.article.visual_separator.annotate_gaps.

        This was previously computed only on the Gemini path
        (gemini_service.py), which left every OpenAI-routed language
        -- Urdu among them -- without the single strongest
        separation signal on a boxed/ruled newspaper layout: printed
        story borders. Confirmed on a real Urdu (THE INQUILAB) page,
        every story is boxed in exactly this way, and the model
        analyzing that page never received evidence of it.
        """

        with open(json_path, "r", encoding="utf-8") as f:
            page = json.load(f)

        page_image = None

        if image_path is not None:
            try:
                page_image = cv2.imread(str(image_path))
            except Exception:
                page_image = None

        compact_blocks = []

        for block in page.get("blocks", []):

            knowledge = block.get("knowledge") or {}

            compact_blocks.append(
                {
                    "id": block.get("id"),
                    "class": block.get("class"),
                    "type": block.get("type"),
                    "bbox": block.get("bbox"),
                    "reading_order": block.get("reading_order"),
                    "column": block.get("column"),
                    "text": block.get("text", ""),
                    "confidence": block.get("ocr_confidence", 0.0),
                    "category": knowledge.get("category") or None,
                }
            )

        median_gap = annotate_gaps(compact_blocks, page_image)

        compact_page = {
            "page": page.get("page"),
            "page_width": page.get("page_width"),
            "page_height": page.get("page_height"),
            "median_block_gap": median_gap,
            "blocks": compact_blocks,
        }

        return json.dumps(compact_page, ensure_ascii=False, separators=(",", ":"))

    # ========================================================
    # ANALYZE PAGE (article block classification / grouping)
    # ========================================================

    def analyze_page(
        self,
        image_path: str,
        json_path: str,
        prompt: str,
    ) -> dict[str, Any]:

        page_json_text = self._compact_blocks_payload(json_path, image_path)

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
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
                        "additionalProperties": False,
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
            {"type": "text", "text": prompt},
            {
                "type": "text",
                "text": (
                    "DETECTED LAYOUT BLOCKS (JSON):\n" + page_json_text
                ),
            },
            self._image_content(image_path),
        ]

        return self._call(
            contents,
            schema_name="article_group_extraction",
            schema=schema,
        )
