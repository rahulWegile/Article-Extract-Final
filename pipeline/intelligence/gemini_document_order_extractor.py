"""
Gemini variant of Stream B (see document_order_extractor.py for the
full design rationale).

Same interface and behavior as DocumentOrderExtractor, routed to
Gemini instead of OpenAI. This exists because gpt-5.6-luna cannot
read Devanagari (confirmed elsewhere this session -- it fabricates
unrelated text on real Hindi crops), so a Hindi/Devanagari document
must use Gemini for Stream B too, exactly as it already does for
Stream A's boundary detection and extraction
(backend/services/pipeline_service.py, "document_llm_provider").

Failures here must never break the main pipeline -- extract_batch()
returns [] rather than raising on any failure, same as the OpenAI
variant.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiDocumentOrderExtractor:

    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        pages_per_batch: int = 2,
    ):

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.model = (
            model
            or os.getenv(
                "GEMINI_DOC_ORDER_MODEL",
                os.getenv(
                    "GEMINI_ARTICLE_MODEL",
                    self.DEFAULT_MODEL,
                ),
            )
        )

        self.pages_per_batch = max(
            1,
            int(pages_per_batch),
        )

        timeout_seconds = int(
            os.getenv(
                "GEMINI_TIMEOUT_SECONDS",
                "120",
            )
        )

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=timeout_seconds * 1000,
            ),
        )

        # Same rationale as the OpenAI variant -- keep well under
        # the provider's total-request image-size limit.
        self.max_batch_image_bytes = (
            int(
                os.getenv(
                    "GEMINI_MAX_BATCH_IMAGE_MB",
                    "30",
                )
            )
            * 1024
            * 1024
        )

    # ========================================================
    # RESPONSE SCHEMA (Gemini's dialect: uppercase type names)
    # ========================================================

    @staticmethod
    def _response_schema() -> dict[str, Any]:

        return {
            "type": "OBJECT",
            "properties": {
                "sections": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page": {
                                "type": "INTEGER",
                            },
                            "heading": {
                                "type": "STRING",
                                "nullable": True,
                            },
                            "text": {
                                "type": "STRING",
                            },
                        },
                        "required": [
                            "page",
                            "heading",
                            "text",
                        ],
                    },
                },
            },
            "required": ["sections"],
        }

    # ========================================================
    # PROMPT
    # ========================================================

    @staticmethod
    def _build_prompt(page_numbers: list[int]) -> str:

        pages_desc = ", ".join(
            str(p) for p in page_numbers
        )

        return (
            "You are reading raw newspaper page images "
            f"(pages {pages_desc}) directly. You have NO other "
            "information about article boundaries or layout "
            "detection -- you are the ONLY source for this task.\n\n"
            "Read each page in natural human reading order (top to "
            "bottom, column by column, left to right across "
            "columns) and break the content into sections. Start a "
            "new section at every visible heading/headline, AND "
            "also whenever a block of body text has NO heading of "
            "its own -- e.g. it continues a story from elsewhere on "
            "the page, or overflows from the bottom of one "
            "page/column to the top of the next.\n\n"
            "For every section return:\n"
            "- page: the page number "
            f"(one of [{pages_desc}]) this section's text appears "
            "on\n"
            "- heading: the exact visible heading/headline text for "
            "this section, or null if this section has NO visible "
            "heading of its own (a heading-less continuation)\n"
            "- text: the COMPLETE verbatim text of this section. "
            "Never summarize, paraphrase, or shorten it. Never "
            "invent text that is not actually printed on the "
            "page.\n\n"
            "Do not try to group sections into articles and do not "
            "guess which sections belong together across pages -- "
            "just report what you see, per page, in reading order."
        )

    # ========================================================
    # EXTRACT ONE BATCH
    # ========================================================

    def extract_batch(
        self,
        page_paths: list[Path],
        page_numbers: list[int],
    ) -> list[dict[str, Any]]:

        page_paths = [Path(p) for p in page_paths]

        total_bytes = sum(
            p.stat().st_size
            for p in page_paths
            if p.exists()
        )

        if (
            len(page_paths) > 1
            and total_bytes > self.max_batch_image_bytes
        ):

            sections: list[dict[str, Any]] = []

            for page_path, page_number in zip(
                page_paths,
                page_numbers,
            ):

                sections.extend(
                    self.extract_batch(
                        [page_path],
                        [page_number],
                    )
                )

            return sections

        contents: list[Any] = [
            types.Part.from_text(
                text=self._build_prompt(page_numbers)
            ),
        ]

        for page_path in page_paths:

            contents.append(
                types.Part.from_bytes(
                    data=page_path.read_bytes(),
                    mime_type="image/png",
                )
            )

        MAX_RETRIES = 5
        RETRY_DELAY = 5

        response = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0,
                            response_mime_type=(
                                "application/json"
                            ),
                            response_schema=(
                                self._response_schema()
                            ),
                        ),
                    )
                )

                break

            except Exception as exc:

                error_text = str(exc)

                is_timeout = isinstance(
                    exc,
                    (httpx.TimeoutException, httpx.ConnectError),
                )

                retryable = (
                    is_timeout
                    or "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "unavailable" in error_text
                )

                if not retryable or attempt == MAX_RETRIES:

                    print(
                        f"⚠ Document-order extraction (Gemini) "
                        f"failed for pages {page_numbers}: {exc}"
                    )

                    return []

                time.sleep(RETRY_DELAY)

        if response is None:
            return []

        response_text = (
            response.text or ""
        ).strip()

        if not response_text:
            return []

        try:

            parsed = json.loads(
                response_text
            )

        except json.JSONDecodeError:
            return []

        sections = parsed.get(
            "sections",
            [],
        )

        if not isinstance(sections, list):
            return []

        return sections

    # ========================================================
    # SUBMIT ALL BATCHES FOR A DOCUMENT
    # ========================================================

    def submit_batches(
        self,
        executor,
        pages: list[Path],
    ) -> list[Any]:

        futures = []

        for start in range(
            0,
            len(pages),
            self.pages_per_batch,
        ):

            batch_paths = pages[
                start:
                start + self.pages_per_batch
            ]

            batch_page_numbers = list(
                range(
                    start + 1,
                    start + 1 + len(batch_paths),
                )
            )

            futures.append(
                executor.submit(
                    self.extract_batch,
                    batch_paths,
                    batch_page_numbers,
                )
            )

        return futures
