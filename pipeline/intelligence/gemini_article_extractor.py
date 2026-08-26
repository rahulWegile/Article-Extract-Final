from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


class GeminiArticleExtractor:
    """
    Direct Gemini Vision extraction from FINAL VERIFIED
    newspaper article crops.

    FINAL ARCHITECTURE
    ------------------

    FINAL VERIFIED ARTICLE CROPS
                ↓
         3 PAGE BATCH
                ↓
          ONE GEMINI CALL
                ↓
        ┌───────────────┐
        │               │
        ↓               ↓
    Article data   Continuation
                       │
                       ↓
              Same-batch target?
                 │          │
                YES         NO
                 │           │
                 ↓           ↓
               MERGE      PENDING
                             │
                             ↓
                     Next 3-page batch
                             │
                             ↓
                      Gemini resolves
                             │
                             ↓
                    Final logical articles

    IMPORTANT
    ---------

    - No EasyOCR.
    - No OCRCleaner.
    - No page-level OCR.
    - No boundary modification.
    - No separate author/date/image/knowledge API request.
    - One Gemini request per page batch performs article extraction,
      structured knowledge extraction, image detection, and
      continuation resolution.
    - Gemini sees verified article crop images.
    - For pending cross-batch continuations, Gemini also receives
      the original source article crop in the same request that
      analyzes the target batch.
    - Gemini handles semantic continuation resolution.
    - Local validation prevents invented article IDs and rejects continuation links involving non-article crops.
    """

    DEFAULT_MODEL = "gemini-3.6-flash"
    DEFAULT_PAGES_PER_BATCH = 3

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        pages_per_batch: int = 3,
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
                "GEMINI_ARTICLE_MODEL",
                self.DEFAULT_MODEL,
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

        print()
        print("=" * 60)
        print("GEMINI ARTICLE EXTRACTOR")
        print("=" * 60)

        print(
            f"Model           : {self.model}"
        )

        print(
            f"Pages per batch : "
            f"{self.pages_per_batch}"
        )

        print(
            "Mode            : "
            "DIRECT IMAGE → JSON"
        )

        print(
            "OCR             : DISABLED"
        )

        print(
            "Continuation    : "
            "GEMINI SAME-BATCH + PENDING"
        )

        print("=" * 60)
        print()

    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    def process_document(
        self,
        document_dir: str | Path,
    ) -> list[dict[str, Any]]:

        document_dir = Path(
            document_dir
        )

        crop_root = (
            document_dir
            / "final_articles_crops"
        )

        if not crop_root.exists():
            raise FileNotFoundError(
                "Final article crop directory "
                f"does not exist:\n{crop_root}"
            )

        pages = self._discover_pages(
            crop_root
        )

        if not pages:

            print(
                "No final article crop pages found."
            )

            return []

        # ----------------------------------------------------
        # Discovery
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("FINAL ARTICLE CROP DISCOVERY")
        print("=" * 60)

        print(
            f"Crop root : {crop_root}"
        )

        print(
            f"Pages     : {len(pages)}"
        )

        page_inventory = {}

        for page_number in pages:

            articles = (
                self._discover_articles(
                    crop_root,
                    page_number,
                )
            )

            page_inventory[
                page_number
            ] = articles

            print(
                f"Page {page_number:03d} : "
                f"{len(articles)} articles"
            )

        print("=" * 60)
        print()

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_root = (
            document_dir
            / "gemini_article_batches"
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Page batches
        # ----------------------------------------------------

        batches = (
            self._make_page_batches(
                pages,
                page_inventory,
            )
        )

        print(
            f"Gemini API requests required: "
            f"{len(batches)}"
        )

        print()

        # ====================================================
        # IMPORTANT
        #
        # pending_continuations contains source articles
        # whose target page has NOT been processed yet.
        #
        # Example:
        #
        # Page 1 article_005
        #     ↓
        # Page 7
        #
        # Batch 1:
        #     PENDING_EXTERNAL
        #
        # Batch 2:
        #     Pages 6-10
        #     ↓
        #     Page 7 available
        #     ↓
        #     Gemini resolves it
        # ====================================================

        pending_continuations = []

        # All extracted article records.
        all_articles = []

        # All confirmed continuation edges.
        all_links = []

        # Batch results.
        results = []

        # ----------------------------------------------------
        # Process batches sequentially
        # ----------------------------------------------------

        for batch_index, batch_pages in enumerate(
            batches,
            start=1,
        ):

            result = self.process_batch(
                document_dir=document_dir,
                page_numbers=batch_pages,
                batch_index=batch_index,
                output_root=output_root,
                pending_continuations=(
                    pending_continuations
                ),
                known_pages=pages,
            )

            results.append(
                result
            )

            # -----------------------------------------------
            # Add current articles
            # -----------------------------------------------

            batch_articles = result.get(
                "articles",
                [],
            )

            all_articles.extend(
                batch_articles
            )

            # -----------------------------------------------
            # Add confirmed links
            # -----------------------------------------------

            batch_links = result.get(
                "continuation_links",
                [],
            )

            all_links.extend(
                batch_links
            )

            # -----------------------------------------------
            # Replace pending queue
            # -----------------------------------------------

            pending_continuations = (
                result.get(
                    "pending_continuations",
                    [],
                )
            )

            # -----------------------------------------------
            # Save current pending queue
            # -----------------------------------------------

            pending_path = (
                output_root
                / "pending_continuations.json"
            )

            with open(
                pending_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        "pending_count":
                            len(
                                pending_continuations
                            ),
                        "pending":
                            pending_continuations,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            print()
            print(
                "=" * 60
            )

            print(
                f"BATCH {batch_index} COMPLETE"
            )

            print(
                f"Articles           : "
                f"{len(batch_articles)}"
            )

            print(
                f"Confirmed links    : "
                f"{len(batch_links)}"
            )

            print(
                f"Pending            : "
                f"{len(pending_continuations)}"
            )

            print(
                "=" * 60
            )
            print()

        # ====================================================
        # FINAL TEXT-BASED CONTINUATION REPAIR
        # ====================================================
        #
        # Gemini can correctly detect that a story continues but may
        # fail to identify the target when the continuation crop has
        # no repeated headline/title.  At this point every page has
        # already been processed, so we can compare the actual Gemini-
        # extracted article text of physical crops.
        #
        # This step NEVER changes crop boundaries and NEVER invents
        # article IDs.  It only creates/replaces continuation links
        # between verified physical article records.
        # ====================================================

        (
            all_links,
            pending_continuations,
            repair_report,
        ) = self._repair_titleless_continuations(
            articles=all_articles,
            continuation_links=all_links,
            pending_continuations=pending_continuations,
        )

        print(
            "Text continuation repair: "
            f"added={repair_report['added']}, "
            f"corrected={repair_report['corrected']}, "
            f"unresolved={repair_report['unresolved']}"
        )

        # ====================================================
        # BUILD FINAL LOGICAL ARTICLES
        # ====================================================

        logical_articles = (
            self._build_logical_articles(
                articles=all_articles,
                continuation_links=all_links,
                pending_continuations=(
                    pending_continuations
                ),
            )
        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        final_output = {
            "schema_version":
                "gemini_article_pipeline_v6_text_repair",

            "document_dir":
                str(document_dir),

            "model":
                self.model,

            "pages_per_batch":
                self.pages_per_batch,

            "article_count":
                len(all_articles),

            "continuation_link_count":
                len(all_links),

            "pending_count":
                len(pending_continuations),

            "logical_article_count":
                len(logical_articles),

            "continuation_repair":
                repair_report,

            "articles":
                all_articles,

            "continuation_links":
                all_links,

            "pending_continuations":
                pending_continuations,

            "logical_articles":
                logical_articles,
        }

        final_path = (
            output_root
            / "final_logical_articles.json"
        )

        with open(
            final_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                final_output,
                f,
                indent=4,
                ensure_ascii=False,
            )

        # ====================================================
        # MANIFEST
        # ====================================================

        manifest = {
            "document_dir":
                str(document_dir),

            "model":
                self.model,

            "pages_per_batch":
                self.pages_per_batch,

            "batch_count":
                len(results),

            "total_articles":
                len(all_articles),

            "continuation_links":
                len(all_links),

            "pending_continuations":
                len(
                    pending_continuations
                ),

            "logical_articles":
                len(logical_articles),

            "continuation_repair":
                repair_report,

            "batches":
                results,
        }

        manifest_path = (
            output_root
            / "manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                manifest,
                f,
                indent=4,
                ensure_ascii=False,
            )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print()
        print("=" * 60)
        print(
            "GEMINI ARTICLE EXTRACTION COMPLETE"
        )
        print("=" * 60)

        print(
            f"Gemini requests       : "
            f"{len(results)}"
        )

        print(
            f"Articles extracted    : "
            f"{len(all_articles)}"
        )

        print(
            f"Continuation links    : "
            f"{len(all_links)}"
        )

        print(
            f"Pending continuations : "
            f"{len(pending_continuations)}"
        )

        print(
            f"Logical articles      : "
            f"{len(logical_articles)}"
        )

        print(
            f"Final output          : "
            f"{final_path}"
        )

        print(
            f"Manifest              : "
            f"{manifest_path}"
        )

        print("=" * 60)
        print()

        return results

    # ========================================================
    # PROCESS ONE BATCH
    # ========================================================

    def process_batch(
        self,
        document_dir: Path,
        page_numbers: list[int],
        batch_index: int,
        output_root: Path,
        pending_continuations: list[
            dict[str, Any]
        ],
        known_pages: list[int],
    ) -> dict[str, Any]:

        crop_root = (
            document_dir
            / "final_articles_crops"
        )

        print()
        print("=" * 60)

        print(
            f"GEMINI ARTICLE BATCH "
            f"{batch_index}"
        )

        print("=" * 60)

        print(
            "Pages:",
            ", ".join(
                f"{p:03d}"
                for p in page_numbers
            ),
        )

        # ====================================================
        # GEMINI CONTENTS
        # ====================================================

        contents: list[Any] = []

        article_manifest = []

        # ----------------------------------------------------
        # Add every article crop
        # ----------------------------------------------------

        for page_number in page_numbers:

            articles = (
                self._discover_articles(
                    crop_root,
                    page_number,
                )
            )

            for article in articles:

                article_id = (
                    article["article_id"]
                )

                image_path = (
                    article["image_path"]
                )

                crop_metadata = (
                    article.get(
                        "crop_metadata",
                        {},
                    )
                )

                # --------------------------------------------
                # Identity marker
                # --------------------------------------------

                contents.append(
                    types.Part.from_text(
                        text=(
                            "\n"
                            "================================================\n"
                            "VERIFIED ARTICLE CROP\n"
                            f"PAGE: {page_number}\n"
                            f"ARTICLE_ID: {article_id}\n"
                            f"IMAGE: {image_path.name}\n"
                            "================================================\n"
                        )
                    )
                )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                metadata = {
                    "page":
                        page_number,

                    "article_id":
                        article_id,

                    "bbox":
                        crop_metadata.get(
                            "bbox"
                        ),

                    "width":
                        crop_metadata.get(
                            "width"
                        ),

                    "height":
                        crop_metadata.get(
                            "height"
                        ),

                    "source_width":
                        crop_metadata.get(
                            "source_width"
                        ),

                    "source_height":
                        crop_metadata.get(
                            "source_height"
                        ),

                    "boundary_source":
                        crop_metadata.get(
                            "boundary_source"
                        ),
                }

                contents.append(
                    types.Part.from_text(
                        text=(
                            "Verified crop metadata:\n"
                            + json.dumps(
                                metadata,
                                ensure_ascii=False,
                            )
                        )
                    )
                )

                # --------------------------------------------
                # Image
                # --------------------------------------------

                image_bytes = (
                    image_path.read_bytes()
                )

                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    )
                )

                article_manifest.append(
                    {
                        "page":
                            page_number,

                        "article_id":
                            article_id,

                        "image":
                            str(image_path),

                        "crop_metadata":
                            crop_metadata,
                    }
                )

        if not article_manifest:

            # A batch with zero article crops is a normal outcome for
            # pages that are entirely advertisements/masthead/chrome
            # (no editorial content at all), not a pipeline failure --
            # skip this batch instead of aborting the whole document.
            # Pending continuations pass through unchanged: nothing in
            # an empty batch could resolve or create any.

            print()
            print(
                f"⚠ No article crops for pages {page_numbers} "
                "(likely all-advertisement/chrome pages) -- "
                "skipping this batch."
            )

            result = {
                "batch_id": f"batch_{batch_index:03d}",
                "pages": page_numbers,
                "model": self.model,
                "requested_article_count": 0,
                "returned_article_count": 0,
                "pending_entering_batch": 0,
                "continuation_links": [],
                "pending_continuations": pending_continuations,
                "validation": {
                    "expected": 0,
                    "returned": 0,
                    "missing": [],
                    "unexpected": [],
                    "duplicates": [],
                    "complete": True,
                },
                "continuation_validation": {
                    "valid": True,
                    "errors": [],
                    "valid_links": [],
                },
                "pending_validation": {
                    "valid": True,
                    "errors": [],
                },
                "articles": [],
                "skipped": True,
            }

            output_path = (
                output_root
                / f"batch_{batch_index:03d}.json"
            )

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            return result

        # ====================================================
        # PENDING CONTINUATION INFORMATION
        # ====================================================

        # Only pending records whose target page is in the
        # CURRENT batch need to be sent to Gemini now.
        #
        # Other pending records remain pending.

        pending_for_current_batch = []

        still_pending_external = []

        current_page_set = set(
            int(p)
            for p in page_numbers
        )

        for pending in (
            pending_continuations
        ):

            target_page = self._safe_int(
                pending.get(
                    "target_page"
                )
            )

            if (
                target_page is not None
                and target_page
                in current_page_set
            ):

                # ------------------------------------------------
                # This pending continuation can now be resolved.
                #
                # Attach the ORIGINAL source article crop so the
                # SAME Gemini request can compare source + target
                # visually and semantically.
                # ------------------------------------------------

                pending_copy = dict(
                    pending
                )

                source = (
                    pending_copy.get(
                        "source"
                    )
                    or {}
                )

                source_page = self._safe_int(
                    source.get(
                        "page"
                    )
                )

                source_article_id = str(
                    source.get(
                        "article_id",
                        ""
                    )
                )

                source_image_path = None

                if (
                    source_page is not None
                    and source_article_id
                ):

                    source_articles = (
                        self._discover_articles(
                            crop_root,
                            source_page,
                        )
                    )

                    for source_article in (
                        source_articles
                    ):

                        if (
                            source_article[
                                "article_id"
                            ]
                            == source_article_id
                        ):

                            source_image_path = (
                                source_article[
                                    "image_path"
                                ]
                            )
                            break

                if source_image_path is not None:

                    pending_copy[
                        "source_image"
                    ] = str(
                        source_image_path
                    )

                else:

                    print(
                        "WARNING: Could not locate "
                        "original source crop for "
                        "pending continuation: "
                        f"page={source_page}, "
                        f"article_id={source_article_id}"
                    )

                pending_for_current_batch.append(
                    pending_copy
                )

            else:

                still_pending_external.append(
                    pending
                )

        # ====================================================
        # PENDING SOURCE ARTICLE CROP IMAGES
        # ====================================================
        #
        # These are ONLY for continuations created by an earlier
        # batch whose target page is now in this batch.
        #
        # They are added to the SAME Gemini request as the current
        # batch's target article crops.
        # ====================================================

        for pending in (
            pending_for_current_batch
        ):

            source = (
                pending.get(
                    "source"
                )
                or {}
            )

            source_page = source.get(
                "page"
            )

            source_article_id = source.get(
                "article_id"
            )

            source_image = pending.get(
                "source_image"
            )

            if not source_image:
                continue

            source_image_path = Path(
                source_image
            )

            if not source_image_path.exists():
                print(
                    "WARNING: Pending source image "
                    "does not exist: "
                    f"{source_image_path}"
                )
                continue

            contents.append(
                types.Part.from_text(
                    text=(
                        "\n"
                        "================================================\n"
                        "PENDING SOURCE ARTICLE CROP\n"
                        "================================================\n"
                        f"PAGE: {source_page}\n"
                        f"ARTICLE_ID: {source_article_id}\n"
                        "This is the ORIGINAL source crop for a "
                        "continuation that must be resolved against "
                        "the current batch.\n"
                        "Compare this source crop with the current "
                        "target-page article crops.\n"
                        "================================================\n"
                    )
                )
            )

            source_image_bytes = (
                source_image_path.read_bytes()
            )

            contents.append(
                types.Part.from_bytes(
                    data=source_image_bytes,
                    mime_type="image/png",
                )
            )

        # ====================================================
        # STRICT CROP INVENTORY PROMPT
        # ====================================================

        prompt = self._build_prompt(
            page_numbers=page_numbers,
            article_manifest=article_manifest,
            pending_continuations=(
                pending_for_current_batch
            ),
            known_pages=known_pages,
        )

        contents.insert(
            0,
            types.Part.from_text(
                text=prompt
            ),
        )

        print(
            f"Article crops in this batch: "
            f"{len(article_manifest)}"
        )

        print(
            f"Pending continuations entering batch: "
            f"{len(pending_for_current_batch)}"
        )

        print()

        # ====================================================
        # GEMINI API CALL
        # ====================================================

        print(
            "Calling Gemini..."
        )

        # ====================================================
        # GEMINI RETRY
        # ====================================================

        MAX_RETRIES = 5
        RETRY_DELAY = 5

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

                if attempt > 1:
                    print(
                        f"✓ Gemini succeeded on retry "
                        f"{attempt}/{MAX_RETRIES}"
                    )

                break

            except Exception as exc:

                error_text = str(exc)

                # Retry temporary Gemini unavailable errors, plus a
                # stalled/timed-out connection (no timeout configured
                # on the client previously meant these hung forever
                # instead of ever reaching this except block).
                is_timeout = isinstance(
                    exc,
                    (httpx.TimeoutException, httpx.ConnectError),
                )

                if (
                    not is_timeout
                    and "503" not in error_text
                    and "UNAVAILABLE" not in error_text
                    and "unavailable" not in error_text
                ):
                    raise

                print()
                print(
                    f"⚠ Gemini unavailable "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                print(
                    f"  Error: {exc}"
                )

                if attempt == MAX_RETRIES:
                    print()
                    print(
                        f"✗ Gemini failed after "
                        f"{MAX_RETRIES} attempts"
                    )
                    raise

                print(
                    f"  Retrying in {RETRY_DELAY} seconds..."
                )
                time.sleep(RETRY_DELAY)

        # ====================================================
        # RESPONSE
        # ====================================================

        response_text = (
            response.text
            or ""
        ).strip()

        if not response_text:
            raise RuntimeError(
                "Gemini returned an empty response "
                f"for batch {batch_index}."
            )

        parsed = self._parse_json(
            response_text
        )

        articles = parsed.get(
            "articles",
            [],
        )

        if not isinstance(
            articles,
            list,
        ):
            articles = []

        continuation_links = (
            parsed.get(
                "continuation_links",
                [],
            )
        )

        if not isinstance(
            continuation_links,
            list,
        ):
            continuation_links = []

        pending_from_gemini = (
            parsed.get(
                "pending_continuations",
                [],
            )
        )

        if not isinstance(
            pending_from_gemini,
            list,
        ):
            pending_from_gemini = []

        # ====================================================
        # SAFETY FILTER FOR PENDING CONTINUATIONS
        # ====================================================

        pending_from_gemini = (
            self._filter_pending_continuations(
                pending_from_gemini,
                current_articles=articles,
            )
        )

        # ====================================================
        # VALIDATE ARTICLE INVENTORY
        # ====================================================

        validation = (
            self._validate_response(
                articles,
                article_manifest,
            )
        )

        # ====================================================
        # VALIDATE CONTINUATION LINKS
        # ====================================================

        link_validation = (
            self._validate_continuation_links(
                links=continuation_links,
                current_articles=articles,
                pending_articles=(
                    pending_for_current_batch
                ),
                current_pages=current_page_set,
            )
        )

        # ====================================================
        # VALIDATE PENDING
        # ====================================================

        pending_validation = (
            self._validate_pending_continuations(
                pending_from_gemini,
                known_article_keys=(
                    self._known_article_keys(
                        article_manifest
                    )
                    | self._pending_article_keys(
                        pending_for_current_batch
                    )
                ),
            )
        )

        # ====================================================
        # BUILD NEXT PENDING QUEUE
        # ====================================================

        next_pending = []

        # ----------------------------------------------------
        # Old pending records that were NOT resolved
        # ----------------------------------------------------

        resolved_sources = set()

        for link in continuation_links:

            source_key = (
                self._link_key(
                    link,
                    "source",
                )
            )

            if source_key:
                resolved_sources.add(
                    source_key
                )

        for pending in (
            pending_for_current_batch
        ):

            source_key = (
                self._pending_source_key(
                    pending
                )
            )

            if (
                source_key
                not in resolved_sources
            ):

                # Gemini did not resolve it even though the
                # target page was available.
                #
                # This is no longer "external".
                # It becomes genuinely unresolved.

                unresolved_pending = (
                    dict(pending)
                )

                unresolved_pending[
                    "status"
                ] = "UNRESOLVED"

                next_pending.append(
                    unresolved_pending
                )

        # ----------------------------------------------------
        # Pending records Gemini created from current batch
        # ----------------------------------------------------

        for pending in (
            pending_from_gemini
        ):

            pending_copy = dict(
                pending
            )

            pending_copy[
                "status"
            ] = "PENDING_EXTERNAL"

            next_pending.append(
                pending_copy
            )

        # ----------------------------------------------------
        # Old pending whose target page has NOT arrived
        # ----------------------------------------------------

        for pending in (
            still_pending_external
        ):

            next_pending.append(
                pending
            )

        # ====================================================
        # Remove duplicates from pending queue
        # ====================================================

        next_pending = (
            self._deduplicate_pending(
                next_pending
            )
        )

        # ====================================================
        # RESULT
        # ====================================================

        result = {
            "batch_id":
                f"batch_{batch_index:03d}",

            "pages":
                page_numbers,

            "model":
                self.model,

            "requested_article_count":
                len(article_manifest),

            "returned_article_count":
                len(articles),

            "pending_entering_batch":
                len(
                    pending_for_current_batch
                ),

            "continuation_links":
                continuation_links,

            "pending_continuations":
                next_pending,

            "validation":
                validation,

            "continuation_validation":
                link_validation,

            "pending_validation":
                pending_validation,

            "articles":
                articles,
        }

        # ====================================================
        # SAVE BATCH
        # ====================================================

        output_path = (
            output_root
            / f"batch_{batch_index:03d}.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False,
            )

        # ====================================================
        # PRINT
        # ====================================================

        print()
        print(
            "✓ Gemini response received"
        )

        print(
            f"✓ Articles returned: "
            f"{len(articles)}"
        )

        print(
            f"✓ Expected articles: "
            f"{len(article_manifest)}"
        )

        print(
            f"✓ Missing articles: "
            f"{len(validation['missing'])}"
        )

        print(
            f"✓ Unexpected articles: "
            f"{len(validation['unexpected'])}"
        )

        print(
            f"✓ Continuation links: "
            f"{len(continuation_links)}"
        )

        print(
            f"✓ Pending after batch: "
            f"{len(next_pending)}"
        )

        print(
            f"✓ Saved: {output_path}"
        )

        # ----------------------------------------------------
        # Validation warnings
        # ----------------------------------------------------

        if validation["unexpected"]:

            print()

            print(
                "⚠ WARNING: Gemini returned article "
                "identities that are NOT present "
                "in the verified crop inventory."
            )

            for item in (
                validation["unexpected"]
            ):

                print(
                    f"  Unexpected: "
                    f"page={item['page']} "
                    f"article_id="
                    f"{item['article_id']}"
                )

        if validation["duplicates"]:

            print()

            print(
                "⚠ WARNING: Gemini returned duplicate "
                "(page, article_id) identities."
            )

        if not validation["complete"]:

            print()

            print(
                "⚠ Batch validation FAILED."
            )

        if not link_validation["valid"]:

            print()

            print(
                "⚠ Continuation validation "
                "found invalid links."
            )

            for error in (
                link_validation["errors"]
            ):

                print(
                    f"  {error}"
                )

        print("=" * 60)

        return result

    # ========================================================
    # PROMPT
    # ========================================================

    @classmethod
    def _build_prompt(
        cls,
        page_numbers: list[int],
        article_manifest: list[
            dict[str, Any]
        ],
        pending_continuations: list[
            dict[str, Any]
        ],
        known_pages: list[int],
    ) -> str:

        pages = ", ".join(
            str(page)
            for page in page_numbers
        )

        known_page_text = ", ".join(
            str(page)
            for page in known_pages
        )

        # ----------------------------------------------------
        # Crop inventory
        # ----------------------------------------------------

        inventory_lines = []

        for page_number in page_numbers:

            page_articles = [
                item
                for item in article_manifest
                if int(item["page"])
                == int(page_number)
            ]

            page_articles.sort(
                key=lambda item:
                cls._article_sort_key(
                    str(
                        item["article_id"]
                    )
                )
            )

            inventory_lines.append(
                f"PAGE {int(page_number):03d}:"
            )

            for item in page_articles:

                inventory_lines.append(
                    f"- article_id="
                    f"{item['article_id']}"
                )

        crop_inventory = "\n".join(
            inventory_lines
        )

        # ----------------------------------------------------
        # Pending continuation inventory
        # ----------------------------------------------------

        pending_lines = []

        if pending_continuations:

            for pending in (
                pending_continuations
            ):

                source = (
                    pending.get(
                        "source"
                    )
                    or {}
                )

                pending_lines.append(
                    json.dumps(
                        pending,
                        ensure_ascii=False,
                    )
                )

        else:

            pending_lines.append(
                "NONE"
            )

        pending_inventory = "\n".join(
            pending_lines
        )

        # ----------------------------------------------------
        # Prompt
        #
        # IMPORTANT:
        #
        # This is NOT an f-string.
        # JSON braces therefore do not need
        # escaping.
        # ----------------------------------------------------

        prompt = """
You are the newspaper article extraction and
continuation-resolution engine.

You are receiving FINAL VERIFIED ARTICLE CROPS
from newspaper pages __PAGES__.

Each image represents ONE FINAL VERIFIED ARTICLE.

You must read each crop directly from the image.

============================================================
PART A — ARTICLE EXTRACTION
============================================================

Return exactly ONE article object for every
(page, article_id) pair in the supplied crop inventory.

Do NOT invent article IDs.

Do NOT create additional articles.

Do NOT split one crop.

Do NOT merge two different crop IDs.

The supplied crop inventory is authoritative.

============================================================
ABSOLUTE CROP INVENTORY
============================================================

__CROP_INVENTORY__

The allowed article identities are EXACTLY the
identities above.

If PAGE 003 contains article_001 through article_016,
DO NOT create article_017.

A large crop may visually contain another story,
sidebar, inset, caption, advertisement, or unrelated
text.

That does NOT create another article object.

Only the verified crop identity is the article.

============================================================
READ THE IMAGE
============================================================

Read the actual newspaper crop visually.

Do NOT use EasyOCR.

Do NOT use external OCR.

Do NOT use page-level OCR.

Do NOT invent unreadable words.

Preserve the actual article wording.

============================================================
BOUNDARY
============================================================

The crop boundary is FINAL.

Do not expand it.

Do not shrink it.

Do not merge it with another crop.

Do not create a new boundary.

============================================================
READING ORDER
============================================================

Newspaper articles can contain multiple columns.

Read each column:

top → bottom

then continue to the next column.

Do NOT read horizontally across unrelated columns.

============================================================
ARTICLE TEXT
============================================================

article_text must contain the actual readable
article text.

Do NOT summarize article_text.

The summary is a separate field.

============================================================
STRUCTURED KNOWLEDGE
============================================================

For every article return:

headline
subheadline
author
location
date
article_text
summary
category
entities
topics
keywords
sentiment
quality

The structured knowledge must be based ONLY on
the target article crop.

============================================================
CATEGORY
============================================================

category must be one of:

National
International
Politics
Business
Sports
Technology
Entertainment
Local
Science
Health
Education
Opinion
Other

============================================================
SENTIMENT
============================================================

sentiment must be:

positive
negative
neutral
mixed
null

============================================================
QUALITY
============================================================

quality.text_readability:

high
medium
low

quality.missing_text:

true only when meaningful article content is
unreadable or genuinely missing.

============================================================
PART B — IMAGE INFORMATION
============================================================

For every verified article crop, inspect the crop for actual
meaningful visual content belonging to that article.

Meaningful visual content includes:

- photographs
- illustrations
- maps
- charts
- graphs
- diagrams
- meaningful article-specific visual figures

Do NOT treat normal text as an image.

Do NOT treat the headline as an image.

Do NOT treat a caption by itself as an image.

Do NOT treat decorative lines, borders, separators, bullets,
background graphics, or newspaper UI elements as images.

For every actual article image, return:

- image_id
- image_description
- image_bbox
- caption
- confidence

image_bbox MUST be measured INSIDE THE SUPPLIED ARTICLE CROP.

Use normalized coordinates from 0 to 1000:

x1 = left
y1 = top
x2 = right
y2 = bottom

Example:

{
    "image_id": "image_001",
    "image_description": "Photograph of two politicians standing...",
    "image_bbox": {
        "x1": 120,
        "y1": 300,
        "x2": 780,
        "y2": 720
    },
    "caption": "Prime Minister ...",
    "confidence": 0.95
}

If there are no meaningful article images, return:

"images": {
    "has_images": false,
    "image_count": 0,
    "items": []
}

If there are images, return:

"images": {
    "has_images": true,
    "image_count": 2,
    "items": [...]
}

The image information must come from the supplied article crop.

Do NOT invent images.

Do NOT use images from another article.

For a pending continuation, you may use the explicitly supplied
ORIGINAL SOURCE ARTICLE CROP only when resolving the continuation
relationship. The target article's own "images" field must describe
images inside the target article crop.

============================================================
PART C — CONTENT TYPE CLASSIFICATION
============================================================

Classify the VERIFIED CROP itself.

content_type MUST be exactly one of:

article
photo_caption
reference
advertisement
other

article
-------
A genuine newspaper article/story with substantial independent
article prose.

photo_caption
-------------
Primarily a photograph, illustration, graphic, or visual with
a caption and little or no independent article prose.

reference
---------
Primarily a page reference/pointer such as "Report on Page 2",
"More on Page 3", or "See Page 5", without substantial
independent article prose.

advertisement
-------------
An advertisement.

other
-----
Anything else.

IMPORTANT:
A continuation marker by itself does NOT make a crop an article.

If a crop contains a headline such as "WINDOWS", a photograph,
a caption, and "Report on Page 2" but does not contain
substantial independent article prose, classify it as
content_type = "photo_caption".

Do NOT classify based only on the continuation marker.
Examine the complete supplied crop.

============================================================
PART D — CONTINUATION DETECTION
============================================================

Look carefully for continuation markers such as:

More on Page 3
Continued on Page 5
Continued from Page 1
See Page 6
Turn to Page 4
Full report on Page 7
Report on Page 2
To be continued

If a marker is visible:

continuation.is_continued = true

Record:

continuation.marker
continuation.next_page

If no continuation marker exists:

continuation.is_continued = false

continuation.marker = null

continuation.next_page = null

============================================================
PART D — CONTINUATION RESOLUTION
============================================================

This request may also contain PENDING continuation
records from previous 3-page batches.

Known newspaper pages:

__KNOWN_PAGES__

Current pages in this request:

__PAGES__

Previous pending continuations:

__PENDING_CONTINUATIONS__

Your job is to resolve continuation relationships
ONLY when the target page is available in the
CURRENT request.

============================================================
CASE 1 — SAME-BATCH CONTINUATION
============================================================

Example:

Page 1 article_006 says:

"More on Page 3"

and Page 3 is present in this request.

Compare the source article against the Page 3
article crops.

Use semantic meaning, not only exact headline matching.

Consider:

- headline
- article subject
- entities
- people
- organizations
- locations
- events
- topics
- keywords
- article text
- story context
- continuation marker
- newspaper wording

If one target article is clearly the continuation,
return a continuation link.

============================================================
CASE 2 — PENDING CONTINUATION FROM PREVIOUS BATCH
============================================================

A previous batch may contain:

Page 1 article_007
marker = "More on Page 7"
next_page = 7

If Page 7 is present in the CURRENT request,
you MUST compare the previous source article
against the Page 7 article crops.

The previous source article is supplied with:

1. structured metadata
2. the ORIGINAL source article crop image

The source crop image is explicitly labeled
"PENDING SOURCE ARTICLE CROP".

You MUST use the source image together with the current
target-page article crop images when resolving the continuation.

Compare both semantic and visual evidence, including:

- source headline
- source article text
- source entities
- source topics
- source people
- source organizations
- source locations
- source events
- source image context
- source image captions
- target headline
- target article text
- target entities
- target topics
- target people
- target organizations
- target locations
- target events
- target image context
- target image captions

Do not rely on image similarity alone.

Resolve the relationship if the combined evidence is strong.

============================================================
CASE 3 — TARGET PAGE NOT AVAILABLE
============================================================

If an article says:

"More on Page 7"

but Page 7 is NOT present in the current request,
DO NOT guess the target article.

Return a pending continuation record.

Status:

PENDING_EXTERNAL

============================================================
CASE 4 — TARGET PAGE AVAILABLE BUT MATCH UNCERTAIN
============================================================

If the target page is present but you cannot confidently
identify the continuation article:

DO NOT guess.

Return:

status = UNRESOLVED

This is different from PENDING_EXTERNAL.

PENDING_EXTERNAL means:

the target page has not been processed yet.

UNRESOLVED means:

the target page is available but the match is uncertain.

============================================================
CONTINUATION LINK RULES
============================================================

Every continuation link must contain:

source_page
source_article_id
target_page
target_article_id
confidence
reason

Only use article IDs that actually exist.

Never invent target IDs.

Never invent pages.

Never merge unrelated articles.

One target article should not be assigned as the
continuation of two different source articles.

An article MAY be both:

- the target of an earlier continuation
- and the source of a later continuation

because an article can continue across multiple pages.

============================================================
CONTINUATION CONTENT-TYPE RULE
============================================================

A continuation link is valid ONLY when:

1. target.content_type == "article"
2. source.content_type == "article" OR source.content_type is "photo_caption"/"reference" WITH an explicit continuation marker pointing to the target page
3. source and target discuss the same underlying story
4. semantic evidence is strong
5. confidence >= 0.75

The following can NEVER be continuation targets:
- photo_caption
- reference
- advertisement
- other

A photo_caption/reference MAY be a continuation SOURCE only when it contains
an explicit marker such as "Report on Page 2", "More on Page 3", or "See Page 5".
Keep the source's original content_type. Do NOT change it to article.
A continuation marker alone is NOT sufficient evidence; semantic evidence
must still connect the source crop to the target article.

============================================================
CONFIDENCE
============================================================

Use:

0.90 - 1.00
Very strong match

0.75 - 0.89
Strong match

0.60 - 0.74
Possible but uncertain

Below 0.60
Do NOT merge

Only return a MERGED continuation link when
the evidence is strong.

============================================================
IMPORTANT EXAMPLE
============================================================

Source:

"SC asks UP SIT to submit report on donation probe"

Target:

"Submit status report: SC to SIT probing Mandir 'theft'"

These may have different headlines but can still be
the same story.

Use semantic evidence.

Another example:

Source:

"Srinagar put under partial lockdown on Martyrs' Day"

Target:

"Lockdown in Srinagar imposed to block Martyrs' Day marches"

These should be recognized as the same story when
the surrounding evidence confirms the relationship.

============================================================
DO NOT FORCE WEAK MATCHES
============================================================

A continuation marker is NOT sufficient evidence.

Before creating a continuation link, verify:
target.content_type == "article"
source.content_type == "article" OR source.content_type in {"photo_caption", "reference"} with an explicit page marker
semantic_match == strong
confidence >= 0.75

IMPORTANT EXAMPLE:
Source: "WINDOWS"
content_type = "photo_caption"
Target: "Atishi joins protest by CJP..."

If the source contains an explicit marker such as "Report on Page 2",
KEEP source.content_type = "photo_caption" but allow it as a continuation
SOURCE. Compare its visible caption/text/entities/topics/keywords with the
real article targets on Page 2.

Create the continuation link ONLY when the semantic evidence strongly
identifies the Page 2 article as the same underlying story.

Do NOT change the crop boundary.
Do NOT change the source article_id.
Do NOT change the source content_type.
Do NOT use the marker alone as proof.

If semantic evidence is insufficient:

UNRESOLVED.

============================================================
PENDING OUTPUT
============================================================

For every continuation whose target page is not
available in the current request, return:

{
    "status": "PENDING_EXTERNAL",
    "source": {
        "page": ...,
        "article_id": ...,
        "headline": ...,
        "content_type": "article" | "photo_caption" | "reference"
    },
    "target_page": ...,
    "marker": ...
}

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

The JSON structure must contain:

{
    "articles": [...],
    "continuation_links": [...],
    "pending_continuations": [...]
}

============================================================
ARTICLE OUTPUT
============================================================

Every article object MUST contain:

page
article_id
headline
subheadline
author
location
date
article_text
summary
category
entities
topics
keywords
sentiment
content_type
continuation
images
quality

============================================================
FINAL RULES
============================================================

Do NOT invent articles.

Do NOT invent article IDs.

Do NOT renumber article IDs.

Do NOT merge separate verified crops during extraction.

Do NOT change boundaries.

Do NOT use OCR.

Do NOT use page-level OCR.

Do NOT guess continuation targets.

NEVER create a continuation link or pending continuation
for a TARGET whose content_type is not "article".
A source with content_type "photo_caption" or "reference" is allowed ONLY
when it has an explicit continuation marker with a target page and the
semantic evidence strongly matches a real article target.

Use semantic reasoning for continuation matching.

Return JSON only.
""".strip()

        prompt = prompt.replace(
            "__PAGES__",
            pages,
        )

        prompt = prompt.replace(
            "__KNOWN_PAGES__",
            known_page_text,
        )

        prompt = prompt.replace(
            "__CROP_INVENTORY__",
            crop_inventory,
        )

        prompt = prompt.replace(
            "__PENDING_CONTINUATIONS__",
            pending_inventory,
        )

        return prompt

    # ========================================================
    # RESPONSE SCHEMA
    # ========================================================

    @staticmethod
    def _response_schema() -> dict[str, Any]:

        return {
            "type": "OBJECT",

            "properties": {

                "articles": {
                    "type": "ARRAY",

                    "items": {
                        "type": "OBJECT",

                        "properties": {

                            "page": {
                                "type": "INTEGER"
                            },

                            "article_id": {
                                "type": "STRING"
                            },

                            "headline": {
                                "type": "STRING"
                            },

                            "subheadline": {
                                "type": "STRING",
                                "nullable": True,
                            },

                            "author": {
                                "type": "STRING",
                                "nullable": True,
                            },

                            "location": {
                                "type": "STRING",
                                "nullable": True,
                            },

                            "date": {
                                "type": "STRING",
                                "nullable": True,
                            },

                            "article_text": {
                                "type": "STRING"
                            },

                            "summary": {
                                "type": "STRING"
                            },

                            "category": {
                                "type": "STRING"
                            },

                            "entities": {
                                "type": "ARRAY",

                                "items": {
                                    "type": "STRING"
                                },
                            },

                            "topics": {
                                "type": "ARRAY",

                                "items": {
                                    "type": "STRING"
                                },
                            },

                            "keywords": {
                                "type": "ARRAY",

                                "items": {
                                    "type": "STRING"
                                },
                            },

                            "sentiment": {
                                "type": "STRING",
                                "nullable": True,
                            },

                            "content_type": {
                                "type": "STRING",
                                "enum": [
                                    "article",
                                    "photo_caption",
                                    "reference",
                                    "advertisement",
                                    "other",
                                ],
                            },

                            "continuation": {
                                "type": "OBJECT",

                                "properties": {

                                    "is_continued": {
                                        "type": "BOOLEAN"
                                    },

                                    "marker": {
                                        "type": "STRING",
                                        "nullable": True,
                                    },

                                    "next_page": {
                                        "type": "INTEGER",
                                        "nullable": True,
                                    },
                                },

                                "required": [
                                    "is_continued",
                                    "marker",
                                    "next_page",
                                ],
                            },

                            "images": {
                                "type": "OBJECT",

                                "properties": {

                                    "has_images": {
                                        "type": "BOOLEAN"
                                    },

                                    "image_count": {
                                        "type": "INTEGER"
                                    },

                                    "items": {
                                        "type": "ARRAY",

                                        "items": {
                                            "type": "OBJECT",

                                            "properties": {

                                                "image_id": {
                                                    "type": "STRING"
                                                },

                                                "image_description": {
                                                    "type": "STRING"
                                                },

                                                "image_bbox": {
                                                    "type": "OBJECT",

                                                    "properties": {

                                                        "x1": {
                                                            "type": "NUMBER"
                                                        },

                                                        "y1": {
                                                            "type": "NUMBER"
                                                        },

                                                        "x2": {
                                                            "type": "NUMBER"
                                                        },

                                                        "y2": {
                                                            "type": "NUMBER"
                                                        },
                                                    },

                                                    "required": [
                                                        "x1",
                                                        "y1",
                                                        "x2",
                                                        "y2",
                                                    ],
                                                },

                                                "caption": {
                                                    "type": "STRING",
                                                    "nullable": True,
                                                },

                                                "confidence": {
                                                    "type": "NUMBER"
                                                },
                                            },

                                            "required": [
                                                "image_id",
                                                "image_description",
                                                "image_bbox",
                                                "caption",
                                                "confidence",
                                            ],
                                        },
                                    },
                                },

                                "required": [
                                    "has_images",
                                    "image_count",
                                    "items",
                                ],
                            },

                            "quality": {
                                "type": "OBJECT",

                                "properties": {

                                    "text_readability": {
                                        "type": "STRING"
                                    },

                                    "missing_text": {
                                        "type": "BOOLEAN"
                                    },

                                    "notes": {
                                        "type": "STRING",
                                        "nullable": True,
                                    },
                                },

                                "required": [
                                    "text_readability",
                                    "missing_text",
                                    "notes",
                                ],
                            },
                        },

                        "required": [
                            "page",
                            "article_id",
                            "headline",
                            "subheadline",
                            "author",
                            "location",
                            "date",
                            "article_text",
                            "summary",
                            "category",
                            "entities",
                            "topics",
                            "keywords",
                            "sentiment",
                            "content_type",
                            "continuation",
                            "images",
                            "quality",
                        ],
                    },
                },

                "continuation_links": {
                    "type": "ARRAY",

                    "items": {
                        "type": "OBJECT",

                        "properties": {

                            "source_page": {
                                "type": "INTEGER"
                            },

                            "source_article_id": {
                                "type": "STRING"
                            },

                            "target_page": {
                                "type": "INTEGER"
                            },

                            "target_article_id": {
                                "type": "STRING"
                            },

                            "confidence": {
                                "type": "NUMBER"
                            },

                            "reason": {
                                "type": "STRING"
                            },
                        },

                        "required": [
                            "source_page",
                            "source_article_id",
                            "target_page",
                            "target_article_id",
                            "confidence",
                            "reason",
                        ],
                    },
                },

                "pending_continuations": {
                    "type": "ARRAY",

                    "items": {
                        "type": "OBJECT",

                        "properties": {

                            "status": {
                                "type": "STRING"
                            },

                            "source": {
                                "type": "OBJECT",

                                "properties": {

                                    "page": {
                                        "type": "INTEGER"
                                    },

                                    "article_id": {
                                        "type": "STRING"
                                    },

                                    "headline": {
                                        "type": "STRING"
                                    },

                                    "content_type": {
                                        "type": "STRING",
                                        "enum": [
                                            "article",
                                            "photo_caption",
                                            "reference",
                                            "advertisement",
                                            "other",
                                        ],
                                    },
                                },

                                "required": [
                                    "page",
                                    "article_id",
                                    "headline",
                                    "content_type",
                                ],
                            },

                            "target_page": {
                                "type": "INTEGER"
                            },

                            "marker": {
                                "type": "STRING",
                                "nullable": True,
                            },
                        },

                        "required": [
                            "status",
                            "source",
                            "target_page",
                            "marker",
                        ],
                    },
                },
            },

            "required": [
                "articles",
                "continuation_links",
                "pending_continuations",
            ],
        }

    # ========================================================
    # DISCOVER PAGES
    # ========================================================

    @staticmethod
    def _discover_pages(
        crop_root: Path,
    ) -> list[int]:

        pages = []

        if not crop_root.exists():
            return pages

        for directory in crop_root.iterdir():

            if not directory.is_dir():
                continue

            name = directory.name

            if not name.startswith(
                "page_"
            ):
                continue

            number_text = (
                name[
                    len("page_"):
                ]
            )

            try:

                page_number = int(
                    number_text
                )

            except ValueError:

                continue

            pages.append(
                page_number
            )

        return sorted(
            set(pages)
        )

    # ========================================================
    # DISCOVER ARTICLES
    # ========================================================

    @classmethod
    def _discover_articles(
        cls,
        crop_root: Path,
        page_number: int,
    ) -> list[dict[str, Any]]:

        page_dir = (
            crop_root
            / f"page_{page_number:03d}"
        )

        if not page_dir.exists():
            return []

        articles = []

        for article_dir in page_dir.iterdir():

            if not article_dir.is_dir():
                continue

            article_id = (
                article_dir.name
            )

            expected_image = (
                article_dir
                / f"page_{page_number:03d}.png"
            )

            if expected_image.exists():

                image_path = (
                    expected_image
                )

            else:

                png_files = sorted(
                    article_dir.glob(
                        "*.png"
                    )
                )

                if not png_files:

                    print(
                        "WARNING: No PNG found:"
                        f" {article_dir}"
                    )

                    continue

                image_path = (
                    png_files[0]
                )

            # ------------------------------------------------
            # crop.json
            # ------------------------------------------------

            crop_metadata = {}

            metadata_path = (
                article_dir
                / "crop.json"
            )

            if metadata_path.exists():

                try:

                    with open(
                        metadata_path,
                        "r",
                        encoding="utf-8",
                    ) as f:

                        crop_metadata = (
                            json.load(f)
                        )

                except Exception as exc:

                    print(
                        "WARNING: Could not read "
                        f"{metadata_path}: {exc}"
                    )

            articles.append(
                {
                    "page":
                        page_number,

                    "article_id":
                        article_id,

                    "image_path":
                        image_path,

                    "crop_metadata":
                        crop_metadata,
                }
            )

        articles.sort(
            key=lambda item:
            cls._article_sort_key(
                item["article_id"]
            )
        )

        return articles

    # ========================================================
    # ARTICLE SORT
    # ========================================================

    @staticmethod
    def _article_sort_key(
        article_id: str,
    ) -> tuple:

        prefix = "article_"

        if article_id.startswith(
            prefix
        ):

            number = (
                article_id[
                    len(prefix):
                ]
            )

            try:

                return (
                    0,
                    int(number),
                )

            except ValueError:
                pass

        return (
            1,
            article_id,
        )

    # ========================================================
    # CREATE PAGE BATCHES
    # ========================================================

    # Defense-in-depth safety margin, not a documented platform limit.
    # A 47-crop batch was observed failing with a connection error and
    # a 61-crop/~29MB batch with a 400 (on the OpenAI side); these
    # caps sit comfortably below both while the timeout increase
    # elsewhere separately addresses the likely slow-upload cause.
    MAX_CROPS_PER_BATCH = 25
    MAX_BATCH_BYTES = 15 * 1024 * 1024

    def _make_page_batches(
        self,
        pages: list[int],
        page_inventory: dict[int, list[dict[str, Any]]] | None = None,
    ) -> list[list[int]]:

        if not page_inventory:
            # No crop-size information available -- fall back to the
            # original fixed page-count chunking.
            batches = []
            for start in range(
                0, len(pages), self.pages_per_batch
            ):
                batches.append(
                    pages[start:start + self.pages_per_batch]
                )
            return batches

        def _page_stats(page_number: int) -> tuple[int, int]:
            articles = page_inventory.get(page_number, [])
            crop_count = len(articles)
            total_bytes = 0
            for article in articles:
                try:
                    total_bytes += (
                        Path(article["image_path"]).stat().st_size
                    )
                except OSError:
                    pass
            return crop_count, total_bytes

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_crops = 0
        current_bytes = 0

        for page_number in pages:

            page_crops, page_bytes = _page_stats(page_number)

            would_exceed = (
                current_batch
                and (
                    len(current_batch) >= self.pages_per_batch
                    or current_crops + page_crops > self.MAX_CROPS_PER_BATCH
                    or current_bytes + page_bytes > self.MAX_BATCH_BYTES
                )
            )

            if would_exceed:
                batches.append(current_batch)
                current_batch = []
                current_crops = 0
                current_bytes = 0

            current_batch.append(page_number)
            current_crops += page_crops
            current_bytes += page_bytes

            if (
                len(current_batch) == 1
                and (
                    page_crops > self.MAX_CROPS_PER_BATCH
                    or page_bytes > self.MAX_BATCH_BYTES
                )
            ):
                print(
                    f"WARNING: page {page_number} alone has "
                    f"{page_crops} crops ({page_bytes / 1024 / 1024:.1f} MB), "
                    f"already over the per-batch safety margin "
                    f"({self.MAX_CROPS_PER_BATCH} crops / "
                    f"{self.MAX_BATCH_BYTES / 1024 / 1024:.0f} MB) -- "
                    "sending it alone anyway since a page's crops "
                    "can't be split across batches."
                )

        if current_batch:
            batches.append(current_batch)

        return batches

    # ========================================================
    # PARSE JSON
    # ========================================================

    @staticmethod
    def _parse_json(
        response_text: str,
    ) -> dict[str, Any]:

        text = (
            response_text
            .strip()
        )

        # ----------------------------------------------------
        # Direct JSON
        # ----------------------------------------------------

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except json.JSONDecodeError:
            pass

        # ----------------------------------------------------
        # Remove markdown fences
        # ----------------------------------------------------

        if text.startswith(
            "```"
        ):

            lines = (
                text.splitlines()
            )

            if lines:
                lines = lines[1:]

            if (
                lines
                and lines[-1]
                .strip()
                .startswith("```")
            ):

                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        # ----------------------------------------------------
        # Parse again
        # ----------------------------------------------------

        try:

            parsed = json.loads(
                text
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Gemini returned invalid JSON.\n\n"
                "Response:\n"
                f"{text[:10000]}"
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):

            raise RuntimeError(
                "Gemini response is not a JSON object."
            )

        return parsed

    # ========================================================
    # VALIDATE ARTICLE RESPONSE
    # ========================================================

    @staticmethod
    def _validate_response(
        articles: list[dict[str, Any]],
        manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:

        expected = set()

        for item in manifest:

            key = (
                int(item["page"]),
                str(
                    item["article_id"]
                ),
            )

            expected.add(
                key
            )

        returned = set()

        duplicates = []

        for article in articles:

            if not isinstance(
                article,
                dict,
            ):
                continue

            try:

                key = (
                    int(
                        article.get(
                            "page"
                        )
                    ),

                    str(
                        article.get(
                            "article_id"
                        )
                    ),
                )

            except Exception:

                continue

            if key in returned:

                duplicates.append(
                    {
                        "page":
                            key[0],

                        "article_id":
                            key[1],
                    }
                )

            returned.add(
                key
            )

        missing = sorted(
            expected
            - returned
        )

        unexpected = sorted(
            returned
            - expected
        )

        return {
            "expected":
                len(expected),

            "returned":
                len(returned),

            "missing": [
                {
                    "page":
                        page,

                    "article_id":
                        article_id,
                }

                for page, article_id
                in missing
            ],

            "unexpected": [
                {
                    "page":
                        page,

                    "article_id":
                        article_id,
                }

                for page, article_id
                in unexpected
            ],

            "duplicates":
                duplicates,

            "complete":
                (
                    not missing
                    and not unexpected
                    and not duplicates
                    and len(expected)
                    == len(returned)
                ),
        }

    # ========================================================
    # VALIDATE CONTINUATION LINKS
    # ========================================================

    @staticmethod
    def _continuation_eligible(
        article: dict[str, Any] | None,
        require_text: bool = True,
    ) -> tuple[bool, str]:
        """Strict eligibility for a continuation TARGET."""
        if not isinstance(article, dict):
            return False, "Article record is missing."

        content_type = str(article.get("content_type", "") or "").strip().lower()
        if content_type != "article":
            return False, f"content_type={content_type or 'missing'}"

        if require_text:
            article_text = str(article.get("article_text", "") or "").strip()
            if len(article_text) < 120:
                return False, "article_text is too short for continuation target"

        headline = str(article.get("headline", "") or "").strip()
        generic_headlines = {
            "windows", "photo", "photograph", "picture",
            "file photo", "pti photo", "more on page",
            "report on page", "continued",
        }
        if " ".join(headline.lower().split()) in generic_headlines:
            return False, f"generic headline={headline!r}"

        return True, "eligible article target"

    @classmethod
    def _continuation_source_eligible(
        cls,
        article: dict[str, Any] | None,
        require_text: bool = True,
    ) -> tuple[bool, str]:
        """
        Eligibility for a continuation SOURCE.

        Normal articles are allowed.  A photo_caption/reference is allowed
        ONLY when it has an explicit page continuation marker.  Its original
        content_type and verified crop boundary are never changed.
        """
        if not isinstance(article, dict):
            return False, "Source record is missing."

        content_type = str(article.get("content_type", "") or "").strip().lower()

        if content_type == "article":
            if require_text:
                article_text = str(article.get("article_text", "") or "").strip()
                if len(article_text) < 60:
                    return False, "article source text is too short"
            return True, "eligible article source"

        if content_type not in {"photo_caption", "reference"}:
            return False, f"content_type={content_type or 'missing'}"

        target_page = cls._continuation_marker_target_page(article)
        if target_page is None:
            return False, (
                f"content_type={content_type} has no explicit continuation target page"
            )

        return True, f"eligible {content_type} continuation source -> page {target_page}"

    @classmethod
    def _filter_pending_continuations(
        cls,
        pending: list[dict[str, Any]],
        current_articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        article_map = {}
        for article in current_articles:
            try:
                key = (
                    int(article.get("page")),
                    str(article.get("article_id")),
                )
                article_map[key] = article
            except Exception:
                continue

        result = []
        for item in pending:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or {}
            try:
                key = (int(source.get("page")), str(source.get("article_id")))
            except Exception:
                continue

            source_article = article_map.get(key)
            if source_article is None:
                result.append(item)
                continue

            eligible, reason = cls._continuation_source_eligible(source_article)
            if not eligible:
                print(
                    "Continuation pending rejected: "
                    f"source={key}; {reason}"
                )
                continue
            result.append(item)

        return result

    @staticmethod
    def _validate_continuation_links(
        links: list[dict[str, Any]],
        current_articles: list[dict[str, Any]],
        pending_articles: list[dict[str, Any]],
        current_pages: set[int],
    ) -> dict[str, Any]:

        errors = []

        # ----------------------------------------------------
        # Known current article identities
        # ----------------------------------------------------

        known_keys = set()

        for article in current_articles:

            try:

                known_keys.add(
                    (
                        int(
                            article.get(
                                "page"
                            )
                        ),

                        str(
                            article.get(
                                "article_id"
                            )
                        ),
                    )
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # Current article map
        # ----------------------------------------------------

        article_map = {}
        for article in current_articles:
            try:
                key = (
                    int(article.get("page")),
                    str(article.get("article_id")),
                )
                article_map[key] = article
            except Exception:
                pass

        # ----------------------------------------------------
        # Known pending source identities
        # ----------------------------------------------------

        pending_keys = set()

        for pending in pending_articles:

            source = (
                pending.get(
                    "source"
                )
                or {}
            )

            try:

                pending_keys.add(
                    (
                        int(
                            source.get(
                                "page"
                            )
                        ),

                        str(
                            source.get(
                                "article_id"
                            )
                        ),
                    )
                )

            except Exception:
                pass

        known_source_keys = (
            known_keys
            | pending_keys
        )

        seen_targets = set()
        seen_sources = set()

        valid_links = []

        for link in links:

            if not isinstance(
                link,
                dict,
            ):

                errors.append(
                    "Continuation link is not an object."
                )

                continue

            source_page = (
                GeminiArticleExtractor
                ._safe_int(
                    link.get(
                        "source_page"
                    )
                )
            )

            target_page = (
                GeminiArticleExtractor
                ._safe_int(
                    link.get(
                        "target_page"
                    )
                )
            )

            source_article_id = str(
                link.get(
                    "source_article_id",
                    "",
                )
            )

            target_article_id = str(
                link.get(
                    "target_article_id",
                    "",
                )
            )

            if source_page is None:
                errors.append(
                    "Link has invalid source_page."
                )
                continue

            if target_page is None:
                errors.append(
                    "Link has invalid target_page."
                )
                continue

            source_key = (
                source_page,
                source_article_id,
            )

            target_key = (
                target_page,
                target_article_id,
            )

            # ------------------------------------------------
            # Source must exist
            # ------------------------------------------------

            if (
                source_key
                not in known_source_keys
            ):

                errors.append(
                    "Unknown continuation source: "
                    f"{source_key}"
                )

                continue

            # ------------------------------------------------
            # Target must be in current pages
            # ------------------------------------------------

            if (
                target_page
                not in current_pages
            ):

                errors.append(
                    "Continuation target page "
                    "is not in current batch: "
                    f"{target_key}"
                )

                continue

            # ------------------------------------------------
            # Target must exist in current articles
            # ------------------------------------------------

            if (
                target_key
                not in known_keys
            ):

                errors.append(
                    "Unknown continuation target: "
                    f"{target_key}"
                )

                continue

            # ------------------------------------------------
            # No self merge
            # ------------------------------------------------

            if source_key == target_key:

                errors.append(
                    "Self continuation detected: "
                    f"{source_key}"
                )

                continue

            # ------------------------------------------------
            # One source should have one target
            # ------------------------------------------------

            if source_key in seen_sources:

                errors.append(
                    "Duplicate continuation source: "
                    f"{source_key}"
                )

                continue

            # ------------------------------------------------
            # One target should not be assigned twice
            # ------------------------------------------------

            if target_key in seen_targets:

                errors.append(
                    "Duplicate continuation target: "
                    f"{target_key}"
                )

                continue

            # ------------------------------------------------
            # Source and target must both be real articles
            # ------------------------------------------------

            source_article = article_map.get(source_key)

            # A source from an earlier batch is not in current_articles.
            # Use its structured pending metadata for the content-type
            # safety check. Current-batch sources use the full article.
            source_require_text = True

            if source_article is None:
                for pending in pending_articles:
                    if (
                        GeminiArticleExtractor
                        ._pending_source_key(pending)
                        == source_key
                    ):
                        source_article = (
                            pending.get("source") or {}
                        )
                        source_require_text = False
                        break

            source_ok, source_reason = (
                GeminiArticleExtractor
                ._continuation_source_eligible(
                    source_article,
                    require_text=source_require_text,
                )
            )

            if not source_ok:
                errors.append(
                    "Continuation rejected: "
                    f"source {source_key}: {source_reason}"
                )
                continue

            target_ok, target_reason = (
                GeminiArticleExtractor
                ._continuation_eligible(
                    article_map.get(target_key)
                )
            )

            if not target_ok:
                errors.append(
                    "Continuation rejected: "
                    f"target {target_key}: {target_reason}"
                )
                continue

            confidence = link.get(
                "confidence"
            )

            try:

                confidence = float(
                    confidence
                )

            except Exception:

                confidence = 0.0

            if confidence < 0.75:

                errors.append(
                    "Continuation confidence below "
                    f"threshold for {source_key} "
                    f"→ {target_key}: "
                    f"{confidence}"
                )

                continue

            seen_sources.add(
                source_key
            )

            seen_targets.add(
                target_key
            )

            valid_links.append(
                link
            )

        return {
            "valid":
                not errors,

            "errors":
                errors,

            "valid_links":
                valid_links,
        }

    # ========================================================
    # VALIDATE PENDING
    # ========================================================

    @staticmethod
    def _validate_pending_continuations(
        pending: list[dict[str, Any]],
        known_article_keys: set[
            tuple[int, str]
        ],
    ) -> dict[str, Any]:

        errors = []

        for item in pending:

            source = (
                item.get(
                    "source"
                )
                or {}
            )

            try:

                source_key = (
                    int(
                        source.get(
                            "page"
                        )
                    ),

                    str(
                        source.get(
                            "article_id"
                        )
                    ),
                )

            except Exception:

                errors.append(
                    "Invalid pending source."
                )

                continue

            if (
                source_key
                not in known_article_keys
            ):

                errors.append(
                    "Pending continuation references "
                    "unknown source: "
                    f"{source_key}"
                )

            target_page = (
                GeminiArticleExtractor
                ._safe_int(
                    item.get(
                        "target_page"
                    )
                )
            )

            if target_page is None:

                errors.append(
                    "Pending continuation has "
                    "invalid target_page."
                )

        return {
            "valid":
                not errors,

            "errors":
                errors,
        }

    # ========================================================
    # KNOWN ARTICLE KEYS
    # ========================================================

    @staticmethod
    def _known_article_keys(
        manifest: list[
            dict[str, Any]
        ],
    ) -> set[tuple[int, str]]:

        result = set()

        for item in manifest:

            try:

                result.add(
                    (
                        int(
                            item["page"]
                        ),

                        str(
                            item["article_id"]
                        ),
                    )
                )

            except Exception:
                pass

        return result

    # ========================================================
    # PENDING ARTICLE KEYS
    # ========================================================

    @staticmethod
    def _pending_article_keys(
        pending: list[
            dict[str, Any]
        ],
    ) -> set[tuple[int, str]]:

        result = set()

        for item in pending:

            source = (
                item.get(
                    "source"
                )
                or {}
            )

            try:

                result.add(
                    (
                        int(
                            source["page"]
                        ),

                        str(
                            source["article_id"]
                        ),
                    )
                )

            except Exception:
                pass

        return result

    # ========================================================
    # LINK KEY
    # ========================================================

    @staticmethod
    def _link_key(
        link: dict[str, Any],
        side: str,
    ) -> tuple[int, str] | None:

        try:

            if side == "source":

                return (
                    int(
                        link[
                            "source_page"
                        ]
                    ),

                    str(
                        link[
                            "source_article_id"
                        ]
                    ),
                )

            return (
                int(
                    link[
                        "target_page"
                    ]
                ),

                str(
                    link[
                        "target_article_id"
                    ]
                ),
            )

        except Exception:

            return None

    # ========================================================
    # PENDING SOURCE KEY
    # ========================================================

    @staticmethod
    def _pending_source_key(
        pending: dict[str, Any],
    ) -> tuple[int, str] | None:

        source = (
            pending.get(
                "source"
            )
            or {}
        )

        try:

            return (
                int(
                    source["page"]
                ),

                str(
                    source["article_id"]
                ),
            )

        except Exception:

            return None

    # ========================================================
    # DEDUPLICATE PENDING
    # ========================================================

    @staticmethod
    def _deduplicate_pending(
        pending: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:

        result = []

        seen = set()

        for item in pending:

            key = (
                GeminiArticleExtractor
                ._pending_source_key(
                    item
                )
            )

            if key is None:
                continue

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                item
            )

        return result

    # ========================================================
    # TEXT-BASED PHYSICAL CONTINUATION REPAIR
    # ========================================================

    @staticmethod
    def _normalize_match_text(value: Any) -> str:
        """Normalize Gemini-extracted text for physical story matching."""
        if value is None:
            return ""

        if isinstance(value, list):
            value = " ".join(str(x) for x in value if x is not None)
        elif isinstance(value, dict):
            value = " ".join(str(x) for x in value.values())

        text = str(value).lower()
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _match_tokens(cls, value: Any) -> list[str]:
        """Return useful unicode word tokens, excluding common newspaper stop words."""
        text = cls._normalize_match_text(value)
        if not text:
            return []

        stop = {
            "the", "a", "an", "and", "or", "but", "of", "to", "in",
            "on", "for", "from", "with", "by", "at", "as", "is", "are",
            "was", "were", "be", "been", "being", "that", "this", "these",
            "those", "it", "its", "into", "over", "after", "before", "than",
            "then", "they", "their", "them", "he", "she", "his", "her", "we",
            "our", "you", "your", "i", "me", "my", "will", "would", "could",
            "should", "has", "have", "had", "not", "no", "also", "said", "says",
            "report", "reports", "page", "continued", "more", "see", "full",
        }

        return [
            token
            for token in text.split()
            if len(token) >= 3 and token not in stop
        ]

    @classmethod
    def _text_overlap_score(
        cls,
        source_text: str,
        target_text: str,
    ) -> tuple[float, int, float, float]:
        """
        Compare the END of a source physical article with the BEGINNING
        of a target physical article.

        Returns:
            combined_score,
            shared_token_count,
            token_jaccard,
            ngram_overlap
        """
        source_tail = str(source_text or "")[-1800:]
        target_head = str(target_text or "")[:1800]

        source_tokens = cls._match_tokens(source_tail)
        target_tokens = cls._match_tokens(target_head)

        if not source_tokens or not target_tokens:
            return 0.0, 0, 0.0, 0.0

        source_set = set(source_tokens)
        target_set = set(target_tokens)
        shared = source_set & target_set

        # Token overlap is intentionally weighted toward the target's
        # beginning because a continuation normally starts immediately
        # with the next part of the story.
        union = source_set | target_set
        token_jaccard = (
            len(shared) / len(union)
            if union else 0.0
        )

        # Word 3-grams are much harder for unrelated newspaper stories
        # to share accidentally than single keywords.
        def ngrams(tokens: list[str], n: int = 3) -> set[tuple[str, ...]]:
            if len(tokens) < n:
                return set()
            return {
                tuple(tokens[i:i + n])
                for i in range(len(tokens) - n + 1)
            }

        source_ngrams = ngrams(source_tokens)
        target_ngrams = ngrams(target_tokens)
        if source_ngrams and target_ngrams:
            ngram_overlap = len(source_ngrams & target_ngrams) / max(
                1,
                min(len(source_ngrams), len(target_ngrams)),
            )
        else:
            ngram_overlap = 0.0

        # Character similarity catches continuation text where Gemini
        # reproduced an almost identical phrase around a page break.
        source_norm = cls._normalize_match_text(source_tail)
        target_norm = cls._normalize_match_text(target_head)
        char_similarity = difflib.SequenceMatcher(
            None,
            source_norm[-900:],
            target_norm[:900],
            autojunk=False,
        ).ratio()

        # Shared-token density gives a stable signal when the target has
        # no headline and begins with proper names/numbers from the source.
        shared_density = len(shared) / max(1, min(len(source_set), len(target_set)))

        score = (
            0.38 * token_jaccard
            + 0.27 * ngram_overlap
            + 0.20 * char_similarity
            + 0.15 * shared_density
        )

        return (
            min(1.0, score),
            len(shared),
            token_jaccard,
            ngram_overlap,
        )

    @classmethod
    def _metadata_overlap_score(
        cls,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> float:
        """Small supporting score from Gemini structured knowledge."""
        fields = ("entities", "topics", "keywords")
        source_values = set()
        target_values = set()

        for field in fields:
            source_values.update(cls._match_tokens(source.get(field)))
            target_values.update(cls._match_tokens(target.get(field)))

        if not source_values or not target_values:
            return 0.0

        return len(source_values & target_values) / max(
            1,
            len(source_values | target_values),
        )

    @staticmethod
    def _headline_is_titleless(article: dict[str, Any]) -> bool:
        headline = str(article.get("headline", "") or "").strip()
        normalized = " ".join(headline.lower().split())
        return not normalized or normalized in {
            "untitled",
            "untitled article",
            "continued",
            "more on page",
            "report on page",
            "continued from page",
        }

    @classmethod
    def _continuation_marker_target_page(
        cls,
        article: dict[str, Any],
    ) -> int | None:
        """Extract an explicit continuation target page without touching crops."""
        continuation = article.get("continuation") or {}
        if not isinstance(continuation, dict):
            continuation = {}

        target = cls._safe_int(continuation.get("next_page"))
        if target is not None:
            return target

        patterns = (
            r"(?:more\s+on|continued\s+on|continued\s+from|report\s+on|full\s+report\s+on|turn\s+to|see|to\s+be\s+continued)[^\n]{0,100}?(?:page|pg\.?|p\.?)\s*[-:]?\s*(\d{1,4})\b",
            r"(?:page|pg\.?|p\.?)\s*[-:]?\s*(\d{1,4})\b",
        )

        candidates=[]
        marker=str(continuation.get("marker", "") or "").strip()
        if marker:
            candidates.append(marker)
        for field in ("article_text", "summary", "headline", "subheadline", "location"):
            value=article.get(field)
            if value:
                candidates.append(str(value))

        for candidate in candidates:
            for pattern in patterns:
                match=re.search(pattern, candidate, flags=re.IGNORECASE)
                if match:
                    page=cls._safe_int(match.group(1))
                    if page is not None:
                        return page
        return None

    @classmethod
    def _score_continuation_pair(
        cls,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any]:
        """Score a physical source -> real article target pair."""
        text_score, shared, jaccard, ngram = cls._text_overlap_score(
            source.get("article_text", ""),
            target.get("article_text", ""),
        )
        metadata_score = cls._metadata_overlap_score(source, target)
        source_type=str(source.get("content_type", "") or "").strip().lower()

        source_headline=cls._normalize_match_text(source.get("headline", ""))
        target_headline=cls._normalize_match_text(target.get("headline", ""))
        headline_score=0.0
        if source_type == "article" and source_headline and target_headline:
            headline_score=difflib.SequenceMatcher(
                None, source_headline, target_headline, autojunk=False
            ).ratio()

        titleless=cls._headline_is_titleless(target)
        if source_type in {"photo_caption", "reference"}:
            # Generic labels such as WINDOWS carry no useful headline signal.
            final_score=0.68*text_score + 0.32*metadata_score
        elif titleless:
            final_score=0.72*text_score + 0.28*metadata_score
        else:
            final_score=0.58*text_score + 0.22*metadata_score + 0.20*headline_score

        return {
            "score": round(min(1.0, final_score), 4),
            "text_score": round(text_score, 4),
            "metadata_score": round(metadata_score, 4),
            "headline_score": round(headline_score, 4),
            "shared_tokens": shared,
            "token_jaccard": round(jaccard, 4),
            "ngram_overlap": round(ngram, 4),
            "target_titleless": titleless,
            "source_content_type": source_type,
        }

    @classmethod
    def _repair_titleless_continuations(
        cls,
        articles: list[dict[str, Any]],
        continuation_links: list[dict[str, Any]],
        pending_continuations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """
        Final deterministic repair pass.

        Gemini remains responsible for reading/extracting the crops and
        detecting continuation markers.  This method only resolves the
        physical source -> target identity using the extracted text after
        ALL pages are available.

        It is specifically designed for the common newspaper case where
        the continuation crop has no repeated headline/title.
        """
        article_map: dict[tuple[int, str], dict[str, Any]] = {}
        for article in articles:
            if not isinstance(article, dict):
                continue
            try:
                key = (
                    int(article.get("page")),
                    str(article.get("article_id")),
                )
            except Exception:
                continue
            article_map[key] = article

        # Start from only valid existing Gemini links.  Invalid Gemini
        # links are not allowed to contaminate the final logical graph.
        clean_links: list[dict[str, Any]] = []
        used_sources: set[tuple[int, str]] = set()
        used_targets: set[tuple[int, str]] = set()

        for link in continuation_links:
            if not isinstance(link, dict):
                continue
            source_key = cls._link_key(link, "source")
            target_key = cls._link_key(link, "target")
            if source_key not in article_map or target_key not in article_map:
                continue
            if source_key == target_key:
                continue
            if source_key in used_sources or target_key in used_targets:
                continue

            source_ok, _ = cls._continuation_source_eligible(article_map[source_key])
            target_ok, _ = cls._continuation_eligible(article_map[target_key])
            if not source_ok or not target_ok:
                continue

            try:
                confidence = float(link.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if confidence < 0.75:
                continue

            clean_links.append(dict(link))
            used_sources.add(source_key)
            used_targets.add(target_key)

        added = 0
        corrected = 0
        unresolved = 0
        repairs = []

        # Sources are considered in page/article order so that a physical
        # story chain is reconstructed deterministically.
        sources = sorted(
            article_map.values(),
            key=lambda a: (
                cls._safe_int(a.get("page")) or 999999,
                cls._article_sort_key(str(a.get("article_id", ""))),
            ),
        )

        # A source may already have a Gemini link. We still evaluate it if
        # its explicit continuation marker points to a page and the target
        # can be text-matched more convincingly. This allows correction of
        # a wrong target_article_id.
        link_by_source = {
            cls._link_key(link, "source"): link
            for link in clean_links
        }

        for source in sources:
            source_key = (
                cls._safe_int(source.get("page")),
                str(source.get("article_id", "")),
            )
            if source_key[0] is None or not source_key[1]:
                continue

            source_ok, _ = cls._continuation_source_eligible(source)
            if not source_ok:
                continue

            target_page = cls._continuation_marker_target_page(source)
            if target_page is None:
                continue

            # A target on the explicitly referenced page is the only page
            # considered. We never search the entire newspaper and guess.
            candidates = [
                article
                for key, article in article_map.items()
                if key[0] == target_page
                and key != source_key
            ]

            candidates = [
                article
                for article in candidates
                if cls._continuation_eligible(article)[0]
            ]

            if not candidates:
                unresolved += 1
                continue

            scored = []
            for target in candidates:
                target_key = (
                    int(target["page"]),
                    str(target["article_id"]),
                )
                score = cls._score_continuation_pair(source, target)
                scored.append((score, target_key, target))

            scored.sort(
                key=lambda item: (
                    item[0]["score"],
                    item[0]["text_score"],
                    item[0]["metadata_score"],
                ),
                reverse=True,
            )

            best_score, best_key, best_target = scored[0]
            second_score = scored[1][0]["score"] if len(scored) > 1 else 0.0

            # Special sources such as WINDOWS can be short. They still require
            # an explicit marker and semantic evidence, but not 120+ chars.
            source_type = str(source.get("content_type", "") or "").strip().lower()
            titleless = bool(best_score["target_titleless"])

            if source_type in {"photo_caption", "reference"}:
                threshold = 0.38
                min_shared = 2
                min_text = 0.18
                min_metadata = 0.10
                special_source = True
            else:
                threshold = 0.62 if titleless else 0.68
                min_shared = 4 if titleless else 3
                min_text = 0.45
                min_metadata = 0.0
                special_source = False

            strong_enough = (
                best_score["score"] >= threshold
                and best_score["shared_tokens"] >= min_shared
                and best_score["text_score"] >= min_text
                and (
                    not special_source
                    or best_score["metadata_score"] >= min_metadata
                    or best_score["text_score"] >= 0.42
                )
            )

            # Avoid ambiguous matches where two target crops are nearly tied.
            if len(scored) > 1 and (best_score["score"] - second_score) < 0.045:
                strong_enough = False

            if not strong_enough:
                unresolved += 1
                repairs.append({
                    "source": {
                        "page": source_key[0],
                        "article_id": source_key[1],
                    },
                    "target_page": target_page,
                    "best_target": {
                        "page": best_key[0],
                        "article_id": best_key[1],
                    },
                    "score": best_score,
                    "second_best_score": round(second_score, 4),
                    "action": "UNRESOLVED",
                })
                continue

            existing = link_by_source.get(source_key)
            new_link = {
                "source_page": source_key[0],
                "source_article_id": source_key[1],
                "target_page": best_key[0],
                "target_article_id": best_key[1],
                "confidence": round(max(0.75, best_score["score"]), 4),
                "reason": (
                    "Text-based physical continuation match. "
                    f"text_score={best_score['text_score']:.3f}, "
                    f"metadata_score={best_score['metadata_score']:.3f}, "
                    f"shared_tokens={best_score['shared_tokens']}, "
                    f"titleless_target={best_score['target_titleless']}"
                ),
                "match_method": "physical_text_matching",
            }

            if existing is None:
                if best_key in used_targets:
                    unresolved += 1
                    continue
                clean_links.append(new_link)
                link_by_source[source_key] = new_link
                used_sources.add(source_key)
                used_targets.add(best_key)
                added += 1
                action = "ADDED"
            else:
                old_key = cls._link_key(existing, "target")
                if old_key != best_key:
                    if best_key in used_targets and old_key != best_key:
                        unresolved += 1
                        continue
                    used_targets.discard(old_key)
                    used_targets.add(best_key)
                    existing.clear()
                    existing.update(new_link)
                    corrected += 1
                    action = "CORRECTED"
                else:
                    # Keep the deterministic text evidence on the final link.
                    existing.update(new_link)
                    action = "CONFIRMED"

            repairs.append({
                "source": {
                    "page": source_key[0],
                    "article_id": source_key[1],
                },
                "target": {
                    "page": best_key[0],
                    "article_id": best_key[1],
                },
                "score": best_score,
                "action": action,
            })

        # ----------------------------------------------------
        # Remove now-resolved pending records.
        # ----------------------------------------------------
        resolved_sources = {
            cls._link_key(link, "source")
            for link in clean_links
        }

        final_pending = []
        for pending in pending_continuations:
            if not isinstance(pending, dict):
                continue
            source_key = cls._pending_source_key(pending)
            if source_key in resolved_sources:
                continue

            # All pages have now been processed. A remaining pending record
            # whose target page exists is no longer PENDING_EXTERNAL; keep it
            # as UNRESOLVED so downstream data does not claim the page is absent.
            pending_copy = dict(pending)
            target_page = cls._safe_int(pending_copy.get("target_page"))
            if target_page in {key[0] for key in article_map}:
                pending_copy["status"] = "UNRESOLVED"
            else:
                pending_copy["status"] = "PENDING_EXTERNAL"
            final_pending.append(pending_copy)

        final_pending = cls._deduplicate_pending(final_pending)

        # ----------------------------------------------------
        # Add target_article_id to source_parts.
        # ----------------------------------------------------
        # This is deliberately redundant with continuation_links. It makes
        # the logical JSON self-describing and fixes the exact field used by
        # consumers that follow source_parts directly.
        source_target_map = {
            cls._link_key(link, "source"): link
            for link in clean_links
        }

        for article in articles:
            source_key = (
                cls._safe_int(article.get("page")),
                str(article.get("article_id", "")),
            )
            link = source_target_map.get(source_key)
            if link:
                article["continuation_target"] = {
                    "target_page": link.get("target_page"),
                    "target_article_id": link.get("target_article_id"),
                    "confidence": link.get("confidence"),
                    "match_method": link.get("match_method", "gemini"),
                }
            else:
                article.pop("continuation_target", None)

        report = {
            "added": added,
            "corrected": corrected,
            "unresolved": unresolved,
            "repairs": repairs,
        }

        return clean_links, final_pending, report

    # ========================================================
    # BUILD FINAL LOGICAL ARTICLES
    # ========================================================

    @classmethod
    def _build_logical_articles(
        cls,
        articles: list[
            dict[str, Any]
        ],
        continuation_links: list[
            dict[str, Any]
        ],
        pending_continuations: list[
            dict[str, Any]
        ],
    ) -> list[
        dict[str, Any]
    ]:

        # ----------------------------------------------------
        # Article map
        # ----------------------------------------------------

        article_map = {}

        for article in articles:

            try:

                key = (
                    int(
                        article.get(
                            "page"
                        )
                    ),

                    str(
                        article.get(
                            "article_id"
                        )
                    ),
                )

            except Exception:

                continue

            article_map[
                key
            ] = article

        # ----------------------------------------------------
        # Union-Find
        # ----------------------------------------------------

        parent = {
            key: key
            for key in article_map
        }

        def find(
            key
        ):

            while parent[key] != key:

                parent[key] = (
                    parent[
                        parent[key]
                    ]
                )

                key = parent[key]

            return key

        def union(
            first,
            second,
        ):

            if (
                first not in parent
                or second not in parent
            ):
                return

            root_first = find(
                first
            )

            root_second = find(
                second
            )

            if (
                root_first
                != root_second
            ):

                parent[
                    root_second
                ] = root_first

        # ----------------------------------------------------
        # Connect continuation edges
        # ----------------------------------------------------

        valid_links = []

        for link in continuation_links:

            source_key = (
                cls._link_key(
                    link,
                    "source",
                )
            )

            target_key = (
                cls._link_key(
                    link,
                    "target",
                )
            )

            if (
                source_key is None
                or target_key is None
            ):
                continue

            if (
                source_key not in article_map
                or target_key not in article_map
            ):
                continue

            union(
                source_key,
                target_key,
            )

            valid_links.append(
                link
            )

        # ----------------------------------------------------
        # Build components
        # ----------------------------------------------------

        components = {}

        for key in article_map:

            root = find(
                key
            )

            components.setdefault(
                root,
                []
            ).append(
                key
            )

        # ----------------------------------------------------
        # Pending source keys
        # ----------------------------------------------------

        pending_source_map = {}

        for pending in (
            pending_continuations
        ):

            source_key = (
                cls._pending_source_key(
                    pending
                )
            )

            if source_key is None:
                continue

            pending_source_map[
                source_key
            ] = pending

        # ----------------------------------------------------
        # Create logical articles
        # ----------------------------------------------------

        logical_articles = []

        logical_counter = 1

        sorted_components = sorted(
            components.values(),
            key=lambda keys:
            min(
                (
                    key[0],
                    cls._article_sort_key(
                        key[1]
                    ),
                )
                for key in keys
            )
        )

        for component_keys in (
            sorted_components
        ):

            component_keys.sort(
                key=lambda key:
                (
                    key[0],
                    cls._article_sort_key(
                        key[1]
                    ),
                )
            )

            component_articles = [
                article_map[key]
                for key in component_keys
            ]

            component_links = []

            for link in valid_links:

                source_key = (
                    cls._link_key(
                        link,
                        "source",
                    )
                )

                if (
                    source_key
                    in component_keys
                ):

                    component_links.append(
                        link
                    )

            component_pending = []

            for key in component_keys:

                if key in pending_source_map:

                    component_pending.append(
                        pending_source_map[
                            key
                        ]
                    )

            if component_pending:

                status = (
                    "pending_external"
                )

            elif component_links:

                status = "merged"

            else:

                status = "standalone"

            source_parts = []

            for article in (
                component_articles
            ):

                source_part = {
                    "page": article.get("page"),
                    "article_id": article.get("article_id"),
                }

                continuation_target = article.get("continuation_target")
                if isinstance(continuation_target, dict):
                    source_part["target_page"] = continuation_target.get("target_page")
                    source_part["target_article_id"] = continuation_target.get("target_article_id")
                    source_part["target_confidence"] = continuation_target.get("confidence")
                    source_part["match_method"] = continuation_target.get("match_method")

                source_parts.append(source_part)

            logical_article = {
                "logical_article_id":
                    f"logical_{logical_counter:04d}",

                "status":
                    status,

                "source_parts":
                    source_parts,

                "continuation_links":
                    component_links,

                "pending_continuations":
                    component_pending,

                "articles":
                    component_articles,
            }

            logical_articles.append(
                logical_article
            )

            logical_counter += 1

        return logical_articles

    # ========================================================
    # SAFE INTEGER
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int | None:

        try:

            return int(
                value
            )

        except Exception:

            return None


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def extract_article_batches(
    document_dir: str | Path,
    api_key: str | None = None,
    model: str | None = None,
    pages_per_batch: int = 3,
) -> list[
    dict[str, Any]
]:

    extractor = (
        GeminiArticleExtractor(
            api_key=api_key,
            model=model,
            pages_per_batch=(
                pages_per_batch
            ),
        )
    )

    return extractor.process_document(
        document_dir
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract verified newspaper "
            "article crops directly using "
            "Gemini Vision with integrated "
            "continuation resolution."
        )
    )

    parser.add_argument(
        "document_dir",
        help=(
            "Document directory.\n"
            "Example:\n"
            "output/documents/doc_000001"
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Gemini model.\n"
            "If omitted, GEMINI_ARTICLE_MODEL "
            "or gemini-3.6-flash is used."
        ),
    )

    parser.add_argument(
        "--pages-per-batch",
        type=int,
        default=3,
        help=(
            "Number of newspaper pages in "
            "one Gemini request. Default: 3."
        ),
    )

    args = parser.parse_args()

    extract_article_batches(
        document_dir=args.document_dir,
        model=args.model,
        pages_per_batch=(
            args.pages_per_batch
        ),
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()