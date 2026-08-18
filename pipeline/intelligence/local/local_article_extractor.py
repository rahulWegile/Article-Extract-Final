from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pipeline.ocr.rapidocr_engine import RapidOCREngine
from pipeline.knowledge.embedding_generator import EmbeddingGenerator
from pipeline.knowledge.entity_extractor import EntityExtractor
from pipeline.knowledge.sentence_features import SentenceFeatures

from pipeline.intelligence.local import field_extractors
from pipeline.intelligence.local.category_classifier import CategoryClassifier
from pipeline.intelligence.local.sentiment_lexicon import classify_sentiment
from pipeline.intelligence.local.topic_keyword_extractor import (
    extract_keywords,
    extract_topics,
)
from pipeline.intelligence.local.image_block_matcher import build_images_field
from pipeline.intelligence.local.block_text_extractor import (
    extract_article_text_from_blocks,
)
from pipeline.intelligence.local import continuation_matching as cm
from pipeline.intelligence.local.validation import validate_manifest_completeness


class LocalArticleExtractor:
    """
    Local (no cloud LLM) replacement for
    `pipeline.intelligence.gemini_article_extractor.GeminiArticleExtractor`.

    Same public contract:

        LocalArticleExtractor(pages_per_batch=3).process_document(document_dir)

    writing the same `gemini_article_batches/*.json` artifacts so
    `pipeline.finalization.logical_article_builder.LogicalArticleBuilder`
    and `pipeline.database.import_final_articles` need no changes.

    Unlike the Gemini engine, there is no per-request cost to amortize,
    so pages/articles are discovered and processed in a single pass
    instead of sequential page-batches; `pages_per_batch` is accepted
    only for constructor-signature compatibility and is not used to
    drive execution.

    Article text is reconstructed from the OCR already computed once
    per page (page_json, see block_text_extractor.py) rather than
    re-running RapidOCR on every crop a second time -- both faster and,
    empirically, more accurate (avoids column-merge noise a second
    crop-level OCR pass can introduce). RapidOCR's own
    `process_article_crop` is kept only as a fallback for crops whose
    block-derived text comes back too short (e.g. crop.json predating
    the `block_ids` field).
    """

    DEFAULT_PAGES_PER_BATCH = 3

    def __init__(
        self,
        pages_per_batch: int = 3,
        ocr_engine: RapidOCREngine | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        self.pages_per_batch = max(1, int(pages_per_batch))

        self.ocr_engine = ocr_engine or RapidOCREngine()
        self.embedder = embedding_generator or EmbeddingGenerator()

        self.entity_extractor = EntityExtractor()
        self.sentence_features = SentenceFeatures()
        self.category_classifier = CategoryClassifier(self.embedder)

        # page_number -> page_json["blocks"], populated on first use per
        # page so text/image extraction across a page's many articles
        # reads page_json once instead of once per article.
        self._page_blocks_cache: dict[int, list[dict[str, Any]]] = {}

        print()
        print("=" * 60)
        print("LOCAL ARTICLE EXTRACTOR")
        print("=" * 60)
        print("OCR             : page_json block reuse (RapidOCR fallback)")
        print("Continuation    : local text/metadata overlap scoring")
        print("Cloud dependency: NONE")
        print("=" * 60)
        print()

    # ========================================================
    # DISCOVERY
    #
    # Same filesystem contract as GeminiArticleExtractor's
    # _discover_pages / _discover_articles.
    # ========================================================

    @staticmethod
    def _discover_pages(crop_root: Path) -> list[int]:

        pages = []

        if not crop_root.exists():
            return pages

        for directory in crop_root.iterdir():

            if not directory.is_dir():
                continue

            name = directory.name

            if not name.startswith("page_"):
                continue

            number_text = name[len("page_"):]

            try:
                page_number = int(number_text)
            except ValueError:
                continue

            pages.append(page_number)

        return sorted(set(pages))

    @staticmethod
    def _article_sort_key(article_id: str) -> tuple:
        return cm.article_sort_key(article_id)

    @classmethod
    def _discover_articles(
        cls,
        crop_root: Path,
        page_number: int,
    ) -> list[dict[str, Any]]:

        page_dir = crop_root / f"page_{page_number:03d}"

        if not page_dir.exists():
            return []

        articles = []

        for article_dir in page_dir.iterdir():

            if not article_dir.is_dir():
                continue

            article_id = article_dir.name

            expected_image = article_dir / f"page_{page_number:03d}.png"

            if expected_image.exists():
                image_path = expected_image
            else:
                png_files = sorted(article_dir.glob("*.png"))
                if not png_files:
                    print(f"WARNING: No PNG found: {article_dir}")
                    continue
                image_path = png_files[0]

            crop_metadata = {}
            metadata_path = article_dir / "crop.json"

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        crop_metadata = json.load(f)
                except Exception as exc:
                    print(f"WARNING: Could not read {metadata_path}: {exc}")

            articles.append({
                "page": page_number,
                "article_id": article_id,
                "image_path": image_path,
                "crop_metadata": crop_metadata,
            })

        articles.sort(key=lambda item: cls._article_sort_key(item["article_id"]))

        return articles

    # ========================================================
    # PER-ARTICLE FIELD EXTRACTION
    # ========================================================

    # Below this, block-derived text is treated as "not enough to
    # trust" and we fall back to a fresh crop-level OCR pass -- covers
    # crop.json predating the block_ids field, and any article whose
    # block_ids didn't map to text-bearing blocks for some reason.
    MIN_BLOCK_TEXT_CHARS = 20

    def _load_page_blocks(
        self,
        document_dir: Path,
        page: int,
    ) -> list[dict[str, Any]]:

        if page in self._page_blocks_cache:
            return self._page_blocks_cache[page]

        path = document_dir / "page_json" / f"page_{page:03d}.json"

        blocks: list[dict[str, Any]] = []

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    blocks = json.load(f).get("blocks") or []
            except Exception:
                blocks = []

        self._page_blocks_cache[page] = blocks

        return blocks

    def _build_article_record(
        self,
        document_dir: Path,
        page: int,
        article_id: str,
        image_path: Path,
        crop_metadata: dict[str, Any],
    ) -> dict[str, Any]:

        block_ids = crop_metadata.get("block_ids") or []
        page_blocks = self._load_page_blocks(document_dir, page)

        # Reuse the OCR already computed once per page (page_json)
        # instead of re-running RapidOCR a second time on this crop --
        # the page-level pass already ran for every page regardless.
        ocr_result = extract_article_text_from_blocks(page_blocks, block_ids)

        if len(str(ocr_result.get("text", "") or "").strip()) < self.MIN_BLOCK_TEXT_CHARS:
            ocr_result = self.ocr_engine.process_article_crop(image_path)

        article_text = str(ocr_result.get("text", "") or "").strip()

        headline, subheadline = field_extractors.extract_headline_and_subheadline(
            ocr_result
        )
        author = field_extractors.extract_author(article_text)
        location, date = field_extractors.extract_location_and_date(article_text)

        cap_score = field_extractors.caption_score(article_text)
        has_figure_block = bool(block_ids)

        content_type = field_extractors.classify_content_type(
            article_text,
            caption_score=cap_score,
            has_figure_block=has_figure_block,
        )

        summary = self.sentence_features.extract(article_text).get(
            "first_sentence", ""
        )

        category = self.category_classifier.classify(f"{headline} {summary}")
        sentiment = classify_sentiment(article_text)

        entities_by_type = self.entity_extractor.extract(article_text)
        entities = sorted({
            value
            for values in entities_by_type.values()
            for value in values
        })

        keywords = extract_keywords(article_text)
        topics = extract_topics(keywords, category)

        crop_bbox = crop_metadata.get("bbox") or {}
        images = build_images_field(
            document_dir=document_dir,
            page=page,
            block_ids=block_ids,
            crop_bbox=crop_bbox,
            blocks=page_blocks,
        )

        avg_confidence = float(ocr_result.get("confidence", 0.0) or 0.0)

        article: dict[str, Any] = {
            "page": page,
            "article_id": article_id,
            "headline": headline,
            "subheadline": subheadline,
            "author": author,
            "location": location,
            "date": date,
            "article_text": article_text,
            "summary": summary,
            "category": category,
            "entities": entities,
            "topics": topics,
            "keywords": keywords,
            "sentiment": sentiment,
            "content_type": content_type,
            "images": images,
            "quality": {
                "text_readability": (
                    "high" if avg_confidence >= 0.85
                    else "medium" if avg_confidence >= 0.6
                    else "low"
                ),
                "missing_text": len(article_text) < 20,
                "notes": None,
            },
            # Absolute, not the raw (often CWD-relative) value stored in
            # crop.json: import_final_articles.py's import_segments()
            # prefers an explicit crop_path over its own resolve_crop_path()
            # fallback (which does produce an absolute path), so a relative
            # value here would flow straight into the DB and silently break
            # article_search.py's URL building (it only recognizes absolute
            # paths containing "/output/").
            "crop_path": (
                str(Path(crop_metadata["crop_path"]).resolve())
                if crop_metadata.get("crop_path")
                else None
            ),
        }

        article["continuation"] = field_extractors.detect_continuation(article)

        return article

    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    def process_document(
        self,
        document_dir: str | Path,
    ) -> list[dict[str, Any]]:

        document_dir = Path(document_dir)

        crop_root = document_dir / "final_articles_crops"

        if not crop_root.exists():
            raise FileNotFoundError(
                f"Final article crop directory does not exist:\n{crop_root}"
            )

        pages = self._discover_pages(crop_root)

        if not pages:
            print("No final article crop pages found.")
            return []

        print()
        print("=" * 60)
        print("FINAL ARTICLE CROP DISCOVERY")
        print("=" * 60)
        print(f"Crop root : {crop_root}")
        print(f"Pages     : {len(pages)}")

        manifest_entries = []
        page_inventory: dict[int, list[dict[str, Any]]] = {}

        for page_number in pages:
            articles = self._discover_articles(crop_root, page_number)
            page_inventory[page_number] = articles
            manifest_entries.extend(
                {"page": item["page"], "article_id": item["article_id"]}
                for item in articles
            )
            print(f"Page {page_number:03d} : {len(articles)} articles")

        print("=" * 60)
        print()

        output_root = document_dir / "gemini_article_batches"
        output_root.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # OCR + field extraction, single pass over every crop.
        #
        # This is the slow part (RapidOCR + embedding-similarity
        # category classification per crop), so it's the part that
        # most needs visible per-item progress -- silence here is what
        # makes the pipeline look stuck.
        # ----------------------------------------------------

        total_articles = sum(len(v) for v in page_inventory.values())

        print()
        print("=" * 60)
        print(f"OCR + FIELD EXTRACTION (0/{total_articles})")
        print("=" * 60)

        all_articles: list[dict[str, Any]] = []
        stage_start = time.perf_counter()
        processed = 0

        for page_number in pages:
            for item in page_inventory[page_number]:

                processed += 1
                item_start = time.perf_counter()

                article = self._build_article_record(
                    document_dir=document_dir,
                    page=item["page"],
                    article_id=item["article_id"],
                    image_path=item["image_path"],
                    crop_metadata=item["crop_metadata"],
                )

                all_articles.append(article)

                item_elapsed = time.perf_counter() - item_start
                total_elapsed = time.perf_counter() - stage_start
                avg = total_elapsed / processed
                remaining = avg * (total_articles - processed)

                print(
                    f"[{processed}/{total_articles}] "
                    f"page {item['page']:03d} {item['article_id']} "
                    f"OCR'd in {item_elapsed:.2f}s "
                    f"(elapsed {total_elapsed:.1f}s, "
                    f"est. remaining {remaining:.1f}s)",
                    flush=True,
                )

        print("=" * 60)
        print(
            f"OCR + field extraction complete: {total_articles} articles "
            f"in {time.perf_counter() - stage_start:.1f}s"
        )
        print("=" * 60)
        print()

        completeness = validate_manifest_completeness(
            all_articles, manifest_entries
        )

        print(
            "Manifest completeness: "
            f"expected={completeness['expected']}, "
            f"returned={completeness['returned']}, "
            f"complete={completeness['complete']}"
        )

        # ----------------------------------------------------
        # Global continuation matching (primary mechanism, see
        # continuation_matching.match_all_continuations docstring)
        # ----------------------------------------------------

        all_links, pending_continuations, repair_report = (
            cm.match_all_continuations(
                articles=all_articles,
                continuation_links=[],
                pending_continuations=[],
            )
        )

        print(
            "Local continuation matching: "
            f"added={repair_report['added']}, "
            f"corrected={repair_report['corrected']}, "
            f"unresolved={repair_report['unresolved']}"
        )

        pending_path = output_root / "pending_continuations.json"
        with open(pending_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pending_count": len(pending_continuations),
                    "pending": pending_continuations,
                },
                f,
                indent=4,
                ensure_ascii=False,
            )

        # ----------------------------------------------------
        # Union-find merge into logical (multi-page) articles.
        # ----------------------------------------------------

        logical_articles = cm.build_logical_articles(
            articles=all_articles,
            continuation_links=all_links,
            pending_continuations=pending_continuations,
        )

        # ----------------------------------------------------
        # Final output — same shape as the Gemini engine's
        # final_logical_articles.json.
        # ----------------------------------------------------

        final_output = {
            "schema_version": "local_article_pipeline_v1",
            "document_dir": str(document_dir),
            "model": "local",
            "pages_per_batch": self.pages_per_batch,
            "article_count": len(all_articles),
            "continuation_link_count": len(all_links),
            "pending_count": len(pending_continuations),
            "logical_article_count": len(logical_articles),
            "continuation_repair": repair_report,
            "manifest_completeness": completeness,
            "articles": all_articles,
            "continuation_links": all_links,
            "pending_continuations": pending_continuations,
            "logical_articles": logical_articles,
        }

        final_path = output_root / "final_logical_articles.json"
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        # ----------------------------------------------------
        # Single synthetic "batch" result, for manifest-shape
        # compatibility with the Gemini engine's batches[] list.
        # ----------------------------------------------------

        batch_result = {
            "batch_index": 1,
            "pages": pages,
            "articles": all_articles,
            "continuation_links": all_links,
            "pending_continuations": pending_continuations,
        }

        batch_path = output_root / "batch_001.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, indent=4, ensure_ascii=False)

        results = [batch_result]

        manifest = {
            "document_dir": str(document_dir),
            "model": "local",
            "pages_per_batch": self.pages_per_batch,
            "batch_count": len(results),
            "total_articles": len(all_articles),
            "continuation_links": len(all_links),
            "pending_continuations": len(pending_continuations),
            "logical_articles": len(logical_articles),
            "continuation_repair": repair_report,
            "batches": results,
        }

        manifest_path = output_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)

        print()
        print("=" * 60)
        print("LOCAL ARTICLE EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Articles extracted    : {len(all_articles)}")
        print(f"Continuation links    : {len(all_links)}")
        print(f"Pending continuations : {len(pending_continuations)}")
        print(f"Logical articles      : {len(logical_articles)}")
        print(f"Final output          : {final_path}")
        print(f"Manifest              : {manifest_path}")
        print("=" * 60)
        print()

        return results
