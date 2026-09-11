"""
Manual boundary editing for the document Viewer.

Lets a user draw a brand-new article boundary (for an article the
pipeline missed) or drag/resize an existing one directly on a page,
then re-crops that region and re-runs text extraction on it using
the same OpenAI vision path the main pipeline uses.

Editing is only supported for single-page (standalone) articles --
an article whose logical record spans multiple pages (a continuation)
is left untouched; merging continuation text correctly is a separate,
much larger problem than fixing one page's boundary.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.database.import_final_articles import (
    deterministic_uuid,
    import_document_directory,
)
from pipeline.database.db import get_connection
from pipeline.intelligence.openai_article_extractor import OpenAIArticleExtractor
from pipeline.languages.registry import resolve_language_pipeline


def _resolve_document_language(document_dir: Path) -> str:
    """
    Reads the language this document was actually processed with
    (backend/services/pipeline_service.py._save_document_metadata
    writes it to document.json's top-level "language" field), so
    manual re-extraction below can use the same LanguagePipeline the
    automatic pipeline used instead of always assuming OpenAI.
    """

    metadata_file = Path(document_dir) / "document.json"

    if not metadata_file.is_file():
        return ""

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f).get("language", "") or ""
    except Exception:
        return ""


def list_page_boundaries(
    document_dir: Path,
    page_number: int,
) -> list[dict]:

    document_dir = Path(document_dir)

    page_dir = (
        document_dir
        / "final_articles_crops"
        / f"page_{page_number:03d}"
    )

    if not page_dir.is_dir():
        return []

    final_data, _ = _load_final_logical_data(document_dir)
    multi_page_ids = _multi_page_article_ids(final_data, page_number)

    boundaries = []

    for article_dir in sorted(page_dir.iterdir()):

        if not article_dir.is_dir():
            continue

        crop_json = article_dir / "crop.json"

        if not crop_json.is_file():
            continue

        with open(crop_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        bbox = metadata.get("bbox")

        if not bbox:
            continue

        article_id = str(
            metadata.get("article_id") or article_dir.name
        )

        logical = _find_logical_article(final_data, page_number, article_id)

        boundaries.append(
            {
                "article_id": article_id,
                "bbox": bbox,
                "is_multi_page": article_id in multi_page_ids,
                # Precise sub-rectangles for display when this
                # article was reshaped to avoid enclosing a
                # neighbouring article (see Article.sub_rects in
                # pipeline/article/article_grouper.py) -- empty for
                # every other article, which keeps rendering as the
                # single `bbox` above exactly as before.
                "sub_rects": metadata.get("sub_rects") or [],
                # Layout blocks (paragraphs/headings/images) merged into
                # this crop -- a real proxy for "how much content is in
                # this article" shown in the Viewer's inspector panel.
                "block_count": len(metadata.get("block_ids") or []),
                # e.g. "final_verified_boundary" (pipeline-detected) or
                # "manual" (drawn/edited in the Viewer) -- surfaced as-is
                # rather than inventing a confidence score the backend
                # doesn't compute.
                "boundary_source": metadata.get("boundary_source"),
                # Lets the Viewer deep-link a boundary to its imported
                # article on the /search article page, when it's already
                # been grouped into a logical article.
                "logical_article_id": (
                    logical.get("logical_article_id") if logical else None
                ),
            }
        )

    return boundaries


def _extract_and_persist(
    document_dir: Path,
    page_number: int,
    article_id: str,
    bbox: dict,
    final_data: dict[str, Any],
    final_logical_path: Path,
) -> dict:
    """
    Shared core of save_boundary_and_reextract and merge_boundaries:
    crop `bbox` out of the page image, re-extract text through this
    document's own language pipeline, and persist the result as the
    physical/logical article `article_id` for `page_number`.

    Never decides WHICH article_id to write under or what to do with
    any other article -- callers own that (a plain edit keeps the
    same id; a merge picks one survivor and separately removes the
    rest -- see merge_boundaries).
    """

    page_image_path = (
        document_dir
        / "pages"
        / f"page_{page_number:03d}.png"
    )

    if not page_image_path.is_file():
        raise FileNotFoundError(
            f"Page image not found: {page_image_path}"
        )

    page_dir = (
        document_dir
        / "final_articles_crops"
        / f"page_{page_number:03d}"
    )

    page_dir.mkdir(parents=True, exist_ok=True)

    image, crop, clamped_bbox = _crop_boundary(page_image_path, bbox)

    article_dir = page_dir / article_id
    article_dir.mkdir(parents=True, exist_ok=True)

    crop_image_path = article_dir / f"page_{page_number:03d}.png"

    crop.save(crop_image_path, format="PNG")

    crop_metadata = {
        "article_id": article_id,
        "page": page_number,
        "source_page": str(page_image_path),
        "crop_path": str(crop_image_path),
        "bbox": clamped_bbox,
        "width": crop.width,
        "height": crop.height,
        "source_width": image.width,
        "source_height": image.height,
        "resized": False,
        "padding": 0,
        "boundary_source": "manual",
        "block_ids": [],
    }

    with open(article_dir / "crop.json", "w", encoding="utf-8") as f:
        json.dump(crop_metadata, f, indent=4, ensure_ascii=False)

    # Use this document's own language pipeline (see
    # pipeline/languages/) so, e.g., a Hindi document re-extracts
    # through the same provider the automatic pipeline used for it --
    # falling back to OpenAIArticleExtractor when that language's
    # extractor doesn't support single-crop re-extraction yet (today:
    # GeminiArticleExtractor only implements the batched
    # process_document() path, not _extract_single_article()).
    lang_pipeline = resolve_language_pipeline(
        _resolve_document_language(document_dir)
    )

    if hasattr(
        lang_pipeline.extractor_class,
        "_extract_single_article",
    ):
        extractor = lang_pipeline.extractor_class(
            prompt_template=(
                lang_pipeline.extraction_prompt_template
            ),
        )

    else:
        # Falling back to a different engine than this language
        # declared, so its own template (written for that other
        # engine) must NOT be forced on OpenAI's extractor -- let
        # OpenAIArticleExtractor use its own built-in template,
        # exactly as this path always has.
        extractor = OpenAIArticleExtractor()

    extracted = extractor._extract_single_article(
        page_number=page_number,
        article_id=article_id,
        image_path=str(crop_image_path),
        crop_metadata=crop_metadata,
    )

    if extracted is None:
        raise RuntimeError(
            "Text re-extraction failed for this boundary. Check "
            "OPENAI_API_KEY / model availability and try again."
        )

    physical_article = dict(extracted)
    physical_article["page"] = page_number
    physical_article["article_id"] = article_id

    _persist_physical_article(
        final_logical_path=final_logical_path,
        final_data=final_data,
        page_number=page_number,
        article_id=article_id,
        physical_article=physical_article,
    )

    return {
        "clamped_bbox": clamped_bbox,
        "physical_article": physical_article,
    }


def save_boundary_and_reextract(
    document_dir: Path,
    document_id: str,
    page_number: int,
    article_id: str | None,
    bbox: dict,
) -> dict:

    document_dir = Path(document_dir)

    final_logical_path = (
        document_dir
        / "gemini_article_batches"
        / "final_logical_articles.json"
    )

    final_data, _ = _load_final_logical_data(document_dir)

    is_new = article_id is None

    if is_new:

        page_dir = (
            document_dir
            / "final_articles_crops"
            / f"page_{page_number:03d}"
        )
        page_dir.mkdir(parents=True, exist_ok=True)
        article_id = _next_article_id(page_dir)

    else:

        existing_logical = _find_logical_article(
            final_data,
            page_number,
            article_id,
        )

        if existing_logical is not None:

            source_parts = existing_logical.get("source_parts") or []

            if len(source_parts) > 1:
                raise ValueError(
                    "This article spans multiple pages; manual "
                    "boundary editing is only supported for "
                    "single-page articles."
                )

    result = _extract_and_persist(
        document_dir=document_dir,
        page_number=page_number,
        article_id=article_id,
        bbox=bbox,
        final_data=final_data,
        final_logical_path=final_logical_path,
    )

    db_synced = False
    db_error = None

    try:
        import_document_directory(str(document_dir))
        db_synced = True

    except Exception as exc:
        db_error = str(exc)

    physical_article = result["physical_article"]

    return {
        "article_id": article_id,
        "bbox": result["clamped_bbox"],
        "headline": physical_article.get("headline"),
        "article_text": physical_article.get("article_text"),
        "is_new": is_new,
        "db_synced": db_synced,
        "db_error": db_error,
    }


def merge_boundaries(
    document_dir: Path,
    document_id: str,
    page_number: int,
    article_ids: list[str],
) -> dict:
    """
    Fuse several existing, separately-detected boundaries on one page
    into ONE article: union their boxes, re-crop that single region,
    and re-extract its text from scratch through the same vision
    extractor a manual single-boundary edit uses (see
    _extract_and_persist) -- never stitched together from the pieces'
    old, separately-extracted text.

    This exists for exactly the failure shape automatic grouping
    cannot always avoid on badly garbled OCR (confirmed on a real
    Urdu page): the SAME geometric signature -- several short items
    sitting in one row with small gaps and no printed rule line
    between them -- is equally the shape of several genuinely
    independent briefs placed side by side. Nothing purely geometric
    can tell the two apart without risking exactly the over-merge
    this codebase has already been burned by once (see
    ArticleGrouper's own "Report unclaimed content blocks" comment).
    A human confirming "these are the same story" is the actual
    disambiguating signal; this endpoint is what turns that human
    judgement into one clean article instead of leaving the reviewer
    to reconcile 2-3 separate crops/records by hand.

    The FIRST id in `article_ids` survives (keeps its identity, its
    logical_article_id, and any links already pointing at it); the
    rest are deleted, from disk and from the DB, once the merged
    article is safely persisted.
    """

    document_dir = Path(document_dir)

    if len(article_ids) < 2:
        raise ValueError(
            "Merging requires at least two article boundaries."
        )

    if len(set(article_ids)) != len(article_ids):
        raise ValueError("Duplicate article ids in merge request.")

    final_logical_path = (
        document_dir
        / "gemini_article_batches"
        / "final_logical_articles.json"
    )

    final_data, _ = _load_final_logical_data(document_dir)

    multi_page_ids = _multi_page_article_ids(final_data, page_number)

    blocked = [aid for aid in article_ids if aid in multi_page_ids]

    if blocked:
        raise ValueError(
            "These articles span multiple pages; manual boundary "
            "editing (including merging) is only supported for "
            "single-page articles: " + ", ".join(blocked)
        )

    page_dir = (
        document_dir
        / "final_articles_crops"
        / f"page_{page_number:03d}"
    )

    bboxes = []

    for article_id in article_ids:

        crop_json = page_dir / article_id / "crop.json"

        if not crop_json.is_file():
            raise FileNotFoundError(
                f"Boundary not found: {article_id}"
            )

        with open(crop_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        bbox = metadata.get("bbox")

        if not bbox:
            raise ValueError(f"Boundary {article_id} has no bbox")

        bboxes.append(bbox)

    merged_bbox = {
        "x1": min(b["x1"] for b in bboxes),
        "y1": min(b["y1"] for b in bboxes),
        "x2": max(b["x2"] for b in bboxes),
        "y2": max(b["y2"] for b in bboxes),
    }

    kept_id, *dropped_ids = article_ids

    # import_document_directory only ever inserts/updates rows that
    # are still present in final_data -- it never deletes a stale one
    # (same requirement delete_boundary already has), so each dropped
    # article's own logical_article_id has to be captured now, before
    # it's removed from final_data below, or its DB row would be
    # orphaned forever.
    dropped_logical_ids = [
        logical.get("logical_article_id")
        for logical in (
            _find_logical_article(final_data, page_number, dropped_id)
            for dropped_id in dropped_ids
        )
        if logical is not None
    ]

    result = _extract_and_persist(
        document_dir=document_dir,
        page_number=page_number,
        article_id=kept_id,
        bbox=merged_bbox,
        final_data=final_data,
        final_logical_path=final_logical_path,
    )

    for dropped_id in dropped_ids:

        crop_dir = page_dir / dropped_id

        if crop_dir.is_dir():
            shutil.rmtree(crop_dir)

        _remove_physical_article(final_data, page_number, dropped_id)

    final_logical_path.parent.mkdir(parents=True, exist_ok=True)

    with open(final_logical_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    db_synced = False
    db_error = None

    try:

        import_document_directory(str(document_dir))

        for logical_id in dropped_logical_ids:
            if logical_id:
                _delete_logical_article_from_db(document_id, logical_id)

        db_synced = True

    except Exception as exc:
        db_error = str(exc)

    physical_article = result["physical_article"]

    return {
        "article_id": kept_id,
        "bbox": result["clamped_bbox"],
        "headline": physical_article.get("headline"),
        "article_text": physical_article.get("article_text"),
        "merged_from": article_ids,
        "removed_article_ids": dropped_ids,
        "db_synced": db_synced,
        "db_error": db_error,
    }


def delete_boundary(
    document_dir: Path,
    document_id: str,
    page_number: int,
    article_id: str,
) -> dict:

    document_dir = Path(document_dir)

    final_data, final_logical_path = _load_final_logical_data(document_dir)

    existing_logical = _find_logical_article(
        final_data,
        page_number,
        article_id,
    )

    if existing_logical is not None:

        source_parts = existing_logical.get("source_parts") or []

        if len(source_parts) > 1:
            raise ValueError(
                "This article spans multiple pages; manual "
                "boundary editing is only supported for "
                "single-page articles."
            )

    logical_article_id = (
        existing_logical.get("logical_article_id")
        if existing_logical
        else None
    )

    crop_dir = (
        document_dir
        / "final_articles_crops"
        / f"page_{page_number:03d}"
        / article_id
    )

    if crop_dir.is_dir():
        shutil.rmtree(crop_dir)

    _remove_physical_article(final_data, page_number, article_id)

    final_logical_path.parent.mkdir(parents=True, exist_ok=True)

    with open(final_logical_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    db_synced = False
    db_error = None

    try:

        if logical_article_id:
            _delete_logical_article_from_db(document_id, logical_article_id)

        db_synced = True

    except Exception as exc:
        db_error = str(exc)

    return {
        "article_id": article_id,
        "deleted": True,
        "db_synced": db_synced,
        "db_error": db_error,
    }


# ================================================================
# HELPERS
# ================================================================


def _load_final_logical_data(
    document_dir: Path,
) -> tuple[dict[str, Any], Path]:

    path = (
        document_dir
        / "gemini_article_batches"
        / "final_logical_articles.json"
    )

    if not path.is_file():
        return {
            "articles": [],
            "logical_articles": [],
        }, path

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("articles", [])
    data.setdefault("logical_articles", [])

    return data, path


def _find_logical_article(
    final_data: dict[str, Any],
    page_number: int,
    article_id: str,
) -> dict | None:

    for logical in final_data.get("logical_articles", []):

        if not isinstance(logical, dict):
            continue

        for part in logical.get("source_parts") or []:

            if not isinstance(part, dict):
                continue

            if (
                str(part.get("article_id")) == str(article_id)
                and int(part.get("page", -1)) == page_number
            ):
                return logical

    return None


def _multi_page_article_ids(
    final_data: dict[str, Any],
    page_number: int,
) -> set[str]:

    ids: set[str] = set()

    for logical in final_data.get("logical_articles", []):

        if not isinstance(logical, dict):
            continue

        source_parts = logical.get("source_parts") or []

        if len(source_parts) <= 1:
            continue

        for part in source_parts:

            if not isinstance(part, dict):
                continue

            if int(part.get("page", -1)) == page_number:
                ids.add(str(part.get("article_id")))

    return ids


def _remove_physical_article(
    final_data: dict[str, Any],
    page_number: int,
    article_id: str,
) -> None:

    def matches(article: Any) -> bool:
        return (
            isinstance(article, dict)
            and str(article.get("article_id")) == str(article_id)
            and int(article.get("page", -1)) == page_number
        )

    final_data["articles"] = [
        article
        for article in final_data.get("articles", [])
        if not matches(article)
    ]

    final_data["logical_articles"] = [
        logical
        for logical in final_data.get("logical_articles", [])
        if not any(
            str(part.get("article_id")) == str(article_id)
            and int(part.get("page", -1)) == page_number
            for part in (logical.get("source_parts") or [])
        )
    ]

    final_data["article_count"] = len(final_data["articles"])
    final_data["logical_article_count"] = len(
        final_data["logical_articles"]
    )


def _delete_logical_article_from_db(
    document_id: str,
    logical_article_id: str,
) -> None:

    document_uuid = deterministic_uuid("document", document_id)

    article_uuid = deterministic_uuid(
        "article",
        f"{document_uuid}:{logical_article_id}",
    )

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                "DELETE FROM article_images WHERE article_id = %s",
                (article_uuid,),
            )

            cur.execute(
                "DELETE FROM article_segments WHERE article_id = %s",
                (article_uuid,),
            )

            cur.execute(
                "DELETE FROM articles WHERE id = %s",
                (article_uuid,),
            )

        conn.commit()


def _next_article_id(page_dir: Path) -> str:

    existing = {
        d.name
        for d in page_dir.iterdir()
        if d.is_dir()
    }

    index = 1

    while f"article_{index:03d}" in existing:
        index += 1

    return f"article_{index:03d}"


def _crop_boundary(
    page_image_path: Path,
    bbox: dict,
) -> tuple[Image.Image, Image.Image, dict]:

    image = Image.open(page_image_path).convert("RGB")

    x1 = max(0, min(int(round(bbox["x1"])), image.width))
    y1 = max(0, min(int(round(bbox["y1"])), image.height))
    x2 = max(0, min(int(round(bbox["x2"])), image.width))
    y2 = max(0, min(int(round(bbox["y2"])), image.height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "Invalid boundary: x2/y2 must be greater than x1/y1"
        )

    crop = image.crop((x1, y1, x2, y2))

    clamped_bbox = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }

    return image, crop, clamped_bbox


def _persist_physical_article(
    final_logical_path: Path,
    final_data: dict[str, Any],
    page_number: int,
    article_id: str,
    physical_article: dict,
) -> None:

    def matches(article: Any) -> bool:
        return (
            isinstance(article, dict)
            and str(article.get("article_id")) == str(article_id)
            and int(article.get("page", -1)) == page_number
        )

    # ------------------------------------------------------------
    # Flat physical-article list
    # ------------------------------------------------------------

    articles = final_data["articles"]

    replaced = False

    for index, article in enumerate(articles):

        if matches(article):
            articles[index] = physical_article
            replaced = True
            break

    if not replaced:
        articles.append(physical_article)

    # ------------------------------------------------------------
    # Logical articles
    # ------------------------------------------------------------

    logical = _find_logical_article(
        final_data,
        page_number,
        article_id,
    )

    if logical is not None:

        nested = logical.get("articles") or []

        nested_replaced = False

        for index, article in enumerate(nested):

            if matches(article):
                nested[index] = physical_article
                nested_replaced = True
                break

        if not nested_replaced:
            nested.append(physical_article)

        logical["articles"] = nested

    else:

        existing_ids = {
            str(item.get("logical_article_id"))
            for item in final_data["logical_articles"]
            if isinstance(item, dict)
        }

        next_index = len(final_data["logical_articles"]) + 1

        while f"logical_{next_index:04d}" in existing_ids:
            next_index += 1

        final_data["logical_articles"].append(
            {
                "logical_article_id": f"logical_{next_index:04d}",
                "status": "standalone",
                "source_parts": [
                    {
                        "page": page_number,
                        "article_id": article_id,
                    },
                ],
                "continuation_links": [],
                "pending_continuations": [],
                "articles": [physical_article],
            }
        )

    final_data["article_count"] = len(final_data["articles"])
    final_data["logical_article_count"] = len(
        final_data["logical_articles"]
    )

    final_logical_path.parent.mkdir(parents=True, exist_ok=True)

    with open(final_logical_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
