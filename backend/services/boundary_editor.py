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
            }
        )

    return boundaries


def save_boundary_and_reextract(
    document_dir: Path,
    document_id: str,
    page_number: int,
    article_id: str | None,
    bbox: dict,
) -> dict:

    document_dir = Path(document_dir)

    page_image_path = (
        document_dir
        / "pages"
        / f"page_{page_number:03d}.png"
    )

    if not page_image_path.is_file():
        raise FileNotFoundError(
            f"Page image not found: {page_image_path}"
        )

    final_logical_path = (
        document_dir
        / "gemini_article_batches"
        / "final_logical_articles.json"
    )

    final_data, _ = _load_final_logical_data(document_dir)

    is_new = article_id is None

    crops_root = document_dir / "final_articles_crops"
    page_dir = crops_root / f"page_{page_number:03d}"

    if is_new:

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

    db_synced = False
    db_error = None

    try:
        import_document_directory(str(document_dir))
        db_synced = True

    except Exception as exc:
        db_error = str(exc)

    return {
        "article_id": article_id,
        "bbox": clamped_bbox,
        "headline": physical_article.get("headline"),
        "article_text": physical_article.get("article_text"),
        "is_new": is_new,
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
