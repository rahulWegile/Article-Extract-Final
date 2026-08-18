"""
Import Gemini logical newspaper articles into PostgreSQL.

Expected document structure:

output/documents/doc_000007/
├── document.json
├── final_articles_crops/
│   ├── page_001/
│   │   ├── article_001/
│   │   │   ├── page_001.png
│   │   │   └── crop.json
│   │   └── ...
│   └── ...
└── gemini_article_batches/
    └── final_logical_articles.json


IMPORTANT
---------
Gemini output contains TWO levels:

1. physical articles
   106 physical article segments

2. logical_articles
   94 logical articles

This importer MUST store:

    articles table
        = 94 logical articles

    article_segments table
        = 106 physical segments

Example:

logical_0001
    ├── page 1 / article_001
    └── page 2 / article_006

becomes:

articles:
    logical_0001

article_segments:
    logical_0001 -> page 1 -> article_001
    logical_0001 -> page 2 -> article_006
"""

from __future__ import annotations

import ast
import json
import re
import sys
import uuid

from datetime import date, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from psycopg.types.json import Jsonb

from pipeline.database.db import get_connection


# ============================================================
# JSON HELPERS
# ============================================================


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def first_value(
    data: dict,
    *keys,
    default=None,
):
    """
    Return first non-empty value.
    """

    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)

        if value is not None and value != "":
            return value

    return default


def as_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def as_json(value):
    """
    Convert Gemini metadata to PostgreSQL JSONB-safe values.
    """

    if value is None:
        return []

    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, (tuple, set)):
        return list(value)

    if isinstance(value, str):

        text = value.strip()

        if not text:
            return []

        # JSON
        try:
            parsed = json.loads(text)

            if isinstance(
                parsed,
                (list, dict),
            ):
                return parsed

            return [parsed]

        except json.JSONDecodeError:
            pass

        # Python literal
        if (
            (
                text.startswith("{")
                and text.endswith("}")
            )
            or (
                text.startswith("[")
                and text.endswith("]")
            )
            or (
                text.startswith("(")
                and text.endswith(")")
            )
        ):
            try:
                parsed = ast.literal_eval(text)

                if isinstance(
                    parsed,
                    (list, tuple, set),
                ):
                    return list(parsed)

                if isinstance(parsed, dict):
                    return parsed

            except (
                ValueError,
                SyntaxError,
            ):
                pass

        if "," in text:
            parts = [
                item.strip().strip("'\"")
                for item in text.split(",")
                if item.strip()
            ]

            if len(parts) > 1:
                return parts

        return [text]

    return [value]


# ============================================================
# DATE
# ============================================================


def parse_date(value):
    if not value:
        return None

    if isinstance(value, date):
        return value

    text = str(value).strip()

    try:
        return date.fromisoformat(
            text[:10]
        )
    except ValueError:
        return None


def get_article_date(
    article: dict,
    article_year: int | None = None,
):
    """
    Extract article's own date.

    The document/newspaper date is NOT used as the article date.
    """

    value = first_value(
        article,
        "article_date",
        "date",
        "published_date",
        "publication_date",
    )

    if value is not None:

        text = str(value).strip()

        if text:

            parsed = parse_date(text)

            if parsed:
                return parsed

            # July 13, 2026
            full_match = re.search(
                r"\b("
                r"January|February|March|April|May|June|July|"
                r"August|September|October|November|December"
                r")\s+(\d{1,2})"
                r"\s*,?\s*(\d{4})\b",
                text,
                flags=re.IGNORECASE,
            )

            if full_match:

                month_name = (
                    full_match.group(1)
                )

                day = int(
                    full_match.group(2)
                )

                year = int(
                    full_match.group(3)
                )

                try:
                    return datetime.strptime(
                        f"{month_name} {day} {year}",
                        "%B %d %Y",
                    ).date()

                except ValueError:
                    pass

            # JULY 13
            month_day_match = re.fullmatch(
                r"\s*("
                r"January|February|March|April|May|June|July|"
                r"August|September|October|November|December"
                r")\s+(\d{1,2})\s*[:.]?\s*",
                text,
                flags=re.IGNORECASE,
            )

            if (
                month_day_match
                and article_year is not None
            ):

                month_name = (
                    month_day_match.group(1)
                )

                day = int(
                    month_day_match.group(2)
                )

                try:
                    return datetime.strptime(
                        f"{month_name} {day} {article_year}",
                        "%B %d %Y",
                    ).date()

                except ValueError:
                    pass

    # Search article text for dateline.
    article_text = first_value(
        article,
        "article_text",
        "text",
        "body",
        "content",
    )

    if article_text:

        if isinstance(
            article_text,
            list,
        ):
            article_text = "\n".join(
                str(x)
                for x in article_text
                if x
            )

        beginning = (
            str(article_text)
            .strip()
            [:1000]
        )

        month_pattern = (
            r"(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
        )

        # New Delhi, July 13, 2026:
        match = re.search(
            rf"^[\s]*(?:[A-Za-z .'-]+,\s*)?"
            rf"{month_pattern}\s+(\d{{1,2}})"
            rf"\s*,?\s*(\d{{4}})\s*:",
            beginning,
            flags=re.IGNORECASE,
        )

        if match:

            month_name = match.group(1)
            day = int(match.group(2))
            year = int(match.group(3))

            try:
                return datetime.strptime(
                    f"{month_name} {day} {year}",
                    "%B %d %Y",
                ).date()

            except ValueError:
                pass

        # New Delhi, July 13:
        match = re.search(
            rf"^[\s]*(?:[A-Za-z .'-]+,\s*)?"
            rf"{month_pattern}\s+(\d{{1,2}})\s*:",
            beginning,
            flags=re.IGNORECASE,
        )

        if (
            match
            and article_year is not None
        ):

            month_name = match.group(1)
            day = int(match.group(2))

            try:
                return datetime.strptime(
                    f"{month_name} {day} {article_year}",
                    "%B %d %Y",
                ).date()

            except ValueError:
                pass

    return None



def get_logical_article_date(
    article: dict,
    publish_date: date | None = None,
    article_year: int | None = None,
):
    """
    Resolve the date for a logical article.

    Rules:
    1. Inspect EVERY physical segment, not only the first segment.
    2. Ignore non-date values such as "Monday" / "Sunday".
    3. Prefer an explicit date found in any segment.
    4. Never allow an article date later than the newspaper publish date.
    5. If no valid article date exists, return the newspaper publish date.
    6. When multiple valid segment dates exist, use the earliest valid
       date. Continued segments normally contain the same article date;
       choosing the earliest valid date is deterministic and prevents a
       later/incorrect segment date from moving the article forward.
    """

    physical_articles = article.get(
        "_physical_articles",
        [],
    )

    if not isinstance(physical_articles, list):
        physical_articles = []

    candidates = []

    # Check every physical segment.
    for physical in physical_articles:

        if not isinstance(physical, dict):
            continue

        segment_date = get_article_date(
            physical,
            article_year,
        )

        if segment_date is None:
            continue

        # Article date must never be greater than the newspaper
        # publication date.
        if (
            publish_date is not None
            and segment_date > publish_date
        ):
            continue

        candidates.append(segment_date)

    # Also check the normalized logical article itself as a fallback.
    # This preserves support for logical-level date fields.
    if not candidates:

        logical_date = get_article_date(
            article,
            article_year,
        )

        if logical_date is not None:

            if (
                publish_date is None
                or logical_date <= publish_date
            ):
                candidates.append(
                    logical_date
                )

    if candidates:
        return min(candidates)

    # No valid article-specific date:
    # use the newspaper publish date.
    return publish_date


# ============================================================
# PHYSICAL ARTICLE HELPERS
# ============================================================


def get_physical_article_id(
    article: dict,
) -> str:

    return str(
        first_value(
            article,
            "article_id",
            "source_article_id",
            "physical_article_id",
            "id",
            default="",
        )
    )


def get_title(
    article: dict,
) -> str:

    value = first_value(
        article,
        "headline",
        "title",
        "article_title",
    )

    if value:
        return str(value).strip()

    return "Untitled Article"


def get_author(
    article: dict,
):
    """
    Extract an author from a single physical/logical article object.

    Empty values and whitespace-only values are ignored.
    """

    value = first_value(
        article,
        "author",
        "byline",
        "writer",
        "reporter",
    )

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def get_logical_article_author(
    article: dict,
):
    """
    Resolve the author for a logical article.

    Rules:
    1. Check every physical segment in page order.
    2. Ignore empty/None author values.
    3. Use the first meaningful author found.
    4. If no physical segment has an author, check the logical
       article itself.
    5. If no author exists anywhere, return None.

    This is important for continued articles where the first page
    has no byline but a continuation page does.
    """

    physical_articles = article.get(
        "_physical_articles",
        [],
    )

    if isinstance(physical_articles, list):
        for physical in physical_articles:
            if not isinstance(physical, dict):
                continue

            author = get_author(physical)

            if author:
                return author

    # Fallback to logical-level author, if present.
    return get_author(article)


def get_text(
    article: dict,
) -> str:

    value = first_value(
        article,
        "article_text",
        "text",
        "body",
        "content",
    )

    if value is None:
        return ""

    if isinstance(
        value,
        list,
    ):

        parts = []

        for item in value:

            if item is None:
                continue

            item_text = str(item).strip()

            if item_text:
                parts.append(item_text)

        return "\n\n".join(parts)

    return str(value).strip()


def get_summary(
    article: dict,
):

    value = first_value(
        article,
        "summary",
        "short_summary",
        "abstract",
    )

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def get_category(
    article: dict,
):

    value = first_value(
        article,
        "category",
        "section",
        "section_name",
    )

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def get_sentiment(
    article: dict,
):

    value = first_value(
        article,
        "sentiment",
    )

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def get_entities(
    article: dict,
):
    return as_json(
        first_value(
            article,
            "entities",
            default=[],
        )
    )


def get_topics(
    article: dict,
):
    return as_json(
        first_value(
            article,
            "topics",
            default=[],
        )
    )


def get_keywords(
    article: dict,
):
    return as_json(
        first_value(
            article,
            "keywords",
            default=[],
        )
    )


def get_location(
    article: dict,
):

    value = first_value(
        article,
        "location",
        "dateline",
        "place",
    )

    if value is None:
        return None

    value = str(value).strip()

    return value or None


# ============================================================
# UNIQUE JSON MERGE
# ============================================================


def merge_json_values(
    values: list[Any],
):
    """
    Merge lists/dicts while removing duplicate list values.
    """

    merged = []

    for value in values:

        normalized = as_json(value)

        if isinstance(
            normalized,
            dict,
        ):
            items = [normalized]
        else:
            items = normalized

        for item in items:

            if item not in merged:
                merged.append(item)

    return merged


# ============================================================
# CROP PATH
# ============================================================


def resolve_crop_path(
    document_path: Path,
    page: int | None,
    article_id: str,
):
    """
    Resolve verified article crop from:

    final_articles_crops/
        page_001/
            article_001/
                page_001.png
    """

    if page is None or not article_id:
        return None

    crop_root = (
        document_path
        / "final_articles_crops"
    )

    article_dir = (
        crop_root
        / f"page_{page:03d}"
        / article_id
    )

    expected = (
        article_dir
        / f"page_{page:03d}.png"
    )

    if expected.is_file():
        return str(expected)

    if article_dir.is_dir():

        pngs = sorted(
            article_dir.glob("*.png")
        )

        if pngs:
            return str(pngs[0])

    return None


# ============================================================
# CROP METADATA
# ============================================================


def load_crop_metadata(
    document_path: Path,
    page: int | None,
    article_id: str,
):
    if page is None or not article_id:
        return {}

    crop_json = (
        document_path
        / "final_articles_crops"
        / f"page_{page:03d}"
        / article_id
        / "crop.json"
    )

    if not crop_json.is_file():
        return {}

    try:
        return load_json(crop_json)

    except Exception as exc:

        print(
            "WARNING: Could not read crop metadata: "
            f"{crop_json} -> {exc}"
        )

        return {}


def get_bbox(
    article: dict,
    crop_metadata: dict,
):
    value = first_value(
        crop_metadata,
        "bbox",
        "bounding_box",
    )

    if value is not None:
        return value

    return first_value(
        article,
        "bbox",
        "bounding_box",
    )


# ============================================================
# IMAGE HELPERS
# ============================================================


def get_images(
    article: dict,
) -> list[dict]:
    """
    Extract Gemini image records.

    Gemini normally returns:

        "images": {
            "has_images": true,
            "image_count": 1,
            "items": [
                {
                    "image_id": "...",
                    "image_description": "...",
                    "image_bbox": {...},
                    "caption": "...",
                    "confidence": 0.95
                }
            ]
        }

    The important part is the nested `items` list.
    """

    images = first_value(
        article,
        "images",
        "article_images",
        "image_information",
        default=[],
    )

    if isinstance(images, dict):
        images = first_value(
            images,
            "items",
            "images",
            default=[],
        )

    if not isinstance(images, list):
        return []

    return [
        item
        for item in images
        if isinstance(item, dict)
    ]


def image_path(
    image: dict,
):
    """
    Return an explicitly supplied image path, if Gemini/output
    already contains one.

    Gemini's current image schema normally contains metadata only,
    so resolve_image_path() is used as the fallback.
    """

    return first_value(
        image,
        "image_path",
        "path",
        "file_path",
        "output_path",
        "image",
    )


def resolve_image_path(
    document_path: Path,
    physical_article: dict,
    image: dict,
):
    """
    Resolve the actual image file for a Gemini-detected image.

    Gemini normally returns image metadata (bbox/caption/etc.) rather
    than a filesystem path. If an explicit path exists, use it.
    Otherwise crop the detected image from the verified article crop
    using Gemini's image_bbox and save the result under:

        document_dir/article_images/page_XXX/article_XXX/

    This makes the image a real file that the frontend can display.
    """

    explicit = image_path(image)

    if explicit:
        p = Path(str(explicit))

        if p.is_file():
            return str(p)

        if not p.is_absolute():
            candidate = document_path / p
            if candidate.is_file():
                return str(candidate)

    page_value = first_value(
        physical_article,
        "page",
        "page_number",
        "source_page",
        "page_num",
    )

    try:
        page = int(page_value)
    except (TypeError, ValueError):
        return None

    source_article_id = get_physical_article_id(
        physical_article
    )

    if not source_article_id:
        return None

    crop_path = resolve_crop_path(
        document_path,
        page,
        source_article_id,
    )

    if not crop_path:
        print(
            "WARNING: Cannot resolve article crop for image: "
            f"page={page}, article={source_article_id}"
        )
        return None

    bbox = image_bbox(image)

    if not bbox:
        print(
            "WARNING: Gemini image has no image_bbox: "
            f"page={page}, article={source_article_id}, "
            f"image_id={image.get('image_id')}"
        )
        return None

    try:
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        print(
            "WARNING: Invalid image_bbox: "
            f"{bbox}"
        )
        return None

    try:
        with Image.open(crop_path) as source:
            source = source.convert("RGB")
            width, height = source.size

            # Gemini often returns image coordinates in a 0..1000
            # coordinate system. If the bbox fits that convention
            # while the actual crop is a different size, scale it.
            max_coord = max(
                abs(x1),
                abs(y1),
                abs(x2),
                abs(y2),
            )

            if max_coord <= 1000 and (
                x2 > width
                or y2 > height
                or (
                    width != 1000
                    and height != 1000
                    and (
                        x2 <= 1000
                        and y2 <= 1000
                    )
                )
            ):
                scale_x = width / 1000.0
                scale_y = height / 1000.0

                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y

            # Clamp to the real image.
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            left = int(round(min(x1, x2)))
            top = int(round(min(y1, y2)))
            right = int(round(max(x1, x2)))
            bottom = int(round(max(y1, y2)))

            if right <= left or bottom <= top:
                print(
                    "WARNING: Image bbox produced an empty crop: "
                    f"{bbox}"
                )
                return None

            # Avoid creating a useless near-full-page image when
            # Gemini's bbox is clearly invalid.
            area_ratio = (
                (right - left) * (bottom - top)
            ) / float(width * height)

            if area_ratio > 0.98:
                print(
                    "WARNING: Image bbox covers almost the entire "
                    "article crop; skipping as image."
                )
                return None

            image_id = str(
                first_value(
                    image,
                    "image_id",
                    "id",
                    default="image",
                )
            )

            safe_image_id = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                image_id,
            )

            output_dir = (
                document_path
                / "article_images"
                / f"page_{page:03d}"
                / source_article_id
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_dir
                / f"{safe_image_id}.jpg"
            )

            cropped = source.crop(
                (
                    left,
                    top,
                    right,
                    bottom,
                )
            )

            cropped.save(
                output_path,
                "JPEG",
                quality=95,
            )

            return str(output_path)

    except Exception as exc:
        print(
            "WARNING: Failed to materialize article image: "
            f"{exc}"
        )
        return None


def image_description(
    image: dict,
):
    return first_value(
        image,
        "description",
        "image_description",
    )


def image_caption(
    image: dict,
):
    return first_value(
        image,
        "caption",
    )


def image_confidence(
    image: dict,
):
    value = first_value(
        image,
        "confidence",
        "score",
    )

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def image_bbox(
    image: dict,
):
    value = first_value(
        image,
        "image_bbox",
        "bbox_pixels",
        "bbox_normalized",
        "bbox",
        "bounding_box",
    )

    if not isinstance(value, dict):
        return {}

    return value


# ============================================================
# LOGICAL ARTICLE DISCOVERY
# ============================================================


def discover_logical_articles(
    final_data: Any,
) -> list[dict]:
    """
    IMPORTANT:

    If final_logical_articles.json contains:

        {
            "articles": [...],
            "logical_articles": [...]
        }

    we MUST use logical_articles.

    We intentionally do NOT use the top-level
    physical "articles" array.
    """

    if isinstance(
        final_data,
        list,
    ):
        raise ValueError(
            "Expected Gemini logical article JSON object "
            "containing 'logical_articles'."
        )

    if not isinstance(
        final_data,
        dict,
    ):
        raise ValueError(
            "final_logical_articles.json must contain "
            "a JSON object."
        )

    logical_articles = (
        final_data.get(
            "logical_articles"
        )
    )

    if not isinstance(
        logical_articles,
        list,
    ):
        raise ValueError(
            "final_logical_articles.json does not contain "
            "a valid 'logical_articles' list."
        )

    result = [
        item
        for item in logical_articles
        if isinstance(item, dict)
    ]

    return result


# ============================================================
# BUILD IMPORTABLE LOGICAL ARTICLE
# ============================================================


def build_logical_article(
    logical: dict,
):
    """
    Convert Gemini logical article:

        {
            logical_article_id,
            status,
            source_parts,
            continuation_links,
            pending_continuations,
            articles: [...]
        }

    into one DB article object.

    Physical articles become article_segments.
    """

    physical_articles = [
        item
        for item in as_list(
            logical.get("articles")
        )
        if isinstance(item, dict)
    ]

    if not physical_articles:

        # A logical article without nested physical articles
        # should not silently become an empty article.
        raise ValueError(
            "Logical article has no nested physical articles: "
            f"{logical.get('logical_article_id')}"
        )

    # --------------------------------------------------------
    # Sort physical segments by page
    # --------------------------------------------------------

    def sort_key(article):

        try:
            page = int(
                first_value(
                    article,
                    "page",
                    "page_number",
                    "source_page",
                    default=999999,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            page = 999999

        return (
            page,
            get_physical_article_id(article),
        )

    physical_articles.sort(
        key=sort_key
    )

    # --------------------------------------------------------
    # Main metadata
    #
    # First physical article is normally the source/original.
    # --------------------------------------------------------

    primary = physical_articles[0]

    logical_id = str(
        first_value(
            logical,
            "logical_article_id",
            default="",
        )
    ).strip()

    if not logical_id:

        raise ValueError(
            "Logical article is missing logical_article_id."
        )

    # --------------------------------------------------------
    # Merge article text
    # --------------------------------------------------------

    text_parts = []

    for physical in physical_articles:

        text = get_text(
            physical
        )

        if text:
            text_parts.append(text)

    combined_text = "\n\n".join(
        text_parts
    )

    # --------------------------------------------------------
    # Merge metadata
    # --------------------------------------------------------

    entities = merge_json_values(
        [
            get_entities(x)
            for x in physical_articles
        ]
    )

    topics = merge_json_values(
        [
            get_topics(x)
            for x in physical_articles
        ]
    )

    keywords = merge_json_values(
        [
            get_keywords(x)
            for x in physical_articles
        ]
    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    all_images = []

    for physical in physical_articles:

        for image in get_images(
            physical
        ):

            image_copy = dict(image)

            if not image_copy.get(
                "article_id"
            ):

                image_copy[
                    "article_id"
                ] = get_physical_article_id(
                    physical
                )

            if not image_copy.get(
                "page"
            ):

                image_copy[
                    "page"
                ] = first_value(
                    physical,
                    "page",
                    "page_number",
                    "source_page",
                )

            all_images.append(
                image_copy
            )

    # --------------------------------------------------------
    # Build logical DB object
    # --------------------------------------------------------

    result = dict(
        primary
    )

    result[
        "logical_article_id"
    ] = logical_id

    result[
        "status"
    ] = logical.get(
        "status"
    )

    result[
        "article_text"
    ] = combined_text

    result[
        "summary"
    ] = get_summary(primary)

    result[
        "category"
    ] = get_category(primary)

    result[
        "sentiment"
    ] = get_sentiment(primary)

    result[
        "entities"
    ] = entities

    result[
        "topics"
    ] = topics

    result[
        "keywords"
    ] = keywords

    result[
        "images"
    ] = all_images

    result[
        "_physical_articles"
    ] = physical_articles

    result[
        "_continuation_links"
    ] = logical.get(
        "continuation_links",
        [],
    )

    result[
        "_pending_continuations"
    ] = logical.get(
        "pending_continuations",
        [],
    )

    return result


# ============================================================
# DETERMINISTIC UUID
# ============================================================


def deterministic_uuid(
    namespace: str,
    value: str,
) -> str:

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"newspaper-archive:{namespace}:{value}",
        )
    )


# ============================================================
# DOCUMENT INSERT
# ============================================================


def import_document(
    cur,
    document_data: dict,
    document_path: Path,
):

    document_id = str(
        document_data["document_id"]
    )

    filename = document_data.get(
        "pdf_name"
    )

    source_path = str(
        document_path
    )

    document_uuid = deterministic_uuid(
        "document",
        document_id,
    )

    cur.execute(
        """
        INSERT INTO documents (
            id,
            filename,
            source_path,
            newspaper_name,
            publish_date
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT (id)
        DO UPDATE SET
            filename = EXCLUDED.filename,
            source_path = EXCLUDED.source_path,
            newspaper_name = EXCLUDED.newspaper_name,
            publish_date = EXCLUDED.publish_date
        """,
        (
            document_uuid,
            filename,
            source_path,
            document_data.get("newspaper_name"),
            parse_date(
                document_data.get("publish_date")
            ),
        ),
    )

    return document_uuid


# ============================================================
# ARTICLE INSERT
# ============================================================


def import_article(
    cur,
    article: dict,
    index: int,
    document_uuid: str,
    article_year: int | None,
    publish_date: date | None,
):

    logical_article_id = str(
        article[
            "logical_article_id"
        ]
    )

    title = get_title(
        article
    )

    author = get_logical_article_author(
        article
    )

    article_date = get_logical_article_date(
        article,
        publish_date,
        article_year,
    )

    text = get_text(
        article
    )

    summary = get_summary(
        article
    )

    category = get_category(
        article
    )

    sentiment = get_sentiment(
        article
    )

    entities = get_entities(
        article
    )

    topics = get_topics(
        article
    )

    keywords = get_keywords(
        article
    )

    images = get_images(
        article
    )

    has_images = (
        len(images) > 0
    )

    composed_image_path = first_value(
        article,
        "composed_image_path",
        "composite_image_path",
        "composed_image",
        "final_image",
    )

    article_uuid = deterministic_uuid(
        "article",
        f"{document_uuid}:{logical_article_id}",
    )

    # Remove old child rows.
    cur.execute(
        """
        DELETE FROM article_images
        WHERE article_id = %s
        """,
        (
            article_uuid,
        ),
    )

    cur.execute(
        """
        DELETE FROM article_segments
        WHERE article_id = %s
        """,
        (
            article_uuid,
        ),
    )

    cur.execute(
        """
        INSERT INTO articles (
            id,
            document_id,
            logical_article_id,
            title,
            author,
            article_date,
            article_text,
            summary,
            category,
            sentiment,
            entities,
            topics,
            keywords,
            has_images,
            image_count,
            composed_image_path
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT (
            document_id,
            logical_article_id
        )
        DO UPDATE SET
            title =
                EXCLUDED.title,
            author =
                EXCLUDED.author,
            article_date =
                EXCLUDED.article_date,
            article_text =
                EXCLUDED.article_text,
            summary =
                EXCLUDED.summary,
            category =
                EXCLUDED.category,
            sentiment =
                EXCLUDED.sentiment,
            entities =
                EXCLUDED.entities,
            topics =
                EXCLUDED.topics,
            keywords =
                EXCLUDED.keywords,
            has_images =
                EXCLUDED.has_images,
            image_count =
                EXCLUDED.image_count,
            composed_image_path =
                EXCLUDED.composed_image_path
        """,
        (
            article_uuid,
            document_uuid,
            logical_article_id,
            title,
            author,
            article_date,
            text,
            summary,
            category,
            sentiment,
            Jsonb(entities),
            Jsonb(topics),
            Jsonb(keywords),
            has_images,
            len(images),
            composed_image_path,
        ),
    )

    return (
        article_uuid,
        images,
    )


# ============================================================
# SEGMENTS
# ============================================================


def import_segments(
    cur,
    article_uuid: str,
    article: dict,
    document_path: Path,
):

    physical_articles = article.get(
        "_physical_articles",
        [],
    )

    if not physical_articles:

        raise ValueError(
            "No physical article segments found for "
            f"logical article {article.get('logical_article_id')}"
        )

    segment_ids = []

    for order, physical in enumerate(
        physical_articles,
        start=1,
    ):

        page_value = first_value(
            physical,
            "page",
            "page_number",
            "source_page",
            "page_num",
        )

        try:
            page = int(
                page_value
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "Invalid/missing page for physical article: "
                f"{get_physical_article_id(physical)}"
            )

        source_article_id = (
            get_physical_article_id(
                physical
            )
        )

        if not source_article_id:

            raise ValueError(
                "Physical article is missing article_id. "
                f"logical_article="
                f"{article.get('logical_article_id')}"
            )

        crop_metadata = (
            physical.get(
                "crop_metadata",
                {},
            )
        )

        if not isinstance(
            crop_metadata,
            dict,
        ):
            crop_metadata = {}

        # If Gemini output does not contain crop metadata,
        # load it from the verified crop directory.
        if not crop_metadata:

            crop_metadata = load_crop_metadata(
                document_path,
                page,
                source_article_id,
            )

        crop_path = first_value(
            physical,
            "crop_path",
            "image_path",
            "source_crop",
        )

        # Gemini may not return the crop path.
        if not crop_path:

            crop_path = resolve_crop_path(
                document_path,
                page,
                source_article_id,
            )

        bbox = get_bbox(
            physical,
            crop_metadata,
        )

        segment_uuid = deterministic_uuid(
            "segment",
            f"{article_uuid}:{order}",
        )

        cur.execute(
            """
            INSERT INTO article_segments (
                id,
                article_id,
                page_number,
                source_article_id,
                segment_order,
                crop_path,
                bbox
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            ON CONFLICT (
                article_id,
                segment_order
            )
            DO UPDATE SET
                page_number =
                    EXCLUDED.page_number,
                source_article_id =
                    EXCLUDED.source_article_id,
                crop_path =
                    EXCLUDED.crop_path,
                bbox =
                    EXCLUDED.bbox
            """,
            (
                segment_uuid,
                article_uuid,
                page,
                source_article_id,
                order,
                crop_path,
                (
                    Jsonb(bbox)
                    if bbox is not None
                    else None
                ),
            ),
        )

        segment_ids.append(
            (
                segment_uuid,
                page,
                source_article_id,
            )
        )

    return segment_ids


# ============================================================
# IMAGES
# ============================================================


def import_images(
    cur,
    article_uuid: str,
    article: dict,
    segment_ids,
    document_path: Path,
):
    """
    Insert every Gemini-detected image from every physical segment
    belonging to the logical article.

    `article["_physical_articles"]` contains the individual physical
    page/article records. This is important for continued articles:
    images from Page 1 and Page 2/3/etc. must all be attached to the
    same logical article.
    """

    physical_articles = article.get(
        "_physical_articles",
        [],
    )

    total_inserted = 0

    for physical in physical_articles:

        page_value = first_value(
            physical,
            "page",
            "page_number",
            "source_page",
            "page_num",
        )

        try:
            page = int(page_value)
        except (TypeError, ValueError):
            page = None

        source_article_id = (
            get_physical_article_id(physical)
        )

        images = get_images(physical)

        if not images:
            continue

        # Find the database segment corresponding to this physical
        # page/article.
        segment_uuid = None

        for (
            candidate_id,
            candidate_page,
            candidate_article_id,
        ) in segment_ids:

            if (
                source_article_id
                and candidate_article_id
                == source_article_id
            ):
                segment_uuid = candidate_id
                break

            if (
                segment_uuid is None
                and page is not None
                and candidate_page == page
            ):
                segment_uuid = candidate_id

        for index, image in enumerate(
            images,
            start=1,
        ):

            path = resolve_image_path(
                document_path,
                physical,
                image,
            )

            if not path:
                print(
                    "WARNING: Gemini detected an image, "
                    "but no individual image file could be "
                    "resolved."
                )
                print(
                    f"  logical_article_id = "
                    f"{article.get('logical_article_id')}"
                )
                print(
                    f"  page               = {page}"
                )
                print(
                    f"  source_article_id = "
                    f"{source_article_id}"
                )
                print(
                    f"  image_id           = "
                    f"{image.get('image_id')}"
                )
                continue

            bbox = image_bbox(image)

            image_id = first_value(
                image,
                "image_id",
                "id",
                default=f"image_{index:03d}",
            )

            image_uuid = deterministic_uuid(
                "image",
                (
                    f"{article_uuid}:"
                    f"{source_article_id}:"
                    f"{image_id}:"
                    f"{path}"
                ),
            )

            width = first_value(
                image,
                "width",
            )

            height = first_value(
                image,
                "height",
            )

            cur.execute(
                """
                INSERT INTO article_images (
                    id,
                    article_id,
                    segment_id,
                    page_number,
                    source_article_id,
                    image_path,
                    description,
                    caption,
                    confidence,
                    bbox_x1,
                    bbox_y1,
                    bbox_x2,
                    bbox_y2,
                    width,
                    height
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (id)
                DO UPDATE SET
                    segment_id =
                        EXCLUDED.segment_id,
                    page_number =
                        EXCLUDED.page_number,
                    source_article_id =
                        EXCLUDED.source_article_id,
                    image_path =
                        EXCLUDED.image_path,
                    description =
                        EXCLUDED.description,
                    caption =
                        EXCLUDED.caption,
                    confidence =
                        EXCLUDED.confidence,
                    bbox_x1 =
                        EXCLUDED.bbox_x1,
                    bbox_y1 =
                        EXCLUDED.bbox_y1,
                    bbox_x2 =
                        EXCLUDED.bbox_x2,
                    bbox_y2 =
                        EXCLUDED.bbox_y2,
                    width =
                        EXCLUDED.width,
                    height =
                        EXCLUDED.height
                """,
                (
                    image_uuid,
                    article_uuid,
                    segment_uuid,
                    page,
                    source_article_id,
                    str(path),
                    image_description(image),
                    image_caption(image),
                    image_confidence(image),
                    bbox.get("x1"),
                    bbox.get("y1"),
                    bbox.get("x2"),
                    bbox.get("y2"),
                    width,
                    height,
                ),
            )

            total_inserted += 1

    return total_inserted


# ============================================================
# MAIN IMPORT
# ============================================================


def import_document_directory(
    document_dir: str,
):

    document_path = (
        Path(document_dir)
        .resolve()
    )

    document_json = (
        document_path
        / "document.json"
    )

    final_logical_json = (
        document_path
        / "gemini_article_batches"
        / "final_logical_articles.json"
    )

    print(
        "=" * 70
    )

    print(
        "NEWSPAPER DATABASE IMPORT"
    )

    print(
        "=" * 70
    )

    print(
        f"Document directory : "
        f"{document_path}"
    )

    print(
        f"Document metadata  : "
        f"{document_json}"
    )

    print(
        f"Articles input     : "
        f"{final_logical_json}"
    )

    print(
        "Input format       : "
        "Gemini logical articles"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not document_json.is_file():

        raise FileNotFoundError(
            f"Missing document metadata: "
            f"{document_json}"
        )

    if not final_logical_json.is_file():

        raise FileNotFoundError(
            "Missing Gemini logical articles file: "
            f"{final_logical_json}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    document_data = load_json(
        document_json
    )

    final_data = load_json(
        final_logical_json
    )

    # --------------------------------------------------------
    # CRITICAL FIX
    #
    # Use logical_articles.
    # NEVER use top-level physical articles.
    # --------------------------------------------------------

    logical_articles_raw = (
        discover_logical_articles(
            final_data
        )
    )

    if not logical_articles_raw:

        raise RuntimeError(
            "No logical articles found in "
            "final_logical_articles.json"
        )

    # --------------------------------------------------------
    # Build importable logical records
    # --------------------------------------------------------

    logical_articles = []

    physical_segment_count = 0

    for logical in logical_articles_raw:

        normalized = (
            build_logical_article(
                logical
            )
        )

        logical_articles.append(
            normalized
        )

        physical_segment_count += len(
            normalized[
                "_physical_articles"
            ]
        )

    # --------------------------------------------------------
    # Document metadata
    # --------------------------------------------------------

    raw_document_id = (
        document_data.get(
            "document_id"
        )
    )

    if raw_document_id in (
        None,
        "",
    ):

        raise ValueError(
            "document.json is missing "
            "a non-empty document_id"
        )

    document_id = str(
        raw_document_id
    )

    newspaper_name = (
        document_data.get(
            "newspaper_name"
        )
    )

    edition = (
        document_data.get(
            "edition"
        )
    )

    document_publish_date = (
        parse_date(
            document_data.get(
                "publish_date"
            )
        )
    )

    article_year = (
        document_publish_date.year
        if document_publish_date
        else None
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        f"Document ID        : "
        f"{document_id}"
    )

    print(
        f"Newspaper          : "
        f"{newspaper_name}"
    )

    print(
        f"Edition            : "
        f"{edition}"
    )

    print(
        f"Article date year  : "
        f"{article_year}"
    )

    print(
        f"Logical articles   : "
        f"{len(logical_articles)}"
    )

    print(
        f"Physical segments  : "
        f"{physical_segment_count}"
    )

    print()

    # --------------------------------------------------------
    # Database transaction
    # --------------------------------------------------------

    with get_connection() as conn:

        with conn.cursor() as cur:

            # ------------------------------------------------
            # Document
            # ------------------------------------------------

            document_uuid = (
                import_document(
                    cur,
                    document_data,
                    document_path,
                )
            )

            print(
                f"✓ Document stored: "
                f"{document_uuid}"
            )

            # ------------------------------------------------
            # Logical articles
            # ------------------------------------------------

            total_segments = 0

            total_images = 0

            for index, article in enumerate(
                logical_articles,
                start=1,
            ):

                logical_article_id = (
                    article[
                        "logical_article_id"
                    ]
                )

                title = get_title(
                    article
                )

                physical_count = len(
                    article[
                        "_physical_articles"
                    ]
                )

                print(
                    f"[{index:03d}/"
                    f"{len(logical_articles):03d}] "
                    f"{logical_article_id} | "
                    f"{title[:70]} "
                    f"[segments={physical_count}]"
                )

                # --------------------------------------------
                # Logical article
                # --------------------------------------------

                (
                    article_uuid,
                    image_records,
                ) = import_article(
                    cur,
                    article,
                    index,
                    document_uuid,
                    article_year,
                    document_publish_date,
                )

                # --------------------------------------------
                # Physical segments
                # --------------------------------------------

                segment_ids = (
                    import_segments(
                        cur,
                        article_uuid,
                        article,
                        document_path,
                    )
                )

                # --------------------------------------------
                # Images
                # --------------------------------------------

                inserted_images = import_images(
                    cur,
                    article_uuid,
                    article,
                    segment_ids,
                    document_path,
                )

                total_segments += len(
                    segment_ids
                )

                total_images += inserted_images

            # ------------------------------------------------
            # Commit
            # ------------------------------------------------

            conn.commit()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "DATABASE IMPORT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Documents          : 1"
    )

    print(
        f"Logical articles   : "
        f"{len(logical_articles)}"
    )

    print(
        f"Physical segments  : "
        f"{total_segments}"
    )

    print(
        f"Images             : "
        f"{total_images}"
    )

    print(
        f"Database            : "
        f"newspaper_archive"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Integrity check
    # --------------------------------------------------------

    if (
        total_segments
        != physical_segment_count
    ):

        raise RuntimeError(
            "SEGMENT COUNT MISMATCH: "
            f"expected {physical_segment_count}, "
            f"inserted {total_segments}"
        )

    return {
        "document_id": document_id,
        "document_uuid": document_uuid,
        "logical_articles": len(
            logical_articles
        ),
        "physical_segments": total_segments,
        "images": total_images,
    }


# ============================================================
# CLI
# ============================================================


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "PYTHONPATH=. python "
            "-m pipeline.database.import_final_articles "
            "output/documents/doc_000007"
        )

        sys.exit(1)

    document_dir = (
        sys.argv[1]
    )

    try:

        import_document_directory(
            document_dir
        )

    except Exception as exc:

        print()

        print(
            "=" * 70
        )

        print(
            "DATABASE IMPORT FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "Transaction was rolled back "
            "by the database connection."
        )

        raise


if __name__ == "__main__":
    main()