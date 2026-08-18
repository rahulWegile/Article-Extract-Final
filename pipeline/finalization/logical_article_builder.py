from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.finalization.image_compositor import (
    ImageCompositor,
)


class LogicalArticleBuilder:
    """
    Finalizes Gemini logical articles.

    Pipeline:

        final_logical_articles.json
                    ↓
            logical article
                    ↓
              physical parts
                    ↓
              article text
                    ↓
          Gemini image information
                    ↓
          normalized image bbox
                    ↓
             actual pixel bbox
                    ↓
              crop actual image
                    ↓
              compose images
                    ↓
              final article JSON
    """

    # Gemini uses normalized coordinates.
    GEMINI_COORDINATE_MAX = 1000.0

    def __init__(
        self,
        document_dir: str | Path,
    ):

        self.document_dir = Path(
            document_dir
        ).resolve()

        self.batch_dir = (
            self.document_dir
            / "gemini_article_batches"
        )

        self.crop_root = (
            self.document_dir
            / "final_articles_crops"
        )

        self.image_root = (
            self.document_dir
            / "article_images"
        )

        self.final_root = (
            self.document_dir
            / "final_articles"
        )

        self.image_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.final_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.compositor = (
            ImageCompositor(
                background="white",
                gap=20,
                padding=20,
                max_height=1000,
            )
        )

    # =========================================================
    # PATH HELPERS
    # =========================================================

    def resolve_path(
        self,
        path: str | Path,
    ) -> Path:

        path = Path(path)

        if path.is_absolute():
            return path

        # First try project working directory.
        candidate = (
            Path.cwd()
            / path
        )

        if candidate.exists():
            return candidate

        # Then try relative to document directory.
        candidate = (
            self.document_dir
            / path
        )

        if candidate.exists():
            return candidate

        return (
            Path.cwd()
            / path
        )

    # =========================================================
    # LOAD LOGICAL ARTICLES
    # =========================================================

    def load_logical_articles(
        self,
    ) -> list[dict[str, Any]]:

        path = (
            self.batch_dir
            / "final_logical_articles.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find:\n{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        # -----------------------------------------------------
        # Possible formats
        # -----------------------------------------------------

        if isinstance(
            data,
            list,
        ):
            return data

        if isinstance(
            data,
            dict,
        ):

            logical_articles = (
                data.get(
                    "logical_articles"
                )
            )

            if isinstance(
                logical_articles,
                list,
            ):
                return logical_articles

            articles = data.get(
                "articles"
            )

            if isinstance(
                articles,
                list,
            ):
                return articles

        raise ValueError(
            "Unknown final_logical_articles.json structure."
        )

    # =========================================================
    # SAFE VALUE
    # =========================================================

    @staticmethod
    def first_value(
        obj: dict[str, Any],
        *keys,
        default=None,
    ):

        for key in keys:

            value = obj.get(
                key
            )

            if value is not None:
                return value

        return default

    # =========================================================
    # PAGE
    # =========================================================

    @staticmethod
    def get_page(
        article: dict[str, Any],
    ):

        value = (
            article.get("page")
            if article.get("page")
            is not None
            else article.get(
                "page_number"
            )
        )

        if value is None:
            return None

        try:
            return int(value)
        except Exception:
            return None

    # =========================================================
    # ARTICLE ID
    # =========================================================

    @staticmethod
    def get_article_id(
        article: dict[str, Any],
    ) -> str | None:

        value = article.get(
            "article_id"
        )

        if value is None:
            return None

        return str(value)

    # =========================================================
    # CROP PATH
    # =========================================================

    def get_crop_path(
        self,
        article: dict[str, Any],
    ) -> Path | None:

        # Gemini may return crop_path/source_crop.
        explicit = self.first_value(
            article,
            "crop_path",
            "source_crop",
        )

        if explicit:

            path = self.resolve_path(
                explicit
            )

            if path.exists():
                return path

        page = self.get_page(
            article
        )

        article_id = (
            self.get_article_id(
                article
            )
        )

        if (
            page is None
            or not article_id
        ):
            return None

        # -----------------------------------------------------
        # Expected verified crop structure
        #
        # final_articles_crops/
        #   page_001/
        #     article_001/
        #       page_001.png
        # -----------------------------------------------------

        candidates = [
            (
                self.crop_root
                / f"page_{page:03d}"
                / article_id
                / f"page_{page:03d}.png"
            ),
            (
                self.crop_root
                / f"page_{page:03d}"
                / article_id
                / f"page_{page}.png"
            ),
        ]

        for candidate in candidates:

            if candidate.exists():
                return candidate

        return None

    # =========================================================
    # GET SEGMENTS
    # =========================================================

    def get_segments(
        self,
        logical_article: dict[str, Any],
    ) -> list[dict[str, Any]]:

        articles = (
            logical_article.get(
                "articles"
            )
            or []
        )

        if not isinstance(
            articles,
            list,
        ):
            return []

        def sort_key(
            article,
        ):

            page = self.get_page(
                article
            )

            article_id = (
                self.get_article_id(
                    article
                )
                or ""
            )

            return (
                page
                if page is not None
                else 999999,
                article_id,
            )

        return sorted(
            articles,
            key=sort_key,
        )

    # =========================================================
    # TEXT
    # =========================================================

    def get_text(
        self,
        article: dict[str, Any],
    ) -> str:

        value = self.first_value(
            article,
            "article_text",
            "text",
            "full_text",
            "content",
            default="",
        )

        if value is None:
            return ""

        return str(
            value
        ).strip()

    # =========================================================
    # TEXT MERGER
    # =========================================================

    @staticmethod
    def merge_texts(
        texts: list[str],
    ) -> str:

        parts = []

        for text in texts:

            text = (
                text or ""
            ).strip()

            if not text:
                continue

            if not parts:
                parts.append(
                    text
                )
                continue

            previous = parts[-1]

            # Exact duplicate.
            if text == previous:
                continue

            # Entire text already present.
            if text in previous:
                continue

            # Previous contains current.
            if previous in text:
                parts[-1] = text
                continue

            previous_words = (
                previous.split()
            )

            current_words = (
                text.split()
            )

            found_overlap = False

            max_overlap = min(
                100,
                len(previous_words),
                len(current_words),
            )

            # -------------------------------------------------
            # Look for overlap at segment boundary.
            # -------------------------------------------------

            for overlap in range(
                max_overlap,
                4,
                -1,
            ):

                if (
                    previous_words[
                        -overlap:
                    ]
                    == current_words[
                        :overlap
                    ]
                ):

                    merged = (
                        previous
                        + " "
                        + " ".join(
                            current_words[
                                overlap:
                            ]
                        )
                    )

                    parts[-1] = merged

                    found_overlap = True

                    break

            if not found_overlap:

                parts.append(
                    text
                )

        return "\n\n".join(
            parts
        )

    # =========================================================
    # IMAGE OBJECT EXTRACTION
    # =========================================================

    def get_image_items(
        self,
        article: dict[str, Any],
    ) -> list[dict[str, Any]]:

        """
        Supports all of these:

        1.
        "images": [
            {...}
        ]

        2.
        "images": {
            "has_images": true,
            "image_count": 2,
            "items": [
                {...},
                {...}
            ]
        }

        3.
        "image_information": {
            "items": [...]
        }
        """

        result = []

        for key in (
            "images",
            "image_information",
            "image_info",
            "extracted_images",
        ):

            value = article.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                for item in value:

                    if isinstance(
                        item,
                        dict,
                    ):
                        result.append(
                            item
                        )

                continue

            if isinstance(
                value,
                dict,
            ):

                items = value.get(
                    "items"
                )

                if isinstance(
                    items,
                    list,
                ):

                    for item in items:

                        if isinstance(
                            item,
                            dict,
                        ):
                            result.append(
                                item
                            )

                    continue

                # Single-image format.
                if (
                    "image_bbox"
                    in value
                    or "bbox"
                    in value
                ):

                    result.append(
                        value
                    )

        return result

    # =========================================================
    # BBOX EXTRACTION
    # =========================================================

    @staticmethod
    def get_bbox(
        image_info: dict[str, Any],
    ):

        bbox = (
            image_info.get(
                "image_bbox"
            )
            or image_info.get(
                "bbox"
            )
        )

        if bbox is None:
            return None

        if isinstance(
            bbox,
            dict,
        ):

            x1 = bbox.get("x1")
            y1 = bbox.get("y1")
            x2 = bbox.get("x2")
            y2 = bbox.get("y2")

            if all(
                value is not None
                for value in (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            ):

                return (
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                )

            # -------------------------------------------------
            # Also support x/y/width/height.
            # -------------------------------------------------

            x = bbox.get("x")
            y = bbox.get("y")
            width = bbox.get(
                "width"
            )
            height = bbox.get(
                "height"
            )

            if all(
                value is not None
                for value in (
                    x,
                    y,
                    width,
                    height,
                )
            ):

                return (
                    float(x),
                    float(y),
                    float(x + width),
                    float(y + height),
                )

        if isinstance(
            bbox,
            (list, tuple),
        ):

            if len(bbox) >= 4:

                return (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )

        return None

    # =========================================================
    # CONVERT NORMALIZED BBOX TO PIXELS
    # =========================================================

    def bbox_to_pixels(
        self,
        bbox,
        image_width: int,
        image_height: int,
    ):

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        # -----------------------------------------------------
        # Gemini image bbox is normalized 0-1000.
        #
        # If all coordinates are <= 1000, treat them as
        # normalized coordinates.
        #
        # If coordinates exceed 1000, assume pixel coordinates.
        # -----------------------------------------------------

        is_normalized = (
            0 <= x1 <= 1000
            and 0 <= y1 <= 1000
            and 0 <= x2 <= 1000
            and 0 <= y2 <= 1000
        )

        if is_normalized:

            x1 = (
                x1
                / self.GEMINI_COORDINATE_MAX
                * image_width
            )

            y1 = (
                y1
                / self.GEMINI_COORDINATE_MAX
                * image_height
            )

            x2 = (
                x2
                / self.GEMINI_COORDINATE_MAX
                * image_width
            )

            y2 = (
                y2
                / self.GEMINI_COORDINATE_MAX
                * image_height
            )

        # -----------------------------------------------------
        # Clamp.
        # -----------------------------------------------------

        x1 = max(
            0,
            min(
                image_width,
                int(round(x1)),
            ),
        )

        y1 = max(
            0,
            min(
                image_height,
                int(round(y1)),
            ),
        )

        x2 = max(
            0,
            min(
                image_width,
                int(round(x2)),
            ),
        )

        y2 = max(
            0,
            min(
                image_height,
                int(round(y2)),
            ),
        )

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        return (
            x1,
            y1,
            x2,
            y2,
        )

    # =========================================================
    # EXTRACT ONE IMAGE
    # =========================================================

    def extract_one_image(
        self,
        article: dict[str, Any],
        image_info: dict[str, Any],
        output_path: Path,
    ):

        crop_path = (
            self.get_crop_path(
                article
            )
        )

        if crop_path is None:

            print(
                "  WARNING: Cannot locate "
                "article crop for image:"
            )

            print(
                "    page:",
                self.get_page(
                    article
                ),
            )

            print(
                "    article:",
                self.get_article_id(
                    article
                ),
            )

            return None

        bbox = self.get_bbox(
            image_info
        )

        if bbox is None:

            print(
                "  WARNING: Image has no bbox"
            )

            return None

        try:

            image = Image.open(
                crop_path
            ).convert("RGB")

        except Exception as exc:

            print(
                "  WARNING: Could not open crop:",
                crop_path,
                exc,
            )

            return None

        pixel_bbox = (
            self.bbox_to_pixels(
                bbox,
                image.width,
                image.height,
            )
        )

        if pixel_bbox is None:

            print(
                "  WARNING: Invalid image bbox:",
                bbox,
            )

            return None

        x1, y1, x2, y2 = (
            pixel_bbox
        )

        extracted = image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

        # -----------------------------------------------------
        # Avoid tiny accidental regions.
        # -----------------------------------------------------

        if extracted.width < 20:
            return None

        if extracted.height < 20:
            return None

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        extracted.save(
            output_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        return {
            "image_path":
                str(output_path),

            "source_crop":
                str(crop_path),

            "bbox_normalized":
                {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },

            "bbox_pixels":
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

            "width":
                extracted.width,

            "height":
                extracted.height,

            "description":
                image_info.get(
                    "image_description",
                    image_info.get(
                        "description"
                    ),
                ),

            "caption":
                image_info.get(
                    "caption"
                ),

            "confidence":
                image_info.get(
                    "confidence"
                ),
        }

    # =========================================================
    # EXTRACT ALL IMAGES
    # =========================================================

    def extract_article_images(
        self,
        logical_article,
        segments,
    ) -> list[dict[str, Any]]:

        logical_id = (
            logical_article.get(
                "logical_article_id",
                "logical_unknown",
            )
        )

        output_dir = (
            self.image_root
            / logical_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        extracted_images = []

        image_counter = 1

        for segment in segments:

            page = self.get_page(
                segment
            )

            article_id = (
                self.get_article_id(
                    segment
                )
            )

            image_items = (
                self.get_image_items(
                    segment
                )
            )

            if not image_items:
                continue

            print(
                f"    P{page:03d} "
                f"{article_id}: "
                f"{len(image_items)} "
                f"Gemini image(s)"
            )

            for image_info in image_items:

                output_path = (
                    output_dir
                    / (
                        f"image_"
                        f"{image_counter:03d}"
                        f"_p{page:03d}"
                        f"_{article_id}.jpg"
                    )
                )

                result = (
                    self.extract_one_image(
                        article=segment,
                        image_info=image_info,
                        output_path=output_path,
                    )
                )

                if result is None:
                    continue

                result.update(
                    {
                        "image_id":
                            (
                                f"{logical_id}_"
                                f"image_"
                                f"{image_counter:03d}"
                            ),

                        "page":
                            page,

                        "article_id":
                            article_id,
                    }
                )

                extracted_images.append(
                    result
                )

                image_counter += 1

        # -----------------------------------------------------
        # Save image metadata.
        # -----------------------------------------------------

        metadata_path = (
            output_dir
            / "images.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "logical_article_id":
                        logical_id,

                    "image_count":
                        len(
                            extracted_images
                        ),

                    "images":
                        extracted_images,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return extracted_images

    # =========================================================
    # MERGE METADATA
    # =========================================================

    def merge_list_fields(
        self,
        segments,
        field,
    ):

        result = []

        seen = set()

        for segment in segments:

            values = segment.get(
                field,
                []
            )

            if not isinstance(
                values,
                list,
            ):
                continue

            for value in values:

                key = json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=False,
                )

                if key in seen:
                    continue

                seen.add(key)

                result.append(
                    value
                )

        return result

    # =========================================================
    # BUILD SOURCE PARTS
    # =========================================================

    def build_parts(
        self,
        segments,
    ):

        parts = []

        for index, segment in enumerate(
            segments,
            start=1,
        ):

            parts.append(
                {
                    "segment_order":
                        index,

                    "page":
                        self.get_page(
                            segment
                        ),

                    "article_id":
                        self.get_article_id(
                            segment
                        ),

                    "crop_path":
                        (
                            str(
                                self.get_crop_path(
                                    segment
                                )
                            )
                            if self.get_crop_path(
                                segment
                            )
                            else None
                        ),

                    "bbox":
                        segment.get(
                            "bbox"
                        ),
                }
            )

        return parts

    # =========================================================
    # BUILD FINAL ARTICLE
    # =========================================================

    def build_article(
        self,
        logical_article,
    ):

        logical_id = logical_article.get(
            "logical_article_id"
        )

        if not logical_id:

            return None

        segments = (
            self.get_segments(
                logical_article
            )
        )

        if not segments:

            return None

        # -----------------------------------------------------
        # Combine article text.
        # -----------------------------------------------------

        texts = [
            self.get_text(
                segment
            )
            for segment in segments
        ]

        combined_text = (
            self.merge_texts(
                texts
            )
        )

        first = segments[0]

        # -----------------------------------------------------
        # Metadata.
        # -----------------------------------------------------

        headline = self.first_value(
            first,
            "headline",
            "title",
            default="",
        )

        subheadline = self.first_value(
            first,
            "subheadline",
            default=None,
        )

        author = self.first_value(
            first,
            "author",
            "byline",
            default=None,
        )

        location = self.first_value(
            first,
            "location",
            default=None,
        )

        date = self.first_value(
            first,
            "date",
            "published_date",
            default=None,
        )

        summary = self.first_value(
            first,
            "summary",
            default=None,
        )

        category = self.first_value(
            first,
            "category",
            default=None,
        )

        sentiment = self.first_value(
            first,
            "sentiment",
            default=None,
        )

        entities = (
            self.merge_list_fields(
                segments,
                "entities",
            )
        )

        topics = (
            self.merge_list_fields(
                segments,
                "topics",
            )
        )

        keywords = (
            self.merge_list_fields(
                segments,
                "keywords",
            )
        )

        # -----------------------------------------------------
        # Extract actual images.
        # -----------------------------------------------------

        print(
            f"  Processing images for "
            f"{logical_id}"
        )

        images = (
            self.extract_article_images(
                logical_article,
                segments,
            )
        )

        # -----------------------------------------------------
        # Compose images.
        # -----------------------------------------------------

        composed_image = None

        image_paths = [
            item["image_path"]
            for item in images
            if item.get(
                "image_path"
            )
        ]

        if image_paths:

            composed_path = (
                self.final_root
                / logical_id
                / "composed_images"
                / "article.jpg"
            )

            composed_image = (
                self.compositor
                .compose_article_images(
                    image_paths,
                    composed_path,
                )
            )

        # -----------------------------------------------------
        # Continuation links.
        # -----------------------------------------------------

        continuation_links = (
            logical_article.get(
                "continuation_links",
                [],
            )
        )

        # Some extractor versions may store
        # links under source_parts.
        if not isinstance(
            continuation_links,
            list,
        ):

            continuation_links = []

        # -----------------------------------------------------
        # Final object.
        # -----------------------------------------------------

        final_article = {

            "logical_article_id":
                logical_id,

            "status":
                logical_article.get(
                    "status"
                ),

            "headline":
                headline,

            "subheadline":
                subheadline,

            "author":
                author,

            "location":
                location,

            "date":
                date,

            "article_text":
                combined_text,

            "summary":
                summary,

            "category":
                category,

            "entities":
                entities,

            "topics":
                topics,

            "keywords":
                keywords,

            "sentiment":
                sentiment,

            "segment_count":
                len(segments),

            "parts":
                self.build_parts(
                    segments
                ),

            "continuation_links":
                continuation_links,

            "images":
                images,

            "has_images":
                bool(images),

            "image_count":
                len(images),

            "composed_image":
                composed_image,
        }

        return final_article

    # =========================================================
    # BUILD ALL
    # =========================================================

    def build_all(self):

        logical_articles = (
            self.load_logical_articles()
        )

        final_articles = []

        print()
        print("=" * 70)
        print(
            "FINAL LOGICAL ARTICLE BUILDER"
        )
        print("=" * 70)

        print(
            "Logical articles:",
            len(logical_articles),
        )

        print()

        for logical_article in (
            logical_articles
        ):

            logical_id = (
                logical_article.get(
                    "logical_article_id",
                    "unknown",
                )
            )

            try:

                article = (
                    self.build_article(
                        logical_article
                    )
                )

                if article is None:

                    print(
                        f"✗ {logical_id} "
                        f"| could not build"
                    )

                    continue

                final_articles.append(
                    article
                )

                print(
                    f"✓ {logical_id} "
                    f"| segments="
                    f"{article['segment_count']} "
                    f"| images="
                    f"{article['image_count']}"
                )

            except Exception as exc:

                print(
                    f"✗ {logical_id} "
                    f"| ERROR: {exc}"
                )

        # -----------------------------------------------------
        # Final JSON.
        # -----------------------------------------------------

        output_path = (
            self.final_root
            / "final_articles.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                final_articles,
                f,
                indent=2,
                ensure_ascii=False,
            )

        # -----------------------------------------------------
        # Summary.
        # -----------------------------------------------------

        total_images = sum(
            article.get(
                "image_count",
                0,
            )
            for article in final_articles
        )

        composed_count = sum(
            1
            for article in final_articles
            if article.get(
                "composed_image"
            )
        )

        merged_count = sum(
            1
            for article in final_articles
            if article.get(
                "segment_count",
                1,
            ) > 1
        )

        print()
        print("=" * 70)
        print(
            "FINALIZATION COMPLETE"
        )
        print("=" * 70)

        print(
            "Final articles:",
            len(final_articles),
        )

        print(
            "Multi-page logical articles:",
            merged_count,
        )

        print(
            "Images extracted:",
            total_images,
        )

        print(
            "Composed images:",
            composed_count,
        )

        print(
            "Saved:",
            output_path,
        )

        print("=" * 70)

        return final_articles


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "document_dir",
        help=(
            "Example: "
            "output/documents/doc_000004"
        ),
    )

    args = parser.parse_args()

    builder = LogicalArticleBuilder(
        args.document_dir
    )

    builder.build_all()


if __name__ == "__main__":
    main()