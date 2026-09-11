from pathlib import Path
from typing import Any
import json

from PIL import Image, ImageDraw


# A foreign block's own rectangle must fall THIS much inside the
# rectangle being cropped before it is worth masking out -- a block
# merely clipped by the crop's edge is not evidence of anything.
# Deliberately the same value as boundary_decomposer.py's
# INTRUSION_INSIDE_RATIO, which already diagnoses this exact
# geometric situation (an article's plain rectangle enclosing
# another article's owned content) for logging; this reuses the same
# threshold so the two stay in agreement about what counts.
_FOREIGN_INSIDE_RATIO = 0.75

# A foreign block that also heavily overlaps one of THIS crop's own
# blocks is a duplicate detection of content this article already
# owns, not a neighbour's content -- never mask that. Same value and
# reasoning as boundary_decomposer.py's INTRUSION_OWN_OVERLAP_MAX.
_FOREIGN_OWN_OVERLAP_MAX = 0.30

_FOREIGN_CONTENT_CLASSES = {"title", "plain text", "figure", "figure_caption"}


def _rect_area(rect):
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def _rect_intersection(a, b):
    return (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )


class FinalArticleCropper:
    """
    Extract every FINAL VERIFIED article boundary
    as a separate image.

    Output structure:

        final_articles_crops/
        ├── page_001/
        │   ├── article_001/
        │   │   ├── crop.json
        │   │   └── page_001.png
        │   ├── article_002/
        │   │   ├── crop.json
        │   │   └── page_001.png
        │   └── ...
        │
        ├── page_002/
        │   ├── article_001/
        │   │   ├── crop.json
        │   │   └── page_002.png
        │   └── ...
        │
        └── page_003/
            └── ...

    IMPORTANT:
    - Uses final boundary coordinates directly.
    - Does NOT detect green pixels.
    - Does NOT use OCR to calculate the crop.
    - Does NOT shrink the boundary.
    - Does NOT expand the boundary.
    - Does NOT add padding.
    - Does NOT resize the crop.

    A crop's rectangle can still geometrically enclose content owned
    by a DIFFERENT article -- boundary_decomposer.py deliberately
    leaves this membership/geometry alone (see its own docstring:
    reshaping a real multi-column article's rectangle to avoid this
    broke more than it fixed). But the crop IMAGE itself is a plain
    pixel rectangle, so that foreign content is still visibly baked
    into it -- confirmed on a real Urdu (THE INQUILAB, doc_000194
    page 1) page: a boxed callout ("عمارت کا مالک...") sits inside
    its own parent story's rectangle (the parent's blocks wrap around
    it in an L-shape), and the parent's crop -- which the vision-
    based article extractor reads directly, see
    GeminiArticleExtractor -- visibly duplicates the callout's full
    text in the corner, right where the callout's OWN separate crop
    already has it. When `page_json_path` is given, any such foreign
    block is white-filled out of the crop before saving -- membership
    and every other article's own geometry are completely untouched.
    """

    def crop_articles(
        self,
        document_dir: Path,
        page_number: int,
        boundaries: list[Any],
        page_image_path=None,
        page_json_path=None,
    ) -> list[dict]:

        document_dir = Path(document_dir)

        # =====================================================
        # SOURCE PAGE IMAGE
        # =====================================================

        if page_image_path is not None:

            page_path = Path(
                page_image_path
            )

        else:

            page_path = (
                document_dir
                / "pages"
                / f"page_{page_number:03d}.png"
            )

        if not page_path.exists():

            raise FileNotFoundError(
                f"Page image not found: {page_path}"
            )

        # =====================================================
        # OPEN ORIGINAL PAGE
        # =====================================================

        image = Image.open(
            page_path
        ).convert("RGB")

        # =====================================================
        # FOREIGN-CONTENT MASKING DATA (optional)
        #
        # See the class docstring. Every block, keyed by id, plus
        # which future article_id (the SAME "article_{index:03d}"
        # scheme the main loop below assigns) owns each one -- so a
        # block belonging to a DIFFERENT article than the one being
        # cropped can be white-filled out of that crop.
        # =====================================================

        blocks_by_id = {}
        owner_of = {}

        if page_json_path is not None:

            blocks_by_id, owner_of = self._load_masking_data(
                page_json_path,
                boundaries,
            )

        # =====================================================
        # PAGE OUTPUT DIRECTORY
        #
        # final_articles_crops/
        #       page_001/
        #       page_002/
        #       page_003/
        # =====================================================

        output_root = (
            document_dir
            / "final_articles_crops"
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        page_dir = (
            output_root
            / f"page_{page_number:03d}"
        )

        page_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []

        # =====================================================
        # PROCESS EACH FINAL BOUNDARY
        # =====================================================

        for index, boundary in enumerate(
            boundaries,
            start=1,
        ):

            print()
            print(
                f"Processing final boundary "
                f"{index}/{len(boundaries)}"
            )

            # =================================================
            # READ FINAL BOUNDARY
            # =================================================

            x1 = int(
                round(
                    self._get_value(
                        boundary,
                        "x1",
                    )
                )
            )

            y1 = int(
                round(
                    self._get_value(
                        boundary,
                        "y1",
                    )
                )
            )

            x2 = int(
                round(
                    self._get_value(
                        boundary,
                        "x2",
                    )
                )
            )

            y2 = int(
                round(
                    self._get_value(
                        boundary,
                        "y2",
                    )
                )
            )

            # =================================================
            # CLAMP TO IMAGE
            # =================================================

            x1 = max(
                0,
                min(
                    x1,
                    image.width,
                ),
            )

            y1 = max(
                0,
                min(
                    y1,
                    image.height,
                ),
            )

            x2 = max(
                0,
                min(
                    x2,
                    image.width,
                ),
            )

            y2 = max(
                0,
                min(
                    y2,
                    image.height,
                ),
            )

            # =================================================
            # VALIDATE BOUNDARY
            # =================================================

            if x2 <= x1:

                print(
                    f"SKIP boundary {index}: "
                    f"invalid X coordinates"
                )

                continue

            if y2 <= y1:

                print(
                    f"SKIP boundary {index}: "
                    f"invalid Y coordinates"
                )

                continue

            # =================================================
            # ARTICLE ID
            #
            # We intentionally generate a PAGE-LOCAL ID:
            #
            # page_001/
            #     article_001
            #     article_002
            #
            # page_002/
            #     article_001
            #     article_002
            #
            # This is exactly what you requested.
            # =================================================

            article_id = (
                f"article_{index:03d}"
            )

            article_dir = (
                page_dir
                / article_id
            )

            article_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # =================================================
            # EXACT CROP
            #
            # No padding.
            # No resizing.
            # No geometry modification -- still exactly x1..y2.
            # =================================================

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            # =================================================
            # MASK FOREIGN CONTENT (optional)
            #
            # See the class docstring. Only ever paints over pixels
            # belonging to a block a DIFFERENT article owns -- never
            # touches this article's own content, never changes the
            # crop's rectangle.
            # =================================================

            if blocks_by_id:

                self._mask_foreign_content(
                    crop,
                    (x1, y1, x2, y2),
                    article_id,
                    blocks_by_id,
                    owner_of,
                )

            # =================================================
            # IMAGE NAME
            # =================================================

            output_path = (
                article_dir
                / f"page_{page_number:03d}.png"
            )

            # =================================================
            # SAVE ORIGINAL RESOLUTION CROP
            # =================================================

            crop.save(
                output_path,
                format="PNG",
            )

            # =================================================
            # METADATA
            # =================================================

            metadata = {

                "article_id": article_id,

                "page": page_number,

                "source_page": str(
                    page_path
                ),

                "crop_path": str(
                    output_path
                ),

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "width": crop.width,

                "height": crop.height,

                "source_width": image.width,

                "source_height": image.height,

                "resized": False,

                "padding": 0,

                "boundary_source": (
                    "final_verified_boundary"
                ),

                "block_ids": (
                    self._get_block_ids(
                        boundary
                    )
                ),

                # Precise sub-rectangles for display, when this
                # article was reshaped to avoid enclosing a
                # neighbour -- see Article.sub_rects. The crop
                # image above is unaffected: it is always the
                # single bbox region regardless, same as before.
                "sub_rects": (
                    self._get_sub_rects(
                        boundary
                    )
                ),
            }

            # =================================================
            # CROP JSON
            # =================================================

            metadata_path = (
                article_dir
                / "crop.json"
            )

            with open(
                metadata_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            results.append(
                metadata
            )

            # =================================================
            # LOG
            # =================================================

            print(
                "=" * 60
            )

            print(
                "ARTICLE CROP CREATED"
            )

            print(
                f"Article : {article_id}"
            )

            print(
                f"Page    : {page_number}"
            )

            print(
                f"BBox    : "
                f"({x1}, {y1}) → "
                f"({x2}, {y2})"
            )

            print(
                f"Size    : "
                f"{crop.width} × "
                f"{crop.height}"
            )

            print(
                f"Image   : "
                f"{output_path}"
            )

            print(
                f"JSON    : "
                f"{metadata_path}"
            )

            print(
                "=" * 60
            )

        # =====================================================
        # PAGE SUMMARY
        # =====================================================

        print()
        print(
            "=" * 60
        )

        print(
            f"PAGE {page_number} "
            f"ARTICLE CROPS"
        )

        print(
            f"Created : {len(results)}"
        )

        print(
            f"Directory : {page_dir}"
        )

        print(
            "=" * 60
        )

        return results

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _get_value(
        boundary: Any,
        key: str,
    ):

        if isinstance(
            boundary,
            dict,
        ):

            value = boundary.get(
                key
            )

        else:

            value = getattr(
                boundary,
                key,
                None,
            )

        if value is None:

            raise ValueError(
                f"Boundary does not contain "
                f"'{key}'"
            )

        return value

    @staticmethod
    def _get_block_ids(
        boundary: Any,
    ) -> list:

        # =================================================
        # block_ids is used downstream (local article
        # extraction) to map DocLayout-YOLO figure /
        # figure_caption blocks onto the article they
        # belong to. Missing/malformed block_ids must
        # never break cropping itself, so this always
        # degrades to an empty list instead of raising.
        # =================================================

        try:

            value = (
                FinalArticleCropper._get_value(
                    boundary,
                    "block_ids",
                )
            )

            return list(value)

        except Exception:

            return []

    @staticmethod
    def _get_sub_rects(
        boundary: Any,
    ) -> list:

        # =================================================
        # sub_rects (see Article.sub_rects / ArticleBoundary.
        # sub_rects) is empty for every article except the
        # rare ones boundary_decomposer.py gave a precise
        # multi-rectangle shape to avoid enclosing a
        # neighbouring article. Missing/malformed sub_rects
        # must never break cropping itself -- the crop image
        # is always taken from the single overall x1..y2 box
        # regardless -- so this degrades to an empty list
        # instead of raising, same as _get_block_ids.
        # =================================================

        try:

            if isinstance(boundary, dict):
                value = boundary.get("sub_rects") or []
            else:
                value = getattr(boundary, "sub_rects", None) or []

            return [
                {
                    "x1": int(round(r[0])),
                    "y1": int(round(r[1])),
                    "x2": int(round(r[2])),
                    "y2": int(round(r[3])),
                }
                for r in value
            ]

        except Exception:

            return []

    @staticmethod
    def _load_masking_data(page_json_path, boundaries):
        """
        Loads page_json's blocks (keyed by id) and, from `boundaries`
        themselves, which future article_id (the same
        "article_{index:03d}" scheme the main loop assigns) owns
        each block id. See the class docstring for why this exists.

        Degrades to ({}, {}) on any failure -- masking is a purely
        additive safety pass and must never break cropping itself,
        same convention as _get_block_ids/_get_sub_rects above.
        """

        try:

            with open(page_json_path, "r", encoding="utf-8") as f:
                page = json.load(f)

            blocks_by_id = {
                item["id"]: item for item in page.get("blocks", [])
            }

            owner_of = {}

            for index, boundary in enumerate(boundaries, start=1):

                article_id = f"article_{index:03d}"

                for block_id in FinalArticleCropper._get_block_ids(
                    boundary
                ):
                    owner_of[block_id] = article_id

            return blocks_by_id, owner_of

        except Exception:

            return {}, {}

    @staticmethod
    def _block_rect(item):
        bbox = item["bbox"]
        return (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])

    @staticmethod
    def _mask_foreign_content(
        crop,
        rect,
        own_article_id,
        blocks_by_id,
        owner_of,
    ):
        """
        White-fills any block a DIFFERENT article owns out of `crop`
        (already cropped to `rect` in page-space coordinates) when
        that block's own rectangle mostly falls inside `rect`. See
        the class docstring for the confirmed real case this exists
        for. Mutates `crop` in place; never touches this article's
        own content.
        """

        x1, y1, x2, y2 = rect

        own_ids = {
            block_id
            for block_id, article_id in owner_of.items()
            if article_id == own_article_id
        }

        own_rects = [
            FinalArticleCropper._block_rect(blocks_by_id[block_id])
            for block_id in own_ids
            if block_id in blocks_by_id
        ]

        draw = None

        for block_id, item in blocks_by_id.items():

            if block_id in own_ids:
                continue

            owner_id = owner_of.get(block_id)

            if owner_id is None or owner_id == own_article_id:
                # Unowned by any article, or owned by this one under
                # a block_id mismatch -- never mask on a guess.
                continue

            if (
                (item.get("class") or "").strip().lower()
                not in _FOREIGN_CONTENT_CLASSES
            ):
                continue

            block_rect = FinalArticleCropper._block_rect(item)

            block_area = _rect_area(block_rect)

            if block_area <= 0:
                continue

            inside_ratio = (
                _rect_area(_rect_intersection(block_rect, rect))
                / block_area
            )

            if inside_ratio < _FOREIGN_INSIDE_RATIO:
                continue

            own_overlap = max(
                (
                    _rect_area(
                        _rect_intersection(block_rect, own_rect)
                    )
                    / block_area
                    for own_rect in own_rects
                ),
                default=0.0,
            )

            if own_overlap > _FOREIGN_OWN_OVERLAP_MAX:
                continue

            local_rect = (
                max(0, block_rect[0] - x1),
                max(0, block_rect[1] - y1),
                min(x2 - x1, block_rect[2] - x1),
                min(y2 - y1, block_rect[3] - y1),
            )

            if (
                local_rect[2] <= local_rect[0]
                or local_rect[3] <= local_rect[1]
            ):
                continue

            if draw is None:
                draw = ImageDraw.Draw(crop)

            draw.rectangle(local_rect, fill=(255, 255, 255))