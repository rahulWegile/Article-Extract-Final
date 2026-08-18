from pathlib import Path
from typing import Any
import json

from PIL import Image


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
    """

    def crop_articles(
        self,
        document_dir: Path,
        page_number: int,
        boundaries: list[Any],
        page_image_path=None,
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
            # No modification.
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