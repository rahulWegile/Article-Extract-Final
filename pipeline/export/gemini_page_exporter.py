import json
from pathlib import Path


class GeminiPageExporter:
    """
    Export the complete page information
    specifically for the Gemini pipeline.

    This exporter is independent from the
    legacy graph pipeline exporter.
    """

    def export(
        self,
        page_number,
        page_width,
        page_height,
        blocks,
        output_dir,
    ):

        page = {

            "page": page_number,

            "page_width": page_width,

            "page_height": page_height,

            "total_blocks": len(blocks),

            "blocks": []

        }

        #
        # Export every detected block
        #

        for reading_order, block in enumerate(blocks):

            knowledge = getattr(
                block,
                "knowledge",
                None,
            )

            width = block.x2 - block.x1
            height = block.y2 - block.y1

            center_x = (block.x1 + block.x2) / 2
            center_y = (block.y1 + block.y2) / 2

            block_json = {

                # -------------------------
                # Identity
                # -------------------------

                "id": block.id,

                "reading_order": reading_order,

                # -------------------------
                # Layout
                # -------------------------

                "class": block.cls,

                "type": getattr(
                    block,
                    "type",
                    block.cls,
                ),

                "is_global": getattr(
                    block,
                    "is_global",
                    False,
                ),

                "is_masthead": getattr(
                    block,
                    "is_masthead",
                    False,
                ),

                "is_page_header": getattr(
                    block,
                    "is_page_header",
                    False,
                ),

                "column": getattr(
                    block,
                    "column",
                    None,
                ),
                "bbox": {

                    "x1": block.x1,
                    "y1": block.y1,
                    "x2": block.x2,
                    "y2": block.y2,

                },

                "width": width,

                "height": height,

                "center": {

                    "x": center_x,
                    "y": center_y,

                },

                # -------------------------
                # OCR
                # -------------------------

                "text": getattr(
                    block,
                    "text",
                    "",
                ),

                "ocr_confidence": getattr(
                    block,
                    "ocr_confidence",
                    0.0,
                ),

                # -------------------------
                # Knowledge
                # -------------------------

                "knowledge": {

                    "language": getattr(
                        knowledge,
                        "language",
                        None,
                    ),

                    "category": getattr(
                        knowledge,
                        "category",
                        "",
                    ),

                    "summary": getattr(
                        knowledge,
                        "summary",
                        "",
                    ),

                    "keywords": getattr(
                        knowledge,
                        "keywords",
                        [],
                    ),

                    "entities": getattr(
                        knowledge,
                        "entities",
                        {},
                    ),

                }

                if knowledge

                else None,

                # -------------------------
                # Gemini Output
                # -------------------------

                "article_id": None,

                "article_confidence": None,

                "gemini_notes": None,

            }

            page["blocks"].append(
                block_json
            )

        output_dir = Path(output_dir)

        output_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        output_path = output_dir / (

            f"page_{page_number:03d}.json"

        )

        with open(

            output_path,

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                page,

                f,

                indent=4,

                ensure_ascii=False,

            )

        print()

        print("=" * 60)

        print(
            f"Saved Page JSON -> {output_path}"
        )

        print("=" * 60)

        print()

        return output_path