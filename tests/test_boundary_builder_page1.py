import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.article.article_grouper import ArticleGrouper
from pipeline.article.boundary_builder import BoundaryBuilder


BASE = Path("output/documents/doc_000001")

PAGE_JSON = BASE / "page_json/page_001.json"
GEMINI_JSON = BASE / "gemini/page_001_response.json"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_blocks(page_data, gemini_data):

    # Original blocks
    original = {
        b["id"]: b
        for b in page_data["blocks"]
    }

    # Gemini block information
    gemini_blocks = {
        b["id"]: b
        for b in gemini_data["blocks"]
    }

    result = []

    for block_id, b in original.items():

        bbox = b.get("bbox", {})

        g = gemini_blocks.get(block_id, {})

        result.append(
            SimpleNamespace(

                id=block_id,

                # IMPORTANT:
                # take role from Gemini response
                role=g.get(
                    "role",
                    "unknown"
                ),

                text=b.get(
                    "text",
                    ""
                ),

                reading_order=b.get(
                    "reading_order",
                    0
                ),

                x1=bbox.get("x1", 0),
                y1=bbox.get("y1", 0),
                x2=bbox.get("x2", 0),
                y2=bbox.get("y2", 0),
            )
        )

    return result


def main():

    print("=" * 70)
    print("BOUNDARY BUILDER PAGE 1 TEST")
    print("=" * 70)

    page_data = load_json(
        PAGE_JSON
    )

    gemini_data = load_json(
        GEMINI_JSON
    )

    blocks = build_blocks(
        page_data,
        gemini_data
    )

    print(
        f"Page blocks : {len(blocks)}"
    )

    print(
        f"Gemini articles : "
        f"{len(gemini_data['articles'])}"
    )

    # ---------------------------------------------------------
    # GROUP ARTICLES
    # ---------------------------------------------------------

    grouper = ArticleGrouper()

    articles = grouper.build(
        gemini_data,
        blocks
    )

    print()
    print(
        f"Articles after grouping: "
        f"{len(articles)}"
    )

    # ---------------------------------------------------------
    # BUILD BOUNDARIES
    # ---------------------------------------------------------

    builder = BoundaryBuilder()

    boundaries = builder.build(
        articles
    )

    # ---------------------------------------------------------
    # PRINT ALL BOUNDARIES
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ARTICLE BOUNDARIES")
    print("=" * 70)

    for b in boundaries:

        print()
        print(
            f"ARTICLE {b.article_id}"
        )

        print(
            f"  bbox = "
            f"({b.x1}, {b.y1}) -> "
            f"({b.x2}, {b.y2})"
        )

        print(
            f"  size = "
            f"{b.width} x {b.height}"
        )

        print(
            f"  blocks = "
            f"{b.block_ids}"
        )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()