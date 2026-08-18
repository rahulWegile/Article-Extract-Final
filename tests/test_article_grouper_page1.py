import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.article.article_grouper import ArticleGrouper


DOCUMENT_ID = "doc_000001"
PAGE_NUMBER = 1

BASE_DIR = Path("output/documents") / DOCUMENT_ID

PAGE_JSON = (
    BASE_DIR
    / "page_json"
    / f"page_{PAGE_NUMBER:03d}.json"
)

GEMINI_JSON = (
    BASE_DIR
    / "gemini"
    / f"page_{PAGE_NUMBER:03d}_response.json"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def convert_blocks(raw_blocks):

    blocks = []

    for b in raw_blocks:

        bbox = b.get(
            "bbox",
            {}
        )

        blocks.append(
            SimpleNamespace(

                id=b.get("id"),

                role=b.get(
                    "role",
                    b.get(
                        "semantic_role",
                        b.get(
                            "class",
                            "unknown"
                        )
                    )
                ),

                text=b.get(
                    "text",
                    ""
                ),

                reading_order=b.get(
                    "reading_order",
                    0
                ),

                x1=bbox.get(
                    "x1",
                    0
                ),

                y1=bbox.get(
                    "y1",
                    0
                ),

                x2=bbox.get(
                    "x2",
                    0
                ),

                y2=bbox.get(
                    "y2",
                    0
                ),
            )
        )

    return blocks


def main():

    print()
    print("=" * 70)
    print("ARTICLE GROUPER PAGE 1 TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    if not PAGE_JSON.exists():

        raise FileNotFoundError(
            f"Missing:\n{PAGE_JSON}"
        )

    if not GEMINI_JSON.exists():

        raise FileNotFoundError(
            f"Missing:\n{GEMINI_JSON}"
        )

    # ---------------------------------------------------------
    # Load page blocks
    # ---------------------------------------------------------

    page_data = load_json(
        PAGE_JSON
    )

    raw_blocks = page_data.get(
        "blocks",
        []
    )

    blocks = convert_blocks(
        raw_blocks
    )

    print(
        f"Page blocks: {len(blocks)}"
    )

    # ---------------------------------------------------------
    # Load ACTUAL Gemini page response
    # ---------------------------------------------------------

    gemini_data = load_json(
        GEMINI_JSON
    )

    gemini_articles = gemini_data.get(
        "articles",
        []
    )

    print(
        f"Gemini articles: "
        f"{len(gemini_articles)}"
    )

    # ---------------------------------------------------------
    # Run ArticleGrouper
    # ---------------------------------------------------------

    grouper = ArticleGrouper()

    articles = grouper.build(
        {
            "articles": gemini_articles
        },
        blocks,
    )

    # ---------------------------------------------------------
    # Duplicate ownership check
    # ---------------------------------------------------------

    ownership = {}

    for article in articles:

        for block_id in article.block_ids:

            ownership.setdefault(
                block_id,
                []
            ).append(
                article.article_id
            )

    duplicates = {
        block_id: owners
        for block_id, owners
        in ownership.items()
        if len(owners) > 1
    }

    print()
    print("=" * 70)
    print("DUPLICATE OWNERSHIP CHECK")
    print("=" * 70)

    if duplicates:

        print(
            f"FOUND {len(duplicates)} duplicate blocks"
        )

        for block_id, owners in duplicates.items():

            print(
                f"Block {block_id} "
                f"-> {owners}"
            )

    else:

        print(
            "✓ No duplicate block ownership"
        )

    # ---------------------------------------------------------
    # Focus on Article 010 and 011
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ARTICLE 010 / 011 ANALYSIS")
    print("=" * 70)

    for article in articles:

        article_id = str(
            article.article_id
        )

        if article_id not in {
            "article_010",
            "article_011",
            "10",
            "11",
        }:

            continue

        print()
        print(
            f"ARTICLE {article_id}"
        )

        print(
            f"Block IDs: "
            f"{article.block_ids}"
        )

        print()

        for block in article.blocks:

            text = str(
                getattr(
                    block,
                    "text",
                    ""
                ) or ""
            ).replace(
                "\n",
                " "
            ).strip()

            if len(text) > 180:

                text = (
                    text[:180]
                    + "..."
                )

            print(
                f"BLOCK {block.id:>4} | "
                f"{getattr(block, 'role', 'unknown'):15} | "
                f"bbox=("
                f"{block.x1}, "
                f"{block.y1}, "
                f"{block.x2}, "
                f"{block.y2})"
            )

            print(
                f"       {text}"
            )

    # ---------------------------------------------------------
    # Search important CM Mann blocks
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("CM MANN / ARTICLE 010-011 BLOCK SEARCH")
    print("=" * 70)

    search_terms = [
        "CM Mann",
        "AAP govt",
        "Gen Z",
        "paper leaks",
        "pharmacy recruitment",
        "opening day",
        "Cong walks out",
    ]

    for block in blocks:

        text = str(
            getattr(
                block,
                "text",
                ""
            ) or ""
        )

        text_lower = text.lower()

        if any(
            term.lower()
            in text_lower
            for term in search_terms
        ):

            print()
            print(
                f"BLOCK {block.id}"
            )

            print(
                f"ROLE: "
                f"{getattr(block, 'role', 'unknown')}"
            )

            print(
                f"READING ORDER: "
                f"{getattr(block, 'reading_order', '?')}"
            )

            print(
                f"BBOX: "
                f"({block.x1}, "
                f"{block.y1}, "
                f"{block.x2}, "
                f"{block.y2})"
            )

            print(
                f"TEXT: {text}"
            )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()