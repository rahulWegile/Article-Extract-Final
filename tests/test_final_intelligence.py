import json
from pathlib import Path

from pipeline.gemini.newspaper_client import NewspaperClient


DOCUMENT_DIR = Path(
    "output/documents/doc_000018"
)

CLEANED_FILE = (
    DOCUMENT_DIR
    / "cleaned_candidates"
    / "batch_001_candidates_cleaned.json"
)

PAGES_DIR = DOCUMENT_DIR / "pages"

OUTPUT_FILE = (
    DOCUMENT_DIR
    / "cleaned_candidates"
    / "batch_001_final_intelligence.json"
)


def main():

    print()
    print("=" * 60)
    print("FINAL NEWSPAPER INTELLIGENCE TEST")
    print("=" * 60)

    # -------------------------------------------------
    # Load cleaned candidates
    # -------------------------------------------------

    print()
    print("Loading cleaned candidates...")

    with open(
        CLEANED_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        cleaned = json.load(f)

    articles = cleaned.get(
        "articles",
        [],
    )

    print(
        f"Cleaned articles : {len(articles)}"
    )

    # -------------------------------------------------
    # Load document metadata
    # -------------------------------------------------

    document_json = (
        DOCUMENT_DIR / "document.json"
    )

    metadata = {}

    if document_json.exists():

        with open(
            document_json,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

    # -------------------------------------------------
    # Build final batch
    # -------------------------------------------------

    batch = {

        "batch_number": 1,

        "start_page": 3,

        "end_page": 3,

        "newspaper_name": metadata.get(
            "newspaper_name",
            "",
        ),

        "edition": metadata.get(
            "edition",
            "",
        ),

        "publish_date": metadata.get(
            "publish_date",
            "",
        ),

        "language": metadata.get(
            "language",
            "English",
        ),

        "articles": articles,

        "page_images": [
            str(
                PAGES_DIR
                / "page_003.png"
            )
        ],
    }

    # -------------------------------------------------
    # Print what will be sent
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL GEMINI INPUT")
    print("=" * 60)

    for article in articles:

        print(
            article.get(
                "candidate_id",
                "",
            ),
            "| blocks:",
            len(
                article.get(
                    "block_ids",
                    [],
                )
            ),
            "| paragraphs:",
            len(
                article.get(
                    "paragraphs",
                    [],
                )
            ),
        )

    # -------------------------------------------------
    # Call final intelligence
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("CALLING FINAL NEWSPAPER INTELLIGENCE")
    print("=" * 60)

    client = NewspaperClient()

    result = client.analyze_batch(
        batch
    )

    # -------------------------------------------------
    # Save result
    # -------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("FINAL INTELLIGENCE COMPLETED")
    print("=" * 60)

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Final articles:",
        len(
            result.get(
                "articles",
                [],
            )
        ),
    )


if __name__ == "__main__":
    main()
