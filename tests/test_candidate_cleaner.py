import json
from pathlib import Path

from pipeline.gemini.newspaper_client import NewspaperClient


DOCUMENT_DIR = Path(
    "output/documents/doc_000001"
)

PAGE_NUMBER = 3

NEWSPAPER_JSON = (
    DOCUMENT_DIR /
    "newspaper.json"
)

PAGE_JSON = (
    DOCUMENT_DIR /
    "page_json" /
    f"page_{PAGE_NUMBER:03}.json"
)

PAGE_IMAGE = (
    DOCUMENT_DIR /
    "pages" /
    f"page_{PAGE_NUMBER:03}.png"
)

OUTPUT_DIR = (
    DOCUMENT_DIR /
    "cleaned_candidates"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLEANER_OUTPUT = (
    OUTPUT_DIR /
    "batch_001_cleaner.json"
)


def main():

    # ------------------------------------------------------
    # Load newspaper candidates
    # ------------------------------------------------------

    with open(
        NEWSPAPER_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        newspaper = json.load(f)

    candidates = newspaper.get(
        "articles",
        []
    )

    # ------------------------------------------------------
    # Load page JSON
    # ------------------------------------------------------

    with open(
        PAGE_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        page_data = json.load(f)

    blocks = page_data.get(
        "blocks",
        []
    )

    print()
    print("=" * 60)
    print("CANDIDATE CLEANER TEST")
    print("=" * 60)

    print(
        f"Document : {DOCUMENT_DIR}"
    )

    print(
        f"Page     : {PAGE_NUMBER}"
    )

    print(
        f"Candidates : {len(candidates)}"
    )

    print(
        f"Page blocks : {len(blocks)}"
    )

    for candidate in candidates:

        print(
            candidate.get(
                "candidate_id",
                ""
            )
        )

    # ------------------------------------------------------
    # Build Gemini payload
    # ------------------------------------------------------

    cleaner_input = {

        "page_number": PAGE_NUMBER,

        "candidates": candidates,

        "page_blocks": blocks,

    }

    print()
    print("=" * 60)
    print("CALLING GEMINI CANDIDATE CLEANER")
    print("=" * 60)

    client = NewspaperClient()

    result = client.service.analyze_newspaper(

        newspaper_json=cleaner_input,

        page_images=[
            str(PAGE_IMAGE)
        ],

        prompt=(
            __import__(
                "pipeline.intelligence.candidate_cleaner_prompt",
                fromlist=[
                    "CANDIDATE_CLEANER_PROMPT"
                ],
            )
            .CANDIDATE_CLEANER_PROMPT
        ),

    )

    # ------------------------------------------------------
    # Save cleaner result
    # ------------------------------------------------------

    with open(
        CLEANER_OUTPUT,
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
    print("CANDIDATE CLEANER FINISHED")
    print("=" * 60)

    print(
        f"Input candidates : "
        f"{len(candidates)}"
    )

    print(
        f"Page blocks      : "
        f"{len(blocks)}"
    )

    print(
        f"Output file      : "
        f"{CLEANER_OUTPUT}"
    )

    # ------------------------------------------------------
    # Quick validation
    # ------------------------------------------------------

    split_count = 0
    block_id_count = 0

    for decision in result.get(
        "candidates",
        []
    ):

        if decision.get(
            "action"
        ) != "split":

            continue

        split_count += 1

        for split in decision.get(
            "split_candidates",
            []
        ):

            block_ids = split.get(
                "block_ids",
                []
            )

            block_id_count += len(
                block_ids
            )

    print(
        f"Split candidates : "
        f"{split_count}"
    )

    print(
        f"Returned block IDs : "
        f"{block_id_count}"
    )

    print()
    print(
        "DO NOT RUN FINAL NEWSPAPER "
        "INTELLIGENCE YET."
    )


if __name__ == "__main__":
    main()