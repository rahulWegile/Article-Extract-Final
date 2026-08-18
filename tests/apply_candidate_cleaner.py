from pathlib import Path

from pipeline.intelligence.candidate_cleaner import (
    apply_cleaner_file,
)


DOCUMENT_DIR = Path(
    "output/documents/doc_000018"
)

apply_cleaner_file(

    newspaper_json=(
        DOCUMENT_DIR /
        "newspaper.json"
    ),

    cleaner_json=(
        DOCUMENT_DIR /
        "cleaned_candidates" /
        "batch_001_cleaner.json"
    ),

    output_json=(
        DOCUMENT_DIR /
        "cleaned_candidates" /
        "batch_001_candidates_cleaned.json"
    ),

    page_json=(
        DOCUMENT_DIR /
        "page_json" /
        "page_003.json"
    ),
)