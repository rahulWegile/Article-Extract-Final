"""
Regression test for LocalArticleExtractor, run against whatever real,
already-cropped document currently sits at output/documents/doc_000001.

This intentionally does NOT hardcode ground truth (specific headlines/
continuation links) for one fixed document -- that document gets
replaced by real uploads during normal use of this app (it did exactly
that once already: the original 3-page "Asian Age" fixture used while
building this extractor was later overwritten by a real user upload).
Instead it asserts properties that must hold for ANY correctly
processed document:

- every discovered crop produces exactly one article record
- no article's text contains an internal duplicate paragraph (the
  overlapping/duplicate-detection-box bug -- see block_text_extractor.py)
- every continuation_link references real, existing articles
- logical_article_count never exceeds article_count

This writes into output/documents/doc_000001/gemini_article_batches/,
so it saves a copy of whatever is already there first and restores it
afterwards -- that directory is always real data (either a prior
Gemini run or a prior local run), never disposable test fixtures.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

from pipeline.intelligence.local.local_article_extractor import LocalArticleExtractor

DOCUMENT_DIR = Path("output/documents/doc_000001")
BATCH_DIR = DOCUMENT_DIR / "gemini_article_batches"


def _find_internal_duplicate_paragraphs(articles):
    """
    Flag any article whose text contains the same non-trivial
    paragraph more than once -- the exact symptom of the overlapping/
    duplicate-block bug this test guards against.
    """
    offenders = []

    for article in articles:

        text = article.get("article_text", "") or ""

        paragraphs = [
            " ".join(p.split())
            for p in text.split("\n")
            if len(p.strip()) >= 40
        ]

        counts = Counter(paragraphs)
        duplicated = [p for p, count in counts.items() if count > 1]

        if duplicated:
            offenders.append({
                "page": article.get("page"),
                "article_id": article.get("article_id"),
                "duplicated_paragraphs": duplicated,
            })

    return offenders


def run_fixture_regression():

    if not BATCH_DIR.exists():
        print(f"SKIP: {BATCH_DIR} not present, no document to test against.")
        return

    backup_dir = BATCH_DIR.parent / "gemini_article_batches.regression_backup"

    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    shutil.copytree(BATCH_DIR, backup_dir)

    try:
        extractor = LocalArticleExtractor(pages_per_batch=3)
        extractor.process_document(DOCUMENT_DIR)

        with open(BATCH_DIR / "final_logical_articles.json", "r", encoding="utf-8") as f:
            result = json.load(f)

        completeness = result.get("manifest_completeness", {})
        assert completeness.get("complete"), (
            f"Manifest incomplete: {completeness}"
        )

        duplicate_offenders = _find_internal_duplicate_paragraphs(result["articles"])
        assert not duplicate_offenders, (
            f"Articles with internally duplicated paragraphs: {duplicate_offenders}"
        )

        article_keys = {
            (article["page"], article["article_id"])
            for article in result["articles"]
        }

        for link in result["continuation_links"]:
            source_key = (link["source_page"], link["source_article_id"])
            target_key = (link["target_page"], link["target_article_id"])
            assert source_key in article_keys, f"Dangling link source: {link}"
            assert target_key in article_keys, f"Dangling link target: {link}"

        assert result["logical_article_count"] <= result["article_count"], (
            f"{result['logical_article_count']} > {result['article_count']}"
        )

        print("=" * 60)
        print("LOCAL ARTICLE EXTRACTOR REGRESSION PASSED")
        print(f"Article count          : {result['article_count']}")
        print(f"Continuation links     : {result['continuation_link_count']}")
        print(f"Logical article count  : {result['logical_article_count']}")
        print(f"Duplicate paragraphs   : {len(duplicate_offenders)}")
        print("=" * 60)

    finally:
        shutil.rmtree(BATCH_DIR)
        shutil.move(str(backup_dir), str(BATCH_DIR))


if __name__ == "__main__":
    run_fixture_regression()
