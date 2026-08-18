"""
Validation helpers, copied from the Gemini-independent parts of
`pipeline.intelligence.gemini_article_extractor.GeminiArticleExtractor`.

Only `validate_manifest_completeness` is actively used by
`LocalArticleExtractor` today (there is no external model response to
validate locally — every article is produced in-process). The rest are
kept as a straight port for parity/testing and possible future reuse
if a semi-automated review step is added.
"""

from __future__ import annotations

from typing import Any


def validate_manifest_completeness(
    articles: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Confirm every discovered crop (per the filesystem manifest) produced
    exactly one article record, with no duplicates/omissions.
    """
    expected = set()

    for item in manifest:
        key = (int(item["page"]), str(item["article_id"]))
        expected.add(key)

    returned = set()
    duplicates = []

    for article in articles:
        if not isinstance(article, dict):
            continue
        try:
            key = (int(article.get("page")), str(article.get("article_id")))
        except Exception:
            continue

        if key in returned:
            duplicates.append({"page": key[0], "article_id": key[1]})

        returned.add(key)

    missing = sorted(expected - returned)
    unexpected = sorted(returned - expected)

    return {
        "expected": len(expected),
        "returned": len(returned),
        "missing": [
            {"page": page, "article_id": article_id}
            for page, article_id in missing
        ],
        "unexpected": [
            {"page": page, "article_id": article_id}
            for page, article_id in unexpected
        ],
        "duplicates": duplicates,
        "complete": (
            not missing
            and not unexpected
            and not duplicates
            and len(expected) == len(returned)
        ),
    }
