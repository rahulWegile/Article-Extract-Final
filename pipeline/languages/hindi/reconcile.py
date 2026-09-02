"""
Reconciles Stream B (document-order extraction) against Stream A
(boundary-based extraction).

Policy: Stream A stays authoritative. Stream B is only used to fill
genuine gaps -- content Stream A's boundary detection lost or split
away -- never to overwrite or out-vote Stream A on content both
sides already cover.

For every Stream B section:
1. If its heading matches an existing article's headline/
   subheadline (on the same or an adjacent page), it's already
   covered -- skipped.
2. Otherwise, check vocabulary overlap against articles on the same
   or an adjacent page. If a plausible parent clears MIN_OVERLAP,
   the section's text is APPENDED to that article's article_text
   (never overwritten).
3. If no plausible parent clears the threshold, the section is left
   unresolved -- returned separately for visibility/logging, never
   fabricated into a new article and never silently dropped.

This reuses the same evidence-based overlap approach already
validated in openai_article_extractor.py's continuation-link guard
(CONTINUATION_MARKER + vocabulary overlap), generalized to work on
both Devanagari and Latin-script text since Stream B can run on
either language.
"""

from __future__ import annotations

import re
from typing import Any

_SIGNIFICANT_TOKEN_PATTERN = re.compile(
    r"[ऀ-ॿ]{4,}|[A-Za-z]{4,}"
)

MIN_OVERLAP = 0.20

PAGE_ADJACENCY = 1


def _significant_tokens(text: str) -> set:

    return {
        token.lower()
        for token in _SIGNIFICANT_TOKEN_PATTERN.findall(
            text or ""
        )
    }


def _vocabulary_overlap(
    source_text: str,
    target_text: str,
) -> float:

    source_tokens = _significant_tokens(source_text)
    target_tokens = _significant_tokens(target_text)

    if not source_tokens or not target_tokens:
        return 0.0

    shared = source_tokens & target_tokens

    return len(shared) / min(
        len(source_tokens),
        len(target_tokens),
    )


def _safe_int(value: Any) -> Any:

    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _normalize_heading(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        (text or "").strip().lower(),
    )


def _heading_covered(
    section_heading: str,
    nearby_articles: list[dict[str, Any]],
) -> bool:

    normalized = _normalize_heading(section_heading)

    if not normalized:
        return False

    for article in nearby_articles:

        for field in ("headline", "subheadline"):

            candidate = _normalize_heading(
                article.get(field) or ""
            )

            if not candidate:
                continue

            if (
                normalized == candidate
                or normalized in candidate
                or candidate in normalized
            ):
                return True

    return False


def reconcile_articles(
    articles: list[dict[str, Any]],
    document_order_sections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (articles, unresolved_sections). `articles` is mutated
    in place (article_text appended where a gap was recovered) and
    also returned for convenience.
    """

    if not document_order_sections:
        return articles, []

    articles_by_page: dict[Any, list[dict[str, Any]]] = {}

    for article in articles:

        articles_by_page.setdefault(
            _safe_int(article.get("page")),
            [],
        ).append(article)

    unresolved = []
    recovered = 0

    for section in document_order_sections:

        page = _safe_int(section.get("page"))
        heading = section.get("heading")
        text = (section.get("text") or "").strip()

        if not text:
            continue

        nearby_articles: list[dict[str, Any]] = []

        for offset in range(
            -PAGE_ADJACENCY,
            PAGE_ADJACENCY + 1,
        ):

            candidate_page = (
                page + offset
                if isinstance(page, int)
                else page
            )

            nearby_articles.extend(
                articles_by_page.get(
                    candidate_page,
                    [],
                )
            )

        if heading and _heading_covered(
            heading,
            nearby_articles,
        ):
            continue

        best_article = None
        best_overlap = 0.0

        for article in nearby_articles:

            overlap = _vocabulary_overlap(
                text,
                article.get("article_text") or "",
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_article = article

        if (
            best_article is not None
            and best_overlap >= MIN_OVERLAP
        ):

            existing_text = (
                best_article.get("article_text") or ""
            )

            if text not in existing_text:

                best_article["article_text"] = (
                    existing_text.rstrip()
                    + "\n\n"
                    + text
                ).strip()

                recovered += 1

        else:

            unresolved.append(section)

    if recovered or unresolved:

        print(
            f"Document-order reconciliation: recovered "
            f"{recovered} gap section(s), {len(unresolved)} "
            "left unresolved"
        )

    return articles, unresolved
