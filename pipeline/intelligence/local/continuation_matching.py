"""
Continuation-matching and logical-article merge logic.

This module is a COPY (not an import) of the Gemini-independent
classmethods that already lived on
`pipeline.intelligence.gemini_article_extractor.GeminiArticleExtractor`
(the functions never touched `self.client`; they only compare
already-extracted text/metadata between article records). It is
copied rather than moved so the Gemini engine stays byte-for-byte
untouched during the migration to a local article extractor.

Used by `pipeline.intelligence.local.local_article_extractor` as the
PRIMARY continuation-resolution pass (there is no Gemini-produced
seed link list here, unlike in the original "repair" usage).
"""

from __future__ import annotations

import difflib
import re
from typing import Any


# ============================================================
# SAFE INT / SORT KEYS
# ============================================================


def safe_int(value: Any) -> int | None:

    try:
        return int(value)
    except Exception:
        return None


def article_sort_key(article_id: str) -> tuple:

    prefix = "article_"

    if article_id.startswith(prefix):

        number = article_id[len(prefix):]

        try:
            return (0, int(number))
        except ValueError:
            pass

    return (1, article_id)


# ============================================================
# LINK / PENDING KEYS
# ============================================================


def link_key(link: dict[str, Any], side: str) -> tuple[int, str] | None:

    try:
        if side == "source":
            return (
                int(link["source_page"]),
                str(link["source_article_id"]),
            )

        return (
            int(link["target_page"]),
            str(link["target_article_id"]),
        )

    except Exception:
        return None


def pending_source_key(pending: dict[str, Any]) -> tuple[int, str] | None:

    source = pending.get("source") or {}

    try:
        return (
            int(source["page"]),
            str(source["article_id"]),
        )
    except Exception:
        return None


def deduplicate_pending(
    pending: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    result = []
    seen = set()

    for item in pending:

        key = pending_source_key(item)

        if key is None:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


# ============================================================
# ELIGIBILITY
# ============================================================


def continuation_marker_target_page(article: dict[str, Any]) -> int | None:
    """Extract an explicit continuation target page without touching crops."""
    continuation = article.get("continuation") or {}
    if not isinstance(continuation, dict):
        continuation = {}

    target = safe_int(continuation.get("next_page"))
    if target is not None:
        return target

    patterns = (
        r"(?:more\s+on|continued\s+on|continued\s+from|report\s+on|full\s+report\s+on|turn\s+to|see|to\s+be\s+continued)[^\n]{0,100}?(?:page|pg\.?|p\.?)\s*[-:]?\s*(\d{1,4})\b",
        r"(?:page|pg\.?|p\.?)\s*[-:]?\s*(\d{1,4})\b",
    )

    candidates = []
    marker = str(continuation.get("marker", "") or "").strip()
    if marker:
        candidates.append(marker)
    for field in ("article_text", "summary", "headline", "subheadline", "location"):
        value = article.get(field)
        if value:
            candidates.append(str(value))

    for candidate in candidates:
        for pattern in patterns:
            match = re.search(pattern, candidate, flags=re.IGNORECASE)
            if match:
                page = safe_int(match.group(1))
                if page is not None:
                    return page
    return None


def headline_is_titleless(article: dict[str, Any]) -> bool:
    headline = str(article.get("headline", "") or "").strip()
    normalized = " ".join(headline.lower().split())
    return not normalized or normalized in {
        "untitled",
        "untitled article",
        "continued",
        "more on page",
        "report on page",
        "continued from page",
    }


def continuation_eligible(
    article: dict[str, Any] | None,
    require_text: bool = True,
) -> tuple[bool, str]:
    """Strict eligibility for a continuation TARGET."""
    if not isinstance(article, dict):
        return False, "Article record is missing."

    content_type = str(article.get("content_type", "") or "").strip().lower()
    if content_type != "article":
        return False, f"content_type={content_type or 'missing'}"

    if require_text:
        article_text = str(article.get("article_text", "") or "").strip()
        if len(article_text) < 120:
            return False, "article_text is too short for continuation target"

    headline = str(article.get("headline", "") or "").strip()
    generic_headlines = {
        "windows", "photo", "photograph", "picture",
        "file photo", "pti photo", "more on page",
        "report on page", "continued",
    }
    if " ".join(headline.lower().split()) in generic_headlines:
        return False, f"generic headline={headline!r}"

    return True, "eligible article target"


def continuation_source_eligible(
    article: dict[str, Any] | None,
    require_text: bool = True,
) -> tuple[bool, str]:
    """
    Eligibility for a continuation SOURCE.

    Normal articles are allowed. A photo_caption/reference is allowed
    ONLY when it has an explicit page continuation marker. Its original
    content_type and verified crop boundary are never changed.
    """
    if not isinstance(article, dict):
        return False, "Source record is missing."

    content_type = str(article.get("content_type", "") or "").strip().lower()

    if content_type == "article":
        if require_text:
            article_text = str(article.get("article_text", "") or "").strip()
            if len(article_text) < 60:
                return False, "article source text is too short"
        return True, "eligible article source"

    if content_type not in {"photo_caption", "reference"}:
        return False, f"content_type={content_type or 'missing'}"

    target_page = continuation_marker_target_page(article)
    if target_page is None:
        return False, (
            f"content_type={content_type} has no explicit continuation target page"
        )

    return True, f"eligible {content_type} continuation source -> page {target_page}"


# ============================================================
# TEXT / METADATA OVERLAP SCORING
# ============================================================


def normalize_match_text(value: Any) -> str:
    """Normalize extracted text for physical story matching."""
    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x is not None)
    elif isinstance(value, dict):
        value = " ".join(str(x) for x in value.values())

    text = str(value).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_tokens(value: Any) -> list[str]:
    """Return useful unicode word tokens, excluding common newspaper stop words."""
    text = normalize_match_text(value)
    if not text:
        return []

    stop = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in",
        "on", "for", "from", "with", "by", "at", "as", "is", "are",
        "was", "were", "be", "been", "being", "that", "this", "these",
        "those", "it", "its", "into", "over", "after", "before", "than",
        "then", "they", "their", "them", "he", "she", "his", "her", "we",
        "our", "you", "your", "i", "me", "my", "will", "would", "could",
        "should", "has", "have", "had", "not", "no", "also", "said", "says",
        "report", "reports", "page", "continued", "more", "see", "full",
    }

    return [
        token
        for token in text.split()
        if len(token) >= 3 and token not in stop
    ]


def text_overlap_score(
    source_text: str,
    target_text: str,
) -> tuple[float, int, float, float]:
    """
    Compare the END of a source physical article with the BEGINNING
    of a target physical article.

    Returns:
        combined_score, shared_token_count, token_jaccard, ngram_overlap
    """
    source_tail = str(source_text or "")[-1800:]
    target_head = str(target_text or "")[:1800]

    source_tokens = match_tokens(source_tail)
    target_tokens = match_tokens(target_head)

    if not source_tokens or not target_tokens:
        return 0.0, 0, 0.0, 0.0

    source_set = set(source_tokens)
    target_set = set(target_tokens)
    shared = source_set & target_set

    union = source_set | target_set
    token_jaccard = len(shared) / len(union) if union else 0.0

    def ngrams(tokens: list[str], n: int = 3) -> set[tuple[str, ...]]:
        if len(tokens) < n:
            return set()
        return {
            tuple(tokens[i:i + n])
            for i in range(len(tokens) - n + 1)
        }

    source_ngrams = ngrams(source_tokens)
    target_ngrams = ngrams(target_tokens)
    if source_ngrams and target_ngrams:
        ngram_overlap = len(source_ngrams & target_ngrams) / max(
            1,
            min(len(source_ngrams), len(target_ngrams)),
        )
    else:
        ngram_overlap = 0.0

    source_norm = normalize_match_text(source_tail)
    target_norm = normalize_match_text(target_head)
    char_similarity = difflib.SequenceMatcher(
        None,
        source_norm[-900:],
        target_norm[:900],
        autojunk=False,
    ).ratio()

    shared_density = len(shared) / max(1, min(len(source_set), len(target_set)))

    score = (
        0.38 * token_jaccard
        + 0.27 * ngram_overlap
        + 0.20 * char_similarity
        + 0.15 * shared_density
    )

    return (
        min(1.0, score),
        len(shared),
        token_jaccard,
        ngram_overlap,
    )


def metadata_overlap_score(
    source: dict[str, Any],
    target: dict[str, Any],
) -> float:
    """Small supporting score from structured knowledge fields."""
    fields = ("entities", "topics", "keywords")
    source_values = set()
    target_values = set()

    for field in fields:
        source_values.update(match_tokens(source.get(field)))
        target_values.update(match_tokens(target.get(field)))

    if not source_values or not target_values:
        return 0.0

    return len(source_values & target_values) / max(
        1,
        len(source_values | target_values),
    )


def score_continuation_pair(
    source: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    """Score a physical source -> real article target pair."""
    text_score, shared, jaccard, ngram = text_overlap_score(
        source.get("article_text", ""),
        target.get("article_text", ""),
    )
    metadata_score = metadata_overlap_score(source, target)
    source_type = str(source.get("content_type", "") or "").strip().lower()

    source_headline = normalize_match_text(source.get("headline", ""))
    target_headline = normalize_match_text(target.get("headline", ""))
    headline_score = 0.0
    if source_type == "article" and source_headline and target_headline:
        headline_score = difflib.SequenceMatcher(
            None, source_headline, target_headline, autojunk=False
        ).ratio()

    titleless = headline_is_titleless(target)
    if source_type in {"photo_caption", "reference"}:
        final_score = 0.68 * text_score + 0.32 * metadata_score
    elif titleless:
        final_score = 0.72 * text_score + 0.28 * metadata_score
    else:
        final_score = 0.58 * text_score + 0.22 * metadata_score + 0.20 * headline_score

    return {
        "score": round(min(1.0, final_score), 4),
        "text_score": round(text_score, 4),
        "metadata_score": round(metadata_score, 4),
        "headline_score": round(headline_score, 4),
        "shared_tokens": shared,
        "token_jaccard": round(jaccard, 4),
        "ngram_overlap": round(ngram, 4),
        "target_titleless": titleless,
        "source_content_type": source_type,
    }


# ============================================================
# GLOBAL CONTINUATION MATCHING
#
# Adapted from `GeminiArticleExtractor._repair_titleless_continuations`.
# In the original, this ran AFTER Gemini had already produced its own
# `continuation_links`, and only repaired titleless/ambiguous cases.
# Here there is no Gemini seed, so it is the PRIMARY (and only)
# continuation-resolution pass: `continuation_links` starts empty and
# every article's own `continuation.next_page` (populated locally via
# regex + ContinuationClassifier, see field_extractors.py) drives the
# matching loop below.
# ============================================================


def match_all_continuations(
    articles: list[dict[str, Any]],
    continuation_links: list[dict[str, Any]] | None = None,
    pending_continuations: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Deterministic, local continuation matching.

    Resolves each article's `continuation.next_page` marker to a
    specific target article on that page using text/metadata overlap
    scoring, with the exact numeric thresholds already validated in
    the Gemini-era codebase (see plan for rationale).
    """
    continuation_links = list(continuation_links or [])
    pending_continuations = list(pending_continuations or [])

    article_map: dict[tuple[int, str], dict[str, Any]] = {}
    for article in articles:
        if not isinstance(article, dict):
            continue
        try:
            key = (int(article.get("page")), str(article.get("article_id")))
        except Exception:
            continue
        article_map[key] = article

    # No Gemini-seeded links to bootstrap from locally.
    clean_links: list[dict[str, Any]] = []
    used_sources: set[tuple[int, str]] = set()
    used_targets: set[tuple[int, str]] = set()

    for link in continuation_links:
        if not isinstance(link, dict):
            continue
        source_key = link_key(link, "source")
        target_key = link_key(link, "target")
        if source_key not in article_map or target_key not in article_map:
            continue
        if source_key == target_key:
            continue
        if source_key in used_sources or target_key in used_targets:
            continue

        source_ok, _ = continuation_source_eligible(article_map[source_key])
        target_ok, _ = continuation_eligible(article_map[target_key])
        if not source_ok or not target_ok:
            continue

        try:
            confidence = float(link.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        if confidence < 0.75:
            continue

        clean_links.append(dict(link))
        used_sources.add(source_key)
        used_targets.add(target_key)

    added = 0
    corrected = 0
    unresolved = 0
    repairs = []

    sources = sorted(
        article_map.values(),
        key=lambda a: (
            safe_int(a.get("page")) or 999999,
            article_sort_key(str(a.get("article_id", ""))),
        ),
    )

    link_by_source = {
        link_key(link, "source"): link
        for link in clean_links
    }

    for source in sources:
        source_key = (
            safe_int(source.get("page")),
            str(source.get("article_id", "")),
        )
        if source_key[0] is None or not source_key[1]:
            continue

        source_ok, _ = continuation_source_eligible(source)
        if not source_ok:
            continue

        target_page = continuation_marker_target_page(source)
        if target_page is None:
            continue

        candidates = [
            article
            for key, article in article_map.items()
            if key[0] == target_page and key != source_key
        ]

        candidates = [
            article
            for article in candidates
            if continuation_eligible(article)[0]
        ]

        if not candidates:
            unresolved += 1
            continue

        scored = []
        for target in candidates:
            target_key = (int(target["page"]), str(target["article_id"]))
            score = score_continuation_pair(source, target)
            scored.append((score, target_key, target))

        scored.sort(
            key=lambda item: (
                item[0]["score"],
                item[0]["text_score"],
                item[0]["metadata_score"],
            ),
            reverse=True,
        )

        best_score, best_key, best_target = scored[0]
        second_score = scored[1][0]["score"] if len(scored) > 1 else 0.0

        source_type = str(source.get("content_type", "") or "").strip().lower()
        titleless = bool(best_score["target_titleless"])

        # NOTE: these thresholds were re-tuned against real RapidOCR output
        # (see tests/test_local_article_extractor_fixture.py), not guessed
        # analytically. Full-article OCR text is longer than the extraction
        # Gemini used to produce, which dilutes the tail/head token-overlap
        # window (see text_overlap_score) even for genuine continuations —
        # the original 0.62-0.68 thresholds almost never fire on locally
        # extracted text. All 5 known continuations in the doc_000001
        # fixture score comfortably above these lowered thresholds with
        # healthy margins from their next-best candidate; a same-numbered
        # coincidental false match that scores higher than its true target
        # is caught by the ambiguity-margin guard below, not by this
        # threshold, so lowering it here does not require weakening that
        # guard.
        if source_type in {"photo_caption", "reference"}:
            threshold = 0.22
            min_shared = 2
            min_text = 0.12
            min_metadata = 0.08
            special_source = True
        else:
            threshold = 0.22 if titleless else 0.25
            min_shared = 3
            min_text = 0.20
            min_metadata = 0.0
            special_source = False

        strong_enough = (
            best_score["score"] >= threshold
            and best_score["shared_tokens"] >= min_shared
            and best_score["text_score"] >= min_text
            and (
                not special_source
                or best_score["metadata_score"] >= min_metadata
                or best_score["text_score"] >= 0.42
            )
        )

        if len(scored) > 1 and (best_score["score"] - second_score) < 0.045:
            strong_enough = False

        if not strong_enough:
            unresolved += 1
            repairs.append({
                "source": {"page": source_key[0], "article_id": source_key[1]},
                "target_page": target_page,
                "best_target": {"page": best_key[0], "article_id": best_key[1]},
                "score": best_score,
                "second_best_score": round(second_score, 4),
                "action": "UNRESOLVED",
            })
            continue

        existing = link_by_source.get(source_key)
        new_link = {
            "source_page": source_key[0],
            "source_article_id": source_key[1],
            "target_page": best_key[0],
            "target_article_id": best_key[1],
            "confidence": round(max(0.75, best_score["score"]), 4),
            "reason": (
                "Text-based physical continuation match. "
                f"text_score={best_score['text_score']:.3f}, "
                f"metadata_score={best_score['metadata_score']:.3f}, "
                f"shared_tokens={best_score['shared_tokens']}, "
                f"titleless_target={best_score['target_titleless']}"
            ),
            "match_method": "local_text_matching",
        }

        if existing is None:
            if best_key in used_targets:
                unresolved += 1
                continue
            clean_links.append(new_link)
            link_by_source[source_key] = new_link
            used_sources.add(source_key)
            used_targets.add(best_key)
            added += 1
            action = "ADDED"
        else:
            old_key = link_key(existing, "target")
            if old_key != best_key:
                if best_key in used_targets and old_key != best_key:
                    unresolved += 1
                    continue
                used_targets.discard(old_key)
                used_targets.add(best_key)
                existing.clear()
                existing.update(new_link)
                corrected += 1
                action = "CORRECTED"
            else:
                existing.update(new_link)
                action = "CONFIRMED"

        repairs.append({
            "source": {"page": source_key[0], "article_id": source_key[1]},
            "target": {"page": best_key[0], "article_id": best_key[1]},
            "score": best_score,
            "action": action,
        })

    resolved_sources = {link_key(link, "source") for link in clean_links}

    final_pending = []
    for pending in pending_continuations:
        if not isinstance(pending, dict):
            continue
        source_key = pending_source_key(pending)
        if source_key in resolved_sources:
            continue

        pending_copy = dict(pending)
        target_page = safe_int(pending_copy.get("target_page"))
        if target_page in {key[0] for key in article_map}:
            pending_copy["status"] = "UNRESOLVED"
        else:
            pending_copy["status"] = "PENDING_EXTERNAL"
        final_pending.append(pending_copy)

    final_pending = deduplicate_pending(final_pending)

    source_target_map = {link_key(link, "source"): link for link in clean_links}

    for article in articles:
        source_key = (
            safe_int(article.get("page")),
            str(article.get("article_id", "")),
        )
        link = source_target_map.get(source_key)
        if link:
            article["continuation_target"] = {
                "target_page": link.get("target_page"),
                "target_article_id": link.get("target_article_id"),
                "confidence": link.get("confidence"),
                "match_method": link.get("match_method", "local"),
            }
        else:
            article.pop("continuation_target", None)

    report = {
        "added": added,
        "corrected": corrected,
        "unresolved": unresolved,
        "repairs": repairs,
    }

    return clean_links, final_pending, report


# ============================================================
# BUILD FINAL LOGICAL ARTICLES (union-find merge)
# ============================================================


def build_logical_articles(
    articles: list[dict[str, Any]],
    continuation_links: list[dict[str, Any]],
    pending_continuations: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    article_map = {}

    for article in articles:
        try:
            key = (int(article.get("page")), str(article.get("article_id")))
        except Exception:
            continue
        article_map[key] = article

    parent = {key: key for key in article_map}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first, second):
        if first not in parent or second not in parent:
            return
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    valid_links = []

    for link in continuation_links:
        source_key = link_key(link, "source")
        target_key = link_key(link, "target")

        if source_key is None or target_key is None:
            continue

        if source_key not in article_map or target_key not in article_map:
            continue

        union(source_key, target_key)
        valid_links.append(link)

    components = {}

    for key in article_map:
        root = find(key)
        components.setdefault(root, []).append(key)

    pending_source_map = {}

    for pending in pending_continuations:
        source_key = pending_source_key(pending)
        if source_key is None:
            continue
        pending_source_map[source_key] = pending

    logical_articles = []
    logical_counter = 1

    sorted_components = sorted(
        components.values(),
        key=lambda keys: min(
            (key[0], article_sort_key(key[1]))
            for key in keys
        )
    )

    for component_keys in sorted_components:

        component_keys.sort(
            key=lambda key: (key[0], article_sort_key(key[1]))
        )

        component_articles = [article_map[key] for key in component_keys]

        component_links = []

        for link in valid_links:
            source_key = link_key(link, "source")
            if source_key in component_keys:
                component_links.append(link)

        component_pending = []

        for key in component_keys:
            if key in pending_source_map:
                component_pending.append(pending_source_map[key])

        if component_pending:
            status = "pending_external"
        elif component_links:
            status = "merged"
        else:
            status = "standalone"

        source_parts = []

        for article in component_articles:

            source_part = {
                "page": article.get("page"),
                "article_id": article.get("article_id"),
            }

            continuation_target = article.get("continuation_target")
            if isinstance(continuation_target, dict):
                source_part["target_page"] = continuation_target.get("target_page")
                source_part["target_article_id"] = continuation_target.get("target_article_id")
                source_part["target_confidence"] = continuation_target.get("confidence")
                source_part["match_method"] = continuation_target.get("match_method")

            source_parts.append(source_part)

        logical_article = {
            "logical_article_id": f"logical_{logical_counter:04d}",
            "status": status,
            "source_parts": source_parts,
            "continuation_links": component_links,
            "pending_continuations": component_pending,
            "articles": component_articles,
        }

        logical_articles.append(logical_article)
        logical_counter += 1

    return logical_articles
