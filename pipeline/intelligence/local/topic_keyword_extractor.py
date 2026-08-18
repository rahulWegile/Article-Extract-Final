"""
Lightweight keyword/topic extraction.

`keywords` uses frequency-filtered capitalized phrases and common nouns
from the article text (a simple, dependency-free proxy for RAKE/TF-IDF).
`topics` reuses the category taxonomy's top matches as coarse tags, kept
as best-effort per the migration plan.
"""

from __future__ import annotations

import re
from collections import Counter

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "from", "with", "by", "at", "as", "is", "are", "was", "were", "be",
    "been", "being", "that", "this", "these", "those", "it", "its",
    "into", "over", "after", "before", "than", "then", "they", "their",
    "them", "he", "she", "his", "her", "we", "our", "you", "your", "i",
    "me", "my", "will", "would", "could", "should", "has", "have", "had",
    "not", "no", "also", "said", "says", "report", "reports", "page",
    "continued", "more", "see", "full",
}

_PHRASE_PATTERN = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
_WORD_PATTERN = re.compile(r"[A-Za-z]{4,}")

_MAX_KEYWORDS = 10


def extract_keywords(text: str) -> list[str]:

    if not text:
        return []

    phrase_counts = Counter(_PHRASE_PATTERN.findall(text))

    word_counts = Counter(
        word.lower()
        for word in _WORD_PATTERN.findall(text)
        if word.lower() not in _STOPWORDS
    )

    ranked_phrases = [phrase for phrase, count in phrase_counts.most_common() if count >= 1]

    ranked_words = [word for word, count in word_counts.most_common() if count >= 2]

    keywords: list[str] = []
    seen_lower = set()

    for candidate in ranked_phrases + ranked_words:
        lowered = candidate.lower()
        if lowered in seen_lower:
            continue
        seen_lower.add(lowered)
        keywords.append(candidate)
        if len(keywords) >= _MAX_KEYWORDS:
            break

    return keywords


def extract_topics(keywords: list[str], category: str) -> list[str]:
    """Keep `topics` as a simple, small subset seeded by category + top keywords."""

    topics: list[str] = []

    if category and category != "other":
        topics.append(category)

    for keyword in keywords[:3]:
        if keyword not in topics:
            topics.append(keyword)

    return topics
