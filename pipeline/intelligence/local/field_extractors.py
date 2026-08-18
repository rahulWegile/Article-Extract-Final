"""
Heuristic field extraction from RapidOCR'd article-crop text.

These are best-effort local replacements for what Gemini's vision+
language understanding used to infer directly from the crop image.
Quality is intentionally lower than Gemini's — see the migration plan
for the accepted trade-off. Fields are never fabricated: when a
heuristic has no confident signal it returns None rather than guessing.
"""

from __future__ import annotations

import re
from typing import Any

from pipeline.knowledge.heading_classifier import HeadingClassifier
from pipeline.knowledge.caption_classifier import CaptionClassifier
from pipeline.knowledge.continuation_classifier import ContinuationClassifier
from pipeline.intelligence.local.continuation_matching import (
    continuation_marker_target_page,
)

_heading_classifier = HeadingClassifier()
_caption_classifier = CaptionClassifier()
_continuation_classifier = ContinuationClassifier()

# CaptionClassifier.AGENCIES is lowercase; keep a matching-ready set here.
_AGENCIES = {agency.upper() for agency in CaptionClassifier.AGENCIES}

_BYLINE_PATTERNS = (
    re.compile(r"^\s*by\s+([A-Z][A-Za-z.\-' ]{2,60})", re.IGNORECASE),
    re.compile(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s*\n?\s*"
        r"(?:Correspondent|Reporter|Bureau|Staff Writer|Special Correspondent)"
    ),
)

# Classic newspaper dateline: "NEW DELHI, Aug 12:" / "MUMBAI:" / "CHANDIGARH, August 12, 2026:"
_DATELINE_PATTERN = re.compile(
    r"^\s*([A-Z][A-Z .]{2,30}?)"
    r"(?:,\s*([A-Za-z]+\.?\s+\d{1,2}(?:,?\s*\d{4})?))?"
    r"\s*[:—\-]\s*"
)

_AD_PATTERNS = (
    re.compile(r"\bwww\.[a-z0-9\-]+\.[a-z]{2,}\b", re.IGNORECASE),
    re.compile(r"\bcall\s+now\b", re.IGNORECASE),
    re.compile(r"\b(?:rs\.?|inr)\s?\d[\d,]*\s*(?:only|/-)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+booking\b", re.IGNORECASE),
    re.compile(r"\b\d{10}\b"),
)

# Small static gazetteer, not exhaustive — only used to raise confidence
# in the dateline heuristic, never to reject a candidate outright.
_PLACE_GAZETTEER = {
    "NEW DELHI", "MUMBAI", "CHANDIGARH", "BENGALURU", "BANGALORE", "CHENNAI",
    "KOLKATA", "HYDERABAD", "PUNE", "AHMEDABAD", "JAIPUR", "LUCKNOW", "PATNA",
    "BHOPAL", "AMRITSAR", "LUDHIANA", "SHIMLA", "DEHRADUN", "GURUGRAM",
    "GURGAON", "NOIDA", "SRINAGAR", "JAMMU", "RAIPUR", "RANCHI", "GUWAHATI",
    "THIRUVANANTHAPURAM", "KOCHI", "NAGPUR", "INDORE", "SURAT", "AGRA",
    "VARANASI", "KANPUR", "LONDON", "WASHINGTON", "NEW YORK", "GENEVA",
    "BEIJING", "ISLAMABAD", "KATHMANDU", "DHAKA", "COLOMBO",
}


def _ocr_lines(ocr_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Lines from RapidOCREngine.process_article_crop, sorted top-to-bottom."""
    lines = ocr_result.get("lines") or []
    return sorted(lines, key=lambda line: line.get("bbox", {}).get("y1", 0.0))


def _looks_like_byline(text: str) -> bool:
    """A fully title-cased byline line can out-score a real headline on
    HeadingClassifier alone (e.g. "By Our Correspondent"); exclude it
    explicitly from headline candidates."""
    stripped = text.strip()
    if _BYLINE_PATTERNS[0].match(stripped):
        return True
    upper_words = set(re.findall(r"[A-Za-z]+", stripped.upper()))
    if upper_words and upper_words <= _AGENCIES:
        return True
    return False


def extract_headline_and_subheadline(
    ocr_result: dict[str, Any],
) -> tuple[str, str | None]:
    """
    Pick the highest-scoring HeadingClassifier candidate among the
    topmost few OCR lines as the headline; the line immediately below
    it becomes the subheadline if it also reads as heading-like.
    """
    lines = _ocr_lines(ocr_result)

    if not lines:
        text = str(ocr_result.get("text", "") or "").strip()
        return (text[:120], None) if text else ("", None)

    # When text was reconstructed from page_json blocks (see
    # block_text_extractor.py), each line already carries the
    # DocLayout-YOLO class that produced it -- a block already
    # classified as "title" is a much stronger signal than re-deriving
    # heading-likelihood from text shape alone.
    title_lines = [line for line in lines if line.get("block_class") == "title"]

    if title_lines:

        def _height(line: dict[str, Any]) -> float:
            bbox = line.get("bbox") or {}
            return max(0.0, bbox.get("y2", 0.0) - bbox.get("y1", 0.0))

        tallest_height = max(_height(line) for line in title_lines)

        # A newspaper headline wrapping across 2 lines has both lines
        # at roughly the same (large) font size -- join those. A line
        # noticeably smaller than the tallest one (e.g. a kicker/
        # eyebrow tag placed above the real headline) is not part of
        # the headline itself; keep it as the subheadline regardless
        # of whether it sits above or below.
        main_lines = [
            line for line in title_lines
            if tallest_height <= 0 or _height(line) >= tallest_height * 0.8
        ]
        other_lines = [line for line in title_lines if line not in main_lines]

        main_lines.sort(key=lambda l: l.get("bbox", {}).get("y1", 0.0))
        other_lines.sort(key=lambda l: l.get("bbox", {}).get("y1", 0.0))

        headline = " ".join(
            str(line.get("text", "") or "").strip() for line in main_lines
        ).strip()

        subheadline = " ".join(
            str(line.get("text", "") or "").strip() for line in other_lines
        ).strip() or None

        return headline, subheadline

    top_candidates = [
        line for line in lines[:6]
        if not _looks_like_byline(str(line.get("text", "")))
    ] or lines[:6]

    scored = [
        (line, _heading_classifier.predict(str(line.get("text", ""))))
        for line in top_candidates
    ]

    scored.sort(key=lambda item: item[1], reverse=True)

    best_line, best_score = scored[0]

    if best_score < 0.3:
        fallback = str(lines[0].get("text", "") or "").strip()
        return (fallback[:120], None)

    headline = str(best_line.get("text", "") or "").strip()

    # Subheadline: the line immediately below the chosen headline line,
    # if it also reads as heading-like but scores lower.
    try:
        best_index = top_candidates.index(best_line)
    except ValueError:
        best_index = -1

    subheadline = None
    if 0 <= best_index < len(top_candidates) - 1:
        candidate = top_candidates[best_index + 1]
        candidate_score = _heading_classifier.predict(str(candidate.get("text", "")))
        if 0.3 <= candidate_score < best_score:
            subheadline = str(candidate.get("text", "") or "").strip() or None

    return headline, subheadline


def extract_author(article_text: str) -> str | None:
    """Byline / news-agency detection in the first 3 and last 2 lines."""
    lines = [line.strip() for line in article_text.splitlines() if line.strip()]
    search_lines = lines[:3] + lines[-2:]

    for line in search_lines:
        for pattern in _BYLINE_PATTERNS:
            match = pattern.search(line)
            if match:
                candidate = match.group(1).strip(" .")
                if candidate:
                    return candidate

    for line in search_lines:
        upper_words = set(re.findall(r"[A-Za-z]+", line.upper()))
        agency_hits = upper_words & _AGENCIES
        if agency_hits:
            return sorted(agency_hits)[0]

    return None


def extract_location_and_date(article_text: str) -> tuple[str | None, str | None]:
    """Classic dateline at the start of the article body."""
    lead = article_text.strip()[:200]

    match = _DATELINE_PATTERN.match(lead)
    if not match:
        return None, None

    location = match.group(1).strip()
    date = match.group(2).strip() if match.group(2) else None

    # Only trust the match if it looks like a real place name: either it's
    # in the gazetteer, or it's short (avoids false-positives on generic
    # ALL-CAPS opening phrases that aren't datelines).
    if location.upper() not in _PLACE_GAZETTEER and len(location.split()) > 3:
        return None, None

    return location, date


def classify_content_type(
    article_text: str,
    caption_score: float,
    has_figure_block: bool,
) -> str:
    """
    Priority-ordered heuristic matching the five values the rest of the
    pipeline already assumes exist: article, photo_caption, reference,
    advertisement, other.
    """
    text = article_text.strip()

    if caption_score >= 0.55 and has_figure_block:
        return "photo_caption"

    if len(text) < 40 and (not text or sum(c.isdigit() for c in text) > len(text) * 0.4):
        return "reference"

    for pattern in _AD_PATTERNS:
        if pattern.search(text):
            return "advertisement"

    if len(text) < 15:
        return "other"

    return "article"


def caption_score(article_text: str) -> float:
    return _caption_classifier.predict(article_text)


def detect_continuation(article: dict[str, Any]) -> dict[str, Any]:
    """
    Populate the `continuation` object using only local signals:
    the regex-based marker/next_page extractor (also reusable for
    validating already-set values) plus ContinuationClassifier's
    heuristic score on the tail of the article text.
    """
    text = str(article.get("article_text", "") or "")
    tail = "\n".join(text.splitlines()[-3:]) if text else ""

    classifier_score = _continuation_classifier.predict(tail)

    next_page = continuation_marker_target_page(article)

    marker = None
    if next_page is not None:
        marker_match = re.search(
            r"[^\n]{0,60}(?:page|pg\.?|p\.?)\s*[-:]?\s*\d{1,4}\b",
            text,
            flags=re.IGNORECASE,
        )
        if marker_match:
            marker = marker_match.group(0).strip()

    is_continued = bool(next_page is not None or classifier_score >= 0.5)

    return {
        "is_continued": is_continued,
        "marker": marker,
        "next_page": next_page,
    }
