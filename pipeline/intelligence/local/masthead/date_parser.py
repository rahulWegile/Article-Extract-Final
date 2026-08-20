from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_MONTH_DAY_YEAR = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
)

_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"
)

_NUMERIC_DMY = re.compile(
    r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"
)


def _safe_date(year, month, day) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def parse_date(text: str) -> str | None:
    if not text:
        return None

    candidates: list[str] = []

    for month_name, day, year in _MONTH_DAY_YEAR.findall(text):
        month = _MONTHS.get(month_name.lower())
        if month:
            parsed = _safe_date(year, month, day)
            if parsed:
                candidates.append(parsed)

    for day, month_name, year in _DAY_MONTH_YEAR.findall(text):
        month = _MONTHS.get(month_name.lower())
        if month:
            parsed = _safe_date(year, month, day)
            if parsed:
                candidates.append(parsed)

    for day, month, year in _NUMERIC_DMY.findall(text):
        parsed = _safe_date(year, month, day)
        if parsed:
            candidates.append(parsed)

    unique = sorted(set(candidates))

    if len(unique) != 1:
        return None

    return unique[0]
