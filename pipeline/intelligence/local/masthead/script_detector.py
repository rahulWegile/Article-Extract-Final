from __future__ import annotations

import re

_SCRIPT_RANGES = {
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "tamil": re.compile(r"[஀-௿]"),
    "telugu": re.compile(r"[ఀ-౿]"),
    "gujarati": re.compile(r"[઀-૿]"),
    "gurmukhi": re.compile(r"[਀-੿]"),
    "latin": re.compile(r"[A-Za-z]"),
}

_UNAMBIGUOUS_SCRIPT_LANGUAGE = {
    "tamil": "Tamil",
    "telugu": "Telugu",
    "gujarati": "Gujarati",
    "gurmukhi": "Punjabi",
    "latin": "English",
}


def dominant_script(text: str) -> str | None:
    if not text:
        return None

    counts = {
        name: len(pattern.findall(text))
        for name, pattern in _SCRIPT_RANGES.items()
    }

    script, count = max(counts.items(), key=lambda item: item[1])

    return script if count > 0 else None


def corroborates(text: str, expected_language: str) -> bool:
    script = dominant_script(text)

    if script is None:
        return True

    unambiguous_language = _UNAMBIGUOUS_SCRIPT_LANGUAGE.get(script)

    if unambiguous_language is None:
        return True

    return unambiguous_language == expected_language
