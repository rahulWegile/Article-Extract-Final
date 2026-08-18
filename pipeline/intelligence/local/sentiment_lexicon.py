"""
Lexicon-based sentiment heuristic.

This is explicitly the lowest-value/highest-noise semantic field per
the migration plan. It defaults to "neutral" whenever the signal is
weak rather than emit a confident-looking wrong label.
"""

from __future__ import annotations

import re

_POSITIVE_WORDS = {
    "win", "wins", "won", "winning", "victory", "success", "successful",
    "growth", "grows", "improve", "improved", "improvement", "praise",
    "praised", "boost", "boosted", "record", "achievement", "celebrate",
    "celebration", "welcome", "welcomed", "progress", "gain", "gains",
    "positive", "hope", "hopeful", "benefit", "benefits", "recovery",
    "award", "awarded", "honour", "honored", "honoured", "surge", "rally",
    "breakthrough", "historic", "milestone",
}

_NEGATIVE_WORDS = {
    "death", "dead", "died", "killed", "kill", "murder", "attack",
    "attacked", "crash", "accident", "disaster", "crisis", "protest",
    "protests", "arrest", "arrested", "fraud", "scam", "corruption",
    "violence", "violent", "riot", "riots", "flood", "floods", "fire",
    "collapse", "collapsed", "loss", "losses", "decline", "declined",
    "fail", "failed", "failure", "concern", "concerns", "warning",
    "warned", "threat", "threatened", "controversy", "scandal", "injured",
    "injury", "clash", "clashes", "outbreak", "shortage",
}

_MIN_HITS = 2
_MIN_MARGIN_RATIO = 1.5


def classify_sentiment(text: str) -> str:

    if not text:
        return "neutral"

    words = re.findall(r"[a-z']+", text.lower())

    if not words:
        return "neutral"

    positive_hits = sum(1 for word in words if word in _POSITIVE_WORDS)
    negative_hits = sum(1 for word in words if word in _NEGATIVE_WORDS)

    if positive_hits < _MIN_HITS and negative_hits < _MIN_HITS:
        return "neutral"

    if positive_hits >= negative_hits * _MIN_MARGIN_RATIO and positive_hits >= _MIN_HITS:
        return "positive"

    if negative_hits >= positive_hits * _MIN_MARGIN_RATIO and negative_hits >= _MIN_HITS:
        return "negative"

    return "neutral"
