import re


class ContinuationClassifier:
    """
    Estimate how likely an OCR block is a
    continuation / jump-line.

    Returns:
        float (0.0 - 1.0)
    """

    PATTERNS = [

        r"continued\s+on\s+page",

        r"continued\s+from\s+page",

        r"see\s+page",

        r"turn\s+to\s+page",

        r"read\s+more",

        r"jump\s+to\s+page",

        r"continued",

        r"see\s+story",

    ]

    PAGE_PATTERN = r"page\s+\d+"

    def predict(self, text: str) -> float:

        if not text:
            return 0.0

        lower = text.lower()

        score = 0.0

        # -----------------------------
        # Pattern matching
        # -----------------------------

        for pattern in self.PATTERNS:

            if re.search(pattern, lower):

                score += 0.60

                break

        # -----------------------------
        # Page number
        # -----------------------------

        if re.search(self.PAGE_PATTERN, lower):

            score += 0.30

        # -----------------------------
        # Very short block
        # -----------------------------

        words = lower.split()

        if len(words) <= 8:

            score += 0.10

        return min(score, 1.0)