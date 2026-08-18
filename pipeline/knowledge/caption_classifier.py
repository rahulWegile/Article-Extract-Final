import re


class CaptionClassifier:
    """
    Estimate how likely an OCR block is to be an image caption.

    Returns:
        float (0.0 - 1.0)
    """

    PREFIXES = {
        "photo",
        "image",
        "caption",
        "source",
        "credit",
        "courtesy",
        "file",
    }

    AGENCIES = {
        "ani",
        "pti",
        "ap",
        "reuters",
        "afp",
        "getty",
        "ians",
    }

    MAX_WORDS = 25

    def predict(self, text: str) -> float:

        if not text:
            return 0.0

        text = text.strip()

        words = text.split()

        score = 0.0

        lower = text.lower()

        # ---------------------------------
        # Very long text is unlikely
        # ---------------------------------

        if len(words) > self.MAX_WORDS:
            return 0.0

        # ---------------------------------
        # Caption prefixes
        # ---------------------------------

        for prefix in self.PREFIXES:

            if lower.startswith(prefix):
                score += 0.45
                break

        # ---------------------------------
        # News agencies
        # ---------------------------------

        for agency in self.AGENCIES:

            if agency in lower:
                score += 0.35
                break

        # ---------------------------------
        # Short descriptive text
        # ---------------------------------

        if 4 <= len(words) <= 18:
            score += 0.15

        # ---------------------------------
        # Ends with period
        # ---------------------------------

        if text.endswith("."):
            score += 0.05

        return min(score, 1.0)