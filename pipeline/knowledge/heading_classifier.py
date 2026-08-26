import re


class HeadingClassifier:
    """
    Estimate how likely an OCR block is to be a headline.

    Returns a probability between 0.0 and 1.0.
    """

    MAX_HEADING_WORDS = 15

    def predict(self, text: str) -> float:

        if not text:
            return 0.0

        text = text.strip()
        words = text.split()

        score = 0.0

        # ---------------------------------
        # Positive signals
        # ---------------------------------

        # Short heading
        if len(words) <= 10:
            score += 0.35

        # Mostly Title Case.
        # No-op for scripts without letter case (e.g. Devanagari) --
        # those blocks simply don't get this signal either way,
        # rather than being penalized for lacking it.
        title_case = sum(
            1
            for w in words
            if len(w) > 1 and w[0].isupper()
        )

        if words:
            ratio = title_case / len(words)

            if ratio > 0.70:
                score += 0.35

        # Headings usually don't end with punctuation.
        # Devanagari sentences end with "।" (danda) or "॥", not ".",
        # so those must count as sentence-ending punctuation too --
        # otherwise every Hindi paragraph falsely gets this signal.
        if text[-1] not in ".!?;।॥":
            score += 0.20

        # Few commas
        if text.count(",") <= 1:
            score += 0.10

        # ---------------------------------
        # Negative signals
        # ---------------------------------

        # Body paragraphs are usually longer
        if len(words) > 15:
            score -= 0.40

        # Paragraphs usually end with a sentence terminator
        if text.endswith((".", "।", "॥")):
            score -= 0.30

        # Long OCR text is unlikely to be a heading
        if len(text) > 100:
            score -= 0.20

        # Multiple sentences → body text
        if (text.count(".") + text.count("।")) > 1:
            score -= 0.20

        # Many commas → paragraph
        if text.count(",") > 2:
            score -= 0.10

        return max(0.0, min(score, 1.0))