import re


class SentenceFeatures:
    """
    Extract sentence-level features from OCR text.

    These features are reused by:
    - EdgeScorer
    - OCR Continuity
    - Gemini
    - Cross-page linking
    """

    def extract(self, text: str) -> dict:

        if not text:
            return {
                "first_sentence": "",
                "last_sentence": "",
                "first_word": "",
                "last_word": "",
                "starts_with_lowercase": False,
                "ends_with_punctuation": False,
            }

        text = text.strip()

        # ---------------------------------
        # Split into sentences
        # ---------------------------------

        # Devanagari sentences end with "।" (danda) / "॥", not ".",
        # so those must split sentences too.
        sentences = re.split(
            r"(?<=[.!?।॥])\s+",
            text,
        )

        sentences = [
            s.strip()
            for s in sentences
            if s.strip()
        ]

        first_sentence = (
            sentences[0]
            if sentences
            else text
        )

        last_sentence = (
            sentences[-1]
            if sentences
            else text
        )

        # ---------------------------------
        # Words
        # ---------------------------------

        # Latin and Devanagari word runs, so `first_word`/`last_word`
        # aren't always empty on Hindi-only text.
        words = re.findall(
            r"[A-Za-zऀ-ॿ]+",
            text,
        )

        first_word = (
            words[0]
            if words
            else ""
        )

        last_word = (
            words[-1]
            if words
            else ""
        )

        # ---------------------------------
        # Simple continuity features
        # ---------------------------------

        starts_with_lowercase = (
            len(text) > 0
            and text[0].islower()
        )

        ends_with_punctuation = (
            len(text) > 0
            and text[-1] in ".!?:;।॥"
        )

        return {

            "first_sentence": first_sentence,

            "last_sentence": last_sentence,

            "first_word": first_word,

            "last_word": last_word,

            "starts_with_lowercase": starts_with_lowercase,

            "ends_with_punctuation": ends_with_punctuation,

        }