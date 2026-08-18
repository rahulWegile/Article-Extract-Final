import re


class LanguageDetector:
    """
    Lightweight language detector for OCR text.

    This is intentionally simple.
    Later versions can replace it with:
        - fastText
        - langdetect
        - CLD3
        - Gemini
    without changing the pipeline.
    """

    DEVANAGARI = re.compile(r"[\u0900-\u097F]")
    LATIN = re.compile(r"[A-Za-z]")

    def detect(self, text: str) -> str:

        if not text:
            return "unknown"

        devanagari = len(self.DEVANAGARI.findall(text))
        latin = len(self.LATIN.findall(text))

        if devanagari > latin:
            return "hi"

        if latin > 0:
            return "en"

        return "unknown"