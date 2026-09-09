from pipeline.languages.base import LanguagePipeline
from pipeline.languages.urdu.grouping_prompt import URDU_GROUPING_PROMPT
from pipeline.languages.urdu.extraction_prompt import (
    URDU_EXTRACTION_PROMPT,
)
from pipeline.ocr.utrnet_engine import UTRNetOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    "Urdu", "ur", "urd", and anything containing "urdu" all count --
    same substring-match convention every other language's matches()
    uses.
    """

    lang = (language or "").strip().lower()

    return lang in ("ur", "urd", "urdu") or "urdu" in lang


# Urdu (Perso-Arabic Nastaliq) is the one language routed through
# UTRNet -- neither RapidOCR, EasyOCR, nor Tesseract has any
# meaningful recognition model for this script. Gemini handles both
# grouping and extraction, same reasoning Hindi's own comment gives:
# gpt-5.6-luna cannot reliably read this script off a real crop.
# is_rtl=True is the piece every other language leaves off -- see
# pipeline/languages/base.py's is_rtl field for exactly what it
# changes (reading order, column indexing) and why it was built with
# Urdu specifically in mind.
URDU = LanguagePipeline(
    code="urdu",
    matches=matches,
    ocr_engine_factory=lambda: UTRNetOCREngine(lang="ur"),
    ocr_engine_label="utrnet (urdu)",
    llm_provider="openai",
    grouping_prompt=URDU_GROUPING_PROMPT,
    extraction_prompt_template=URDU_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
    is_rtl=True,
    # See LanguagePipeline.layout_confidence -- confirmed on a real
    # Urdu (THE INQUILAB) document that the shared 0.20 default drops
    # dozens of real articles on a dense Nastaliq page.
    layout_confidence=0.08,
)
