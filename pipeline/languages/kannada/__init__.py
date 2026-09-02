from pipeline.languages.base import LanguagePipeline
from pipeline.languages.kannada.grouping_prompt import (
    KANNADA_GROUPING_PROMPT,
)
from pipeline.languages.kannada.extraction_prompt import (
    KANNADA_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_kannada_language.
    RapidOCR ships a Kannada recognition model, but a direct
    side-by-side test found its reading badly garbled (dropped/wrong
    conjuncts) while Tesseract read the identical crop correctly
    except one character. Routed to Tesseract for that measured
    reason.
    """

    lang = (language or "").strip().lower()

    return lang in ("kn", "kan", "kannada") or "kannada" in lang


KANNADA = LanguagePipeline(
    code="kannada",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="kan"),
    ocr_engine_label="tesseract (kannada)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati/
    # Assamese/Bengali use.
    #
    # TEMPORARY: no completed Kannada document existed yet to measure
    # (unlike the languages above, where extracted-text length vs. OCR
    # text length was directly compared) -- this is a preemptive match
    # to the same fix, on the same reasoning: OpenAIArticleExtractor's
    # pinned model (OPENAI_ARTICLE_MODEL=gpt-5.6-luna) is confirmed
    # failing on multiple other non-Latin, non-Devanagari scripts, and
    # Kannada shares that same architecture (Tesseract OCR + OpenAI
    # extraction). Verify against a real completed Kannada document
    # the same way Urdu/Punjabi crops were checked before trusting
    # this. To revert: change extractor_class back to
    # OpenAIArticleExtractor (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=KANNADA_GROUPING_PROMPT,
    extraction_prompt_template=KANNADA_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
