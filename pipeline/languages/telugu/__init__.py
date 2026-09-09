from pipeline.languages.base import LanguagePipeline
from pipeline.languages.telugu.grouping_prompt import (
    TELUGU_GROUPING_PROMPT,
)
from pipeline.languages.telugu.extraction_prompt import (
    TELUGU_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_telugu_language.
    RapidOCR ships a real Telugu recognition model, but a direct
    side-by-side test found its shared text detector returns a
    silent empty result for some real Telugu conjunct clusters
    (not a threshold/padding/upscaling issue). Tesseract read the
    identical crop correctly, so Telugu is routed there instead.
    """

    lang = (language or "").strip().lower()

    return lang in ("te", "tel", "telugu") or "telugu" in lang


TELUGU = LanguagePipeline(
    code="telugu",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="tel"),
    ocr_engine_label="tesseract (telugu)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati/
    # Assamese/Bengali/Kannada use: OpenAIArticleExtractor's pinned
    # model (OPENAI_ARTICLE_MODEL=gpt-5.6-luna) is confirmed failing on
    # multiple non-Latin, non-Devanagari scripts, and Telugu shares
    # that same architecture (Tesseract OCR + OpenAI extraction). To
    # revert: change extractor_class back to OpenAIArticleExtractor
    # (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=TELUGU_GROUPING_PROMPT,
    extraction_prompt_template=TELUGU_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
