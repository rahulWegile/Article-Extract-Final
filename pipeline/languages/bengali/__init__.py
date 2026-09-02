from pipeline.languages.base import LanguagePipeline
from pipeline.languages.bengali.grouping_prompt import (
    BENGALI_GROUPING_PROMPT,
)
from pipeline.languages.bengali.extraction_prompt import (
    BENGALI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_bengali_language.
    RapidOCR has no Bengali recognition model, so this routes to
    TesseractOCREngine(lang="ben").
    """

    lang = (language or "").strip().lower()

    return lang in ("bn", "ben", "bengali") or "bengali" in lang


BENGALI = LanguagePipeline(
    code="bengali",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="ben"),
    ocr_engine_label="tesseract (bengali)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati/
    # Assamese use.
    #
    # TEMPORARY: cross-language analysis (extracted-text length vs.
    # each article's own OCR text) showed OpenAIArticleExtractor's
    # pinned model (OPENAI_ARTICLE_MODEL=gpt-5.6-luna) under-extracting
    # Bengali crops (extracted length averaged ~0.39x the underlying
    # OCR text). Not yet visually confirmed on a real Bengali crop the
    # way Urdu and Punjabi were -- do that check on the next processed
    # document. To revert: change extractor_class back to
    # OpenAIArticleExtractor (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=BENGALI_GROUPING_PROMPT,
    extraction_prompt_template=BENGALI_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
