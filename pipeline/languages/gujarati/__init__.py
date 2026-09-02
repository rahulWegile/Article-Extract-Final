from pipeline.languages.base import LanguagePipeline
from pipeline.languages.gujarati.grouping_prompt import (
    GUJARATI_GROUPING_PROMPT,
)
from pipeline.languages.gujarati.extraction_prompt import (
    GUJARATI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_gujarati_language.
    Gujarati is not Devanagari script, and neither RapidOCR nor
    EasyOCR ship a Gujarati recognition model, so it is routed to
    TesseractOCREngine below instead of RapidOCREngine.
    """

    lang = (language or "").strip().lower()

    return lang in ("gu", "guj", "gujarati") or "gujarati" in lang


GUJARATI = LanguagePipeline(
    code="gujarati",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="guj"),
    ocr_engine_label="tesseract (gujarati)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi already use.
    #
    # TEMPORARY: cross-language analysis (extracted-text length vs.
    # each article's own OCR text) showed OpenAIArticleExtractor's
    # pinned model (OPENAI_ARTICLE_MODEL=gpt-5.6-luna) under-extracting
    # Gujarati crops (extracted length averaged ~0.65x the underlying
    # OCR text, vs. ~1x+ for Hindi/Marathi once routed to Gemini). Not
    # yet visually confirmed on a real Gujarati crop the way Urdu and
    # Punjabi were -- do that check on the next processed document. To
    # revert: change extractor_class back to OpenAIArticleExtractor
    # (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=GUJARATI_GROUPING_PROMPT,
    extraction_prompt_template=GUJARATI_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
