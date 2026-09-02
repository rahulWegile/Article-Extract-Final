from pipeline.languages.base import LanguagePipeline
from pipeline.languages.tamil.grouping_prompt import (
    TAMIL_GROUPING_PROMPT,
)
from pipeline.languages.tamil.extraction_prompt import (
    TAMIL_EXTRACTION_PROMPT,
)
from pipeline.ocr.rapidocr_engine import RapidOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """Copied verbatim from the old PipelineService._is_tamil_language."""

    lang = (language or "").strip().lower()

    return lang in ("ta", "tam", "tamil") or "tamil" in lang


TAMIL = LanguagePipeline(
    code="tamil",
    matches=matches,
    ocr_engine_factory=lambda: RapidOCREngine(lang="ta"),
    ocr_engine_label="rapidocr (tamil)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati/
    # Assamese/Bengali/Kannada/Urdu use.
    #
    # TEMPORARY: unlike those languages, a direct spot check of real
    # Tamil articles did NOT show the "generic paraphrase" failure
    # signature -- a 1611-char article came back richly detailed and
    # specific, and short articles read like genuine news briefs, not
    # boilerplate. Tamil's lower average extracted-length ratio (0.48x
    # OCR text) looks like ordinary short/long editorial variance
    # rather than the tight-clustering tell confirmed on Urdu/Punjabi/
    # Assamese. Switched to Gemini anyway per explicit instruction, not
    # because this was independently confirmed broken -- verify against
    # a real document (ideally compare against the previous OpenAI
    # output for the same document) before treating this as settled.
    # To revert: change extractor_class back to OpenAIArticleExtractor
    # (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=TAMIL_GROUPING_PROMPT,
    extraction_prompt_template=TAMIL_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
