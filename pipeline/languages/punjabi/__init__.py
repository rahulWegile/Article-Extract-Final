from pipeline.languages.base import LanguagePipeline
from pipeline.languages.punjabi.grouping_prompt import (
    PUNJABI_GROUPING_PROMPT,
)
from pipeline.languages.punjabi.extraction_prompt import (
    PUNJABI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_punjabi_language.
    Newspaper Punjabi is Gurmukhi script, which neither RapidOCR nor
    EasyOCR cover, so this routes to TesseractOCREngine(lang="pan").
    """

    lang = (language or "").strip().lower()

    return lang in ("pa", "pan", "punjabi") or "punjabi" in lang


PUNJABI = LanguagePipeline(
    code="punjabi",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="pan"),
    ocr_engine_label="tesseract (punjabi)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi already uses (also
    # OpenAI for grouping, GeminiArticleExtractor for extraction).
    #
    # TEMPORARY: confirmed on a real document (THE INQUILAB Jalandhar
    # Punjabi Jagran page) that OpenAIArticleExtractor's pinned model
    # (OPENAI_ARTICLE_MODEL=gpt-5.6-luna) does not actually transcribe
    # Gurmukhi crops -- article_text and summary came back at the same
    # level of generality (a paraphrase, not a transcription), and
    # extracted length clustered tightly (~162 chars, std=52) regardless
    # of how much text the crop actually contained. To revert: change
    # this back to OpenAIArticleExtractor (and restore the
    # `from pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=PUNJABI_GROUPING_PROMPT,
    extraction_prompt_template=PUNJABI_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
