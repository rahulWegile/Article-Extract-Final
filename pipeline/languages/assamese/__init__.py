from pipeline.languages.base import LanguagePipeline
from pipeline.languages.assamese.grouping_prompt import (
    ASSAMESE_GROUPING_PROMPT,
)
from pipeline.languages.assamese.extraction_prompt import (
    ASSAMESE_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old
    PipelineService._is_assamese_language. Assamese uses its own
    script variant (distinct from Bengali in a handful of
    letterforms), and Tesseract ships a dedicated Assamese model
    (tessdata_best "asm"), so this uses that rather than reusing the
    Bengali model for a script it was not trained on.
    """

    lang = (language or "").strip().lower()

    return lang in ("as", "asm", "assamese") or "assamese" in lang


ASSAMESE = LanguagePipeline(
    code="assamese",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="asm"),
    ocr_engine_label="tesseract (assamese)",
    llm_provider="openai",
    grouping_prompt=ASSAMESE_GROUPING_PROMPT,
    extraction_prompt_template=ASSAMESE_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,
    extraction_pages_per_batch=2,
    extraction_max_articles_per_batch=25,
)
