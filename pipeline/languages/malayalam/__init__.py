from pipeline.languages.base import LanguagePipeline
from pipeline.languages.malayalam.grouping_prompt import (
    MALAYALAM_GROUPING_PROMPT,
)
from pipeline.languages.malayalam.extraction_prompt import (
    MALAYALAM_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old
    PipelineService._is_malayalam_language. Neither RapidOCR nor
    EasyOCR ship a Malayalam recognition model, so this routes to
    TesseractOCREngine(lang="mal").
    """

    lang = (language or "").strip().lower()

    return lang in ("ml", "mal", "malayalam") or "malayalam" in lang


MALAYALAM = LanguagePipeline(
    code="malayalam",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="mal"),
    ocr_engine_label="tesseract (malayalam)",
    llm_provider="openai",
    grouping_prompt=MALAYALAM_GROUPING_PROMPT,
    extraction_prompt_template=MALAYALAM_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,
)
