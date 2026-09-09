from pipeline.languages.base import LanguagePipeline
from pipeline.languages.odia.grouping_prompt import (
    ODIA_GROUPING_PROMPT,
)
from pipeline.languages.odia.extraction_prompt import (
    ODIA_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_odia_language.
    RapidOCR has no Odia recognition model, so this routes to
    TesseractOCREngine(lang="ori").
    """

    lang = (language or "").strip().lower()

    return (
        lang in ("or", "ori", "odia", "oriya")
        or "odia" in lang
        or "oriya" in lang
    )


ODIA = LanguagePipeline(
    code="odia",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="ori"),
    ocr_engine_label="tesseract (odia)",
    llm_provider="openai",
    grouping_prompt=ODIA_GROUPING_PROMPT,
    extraction_prompt_template=ODIA_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
    # The shared 0.20 default drops genuine Odia headlines/columns.
    # Confirmed on a real Odia (Sambad, doc_000177 page 1) page: the
    # lead headline ("ସତର୍କ କରାଇଲା ଚୀନ୍") was detected at YOLO
    # confidence 0.15 and discarded at the default threshold --
    # across all 4 pages of that document, detections dropped ~30-40%
    # at 0.20 vs. 0.10. See LanguagePipeline.layout_confidence.
    layout_confidence=0.10,
)
