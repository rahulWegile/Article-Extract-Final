from pipeline.languages.base import LanguagePipeline
from pipeline.languages.marathi.grouping_prompt import (
    MARATHI_GROUPING_PROMPT,
)
from pipeline.languages.marathi.extraction_prompt import (
    MARATHI_EXTRACTION_PROMPT,
)
from pipeline.ocr.rapidocr_engine import RapidOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old
    PipelineService._is_marathi_language. Marathi shares the
    Devanagari script with Hindi, so it reuses the Hindi OCR model
    family (RapidOCREngine(lang="mr") builds the same Devanagari
    recognizer), but keeps its own match check and its own OCR
    engine instance so it can never be confused with an actual Hindi
    document downstream -- Marathi still routes to OpenAI/generic
    grouping below, unlike Hindi.
    """

    lang = (language or "").strip().lower()

    return lang in ("mr", "mar", "marathi") or "marathi" in lang


MARATHI = LanguagePipeline(
    code="marathi",
    matches=matches,
    ocr_engine_factory=lambda: RapidOCREngine(lang="mr"),
    ocr_engine_label="devanagari (marathi)",
    llm_provider="openai",
    grouping_prompt=MARATHI_GROUPING_PROMPT,
    extraction_prompt_template=MARATHI_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
