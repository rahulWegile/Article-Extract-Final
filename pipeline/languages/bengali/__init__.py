from pipeline.languages.base import LanguagePipeline
from pipeline.languages.bengali.grouping_prompt import (
    BENGALI_GROUPING_PROMPT,
)
from pipeline.languages.bengali.extraction_prompt import (
    BENGALI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
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
    # Both grouping and article-level text extraction run on OpenAI,
    # matching the Hindi/Odia/Punjabi pattern. Switched back from the
    # temporary GeminiArticleExtractor (see git history) now that this
    # prompt carries the same verbatim-transcription discipline that
    # fixed the equivalent Gurmukhi/Devanagari under-extraction issue.
    llm_provider="openai",
    grouping_prompt=BENGALI_GROUPING_PROMPT,
    extraction_prompt_template=BENGALI_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,
    # Bengali only: 1 page per batch, max 8 article crops per batch --
    # keeps each OpenAI call small, matching Punjabi (see
    # pipeline/languages/punjabi/__init__.py).
    extraction_pages_per_batch=1,
    extraction_max_articles_per_batch=8,
    # Boundary post-processing safeguards enabled (these are already
    # the shared defaults -- listed explicitly here so Bengali's
    # intent doesn't depend on base.py's defaults staying unchanged).
    use_article_splitter=True,
    use_boundary_decomposition=True,
    use_orphan_block_reassignment=True,
    # Disabled (overriding the shared True default): Bengali mastheads
    # commonly place a wide top banner over narrower columns that
    # legitimately belong with it, so it should not be auto-detached.
    use_wide_top_banner_detachment=False,
)
