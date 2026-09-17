from pipeline.languages.base import LanguagePipeline
from pipeline.languages.gujarati.grouping_prompt import (
    GUJARATI_GROUPING_PROMPT,
)
from pipeline.languages.gujarati.extraction_prompt import (
    GUJARATI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
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
    # Both grouping and article-level text extraction run on OpenAI,
    # matching Hindi/Punjabi/Odia's pipelines. extraction_prompt.py
    # carries the same verbatim-transcription discipline ported from
    # Hindi/Odia (anti-hallucination, multi-column completeness,
    # script preservation, bottom-margin continuation-marker scan).
    llm_provider="openai",
    grouping_prompt=GUJARATI_GROUPING_PROMPT,
    extraction_prompt_template=GUJARATI_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,

    # Gujarati only: 1 page per batch, max 8 article crops per batch --
    # keeps each OpenAI call small so a page with many crops doesn't get
    # bundled into a slow, oversized request. Matches Punjabi/Odia's
    # own settings (see their own __init__.py).
    extraction_pages_per_batch=1,
    extraction_max_articles_per_batch=8,

    # Boundary post-processing safeguards kept enabled (these are the
    # shared defaults already -- kept explicit here so this choice
    # isn't mistaken for an oversight).
    use_article_splitter=True,
    use_boundary_decomposition=True,
    use_orphan_block_reassignment=True,

    # Gujarati broadsheet headlines commonly span across columns, so
    # detaching a wide top block from the narrow columns below it would
    # incorrectly split the masthead/headline away from its own story.
    use_wide_top_banner_detachment=False,
)
