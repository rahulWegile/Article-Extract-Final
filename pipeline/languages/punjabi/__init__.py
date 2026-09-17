from pipeline.languages.base import LanguagePipeline
from pipeline.languages.punjabi.grouping_prompt import (
    PUNJABI_GROUPING_PROMPT,
)
from pipeline.languages.punjabi.extraction_prompt import (
    PUNJABI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
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
    # Both grouping and article-level text extraction run on OpenAI,
    # matching Hindi's pipeline (pipeline/languages/hindi/__init__.py).
    #
    # NOTE: an earlier attempt at OpenAIArticleExtractor for Punjabi was
    # reverted to GeminiArticleExtractor after a real document (THE
    # INQUILAB Jalandhar Punjabi Jagran page) showed the pinned model
    # paraphrasing Gurmukhi crops instead of transcribing them. Hindi's
    # extraction_prompt was since strengthened with an explicit
    # exhaustive-verbatim-transcription section and later confirmed
    # (2026-09-16) to transcribe Devanagari with full fidelity; this
    # switch ports that same reading-order/verbatim-transcription
    # discipline to Punjabi's prompt. Verify on a real Gurmukhi page
    # before relying on it -- if it regresses the same way, revert to
    # GeminiArticleExtractor (restore the
    # `from pipeline.intelligence.gemini_article_extractor import
    # GeminiArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=PUNJABI_GROUPING_PROMPT,
    extraction_prompt_template=PUNJABI_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,
    # Punjabi only: 1 page per batch, max 8 article crops per batch --
    # keeps each OpenAI call small so a page with many crops (or a
    # dense, small-font page needing more attention per crop) doesn't
    # get bundled into a slow, oversized request. Hindi/English/Urdu
    # keep their own settings unchanged (see their own __init__.py).
    extraction_pages_per_batch=1,
    extraction_max_articles_per_batch=8,

    # Disable heuristic splitters and repairs to preserve OpenAI grouping decisions
    use_orphan_block_reassignment=False,
    use_orphan_title_root_repair=False,
    use_unclaimed_kicker_recovery=False,
    use_unclaimed_image_recovery=False,
    use_unclaimed_footprint_recovery=False,
    use_article_splitter=False,
    use_wide_top_banner_detachment=False,
    use_dropped_article_recovery=False,
    use_boundary_decomposition=False,
)
