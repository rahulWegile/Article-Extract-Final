from pipeline.languages.base import LanguagePipeline
from pipeline.languages.hindi.grouping_prompt import HINDI_GROUPING_PROMPT
from pipeline.languages.hindi.extraction_prompt import (
    HINDI_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)


def matches(language: str) -> bool:
    """
    "Hindi", "hi", "hin", and anything containing "hindi" (e.g.
    "Hindi (Devanagari)") all count.
    """

    lang = (language or "").strip().lower()

    return lang in ("hi", "hin", "hindi") or "hindi" in lang


HINDI = LanguagePipeline(
    code="hindi",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="hin"),
    ocr_engine_label="tesseract (hindi)",
    llm_provider="openai",
    grouping_prompt=HINDI_GROUPING_PROMPT,
    extraction_prompt_template=HINDI_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,
    use_contested_block_arbitration=False,

    # 2-page extraction batching: shares the extraction prompt's fixed
    # overhead across 2 pages instead of paying it on every single
    # page. Boundary detection is unaffected -- it stays one page,
    # one call, regardless of this setting.
    extraction_pages_per_batch=2,

    # Safety cap: even within a 2-page group, never send more than 25
    # article crops in one extraction call -- a page with an unusually
    # high article count (20+ seen on real Hindi pages) paired with a
    # second page could otherwise produce an oversized request. A page
    # that would push the running total past this cap is held back and
    # starts the NEXT batch instead of being force-fit into this one.
    extraction_max_articles_per_batch=25,

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
