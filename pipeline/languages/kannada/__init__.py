from pipeline.languages.base import LanguagePipeline
from pipeline.languages.kannada.grouping_prompt import (
    KANNADA_GROUPING_PROMPT,
)
from pipeline.languages.kannada.extraction_prompt import (
    KANNADA_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_kannada_language.
    RapidOCR ships a Kannada recognition model, but a direct
    side-by-side test found its reading badly garbled (dropped/wrong
    conjuncts) while Tesseract read the identical crop correctly
    except one character. Routed to Tesseract for that measured
    reason.
    """

    lang = (language or "").strip().lower()

    return lang in ("kn", "kan", "kannada") or "kannada" in lang


KANNADA = LanguagePipeline(
    code="kannada",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="kan"),
    ocr_engine_label="tesseract (kannada)",
    llm_provider="openai",
    grouping_prompt=KANNADA_GROUPING_PROMPT,
    extraction_prompt_template=KANNADA_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,

    # Matches Odia's batching: share the extraction prompt's fixed
    # overhead across 1 page at a time, capped at 8 article crops per
    # call so gpt-5.6-luna has enough completion-token headroom to
    # transcribe every column verbatim instead of compressing/
    # truncating articles (see the MULTI-COLUMN COMPLETENESS MANDATE
    # in extraction_prompt.py).
    extraction_pages_per_batch=1,
    extraction_max_articles_per_batch=8,

    use_article_splitter=True,
    use_boundary_decomposition=True,
    use_orphan_block_reassignment=True,

    # Kannada headlines/banners can span across columns; disable
    # wide-top-banner detachment so a genuine multi-column headline
    # isn't cut away from its own body.
    use_wide_top_banner_detachment=False,
)
