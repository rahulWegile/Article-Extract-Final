from pipeline.languages.base import LanguagePipeline
from pipeline.languages.tamil.grouping_prompt import (
    TAMIL_GROUPING_PROMPT,
)
from pipeline.languages.tamil.extraction_prompt import (
    TAMIL_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)


def matches(language: str) -> bool:
    """Copied verbatim from the old PipelineService._is_tamil_language."""

    lang = (language or "").strip().lower()

    return lang in ("ta", "tam", "tamil") or "tamil" in lang


TAMIL = LanguagePipeline(
    code="tamil",
    matches=matches,
    # RapidOCR's ta_PP-OCRv5_rec_mobile.onnx was measured returning
    # single-character gibberish ("L", "Aup") on real vernacular
    # broadsheet fonts (The Hindu Tamil Thisai). Routed to Tesseract
    # instead, same precedent as Telugu/Kannada (see those languages'
    # __init__.py) -- requires models/tessdata/tam.traineddata (add
    # "tam" to the Dockerfile tessdata download loop; not bundled
    # locally since models/tessdata is gitignored).
    ocr_engine_factory=lambda: TesseractOCREngine(lang="tam"),
    ocr_engine_label="tesseract (tamil)",
    # Grouping/boundary detection AND article-level text extraction both
    # stay on OpenAI, matching Hindi/Odia/Assamese. Moved off Gemini
    # extraction to prevent the same failure modes fixed for those
    # languages: photo/caption hallucination, dropped trailing columns,
    # and missed continuation markers.
    llm_provider="openai",
    grouping_prompt=TAMIL_GROUPING_PROMPT,
    extraction_prompt_template=TAMIL_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,

    # 2-page extraction batching, matching Hindi/Odia's move to OpenAI:
    # shares the extraction prompt's fixed overhead across 2 pages
    # instead of paying it on every single page. Boundary detection is
    # unaffected -- it stays one page, one call, regardless of this
    # setting.
    extraction_pages_per_batch=2,

    # Safety cap: even within a 2-page group, never send more than 25
    # article crops in one extraction call. A page that would push the
    # running total past this cap is held back and starts the NEXT
    # batch instead of being force-fit into this one.
    extraction_max_articles_per_batch=25,

    # Dense Tamil script (pulli, kombu vowel marks) and compact
    # broadsheet fonts push DocLayout-YOLO's own confidence below the
    # shared 0.20 default, the same failure mode measured on Urdu
    # (see LanguagePipeline.layout_confidence's docstring) -- dropping
    # real text blocks before OCR/grouping ever see them.
    layout_confidence=0.10,

    # Align with English: these repair passes were found to mangle
    # boundaries (boundary_decomposer slicing overlapping column
    # rectangles, footprint union merging unrelated stories across
    # columns) rather than fix them. See
    # pipeline/languages/english/__init__.py for the same disablement.
    use_orphan_block_reassignment=False,
    use_orphan_title_root_repair=False,
    use_unclaimed_kicker_recovery=False,
    use_unclaimed_image_recovery=False,
    use_unclaimed_footprint_recovery=False,

    # Re-enabled: real Tamil pages showed the LLM grouping pass
    # dropping an article's own title block (leaving it recoverable
    # only via use_dropped_article_recovery) and merging two separate
    # stories that shared an L-shaped photo layout (only separable via
    # use_article_splitter/use_boundary_decomposition slicing the
    # merged footprint back into per-article sub-rectangles).
    use_article_splitter=True,
    use_dropped_article_recovery=True,
    use_boundary_decomposition=True,
)
