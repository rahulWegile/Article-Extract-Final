from pipeline.languages.base import LanguagePipeline
from pipeline.languages.tamil.grouping_prompt import (
    TAMIL_GROUPING_PROMPT,
)
from pipeline.languages.tamil.extraction_prompt import (
    TAMIL_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
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
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati/
    # Assamese/Bengali/Kannada/Urdu use.
    #
    # TEMPORARY: unlike those languages, a direct spot check of real
    # Tamil articles did NOT show the "generic paraphrase" failure
    # signature -- a 1611-char article came back richly detailed and
    # specific, and short articles read like genuine news briefs, not
    # boilerplate. Tamil's lower average extracted-length ratio (0.48x
    # OCR text) looks like ordinary short/long editorial variance
    # rather than the tight-clustering tell confirmed on Urdu/Punjabi/
    # Assamese. Switched to Gemini anyway per explicit instruction, not
    # because this was independently confirmed broken -- verify against
    # a real document (ideally compare against the previous OpenAI
    # output for the same document) before treating this as settled.
    # To revert: change extractor_class back to OpenAIArticleExtractor
    # (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=TAMIL_GROUPING_PROMPT,
    extraction_prompt_template=TAMIL_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,

    # Dense Tamil script (pulli, kombu vowel marks) and compact
    # broadsheet fonts push DocLayout-YOLO's own confidence below the
    # shared 0.20 default, the same failure mode measured on Urdu
    # (see LanguagePipeline.layout_confidence's docstring) -- dropping
    # real text blocks before OCR/grouping ever see them.
    layout_confidence=0.10,

    # Align with English: these 8 heuristic repair passes were found
    # to mangle boundaries (boundary_decomposer slicing overlapping
    # column rectangles, footprint union merging unrelated stories
    # across columns) rather than fix them. See
    # pipeline/languages/english/__init__.py for the same disablement.
    use_orphan_block_reassignment=False,
    use_orphan_title_root_repair=False,
    use_unclaimed_kicker_recovery=False,
    use_unclaimed_image_recovery=False,
    use_unclaimed_footprint_recovery=False,
    use_article_splitter=False,
    use_dropped_article_recovery=False,
    use_boundary_decomposition=False,
)
