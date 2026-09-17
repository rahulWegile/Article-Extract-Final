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
    extractor_class=OpenAIArticleExtractor,

    # 2-page extraction batching, matching Hindi's move to OpenAI: shares
    # the extraction prompt's fixed overhead across 2 pages instead of
    # paying it on every single page. Boundary detection is unaffected --
    # it stays one page, one call, regardless of this setting.
    extraction_pages_per_batch=2,

    # Safety cap: even within a 2-page group, never send more than 8
    # article crops in one extraction call. A page that would push the
    # running total past this cap is held back and starts the NEXT batch
    # instead of being force-fit into this one.
    #
    # Lowered from 25 to 8: a batch of ~21 articles at ~1,500 Odia chars
    # each needs 35,000+ completion tokens, which pushed gpt-5.6-luna to
    # compress/truncate individual articles instead of transcribing every
    # column verbatim (see the COMPLETENESS MANDATE in
    # extraction_prompt.py). 8 articles per call leaves ample token
    # headroom per article for full multi-column transcription.
    extraction_max_articles_per_batch=8,

    # Unlike Hindi (which disables every heuristic repair pass to
    # preserve OpenAI's own grouping decisions), Odia keeps the
    # geometric splitter and boundary-decomposition repair passes
    # enabled -- these are the shared defaults already, kept explicit
    # here so this choice isn't mistaken for an oversight.
    use_article_splitter=True,
    use_boundary_decomposition=True,

    # The shared 0.20 default drops genuine Odia headlines/columns.
    # Confirmed on a real Odia (Sambad, doc_000177 page 1) page: the
    # lead headline ("ସତର୍କ କରାଇଲା ଚୀନ୍") was detected at YOLO
    # confidence 0.15 and discarded at the default threshold --
    # across all 4 pages of that document, detections dropped ~30-40%
    # at 0.20 vs. 0.10. See LanguagePipeline.layout_confidence.
    layout_confidence=0.10,
)
