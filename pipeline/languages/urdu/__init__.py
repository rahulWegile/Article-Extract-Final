from pipeline.languages.base import LanguagePipeline
from pipeline.languages.urdu.grouping_prompt import (
    URDU_GROUPING_PROMPT,
)
from pipeline.languages.urdu.extraction_prompt import (
    URDU_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_urdu_language.
    Urdu is written in the Perso-Arabic Nastaliq style, which
    RapidOCR's "arabic" (Naskh-oriented) recognition model is not
    tuned for, so this routes to TesseractOCREngine(lang="urd")
    instead.
    """

    lang = (language or "").strip().lower()

    return lang in ("ur", "urd", "urdu") or "urdu" in lang


URDU = LanguagePipeline(
    code="urdu",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="urd"),
    ocr_engine_label="tesseract (urdu)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati/
    # Assamese/Bengali/Kannada use.
    #
    # TEMPORARY: this was the worst-confirmed case in the cross-
    # language analysis -- extracted length averaged 0.14x the
    # underlying OCR text (vs. ~1x for Hindi/Marathi on Gemini), and
    # visually confirmed on a real crop (THE INQUILAB) that
    # OpenAIArticleExtractor's pinned model (OPENAI_ARTICLE_MODEL=
    # gpt-5.6-luna) was returning a generic paraphrase -- article_text
    # and summary at the same level of generality -- instead of a
    # transcription. NOTE: every other language moved to Gemini so far
    # is Devanagari (Hindi/Marathi) or a Brahmic script; Urdu's
    # Perso-Arabic Nastaliq is visually very different (highly
    # cursive/calligraphic), and Gemini's reading of THIS script has
    # not been separately confirmed the way its Devanagari reading
    # was -- verify on a real crop before trusting this fully. To
    # revert: change extractor_class back to OpenAIArticleExtractor
    # (and restore the `from
    # pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=URDU_GROUPING_PROMPT,
    extraction_prompt_template=URDU_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
    is_rtl=True,
)
