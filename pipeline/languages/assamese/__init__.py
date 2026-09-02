from pipeline.languages.base import LanguagePipeline
from pipeline.languages.assamese.grouping_prompt import (
    ASSAMESE_GROUPING_PROMPT,
)
from pipeline.languages.assamese.extraction_prompt import (
    ASSAMESE_EXTRACTION_PROMPT,
)
from pipeline.ocr.tesseract_engine import TesseractOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old
    PipelineService._is_assamese_language. Assamese uses its own
    script variant (distinct from Bengali in a handful of
    letterforms), and Tesseract ships a dedicated Assamese model
    (tessdata_best "asm"), so this uses that rather than reusing the
    Bengali model for a script it was not trained on.
    """

    lang = (language or "").strip().lower()

    return lang in ("as", "asm", "assamese") or "assamese" in lang


ASSAMESE = LanguagePipeline(
    code="assamese",
    matches=matches,
    ocr_engine_factory=lambda: TesseractOCREngine(lang="asm"),
    ocr_engine_label="tesseract (assamese)",
    # Grouping/boundary detection stays on OpenAI (llm_provider below)
    # -- only ARTICLE-LEVEL TEXT EXTRACTION is switched to Gemini here,
    # via extractor_class. Same pattern Marathi/Punjabi/Gujarati use.
    #
    # TEMPORARY: cross-language analysis (extracted-text length vs.
    # each article's own OCR text) showed OpenAIArticleExtractor's
    # pinned model (OPENAI_ARTICLE_MODEL=gpt-5.6-luna) failing on
    # Assamese the same way it failed on Urdu/Punjabi -- extracted
    # length clustered tightly (~190 chars, std=41) regardless of how
    # much text the crop actually contained, the same "paraphrase, not
    # transcription" signature confirmed visually on Urdu and Punjabi
    # crops. Not yet visually confirmed on a real Assamese crop -- do
    # that check on the next processed document. To revert: change
    # extractor_class back to OpenAIArticleExtractor (and restore the
    # `from pipeline.intelligence.openai_article_extractor import
    # OpenAIArticleExtractor` import above).
    llm_provider="openai",
    grouping_prompt=ASSAMESE_GROUPING_PROMPT,
    extraction_prompt_template=ASSAMESE_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
)
