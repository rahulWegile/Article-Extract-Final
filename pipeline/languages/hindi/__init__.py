from pipeline.languages.base import LanguagePipeline
from pipeline.languages.hindi.grouping_prompt import HINDI_GROUPING_PROMPT
from pipeline.languages.hindi.extraction_prompt import (
    HINDI_EXTRACTION_PROMPT,
)
from pipeline.ocr.rapidocr_engine import RapidOCREngine
from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)
from pipeline.languages.hindi.document_order_extractor import (
    GeminiDocumentOrderExtractor,
)
from pipeline.languages.hindi.heading_repair import (
    repair_missing_headings,
)
from pipeline.languages.hindi.reconcile import (
    reconcile_articles,
)


def matches(language: str) -> bool:
    """
    Copied verbatim from the old PipelineService._is_hindi_language.
    "Hindi", "hi", "hin", and anything containing "hindi" (e.g.
    "Hindi (Devanagari)") all count.
    """

    lang = (language or "").strip().lower()

    return lang in ("hi", "hin", "hindi") or "hindi" in lang


# The one language that differs from the shared/default pipeline:
# Gemini for grouping + extraction (gpt-5.6-luna cannot read
# Devanagari, confirmed on real headline crops), layout-distance
# contested-block arbitration instead of first-claim-wins, and a
# concurrent document-order extraction pass ("Stream B") to recover
# headings the boundary pipeline missed.
HINDI = LanguagePipeline(
    code="hindi",
    matches=matches,
    ocr_engine_factory=lambda: RapidOCREngine(lang="hi"),
    ocr_engine_label="devanagari (hindi)",
    llm_provider="gemini",
    grouping_prompt=HINDI_GROUPING_PROMPT,
    extraction_prompt_template=HINDI_EXTRACTION_PROMPT,
    extractor_class=GeminiArticleExtractor,
    use_contested_block_arbitration=True,
    document_order_extractor_factory=GeminiDocumentOrderExtractor,
    document_order_repair_headings=repair_missing_headings,
    document_order_reconcile_articles=reconcile_articles,
)
