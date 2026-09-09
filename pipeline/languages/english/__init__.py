from pipeline.languages.base import LanguagePipeline
from pipeline.languages.english.grouping_prompt import (
    ENGLISH_GROUPING_PROMPT,
)
from pipeline.languages.english.extraction_prompt import (
    ENGLISH_EXTRACTION_PROMPT,
)
from pipeline.ocr.rapidocr_engine import RapidOCREngine
from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)


def matches(language: str) -> bool:
    lang = (language or "").strip().lower()

    return lang in ("en", "eng", "english") or "english" in lang


# We construct a dedicated OCR engine for English, explicitly disabling
# the misclassified-figure and low-confidence-retry passes, to match
# the reference "english Pproper" pipeline. The default engine used
# for local masthead extraction still keeps them enabled so other
# languages' masthead OCR is unaffected.
ENGLISH = LanguagePipeline(
    code="english",
    matches=matches,
    ocr_engine_factory=lambda: RapidOCREngine(
        enable_misclassified_figure_recovery=False,
        enable_low_confidence_retry=False,
    ),
    ocr_engine_label="default",
    llm_provider="openai",
    grouping_prompt=ENGLISH_GROUPING_PROMPT,
    extraction_prompt_template=ENGLISH_EXTRACTION_PROMPT,
    extractor_class=OpenAIArticleExtractor,

    # Disable all new repair passes to match Path B behavior
    use_orphan_block_reassignment=False,
    use_orphan_title_root_repair=False,
    use_unclaimed_kicker_recovery=False,
    use_unclaimed_image_recovery=False,
    use_unclaimed_footprint_recovery=False,
    use_article_splitter=False,
    use_dropped_article_recovery=False,
    use_boundary_decomposition=False,
)
