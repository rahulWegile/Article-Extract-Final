"""
Per-language pipeline configuration.

Each supported language declares, in its own module under
pipeline/languages/, which OCR engine, LLM provider, grouping
prompt/policy, article extractor, and recovery/reconciliation
behavior it uses. This replaces the previous approach of scattering
`_is_<language>_language()` checks and `is_hindi`/`doc_is_hindi`
booleans across backend/services/pipeline_service.py,
pipeline/article/article_grouper.py, and the Gemini/OCR modules.

Most languages simply point at the same shared implementations
(OpenAIArticleExtractor, the default grouping prompt, no
contested-block arbitration, no document-order Stream B) -- only
Hindi currently differs. That is intentional: the shared
implementations are not duplicated per language here, only the
wiring/dispatch is made explicit per language. See
pipeline/languages/registry.py for how a document's detected
language resolves to one of these.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class LanguagePipeline:
    # Short internal identifier, also used as the OCR-engine cache key
    # in backend/services/pipeline_service.py.
    code: str

    # Copied verbatim from this language's old _is_<language>_language
    # method -- same substring-match semantics, not just equality.
    matches: Callable[[str], bool]

    # Builds this language's OCR engine instance. Called at most once
    # per PipelineService instance; the result is cached by the
    # caller, matching the previous per-language lazy-cache behavior.
    ocr_engine_factory: Callable[[], object]

    # Human-readable label for the "OCR engine : ..." log line.
    ocr_engine_label: str

    # "gemini" or "openai". ("local" is a global PipelineService-level
    # override, not a per-language choice, and is applied by the
    # caller on top of this value.)
    llm_provider: str

    # The grouping prompt sent to whichever provider above analyzes
    # this document's pages for article boundaries.
    grouping_prompt: str

    # The article-extraction prompt template handed to extractor_class.
    # Its placeholders (__PAGES__, __CROP_INVENTORY__, ...) are filled
    # in by the extractor's _build_prompt at call time.
    extraction_prompt_template: str

    # GeminiArticleExtractor or OpenAIArticleExtractor.
    extractor_class: type

    # Whether ArticleGrouper.build() should run layout-distance
    # arbitration for contested blocks, vs. first-claim-wins.
    use_contested_block_arbitration: bool = False

    # ----------------------------------------------------------
    # Post-grouping repair passes.
    #
    # All default to True, i.e. the behaviour every language has had
    # since these passes were added. English sets them to False to
    # match the reference "english Pproper" pipeline, whose
    # ArticleGrouper has none of them. Adding a language without
    # touching these keeps today's behaviour unchanged.
    # ----------------------------------------------------------

    # Move a block to the claiming article it actually sits with
    # geometrically (ArticleGrouper._reassign_orphan_blocks).
    use_orphan_block_reassignment: bool = True

    # Re-attach a body-less kicker/eyebrow title stranded in its own
    # single-block article (ArticleGrouper._reassign_orphan_title_roots).
    use_orphan_title_root_repair: bool = True

    # Fold leftover unclaimed kicker titles into the article beneath
    # them (ArticleGrouper._recover_unclaimed_kicker_titles).
    use_unclaimed_kicker_recovery: bool = True

    # Fold an unclaimed article_image into whichever article claimed
    # its own caption (ArticleGrouper._recover_unclaimed_images_via_caption).
    use_unclaimed_image_recovery: bool = True

    # Last-resort catch-all: fold any block still unclaimed after
    # every recovery pass above into the one article whose own
    # footprint already geometrically contains it
    # (ArticleGrouper._recover_unclaimed_blocks_by_footprint).
    use_unclaimed_footprint_recovery: bool = True

    # Re-check any article holding more than one title block against
    # local_grouper's geometry and split it when they disagree
    # (pipeline/article/article_splitter.split_oversized_articles).
    use_article_splitter: bool = True

    # Detach a wide top block (e.g. masthead banner) from narrow columns
    # below it (pipeline/article/article_splitter.detach_wide_top_banner_blocks).
    # Disabled for broadsheet languages where main headlines span across columns.
    use_wide_top_banner_detachment: bool = True

    # Whether this language is printed right-to-left (currently only
    # Urdu -- Perso-Arabic Nastaliq). Affects two purely geometric,
    # pre-LLM computations that both assume left-to-right print by
    # default:
    #
    # - pipeline/sort_blocks.py's reading_order (top-to-bottom, then
    #   RIGHT-to-left instead of left-to-right within a row);
    # - PageCleaner.assign_columns's column index (0 = the
    #   RIGHTMOST of the page's six print columns instead of the
    #   leftmost), so "column 0" still means "the first column a
    #   reader's eye reaches" for both directions.
    #
    # Confirmed on a real Urdu (THE INQUILAB) document that neither
    # was RTL-aware: every page's blocks carried LTR reading_order
    # and column numbers into the grouping prompt, which explicitly
    # tells the model to use "column flow and reading order" and
    # "headline reading direction" to resolve ambiguous boundary
    # blocks (see ARTICLE OWNERSHIP DECISION ORDER) -- backwards
    # values there actively point the model's tie-breaker in the
    # wrong direction rather than merely omitting a signal.
    is_rtl: bool = False

    # Rebuild a story the model threw away by giving every one of its
    # blocks a non-article role such as `advertisement` or `weather`
    # (pipeline/article/dropped_article_recovery.recover_dropped_articles).
    # Purely additive -- it only ever forms new articles out of blocks
    # no existing article owns.
    use_dropped_article_recovery: bool = True

    # Re-cut an article whose min/max RECTANGLE swallows content it
    # does not own -- an L-shaped story whose stranded photograph
    # stretches its box across a neighbouring story
    # (pipeline/article/boundary_decomposer.decompose_overlapping_articles).
    use_boundary_decomposition: bool = True

    # Factory for this language's document-order ("Stream B")
    # extractor, or None if this language doesn't run Stream B.
    document_order_extractor_factory: Optional[Callable[[], object]] = None

    # Repairs headings Stream B saw but that never made it into any
    # article crop's block_ids -- signature:
    # fn(document_dir, batch_sections). Set together with
    # document_order_extractor_factory; None for languages with no
    # Stream B.
    document_order_repair_headings: Optional[Callable] = None

    # Confidence floor passed to LayoutDetector.detect() (see
    # pipeline/layout_detector.py's DEFAULT_CONFIDENCE) for this
    # language's pages. Confirmed on a real Urdu (THE INQUILAB)
    # document that the shared 0.20 default drops dozens of real
    # articles on a dense Nastaliq page (20 blocks detected vs. 50+ at
    # 0.08 on the SAME page) -- dense, cursive/ligature-heavy scripts
    # push DocLayout-YOLO's own confidence lower than the Latin/
    # Devanagari pages that default was tuned on. Per-language rather
    # than global so lowering it for Urdu cannot loosen detection (and
    # invite more false-positive/noise blocks) for every other script
    # that already works well at 0.20.
    layout_confidence: float = 0.20

    # Folds Stream B's text into Stream A's articles -- signature:
    # fn(articles, document_order_sections) -> (articles, unresolved).
    # Set together with document_order_extractor_factory; None for
    # languages with no Stream B.
    document_order_reconcile_articles: Optional[Callable] = None

    # Number of pages' crops grouped into one article-extraction call,
    # in the page-by-page pipeline (backend/services/pipeline_service.py's
    # interleave_extraction path). 1 (the default, every language except
    # Hindi) means extraction fires immediately after each page's own
    # crops are written. A value > 1 buffers that many pages' crops
    # before firing one shared extraction call across them, trading a
    # small per-page cost reduction (shared prompt overhead -- see
    # OpenAIArticleExtractor.process_batch) for less even, larger
    # requests. Purely a batching-size knob: the pipeline stays fully
    # sequential either way (one API call in flight at a time), so this
    # never introduces concurrent/overlapping API calls or "overload"
    # risk against boundary-grouping, which always remains one page,
    # one call, regardless of this setting.
    extraction_pages_per_batch: int = 1

    # Safety cap on total article crops per extraction call, on top of
    # extraction_pages_per_batch. None (the default, every language
    # except Hindi) means no cap beyond the page-count one. When set,
    # a page whose crops would push the running total past this value
    # is held back to start the NEXT batch instead of being force-fit
    # into the current one -- mirrors the existing byte-size safety
    # cap in OpenAIArticleExtractor._make_page_batches ("a single
    # oversized page still ships alone").
    extraction_max_articles_per_batch: Optional[int] = None
