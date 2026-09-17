import json

import cv2

from pipeline.block_parser import LayoutBlock

from pipeline.gemini.gemini_parser import GeminiParser

from pipeline.article.article_grouper import (
    ArticleGrouper,
)

from pipeline.article.article_splitter import (
    detach_wide_top_banner_blocks,
    drop_distant_orphan_blocks,
    split_oversized_articles,
)

from pipeline.article.dropped_article_recovery import (
    recover_dropped_articles,
)

from pipeline.article.headless_headline_relink import (
    relink_headless_headlines,
)

from pipeline.article.sidebox_absorption import (
    absorb_composite_sideboxes,
)

from pipeline.article.boundary_decomposer import (
    decompose_overlapping_articles,
)

from pipeline.article.boundary_builder import (
    BoundaryBuilder,
)

from pipeline.visualization.draw_final_boundaries import (
    FinalBoundaryVisualizer,
)


class GeminiBoundaryPipeline:
    """
    Build final article boundaries from
    Gemini grouped articles.
    """

    # -----------------------------------------------------

    def _load_blocks(
        self,
        page_json_path,
    ):

        with open(
            page_json_path,
            "r",
            encoding="utf-8",
        ) as f:

            page = json.load(f)

        blocks = []

        for item in page["blocks"]:

            block = LayoutBlock(

                id=item["id"],

                cls=item["class"],

                x1=item["bbox"]["x1"],
                y1=item["bbox"]["y1"],
                x2=item["bbox"]["x2"],
                y2=item["bbox"]["y2"],

                confidence=1.0,

            )

            block.text = item.get(
                "text",
                "",
            )

            block.reading_order = item.get(
                "reading_order",
                0,
            )

            blocks.append(
                block
            )

        return blocks

    # -----------------------------------------------------

    def run(
        self,
        image_path,
        page_json_path,
        gemini_response_path,
        output_path,
        use_contested_block_arbitration: bool = True,
        use_orphan_block_reassignment: bool = True,
        use_orphan_title_root_repair: bool = True,
        use_unclaimed_kicker_recovery: bool = True,
        use_unclaimed_image_recovery: bool = True,
        use_unclaimed_footprint_recovery: bool = True,
        use_article_splitter: bool = True,
        use_wide_top_banner_detachment: bool = True,
        use_dropped_article_recovery: bool = True,
        use_headless_headline_relink: bool = True,
        use_sidebox_absorption: bool = True,
        use_boundary_decomposition: bool = True,
    ):

        print()
        print("=" * 60)
        print("GEMINI BOUNDARY PIPELINE")
        print("=" * 60)

        #
        # Load page blocks
        #

        blocks = self._load_blocks(
            page_json_path,
        )

        #
        # Load Gemini response
        #

        with open(
            gemini_response_path,
            "r",
            encoding="utf-8",
        ) as f:

            response = json.load(f)

        #
        # Parse Gemini response
        #

        parser = GeminiParser()

        parsed = parser.parse(

            response,

            blocks,

        )

        #
        # Load the page image once, up front, whenever ANY
        # pixel-probing pass below needs it -- orphan-block
        # reassignment's hard-separator veto (ArticleGrouper.build,
        # see _has_separator_between) now needs it too, not just the
        # splitter/recovery passes that used to be the only consumers.
        # Sidebox absorption's own hard-separator veto (see
        # sidebox_absorption.py) is the same probe again, applied to a
        # different pair of blocks -- it works fine without a page
        # image (every OTHER guard there is pure geometry), so it is
        # included here only to give that one extra check a page image
        # to use when everything else already loaded one anyway.
        #
        # Only these passes probe page pixels, so a language with all
        # of them switched off never pays for decoding a
        # full-resolution page scan here.
        #

        if (
            use_orphan_block_reassignment
            or use_article_splitter
            or use_dropped_article_recovery
            or use_sidebox_absorption
        ):

            try:
                page_image = cv2.imread(str(image_path))
            except Exception:
                page_image = None

        else:

            page_image = None

        #
        # Build articles
        #

        grouper = ArticleGrouper()

        articles = grouper.build(

            parsed,

            blocks,

            use_contested_block_arbitration=(
                use_contested_block_arbitration
            ),

            use_orphan_block_reassignment=(
                use_orphan_block_reassignment
            ),

            use_orphan_title_root_repair=(
                use_orphan_title_root_repair
            ),

            use_unclaimed_kicker_recovery=(
                use_unclaimed_kicker_recovery
            ),

            use_unclaimed_image_recovery=(
                use_unclaimed_image_recovery
            ),

            use_unclaimed_footprint_recovery=(
                use_unclaimed_footprint_recovery
            ),

            page_image=page_image,

        )

        #
        # Split any article the model over-merged: re-checks any
        # article containing more than one title block against
        # local_grouper's own geometric root-detection, splitting it
        # only when that algorithm actually finds separate stories
        # (never touches ArticleGrouper's or the LLM's own decision
        # otherwise -- see article_splitter.py).
        #
        # Runs for every language, not just Hindi: this whole check
        # is pure geometry and raw layout-detector class (bbox
        # overlap, gap size, rule-line pixels) -- nothing in
        # split_oversized_articles, group_blocks, or the orphan-
        # content merge reads or assumes anything about the text's
        # language. Confirmed on a real Tamil page: an OpenAI-routed
        # (non-Hindi) article with 4 title-class blocks -- well past
        # the >1 trigger -- merged 4 unrelated stories (a returnee
        # story, a Nepal rescue photo, a temperature forecast, a
        # group-photo item) into one 16-block boundary, with no
        # safety net at all to catch it while this ran Hindi-only.
        # Was previously gated behind is_hindi only because the
        # reference English pipeline had no equivalent step to draw
        # on, not because of any language-specific assumption here.
        # Already a no-op whenever an article has <=1 title block, so
        # this cannot change behavior on pages that were already
        # fine.
        #

        # Recover a whole story the model threw away by giving every
        # one of its blocks a non-article role. ArticleGrouper's
        # IGNORE_ROLES drop is terminal -- none of its repair passes
        # can see a block whose role was excluded -- so a single
        # wrong role classification silently costs a full article,
        # boundary and crop. See dropped_article_recovery.py for the
        # story-shape guards that keep this from resurrecting real
        # advertisements.
        #
        # Runs BEFORE the splitter below so that anything it recovers
        # is re-checked by the same over-merge logic every other
        # article gets, and before decomposition so a recovered story
        # is already a known owner by the time rectangles are checked
        # for enclosing content they do not own.
        #

        if use_dropped_article_recovery:

            articles = recover_dropped_articles(
                articles,
                blocks,
                page_image,
            )

        if use_article_splitter:

            articles = split_oversized_articles(
                articles,
                page_image,
            )

            # A different failure than the multi-title over-merge
            # above: a single-title article that also picked up a
            # body-text block hundreds of pixels away, in a
            # different/misaligned column, with other complete
            # unrelated stories printed in between -- confirmed on a
            # real Urdu page. split_oversized_articles never
            # re-examines this (it only re-checks articles with >1
            # title), so it would otherwise reach the crop stage
            # untouched. See DISTANT ORPHAN BLOCKS in
            # article_splitter.py.
            articles = drop_distant_orphan_blocks(articles)

        #
        # A masthead/banner block PageCleaner failed to flag as
        # global chrome can still get pulled into a real story when
        # it sits directly above that story's own narrow column
        # blocks -- the article's bounding rectangle then stretches
        # to the banner's near-full-page width even though the real
        # content underneath stays in one or two narrow columns. See
        # WIDE TOP BANNER DETACHMENT in article_splitter.py.
        #
        # Runs after the splitter/distant-orphan passes above (so it
        # acts on membership those have already settled) and before
        # decomposition below, for the same reason as the other
        # passes in this sequence.
        #

        if use_wide_top_banner_detachment and use_article_splitter:

            page_width_estimate = max(
                (block.x2 for block in blocks),
                default=0.0,
            )

            articles = detach_wide_top_banner_blocks(
                articles,
                page_width=page_width_estimate,
            )

        #
        # Re-link a headline a DIFFERENT article claims back to a
        # headless neighbour it actually sits directly above -- a
        # narrower, more targeted case than the general orphan-block
        # reassignment above (ArticleGrouper._reassign_orphan_blocks),
        # which requires a block to be nearly isolated from its own
        # article's other blocks before it is even reconsidered. See
        # headless_headline_relink.py.
        #
        # Runs after the splitter/distant-orphan passes above (so it
        # acts on membership those have already settled) and before
        # decomposition below (so a corrected headline is already in
        # place by the time rectangles are checked for enclosing
        # content they do not own).
        #

        if use_headless_headline_relink:

            articles = relink_headless_headlines(articles)

        #
        # Fold a small callout/sidebar article (a pull-quote panel, a
        # boxed reactions strip) that the model gave its own complete
        # article into the dominant, plain story it actually belongs
        # to -- see sidebox_absorption.py for the full geometric-
        # safety writeup (why this only ever fires on a narrow,
        # corpus-validated shape).
        #
        # Runs after the splitter/relink passes above (so it acts on
        # membership those have already settled) and before
        # decomposition below (so a newly-absorbed satellite is
        # already part of its parent by the time rectangles are
        # checked for enclosing content they do not own).
        #

        if use_sidebox_absorption:

            articles = absorb_composite_sideboxes(articles, page_image)

        #
        # Re-cut any article whose min/max RECTANGLE encloses content
        # it does not own -- the L-shaped-article problem the
        # grouping prompts describe but nothing enforced on our side.
        # See boundary_decomposer.py.
        #

        if use_boundary_decomposition:

            articles = decompose_overlapping_articles(
                articles,
                blocks,
            )

        #
        # Build boundaries
        #

        builder = BoundaryBuilder()

        boundaries = builder.build(

            articles,

        )

        #
        # Render boundaries
        #

        FinalBoundaryVisualizer().save(

            image_path=image_path,

            boundaries=boundaries,

            output_path=output_path,

        )

        print()
        print("=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Articles Built   : {len(articles)}")
        print(f"Boundaries Built : {len(boundaries)}")
        print(f"Saved            : {output_path}")
        print("=" * 60)
        print()

        return boundaries