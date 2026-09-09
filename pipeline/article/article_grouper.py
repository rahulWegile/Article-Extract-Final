from dataclasses import dataclass, field
from typing import List, Tuple

from pipeline.article.visual_separator import (
    has_vertical_separator_between,
    has_visual_separator,
)


@dataclass
class Article:

    article_id: int

    blocks: List

    block_ids: List[int]

    confidence: float = 1.0

    # Precise (x1, y1, x2, y2) sub-rectangles covering this article's
    # own blocks, set only when a single bounding rectangle would
    # improperly enclose a neighbouring article's content and the
    # neighbour's content can't safely be split off as its own
    # article (see pipeline/article/boundary_decomposer.py). Empty
    # for the overwhelming majority of articles, which stay one plain
    # rectangle exactly as before.
    sub_rects: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )


class ArticleGrouper:
    """
    Convert Gemini output into internal
    Article objects.

    Only article-related semantic roles
    are kept.
    """

    #
    # Non-article semantic roles
    #

    IGNORE_ROLES = {

        "masthead",

        "page_header",

        "page_footer",

        "page_number",

        "section_header",

        "advertisement",

        "comic",

        "weather",

        "logo",

        "decoration",

        "unknown",

    }

    @staticmethod
    def _is_uncertain_title(block):
        """
        True when the LAYOUT detector called this block a headline
        (block.cls == "title") but the grouping model's own SEMANTIC
        role classification came back "unknown" -- i.e. the model
        never listed this block in any article's "blocks" array and,
        when asked to name its role directly, couldn't.

        "unknown" is not the model confidently saying "this is not a
        story" the way "masthead"/"decoration"/"advertisement" are --
        it is the model admitting it couldn't classify the block at
        all, which happens most often when the OCR text handed to it
        is themself unreadable (garbled/mis-segmented, a known issue
        for RTL scripts -- see pipeline/ocr/tesseract_engine.py). By
        default IGNORE_ROLES treats "unknown" identically to genuine
        page chrome, so a headline that isn't confidently readable
        vanishes with no boundary, no crop, and no warning.

        Confirmed on a real Urdu page (THE INQUILAB): two title-class
        blocks with unreadable OCR text got role "unknown" and were
        dropped before _recover_unclaimed_kicker_titles -- built
        specifically to reattach exactly this shape of stray headline
        -- ever got a chance to see them.

        Scoped to title-class blocks only: a "plain text"/"figure"
        block the model marked "unknown" is a genuinely different,
        more ambiguous situation (could easily be an ad or other
        chrome the model correctly hedged on) that IGNORE_ROLES should
        keep excluding by default.
        """

        role = getattr(block, "role", None)

        if role != "unknown":
            return False

        return (getattr(block, "cls", "") or "").strip().lower() == "title"

    @staticmethod
    def _vertical_gap(block_a, block_b):
        """
        Blank vertical distance between two blocks, 0 when they
        overlap vertically at all.
        """

        if block_a.y2 <= block_b.y1:
            return float(block_b.y1 - block_a.y2)

        if block_b.y2 <= block_a.y1:
            return float(block_a.y1 - block_b.y2)

        return 0.0

    @staticmethod
    def _layout_distance(block_a, block_b):
        """
        Newspaper-layout distance between two blocks.

        Blocks in the same column that vertically abut each other are
        near-zero apart; blocks in different columns are pushed far
        apart regardless of how close they happen to be vertically,
        because column structure -- not raw pixel distance -- is what
        decides story ownership on a newspaper page.
        """

        a_width = max(1.0, float(block_a.x2 - block_a.x1))
        b_width = max(1.0, float(block_b.x2 - block_b.x1))

        overlap_x = max(
            0.0,
            min(block_a.x2, block_b.x2) - max(block_a.x1, block_b.x1),
        )

        # IoU-style overlap (over the UNION of both widths), not over
        # the narrower block's width alone. Normalizing by the
        # narrower width gives a column-spanning block (a wide photo,
        # a banner headline) an overlap_ratio near 1.0 against almost
        # any narrower block it happens to contain horizontally, even
        # when that narrower block sits in one specific sub-column
        # unrelated to the spanning block's own story -- collapsing
        # column_penalty to ~0 exactly when it should be discriminating
        # most. IoU still gives ~1.0 for the common case (two similarly
        # -sized blocks stacked in the same column), but correctly
        # drops for a large size mismatch.
        union_width = a_width + b_width - overlap_x

        overlap_ratio = overlap_x / union_width if union_width > 0 else 0.0

        vertical_gap = ArticleGrouper._vertical_gap(block_a, block_b)

        # A different column is never the better explanation for
        # ownership, so anything poorly overlapped horizontally gets a
        # penalty larger than any plausible in-column vertical gap.
        column_penalty = (1.0 - min(1.0, overlap_ratio)) * 100000.0

        return vertical_gap + column_penalty

    # Mirrors article_splitter.py's BANNER_FRAGMENT_EDGE_TOLERANCE /
    # BANNER_FRAGMENT_HEIGHT_RATIO / BANNER_FRAGMENT_MAX_GAP_RATIO --
    # same geometric test (two title blocks that are really one
    # printed headline line, cut apart at a column gutter), duplicated
    # here rather than imported because article_splitter.py already
    # imports FROM this module (importing back would be circular).
    # Deliberately WITHOUT that file's final has_vertical_separator_
    # between pixel probe: ArticleGrouper.build() runs before any
    # page image is loaded anywhere in the pipeline (see
    # gemini_boundary_pipeline.py -- page_image is only read later,
    # for the splitter/dropped-article-recovery stages), so only
    # bbox/text evidence is available here. See
    # _is_plausible_banner_pair below for why this exists.
    BANNER_PAIR_MIN_ROW_OVERLAP_RATIO = 0.5
    BANNER_PAIR_HEIGHT_RATIO = 0.75
    BANNER_PAIR_MAX_GAP_RATIO = 0.5

    @classmethod
    def _is_plausible_banner_pair(cls, block_a, block_b) -> bool:
        """
        Whether two title-class blocks are plausibly two fragments of
        ONE printed headline that a column gutter cut apart (e.g. a
        banner running "GDP...fake claim" split into a left and right
        title box) rather than two independent headlines that happen
        to share a row.

        _layout_distance structurally cannot see this: a banner
        fragment's own bbox never column-overlaps its OTHER half --
        that lack of overlap is exactly what makes it a cut-apart
        fragment -- so the generic column-penalty formula scores two
        genuine banner halves as maximally far apart, near its own
        100000 ceiling. Confirmed on a real Urdu (THE INQUILAB) page:
        this made _reassign_orphan_blocks repeatedly judge a banner
        fragment "isolated" from its own true other half and instead
        anchor it on whatever ordinary body block happened to sit
        directly below it -- geometrically tighter by the generic
        metric, but semantically the wrong pairing, since that body
        block itself belongs with the OTHER fragment.

        Uses ROW OVERLAP rather than article_splitter.py's stricter
        single-line-text + exact-edge-tolerance test: a banner cut
        into two narrower boxes routinely has its OCR text wrap
        across 2-3 lines within each box's own narrower width (the
        printed line is one row of large type; the per-box crop is
        not), which would fail a literal single-line check even
        though the two boxes are still visibly one banner row.
        Confirmed necessary on a real Urdu page: the "ہر گھر تک جانچ
        ہوگی" banner's own two fragments both carried multi-line OCR
        text purely from this box-width wrapping.
        """

        if (
            (getattr(block_a, "cls", None) or "").strip().lower()
            != "title"
            or (getattr(block_b, "cls", None) or "").strip().lower()
            != "title"
        ):
            return False

        height_a = block_a.y2 - block_a.y1
        height_b = block_b.y2 - block_b.y1

        shorter = min(height_a, height_b)
        taller = max(height_a, height_b)

        if shorter <= 0 or shorter / taller < cls.BANNER_PAIR_HEIGHT_RATIO:
            return False

        row_top = max(block_a.y1, block_b.y1)
        row_bottom = min(block_a.y2, block_b.y2)
        row_overlap = max(0.0, float(row_bottom - row_top))

        if row_overlap / shorter < cls.BANNER_PAIR_MIN_ROW_OVERLAP_RATIO:
            return False

        left, right = (
            (block_a, block_b)
            if block_a.x1 <= block_b.x1
            else (block_b, block_a)
        )

        gap = right.x1 - left.x2

        if gap < 0 or gap > cls.BANNER_PAIR_MAX_GAP_RATIO * shorter:
            return False

        return True

    @classmethod
    def _reassignment_distance(cls, block_a, block_b):
        """
        _layout_distance, except a genuine banner-headline fragment
        pair (see _is_plausible_banner_pair) is treated as touching
        (0.0) instead of running through the generic column-penalty
        formula. Used ONLY by _reassign_orphan_blocks_once -- every
        OTHER caller of _layout_distance (contested-block arbitration,
        orphan-title-root repair) is left exactly as before.
        """

        if cls._is_plausible_banner_pair(block_a, block_b):
            return 0.0

        return cls._layout_distance(block_a, block_b)

    @staticmethod
    def _has_separator_between(page_image, block_a, block_b) -> bool:
        """
        Whether a printed rule line or colored border sits between
        block_a and block_b -- a HARD signal that the newspaper itself
        already marked these as two separate editorial items, which
        must never be overridden by geometric proximity alone.

        Checks whichever relationship the two blocks actually have:
        a horizontal rule in the blank band between them when they are
        vertically stacked (the usual _reassign_orphan_blocks shape --
        a block being pulled up/down into a neighboring article), or a
        vertical divider in the gap between them when they sit side by
        side. Reuses the SAME pixel probes article_splitter.py already
        relies on for its own hard-separator checks
        (has_visual_separator / has_vertical_separator_between) rather
        than a new detector -- see visual_separator.py for the tuning
        history behind both.

        Returns False (no veto) when there is no page image to probe,
        or when the two blocks neither stack nor sit side by side
        (e.g. they overlap on both axes) -- this check only ever
        BLOCKS a reassignment it can positively confirm crosses a
        printed divider; it never blocks one it cannot evaluate.
        """

        if page_image is None:
            return False

        if block_a.y2 <= block_b.y1 or block_b.y2 <= block_a.y1:

            top, bottom = (
                (block_a, block_b)
                if block_a.y2 <= block_b.y1
                else (block_b, block_a)
            )

            if has_visual_separator(
                page_image,
                min(top.x1, bottom.x1),
                max(top.x2, bottom.x2),
                top.y2,
                bottom.y1,
            ):
                return True

        if block_a.x2 <= block_b.x1 or block_b.x2 <= block_a.x1:

            left, right = (
                (block_a, block_b)
                if block_a.x2 <= block_b.x1
                else (block_b, block_a)
            )

            if has_vertical_separator_between(
                page_image,
                left.x2,
                right.x1,
                min(block_a.y1, block_b.y1),
                max(block_a.y2, block_b.y2),
            ):
                return True

        return False

    # Fraction of the narrower block's width that must overlap
    # horizontally for two blocks to count as sharing a column, for
    # the orphan-title-root repairs below. Matches
    # local_grouper.py's COLUMN_OVERLAP -- same geometric question
    # (does a title-only block belong to the headline below it in
    # print), just re-expressed against Block objects instead of
    # page_json dicts.
    ORPHAN_TITLE_COLUMN_OVERLAP = 0.35

    # A genuine kicker/eyebrow line is a single short line of text
    # printed above the real headline -- never a wrapped multi-
    # sentence paragraph. Used by _reassign_orphan_title_roots to
    # tell a truly "body-less" title apart from a block the layout
    # detector cut wide enough to also contain its own body text (see
    # that method's docstring).
    KICKER_MAX_LINES = 2
    KICKER_MAX_CHARS = 90

    # Max vertical gap between an unclaimed article_image and its own
    # claimed caption for _recover_unclaimed_images_via_caption below
    # to treat them as one photo/caption pair. Measured over every
    # figure/figure_caption pair in the saved corpus: a real pair
    # sits within ~240px of each other; the next-nearest UNRELATED
    # caption in the same column sits 5000+px away. This leaves a
    # wide margin on the "real pair" side while staying nowhere near
    # the "no real caption exists" tail.
    IMAGE_CAPTION_MAX_GAP = 250.0

    # Fraction of page height counted as the page's own header band
    # for _recover_unclaimed_title_graphics's masthead guard -- matches
    # PageCleaner.detect_page_header's own top_limit (0.15), the
    # broader "this is chrome, not a story" band, rather than
    # detect_masthead's narrower 0.12 (which only covers the masthead
    # proper on page 1).
    MASTHEAD_HEADER_BAND_RATIO = 0.15

    @staticmethod
    def _title_column_overlap(block_a, block_b):

        width_a = max(1.0, float(block_a.x2 - block_a.x1))
        width_b = max(1.0, float(block_b.x2 - block_b.x1))

        overlap = min(block_a.x2, block_b.x2) - max(block_a.x1, block_b.x1)

        return max(0.0, overlap) / min(width_a, width_b)

    @classmethod
    def _resolve_contested_blocks(
        cls,
        response_articles,
        block_lookup,
    ):
        """
        Decide the single owner of every block claimed by more than
        one article.

        Returns {block_id: winning_article_id} for contested blocks
        only. Blocks claimed exactly once are absent from the result
        and need no arbitration.
        """

        claim_counts = {}

        for article in response_articles:
            for block_id in article["blocks"]:
                claim_counts[block_id] = (
                    claim_counts.get(block_id, 0) + 1
                )

        contested = {
            block_id
            for block_id, count in claim_counts.items()
            if count > 1
        }

        if not contested:
            return {}

        resolved = {}

        # A geometric alternative must beat the model's OWN first
        # claim by more than this margin before it overrides that
        # claim. The model read the block's actual text and made a
        # semantic call; pure column/vertical-gap geometry should
        # correct that only when it is clearly wrong, not second-guess
        # a near-tie caused by one stray nearby element or a slightly
        # -off gap measurement. Below this margin, the first article to
        # claim the block in the model's own response order wins --
        # matching the safer "first claim wins" default this arbitration
        # only exists to improve on for clear-cut cases.
        ARBITRATION_MARGIN = 50.0

        for block_id in contested:

            block = block_lookup.get(block_id)

            if block is None:
                continue

            first_claim_article_id = None
            first_claim_distance = None

            best_article_id = None
            best_distance = None

            for article in response_articles:

                if block_id not in article["blocks"]:
                    continue

                # Compare against this article's UNCONTESTED blocks
                # only: using other contested blocks as evidence would
                # make the outcome depend on arbitration order.
                reference_ids = [
                    other_id
                    for other_id in article["blocks"]
                    if other_id != block_id
                    and other_id not in contested
                ]

                distances = [
                    cls._layout_distance(
                        block,
                        block_lookup[other_id],
                    )
                    for other_id in reference_ids
                    if other_id in block_lookup
                ]

                if not distances:
                    continue

                distance = min(distances)

                if first_claim_article_id is None:
                    first_claim_article_id = article["article_id"]
                    first_claim_distance = distance

                # Ties broken by article_id so the result is stable.
                if (
                    best_distance is None
                    or distance < best_distance
                    or (
                        distance == best_distance
                        and article["article_id"] < best_article_id
                    )
                ):
                    best_distance = distance
                    best_article_id = article["article_id"]

            if (
                best_article_id is not None
                and first_claim_article_id is not None
                and first_claim_article_id != best_article_id
                and first_claim_distance is not None
                and first_claim_distance <= best_distance + ARBITRATION_MARGIN
            ):
                resolved[block_id] = first_claim_article_id

            elif best_article_id is not None:
                resolved[block_id] = best_article_id

        return resolved

    @classmethod
    def _reassign_orphan_blocks_once(
        cls, response_articles, block_lookup, page_image=None
    ):
        """
        Compute every CANDIDATE reassignment for the current article
        state (see _reassign_orphan_blocks for how these candidates
        get applied one at a time): a block that is geometrically
        inconsistent with every other block in the one article that
        claims it, when some OTHER article's blocks fit it far
        better.

        Distinct from _resolve_contested_blocks: that resolves a
        block claimed by MULTIPLE articles (an ownership dispute).
        This handles a block claimed by exactly one article, where
        the claim itself doesn't hold up geometrically -- observed
        case: the model attributed a body paragraph to the wrong one
        of two topically-similar articles sharing adjacent columns,
        with no other article ever contesting the same block. Left
        uncorrected, that single block's bbox drags its article's
        boundary across the neighboring article's entire column.

        Both thresholds below are deliberately conservative (a large
        gap between them, not a hair-trigger margin): a legitimately
        wide, multi-column article's own blocks are never anywhere
        near ORPHAN_ISOLATION_FLOOR apart from ALL of their siblings
        at once (see the ArticleGrouper docstring examples), so this
        should only ever fire on a genuine mismatch.

        A candidate alternative is REJECTED outright (never becomes
        `best_article_id`, however good its distance score) when a
        printed rule line or border sits between the block and its
        nearest neighbor in that candidate article -- see
        _has_separator_between. This check runs BEFORE the distance
        comparison, so a hard separator can never be outscored by
        raw proximity: a confirmed editorial boundary is not a
        candidate at all, not merely a worse-scoring one. `page_image`
        is optional (None skips the check, same as every other
        pixel-probing pass in this codebase when no image is given).

        Returns {block_id: (new_article_id, distance)} for every
        candidate reassignment, relative to THIS call's own
        `response_articles` snapshot -- `distance` lets the caller
        pick the single best candidate across the whole page (see
        _reassign_orphan_blocks). Any block absent from the result is
        not currently a reassignment candidate.
        """

        # A block's own claim is trusted at face value below this --
        # only when it shares essentially NO horizontal overlap with
        # ANY other block in its own article (column_penalty alone
        # is already 90% of _layout_distance's max) does its claim
        # even become a candidate for reassignment.
        ORPHAN_ISOLATION_FLOOR = 90000.0

        # An alternative article only qualifies as a genuine home if
        # the block sits in real column proximity to it -- comfortably
        # below the column_penalty scale, not merely "less isolated"
        # than its own article.
        ORPHAN_ALTERNATIVE_CEILING = 10000.0

        # ...and if it is genuinely ADJACENT to that article, not
        # merely somewhere in the same column.
        #
        # ORPHAN_ALTERNATIVE_CEILING alone cannot express that.
        # _layout_distance is `vertical_gap + column_penalty` with
        # column_penalty on a 0-100000 scale and vertical_gap in RAW
        # PIXELS, so on any real page every vertical gap is already
        # far below the 10000 ceiling: the ceiling ends up testing the
        # column and nothing else, and a block a thousand pixels away
        # passes it as easily as one directly abutting.
        #
        # Confirmed on a real Assamese sports page: a body block was
        # moved 618px UP into an unrelated story (distance 1139) and a
        # photo was moved 1050px DOWN into another (distance 3682),
        # purely because each shared a column with its new article.
        # Since an article's crop is the bounding box of its blocks,
        # one such move stretched the boundary across two thirds of
        # the page and swallowed three other stories whole.
        #
        # A second confirmed case, on a Bengali study page: a narrow
        # right-hand column of its article's own Q&A was moved 477px
        # down into an unrelated study-plan article, whose boundary
        # then grew upward across the whole first article.
        #
        # Measured over every LLM-grouped article in three full
        # editions (16-page Assamese, 12-page Bengali, 4-page
        # Gujarati), the vertical gap between a block and its nearest
        # same-column sibling is 5px at the median, 273px at p99 and
        # 419px at the maximum. Over the same three editions this pass
        # produces 17 reassignment candidates: 9 sit at or below
        # 300px, and every single one above it is a confirmed
        # misplacement (434, 477, 618, 719, 802, 1050, 1955, 2189px).
        # A block that genuinely belongs to another article abuts it;
        # it does not sit a third of a page away.
        #
        # This bounds only what this REPAIR PASS may override. An
        # article the model itself grouped across a longer span is
        # untouched -- the pass simply stops relocating a block that
        # far on geometry alone.
        ORPHAN_MAX_VERTICAL_GAP = 300.0

        claim_counts = {}

        for article in response_articles:
            for block_id in article["blocks"]:
                claim_counts[block_id] = (
                    claim_counts.get(block_id, 0) + 1
                )

        # Contested blocks are a separate mechanism (see above) --
        # excluded both as candidates here and as reference points,
        # since their own ownership isn't settled yet.
        contested = {
            block_id
            for block_id, count in claim_counts.items()
            if count > 1
        }

        reassignments = {}

        for article in response_articles:

            own_id = article["article_id"]

            sibling_ids = [
                block_id
                for block_id in article["blocks"]
                if block_id not in contested
                and block_id in block_lookup
            ]

            for block_id in article["blocks"]:

                if block_id in contested:
                    continue

                block = block_lookup.get(block_id)

                if block is None:
                    continue

                own_reference_ids = [
                    other_id
                    for other_id in sibling_ids
                    if other_id != block_id
                ]

                if not own_reference_ids:
                    # Only block in its article -- no sibling to
                    # judge consistency against, so trust the claim.
                    continue

                own_distance = min(
                    cls._reassignment_distance(
                        block,
                        block_lookup[other_id],
                    )
                    for other_id in own_reference_ids
                )

                # A printed vertical divider between this block and
                # its OWN article's headline overrides the geometric
                # isolation floor below: the newspaper itself already
                # marked the two as separate editorial items, so the
                # model's claim doesn't get to stand on "it scores
                # close enough" alone. Confirmed on a real Odia
                # (Sambad, doc_000177 page 2) page: a 4-column lead
                # story's Column 4 photo was claimed by a single-
                # column Column 5 article, cutting through the lead
                # story's own headline -- the photo and that article's
                # headline sit on opposite sides of a printed rule
                # line, but were otherwise close enough in the raw
                # column-penalty metric to pass as a normal claim.
                own_title_ids = [
                    other_id
                    for other_id in own_reference_ids
                    if (
                        getattr(block_lookup[other_id], "cls", "")
                        or ""
                    ).strip().lower() == "title"
                ]

                separated_from_own_title = any(
                    cls._has_separator_between(
                        page_image,
                        block,
                        block_lookup[other_id],
                    )
                    for other_id in own_title_ids
                )

                if (
                    own_distance < ORPHAN_ISOLATION_FLOOR
                    and not separated_from_own_title
                ):
                    continue

                best_article_id = None
                best_distance = None
                best_vertical_gap = None

                for other_article in response_articles:

                    if other_article["article_id"] == own_id:
                        continue

                    candidate_ids = [
                        other_id
                        for other_id in other_article["blocks"]
                        if other_id not in contested
                        and other_id in block_lookup
                    ]

                    if not candidate_ids:
                        continue

                    nearest_id = min(
                        candidate_ids,
                        key=lambda other_id: cls._reassignment_distance(
                            block,
                            block_lookup[other_id],
                        ),
                    )

                    # HARD BOUNDARY CHECK -- runs BEFORE the distance
                    # comparison below, so a printed separator can
                    # never be outscored by raw proximity: a
                    # candidate article the newspaper itself already
                    # separated this block from is not a worse
                    # candidate, it is not a candidate at all. See
                    # _has_separator_between.
                    if cls._has_separator_between(
                        page_image,
                        block,
                        block_lookup[nearest_id],
                    ):
                        continue

                    distance = cls._reassignment_distance(
                        block,
                        block_lookup[nearest_id],
                    )

                    if (
                        best_distance is None
                        or distance < best_distance
                        or (
                            distance == best_distance
                            and other_article["article_id"]
                            < best_article_id
                        )
                    ):
                        best_distance = distance
                        best_article_id = other_article["article_id"]

                        # Measured against the SAME block the
                        # distance was won on, so the gap and the
                        # distance always describe one pairing.
                        best_vertical_gap = cls._vertical_gap(
                            block,
                            block_lookup[nearest_id],
                        )

                if (
                    best_article_id is not None
                    and best_distance < ORPHAN_ALTERNATIVE_CEILING
                    and best_vertical_gap is not None
                    and best_vertical_gap <= ORPHAN_MAX_VERTICAL_GAP
                ):
                    reassignments[block_id] = (best_article_id, best_distance)

        return reassignments

    # A block's own best-fit alternative can depend on ANOTHER block
    # that is itself a reassignment candidate -- confirmed on a real
    # Urdu page: block 33 (a banner-headline fragment) belongs with
    # article 5 (its own other fragment + photo), but article 5's OWN
    # body paragraph (block 5) was, at that same moment, the single
    # nearest thing to block 33 in article 5's ORIGINAL contents --
    # and symmetrically, block 33 was the nearest thing to block 5 in
    # ITS article's original contents. Deciding both moves from the
    # same frozen snapshot and applying them together swapped the two
    # articles' content instead of ever landing them together: article
    # 5 ended up missing its own body text, and article 6 ended up
    # holding two completely unrelated stories' body paragraphs fused
    # into one. Each individual decision was correct against the
    # snapshot it was computed from; only the compound, SIMULTANEOUS
    # effect was wrong.
    #
    # Capped defensively -- normal pages reassign only a handful of
    # blocks in total, so this is a generous ceiling on the number of
    # SINGLE moves this pass will ever make on one page, not a tuned
    # convergence-round count.
    MAX_ORPHAN_REASSIGNMENTS = 25

    @classmethod
    def _reassign_orphan_blocks(cls, response_articles, block_lookup, page_image=None):
        """
        State-aware wrapper around _reassign_orphan_blocks_once:

            1. Compute every candidate reassignment against the
               CURRENT article contents.
            2. Accept only the SINGLE best candidate (by distance)
               across the entire page.
            3. Apply it, updating current article membership.
            4. Recompute every candidate's score from scratch against
               that updated membership.
            5. Repeat for the next-best candidate.

        Never accepts two reassignments from the same frozen snapshot
        -- a target article's membership is always re-scored before
        the next block is allowed to move, so a block can never be
        moved out of a story on the strength of a neighbor that is
        itself only there because of a reassignment not yet applied
        (see MAX_ORPHAN_REASSIGNMENTS above for the confirmed
        real-page failure this prevents).

        Returns {block_id: FINAL new_article_id}, relative to the
        ORIGINAL `response_articles` passed in -- safe to apply once,
        via _apply_orphan_reassignments, against that original list.
        """

        working_articles = response_articles
        final_reassignments: dict = {}

        for _ in range(cls.MAX_ORPHAN_REASSIGNMENTS):

            candidates = cls._reassign_orphan_blocks_once(
                working_articles,
                block_lookup,
                page_image=page_image,
            )

            if not candidates:
                break

            # The SINGLE best candidate across the whole page -- ties
            # broken by block_id for a stable, deterministic result.
            best_block_id = min(
                candidates,
                key=lambda block_id: (
                    candidates[block_id][1],
                    block_id,
                ),
            )

            best_article_id, _best_distance = candidates[best_block_id]

            working_articles = cls._apply_orphan_reassignments(
                working_articles,
                {best_block_id: best_article_id},
            )

            final_reassignments[best_block_id] = best_article_id

        return final_reassignments

    @classmethod
    def _reassign_orphan_title_roots(cls, response_articles, block_lookup):
        """
        Merge a single-block, title-only article into the next
        headline's article below it in the same column.

        A body-less title the model gave its own single-block
        article is almost always a kicker/eyebrow line sitting just
        above the real headline, not an independent story -- the
        same shape local_grouper.py's offline path already resolves
        via its orphan-root cleanup (_find_next_root_below).
        _reassign_orphan_blocks above deliberately trusts a lone
        block's claim at face value ("no sibling to judge
        consistency against"), so it never catches this shape; this
        pass targets exactly the single-title-article case that one
        skips.

        Returns {block_id: target_article_id} for kicker blocks to
        move. An article is a merge candidate only when it has
        exactly one block and that block is title-class -- this
        never touches a caption, image, or body paragraph, so it
        can't repeat the over-eager nearest-ANY-geometry regression
        the unclaimed-block auto-attach was reverted for (see the
        "Report unclaimed content blocks" comment in build()).
        """

        reassignments = {}

        for article in response_articles:

            if len(article["blocks"]) != 1:
                continue

            block_id = article["blocks"][0]
            block = block_lookup.get(block_id)

            if block is None:
                continue

            if getattr(block, "role", None) != "article_title":
                continue

            # The model labels a block "article_title" whenever it
            # STARTS with a headline, even when the layout detector
            # cut the region wide enough to also contain that
            # headline's own body paragraph below it in the same
            # block -- this block is not actually "body-less", so
            # merging it away as a kicker would silently swallow an
            # entire separate story into its neighbour. Confirmed on
            # a real Urdu page (Siasat Daily): two single-block
            # articles -- "GST collection..." and "Congress workers
            # scuffle with police", each a complete story with its
            # own headline as the block's first line(s) -- were
            # merged into one article this way. A genuine kicker/
            # eyebrow line is one short line of text; anything longer
            # is body content the block already carries on its own.
            text = (getattr(block, "text", "") or "").strip()
            text_lines = [line for line in text.splitlines() if line.strip()]

            if len(text_lines) > cls.KICKER_MAX_LINES or len(text) > cls.KICKER_MAX_CHARS:
                continue

            best_article_id = None
            best_distance = None

            for other_article in response_articles:

                if other_article["article_id"] == article["article_id"]:
                    continue

                for other_id in other_article["blocks"]:

                    other_block = block_lookup.get(other_id)

                    if other_block is None:
                        continue

                    if getattr(other_block, "role", None) != "article_title":
                        continue

                    # Strictly below, same column -- the headline
                    # this kicker introduces in print.
                    if other_block.y1 < block.y2:
                        continue

                    if (
                        cls._title_column_overlap(block, other_block)
                        < cls.ORPHAN_TITLE_COLUMN_OVERLAP
                    ):
                        continue

                    distance = float(other_block.y1 - block.y2)

                    if (
                        best_distance is None
                        or distance < best_distance
                        or (
                            distance == best_distance
                            and other_article["article_id"] < best_article_id
                        )
                    ):
                        best_distance = distance
                        best_article_id = other_article["article_id"]

            if best_article_id is not None:
                reassignments[block_id] = best_article_id

        return reassignments

    @staticmethod
    def _apply_orphan_reassignments(response_articles, reassignments):
        """
        Move each reassigned block from its original article's
        "blocks" list to its new one, leaving every other article
        untouched. See _reassign_orphan_blocks for how targets are
        chosen.
        """

        if not reassignments:
            return response_articles

        additions = {}

        for block_id, target_article_id in reassignments.items():
            additions.setdefault(target_article_id, []).append(
                block_id
            )

        updated = []

        for article in response_articles:

            kept_blocks = [
                block_id
                for block_id in article["blocks"]
                if block_id not in reassignments
            ]

            kept_blocks.extend(
                additions.get(article["article_id"], [])
            )

            updated.append(
                {
                    **article,
                    "blocks": kept_blocks,
                }
            )

        return updated

    @classmethod
    def _recover_unclaimed_kicker_titles(cls, unclaimed_blocks, articles):
        """
        Fold a still-unclaimed lone title into the nearest article it
        is the headline (or kicker) FOR: preferentially another
        article's OWN title-class block sitting below it in the same
        column (a kicker/eyebrow line above an already-recognised
        headline); falling back, only when no such title-to-title
        match exists anywhere, to a currently HEADLINE-LESS article
        whose own topmost block (any role) sits directly below it in
        the same column (this block IS that article's missing
        headline, not a kicker for one).

        Distinct from _reassign_orphan_title_roots: that repairs a
        kicker the model DID list inside some (single-block)
        article. This handles the other failure shape -- the model
        never listed the kicker in ANY article's "blocks" array at
        all, so without this it only ever shows up in the
        unclaimed-blocks warning below and vanishes from the final
        output entirely.

        The headline-less fallback closes a real gap the title-to-
        title match alone cannot: confirmed on a real Urdu page (THE
        INQUILAB), a genuine headline with unreadable OCR text (role
        "unknown", cls "title" -- see _is_uncertain_title) sat 10px
        above its own article's first body paragraph, but that
        article had no OTHER title-class block for the loop below to
        match against at all -- an entire story's headline vanished
        from the final output with only a WARNING, never actually
        reattached.

        Both the title-to-title match and this fallback are bounded
        by IMAGE_CAPTION_MAX_GAP: a kicker/headline pair, or a
        headline/first-paragraph pair, is never more than a
        line-height or two apart in print, so neither anchor is
        strong enough evidence to justify an unbounded search. This
        was confirmed the hard way on the same Urdu page's page 2: a
        body paragraph mis-boxed as "title" by the layout detector
        (correctly hedged "unknown" by the grouping model) shared
        column alignment with an unrelated article's real title 404px
        below it and, once an unrelated bug in the
        has_own_unclaimed_body guard below was fixed, the title-to-
        title loop -- uncapped at the time -- happily merged it in.
        Capping both searches at IMAGE_CAPTION_MAX_GAP fixed that
        false match while leaving every confirmed-real pair (a kicker
        sitting well under 250px from its headline; block 73 above,
        10px from its body) untouched.

        Deliberately as narrow as _reassign_orphan_title_roots: only
        a title-class block with no unclaimed article_text sibling
        of its own directly below it (i.e. not a genuine standalone
        story the model dropped outright, which this must not
        swallow into an unrelated neighbor) is eligible. This cannot
        reproduce the nearest-ANY-geometry regression the unclaimed-
        block auto-attach was reverted for (see the "Report unclaimed
        content blocks" comment in build()) -- it never touches a
        caption or body paragraph, and never attaches sideways or
        upward.

        Mutates the matched Article in place (appends the block,
        re-sorts by reading_order). Returns the set of recovered
        block ids.
        """

        recovered = set()

        for block in unclaimed_blocks:

            role = getattr(block, "role", None)

            # Accept both a confidently-classified kicker and a
            # title-shaped block the model couldn't classify at all
            # (role "unknown", usually from unreadable OCR text) --
            # see _is_uncertain_title.
            if role != "article_title" and not cls._is_uncertain_title(
                block
            ):
                continue

            # "Directly below it" (see docstring) means genuinely
            # adjacent, not merely somewhere further down the same
            # column -- capped at IMAGE_CAPTION_MAX_GAP for the same
            # reason as every other proximity check in this class.
            # Confirmed on a real Urdu page: an unrelated unclaimed
            # paragraph sat 856px below an orphaned headline, sharing
            # its column purely by coincidence (both are narrow blocks
            # in a page with few columns) -- without this cap it read
            # as "this title has its own dropped body", permanently
            # blocking the headline-less-article recovery below even
            # though the two blocks belong to entirely different
            # stories.
            has_own_unclaimed_body = any(
                other is not block
                and getattr(other, "role", None) == "article_text"
                and other.y1 >= block.y2
                and (other.y1 - block.y2) <= cls.IMAGE_CAPTION_MAX_GAP
                and cls._title_column_overlap(block, other)
                >= cls.ORPHAN_TITLE_COLUMN_OVERLAP
                for other in unclaimed_blocks
            )

            if has_own_unclaimed_body:
                continue

            best_article = None
            best_distance = None

            for article in articles:

                for candidate in article.blocks:

                    if getattr(candidate, "role", None) != "article_title":
                        continue

                    if candidate.y1 < block.y2:
                        continue

                    if (
                        cls._title_column_overlap(block, candidate)
                        < cls.ORPHAN_TITLE_COLUMN_OVERLAP
                    ):
                        continue

                    distance = float(candidate.y1 - block.y2)

                    # A kicker/eyebrow line sits immediately above its
                    # own headline in print -- a real pair is never
                    # more than a line-height or two apart. Without
                    # this cap a title-class block far up the page can
                    # latch onto a distant, unrelated article's title
                    # sharing its column purely by coincidence; see
                    # the has_own_unclaimed_body cap above for a
                    # confirmed real case of exactly that coincidence
                    # (different guard, same underlying risk).
                    if distance > cls.IMAGE_CAPTION_MAX_GAP:
                        continue

                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_article = article

            if best_article is None:

                # No article anywhere already has a title-class block
                # positioned below this one -- try it as the MISSING
                # headline of a currently headline-less article
                # instead (see docstring). Only considered once the
                # title-to-title search above has fully failed, so a
                # genuine kicker/eyebrow match always wins over this
                # weaker, distance-capped fallback.

                for article in articles:

                    if not article.blocks:
                        continue

                    has_own_title = any(
                        getattr(existing, "role", None)
                        == "article_title"
                        for existing in article.blocks
                    )

                    if has_own_title:
                        continue

                    topmost = min(article.blocks, key=lambda b: b.y1)

                    if topmost.y1 < block.y2:
                        continue

                    if (
                        cls._title_column_overlap(block, topmost)
                        < cls.ORPHAN_TITLE_COLUMN_OVERLAP
                    ):
                        continue

                    distance = float(topmost.y1 - block.y2)

                    if distance > cls.IMAGE_CAPTION_MAX_GAP:
                        continue

                    if (
                        best_distance is None
                        or distance < best_distance
                    ):
                        best_distance = distance
                        best_article = article

            if best_article is None:
                continue

            best_article.blocks.append(block)
            best_article.block_ids.append(block.id)

            best_article.blocks.sort(
                key=lambda b: getattr(b, "reading_order", 0)
            )

            recovered.add(block.id)

        return recovered

    @classmethod
    def _recover_unclaimed_title_graphics(
        cls,
        unclaimed_blocks,
        articles,
        header_band_bottom=None,
        masthead_blocks=None,
    ):
        """
        Fold an unclaimed graphic-style headline banner into the
        article whose own topmost content sits directly beneath it in
        the same column.

        Some newspapers render a story's own headline as a colored
        graphic banner (custom typography over an art background)
        rather than plain OCR-able text. The layout detector correctly
        boxes it as a "figure", but with no real-world photo content
        to recognize and no prompt guidance distinguishing "this
        story's own stylized title" from "the newspaper's own
        recurring branding", the grouping model falls back to "logo"
        or "decoration" -- both hard-excluded from articles by
        IGNORE_ROLES, and so permanently invisible to every other
        repair pass (they never reach unclaimed_content_blocks in the
        first place -- see the caller). Confirmed on a real Tamil
        page: an electric-vehicle-jobs story's own banner headline sat
        immediately above the story's own lead photo, was tagged
        "logo", and the story's final crop started below it -- cutting
        its own headline out of its own boundary.

        Deliberately as narrow as _recover_unclaimed_kicker_titles:
        the caller already restricts candidates to cls == "figure"
        with role "logo"/"decoration"; this only additionally accepts
        one sitting directly ABOVE -- never beside, never overlapping
        -- some article's own block, with substantial column overlap
        and within IMAGE_CAPTION_MAX_GAP, and only when it is the
        single nearest such article. A genuine page-level logo (a
        masthead-adjacent section flag, say) sitting well above the
        page's first real story is never this close to any article's
        own content, so this cannot misattach real page chrome the
        way an unbounded nearest-geometry attach would (see build()'s
        "Report unclaimed content blocks" comment for why that was
        reverted elsewhere).

        Guards against a SPECIFIC over-reach of "sits directly above
        some article": the newspaper's own masthead/nameplate logo,
        sitting at the very top of the page above the lead story, also
        satisfies "directly above, same column, close enough" once
        that lead story has no headline of its own to compete with it.
        Confirmed on a real Odia (Sambad, doc_000177 page 1) page: the
        lead story had no headline block, and the masthead logo above
        it (never textually recognized as a masthead by
        PageCleaner.detect_masthead, since that check matches known
        paper names and this masthead is a pure graphic) was folded in
        as the story's own "title graphic", stretching the article's
        crop up to swallow the paper's nameplate. `header_band_bottom`
        (a y-coordinate) and `masthead_blocks` (already positively
        identified by PageCleaner) both veto that: a candidate block
        sitting above the header band, or overlapping a confirmed
        masthead block, is never eligible here, regardless of which
        article's content happens to sit beneath it.

        Mutates the matched Article in place (appends the block,
        re-sorts by reading_order). Returns the set of recovered
        block ids.
        """

        def overlaps_masthead(candidate_block):

            if not masthead_blocks:
                return False

            for masthead in masthead_blocks:

                ix1 = max(candidate_block.x1, masthead.x1)
                iy1 = max(candidate_block.y1, masthead.y1)
                ix2 = min(candidate_block.x2, masthead.x2)
                iy2 = min(candidate_block.y2, masthead.y2)

                if ix2 > ix1 and iy2 > iy1:
                    return True

            return False

        recovered = set()

        for block in unclaimed_blocks:

            if (
                getattr(block, "is_global", False)
                or (getattr(block, "type", "") or "").strip().lower()
                == "masthead"
            ):
                continue

            if (
                header_band_bottom is not None
                and block.y1 < header_band_bottom
            ):
                continue

            if overlaps_masthead(block):
                continue

            best_article = None
            best_distance = None

            for article in articles:

                for candidate in article.blocks:

                    if candidate.y1 < block.y2:
                        continue

                    if (
                        cls._title_column_overlap(block, candidate)
                        < cls.ORPHAN_TITLE_COLUMN_OVERLAP
                    ):
                        continue

                    distance = float(candidate.y1 - block.y2)

                    if distance > cls.IMAGE_CAPTION_MAX_GAP:
                        continue

                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_article = article

            if best_article is None:
                continue

            best_article.blocks.append(block)
            best_article.block_ids.append(block.id)

            best_article.blocks.sort(
                key=lambda b: getattr(b, "reading_order", 0)
            )

            recovered.add(block.id)

        return recovered

    @classmethod
    def _recover_unclaimed_images_via_caption(
        cls,
        unclaimed_blocks,
        articles,
        block_lookup,
    ):
        """
        Attach an unclaimed article_image back to whichever article
        claimed ITS OWN caption.

        The model can tag a photo with the correct "article_image"
        role and still never list its id in ANY article's "blocks"
        array -- confirmed on a real page: a photo's own
        figure_caption was correctly claimed by the article directly
        below it, but the photo itself never appeared in any
        article's block list at all, so it vanished from the final
        crop entirely even though its caption -- and the headline
        below that -- rendered fine. The generic unclaimed-content
        warning in build() surfaces this but does not recover it.

        Deliberately anchored on the caption, not on bare nearest-
        geometry: this is exactly the reverted whole-page nearest-ANY
        attach (see the "Report unclaimed content blocks" comment in
        build(), which merged 35% of one page into a single blob),
        EXCEPT restricted to a caption's own governing article. A
        caption is only ever printed for the ONE photo directly
        beside it, never shared between stories, so this can only
        ever fold an image into the story its own caption was already
        placed in -- never merge two unrelated stories the way plain
        nearest-distance would.

        IMAGE_CAPTION_MAX_GAP keeps this to genuinely adjacent pairs
        -- see its definition above.

        Mutates the matched Article in place (appends the block,
        re-sorts by reading_order). Returns the set of recovered
        block ids.
        """

        if not unclaimed_blocks:
            return set()

        caption_owner = {}

        for article in articles:
            for candidate in article.blocks:
                if getattr(candidate, "role", None) == "caption":
                    caption_owner[candidate.id] = article

        if not caption_owner:
            return set()

        recovered = set()

        for block in unclaimed_blocks:

            if getattr(block, "role", None) != "article_image":
                continue

            best_article = None
            best_gap = None

            for caption_id, owner_article in caption_owner.items():

                caption_block = block_lookup.get(caption_id)

                if caption_block is None:
                    continue

                if (
                    cls._title_column_overlap(block, caption_block)
                    < cls.ORPHAN_TITLE_COLUMN_OVERLAP
                ):
                    continue

                gap = cls._vertical_gap(block, caption_block)

                if gap > cls.IMAGE_CAPTION_MAX_GAP:
                    continue

                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_article = owner_article

            if best_article is None:
                continue

            best_article.blocks.append(block)
            best_article.block_ids.append(block.id)

            best_article.blocks.sort(
                key=lambda b: getattr(b, "reading_order", 0)
            )

            recovered.add(block.id)

        return recovered

    # A block recovered by footprint containment must sit within this
    # many pixels of an article's own outer bbox to count as
    # "immediately adjacent" when it isn't already strictly inside it
    # -- generous enough to cover ordinary column-gutter/margin slack
    # around a real footprint without approaching the scale of a
    # genuinely separate story sitting in the next column or block.
    FOOTPRINT_ADJACENT_TOLERANCE = 50.0

    @classmethod
    def _recover_unclaimed_blocks_by_footprint(
        cls,
        unclaimed_blocks,
        articles,
    ):
        """
        Attach a still-unclaimed non-chrome block to the ONE article
        whose own outer footprint (the union bbox of its already-
        claimed blocks) already geometrically contains it -- the last
        resort after every more specific recovery above (kicker,
        title-graphic, image-via-caption) has had its narrower chance.

        Confirmed on a real Malayalam page (doc_000166, page 2): a
        story's own byline (role "byline") and its own lead paragraph
        (role "article_text") were both correctly role-classified by
        the grouping model but never listed in ANY article's "blocks"
        array -- neither a kicker-shaped title nor an image-via-
        caption match, so both fell through every existing recovery
        pass and stayed permanently unclaimed, printed only as a
        WARNING. Both blocks sit, in x AND y, entirely inside the
        bounding rectangle of the one real article whose headline and
        remaining body/photo already surround them on every side --
        exactly the shape a plain rectangular crop (see
        boundary_builder.py) already includes on the page regardless,
        but which the article's own block-membership list (and
        anything downstream that reads it, e.g. text assembly) was
        still silently missing.

        Containment (block bbox falls entirely within the article's
        own footprint, plus FOOTPRINT_ADJACENT_TOLERANCE of slack for
        a block sitting just outside it) is deliberately a much
        narrower question than the reverted whole-page nearest-ANY
        attach (see the "Report unclaimed content blocks" comment in
        build(), which merged 35% of one page into a single blob):
        that attach considered every block a candidate for whichever
        article was geometrically NEAREST, even a page away; this only
        ever considers a block already sitting inside (or barely
        outside) ONE specific article's own established boundary.

        Ambiguous when more than one article's footprint contains the
        block (can happen for an L-shaped/overlapping-rectangle
        article, see boundary_decomposer.py) -- skipped rather than
        guessed, same conservative default as every other recovery
        pass here.

        Deliberately does NOT also veto on _has_separator_between the
        way _reassign_orphan_blocks_once does: confirmed on the same
        real Malayalam page, a byline's own printed underline (routine
        typographic styling directly under a byline, not a story
        divider) sits in the immediate gap between the byline and its
        own very next paragraph, and reads as a hard "rule line" to
        that same-purposed probe -- which would permanently veto
        recovering a block from the ONLY candidate that could ever
        claim it. That veto earns its keep in _reassign_orphan_blocks_
        once because it is choosing between competing ALTERNATIVE
        articles and a wrong choice silently reattributes content;
        here there is exactly one candidate to begin with (the
        ambiguity check above already rejects every case where a
        choice would even need making), so the failure mode a
        separator veto guards against does not apply.

        Mutates the matched Article in place (appends the block,
        re-sorts by reading_order). Returns the set of recovered
        block ids.
        """

        if not unclaimed_blocks:
            return set()

        recovered = set()

        for block in unclaimed_blocks:

            candidates = []

            for article in articles:

                if not article.blocks:
                    continue

                min_x = min(b.x1 for b in article.blocks)
                min_y = min(b.y1 for b in article.blocks)
                max_x = max(b.x2 for b in article.blocks)
                max_y = max(b.y2 for b in article.blocks)

                inside = (
                    block.x1 >= min_x - cls.FOOTPRINT_ADJACENT_TOLERANCE
                    and block.x2 <= max_x + cls.FOOTPRINT_ADJACENT_TOLERANCE
                    and block.y1 >= min_y - cls.FOOTPRINT_ADJACENT_TOLERANCE
                    and block.y2 <= max_y + cls.FOOTPRINT_ADJACENT_TOLERANCE
                )

                if not inside:
                    continue

                candidates.append(article)

            if len(candidates) != 1:
                continue

            best_article = candidates[0]

            best_article.blocks.append(block)
            best_article.block_ids.append(block.id)

            best_article.blocks.sort(
                key=lambda b: getattr(b, "reading_order", 0)
            )

            recovered.add(block.id)

        return recovered

    def build(
        self,
        parsed_response,
        blocks,
        use_contested_block_arbitration: bool = True,
        use_orphan_block_reassignment: bool = True,
        use_orphan_title_root_repair: bool = True,
        use_unclaimed_kicker_recovery: bool = True,
        use_unclaimed_image_recovery: bool = True,
        use_unclaimed_footprint_recovery: bool = True,
        page_image=None,
    ):

        #
        # Fast lookup
        #

        block_lookup = {

            block.id: block

            for block in blocks

        }

        articles = []

        removed = 0

        duplicate_conflicts = 0

        #
        # A block may only belong to ONE article. The model is asked
        # for exclusive ownership, but is not always reliable about
        # it on dense pages -- without this guard a block claimed by
        # multiple articles makes their boundaries overlap/merge in
        # BoundaryBuilder.
        #
        # Contested blocks are resolved by page LAYOUT rather than by
        # response order. "First claim wins" was observed handing a
        # story's only body paragraph to an unrelated article listed
        # earlier in the response, which left the real owner as a
        # headline-only article -- and a headline-only crop then makes
        # the downstream extractor fabricate body text for it. The
        # owner is instead the claiming article whose other blocks sit
        # closest to the contested block in the same column, which for
        # newspaper layout is the article whose headline/body directly
        # abuts it.
        #

        orphan_reassignments = {}
        orphan_title_reassignments = {}

        # Orphan-block reassignment corrects a block that is
        # geometrically inconsistent with the one article that
        # (uniquely) claims it -- a raw model mistake rather than a
        # per-language policy. Opted into per language (see
        # pipeline/languages/): every language enables it except
        # English, which matches the reference pipeline that has no
        # such pass.
        if use_orphan_block_reassignment:

            orphan_reassignments = self._reassign_orphan_blocks(
                parsed_response["articles"],
                block_lookup,
                page_image=page_image,
            )

            response_articles = self._apply_orphan_reassignments(
                parsed_response["articles"],
                orphan_reassignments,
            )

        else:

            response_articles = parsed_response["articles"]

        # Kicker/eyebrow-line repair, same rationale and same
        # per-language opt-in as orphan-block reassignment above: a
        # body-less title stranded in its own single-block article is
        # a raw model mistake. See _reassign_orphan_title_roots.
        if use_orphan_title_root_repair:

            orphan_title_reassignments = (
                self._reassign_orphan_title_roots(
                    response_articles,
                    block_lookup,
                )
            )

            response_articles = self._apply_orphan_reassignments(
                response_articles,
                orphan_title_reassignments,
            )

        # Languages that don't opt into contested-block arbitration
        # (see pipeline/languages/) use repo B's original policy
        # here: first claim wins, no layout-distance arbitration --
        # an empty resolved_owner map means the ownership check below
        # never overrides the natural first-claim-wins order the
        # claimed_block_ids loop already provides on its own.
        resolved_owner = (
            self._resolve_contested_blocks(
                response_articles,
                block_lookup,
            )
            if use_contested_block_arbitration
            else {}
        )

        claimed_block_ids = set()

        #
        # Build each article
        #

        for article in response_articles:

            article_blocks = []

            article_block_ids = []

            #
            # WEATHER MISCLASSIFICATION SAFETY NET
            #
            # "weather" is meant for a small numbers-only forecast
            # panel, but the model sometimes tags an entire narrative
            # news report about rain/monsoon conditions "weather"
            # too (see the grouping prompts' WEATHER section) while
            # still, correctly, listing that block inside a real
            # article alongside a genuine article_title and
            # article_text. IGNORE_ROLES is meant for page chrome
            # (masthead, advertisement, etc.) that never belongs in
            # an article regardless of context -- it should not also
            # amputate real content the model's OWN article grouping
            # already vouches for by surrounding it with a real
            # headline and body text. Scoped to "weather" only: every
            # other IGNORE_ROLES role (advertisement, masthead, ...)
            # stays a hard exclusion, per the prompts' own "NEVER
            # place advertisements inside news articles".
            #
            article_role_values = {
                getattr(block_lookup.get(bid), "role", None)
                for bid in article["blocks"]
            }

            article_has_real_story = (
                "article_title" in article_role_values
                and "article_text" in article_role_values
            )

            for block_id in article["blocks"]:

                if block_id in claimed_block_ids:

                    duplicate_conflicts += 1

                    continue

                #
                # Contested block: only its layout-resolved owner
                # keeps it (see _resolve_contested_blocks).
                #

                owner = resolved_owner.get(block_id)

                if (
                    owner is not None
                    and owner != article["article_id"]
                ):

                    duplicate_conflicts += 1

                    continue

                block = block_lookup.get(block_id)

                if block is None:
                    continue

                #
                # GeminiParser only sets block.role for IDs that
                # appear in response["blocks"]. The model sometimes
                # lists a block inside an article's "blocks" array
                # without also giving it a role classification --
                # in that case the attribute is simply missing here,
                # not "unknown" by the model's own judgement. That
                # explicit article membership is a stronger, more
                # specific signal than a missing role, so such a
                # block is treated as valid content rather than
                # silently dropped via IGNORE_ROLES.
                #

                has_role = hasattr(
                    block,
                    "role",
                )

                role = getattr(
                    block,
                    "role",
                    "unknown",
                )

                #
                # Ignore non-article blocks
                #

                if (
                    has_role
                    and role in self.IGNORE_ROLES
                    and not (
                        role == "weather" and article_has_real_story
                    )
                ):

                    removed += 1

                    continue

                claimed_block_ids.add(block_id)

                article_blocks.append(block)

                article_block_ids.append(block_id)

            #
            # Skip empty articles
            #

            if not article_blocks:
                continue

            #
            # Preserve reading order
            #

            article_blocks.sort(

                key=lambda b:

                getattr(

                    b,

                    "reading_order",

                    0,

                )

            )

            articles.append(

                Article(

                    article_id=article["article_id"],

                    blocks=article_blocks,

                    block_ids=article_block_ids,

                    confidence=1.0,

                )

            )

        #
        # Report unclaimed content blocks. The model sometimes gives a
        # block a genuine content role (e.g. caption, article_image,
        # article_text) but never includes that block's id in any
        # article's "blocks" list -- the loop above only ever visits
        # blocks that appear in some article, so such blocks are
        # silently missing from the final output. Attaching them by
        # nearest-geometry was tried and reverted: when the model
        # leaves a large fraction of a page unclaimed (observed up to
        # ~35% on a dense page), nearest-article merges unrelated
        # stories into one giant block instead of recovering a stray
        # caption. Surface it instead so it's visible rather than
        # silently corrupting an otherwise-correct grouping.
        #

        unclaimed_content_blocks = [
            block
            for block in blocks
            if block.id not in claimed_block_ids
            and (
                getattr(block, "role", "unknown") not in self.IGNORE_ROLES
                or self._is_uncertain_title(block)
            )
        ]

        # Per-language opt-in (see pipeline/languages/): enabled for
        # every language except English, which matches the reference
        # pipeline that leaves unclaimed kickers unclaimed.
        recovered_kicker_ids = (
            self._recover_unclaimed_kicker_titles(
                unclaimed_content_blocks,
                articles,
            )
            if use_unclaimed_kicker_recovery
            else set()
        )

        if recovered_kicker_ids:

            claimed_block_ids.update(recovered_kicker_ids)

            unclaimed_content_blocks = [
                block
                for block in unclaimed_content_blocks
                if block.id not in recovered_kicker_ids
            ]

        # A "logo"/"decoration" role is excluded from
        # unclaimed_content_blocks above by the IGNORE_ROLES filter --
        # by design, since most blocks with those roles really are
        # page chrome. Computed as its own separate pool, restricted
        # to cls == "figure" so a plain-text/title block that got
        # "decoration" stays excluded exactly as IGNORE_ROLES intends;
        # see _recover_unclaimed_title_graphics for why a figure-shaped
        # exception is needed. Same per-language opt-in as the kicker
        # recovery above -- this is the same shape of repair (an
        # unclaimed headline-role block reattached to the article
        # beneath it), just for a graphic banner instead of OCR text.
        unclaimed_graphic_blocks = [
            block
            for block in blocks
            if block.id not in claimed_block_ids
            and (getattr(block, "role", None) or "").strip().lower()
            in ("logo", "decoration")
            and (getattr(block, "cls", "") or "").strip().lower()
            == "figure"
        ]

        # Masthead guard for _recover_unclaimed_title_graphics below --
        # see that method's docstring. A true masthead is already
        # flagged by PageCleaner.detect_masthead (block.type ==
        # "masthead"), but that only fires when the paper's NAME is
        # recognized in its OCR text; a masthead rendered as a pure
        # graphic (no matching text) is invisible to that check, so a
        # positional header-band cutoff is also needed. page_height
        # isn't passed to build(); estimated the same way
        # detect_page_header estimates page WIDTH from the blocks
        # themselves.
        estimated_page_height = max(
            (getattr(b, "y2", 0) for b in blocks),
            default=0,
        )

        header_band_bottom = (
            estimated_page_height * self.MASTHEAD_HEADER_BAND_RATIO
            if estimated_page_height
            else None
        )

        masthead_blocks = [
            b
            for b in blocks
            if (getattr(b, "type", "") or "").strip().lower()
            == "masthead"
        ]

        recovered_title_graphic_ids = (
            self._recover_unclaimed_title_graphics(
                unclaimed_graphic_blocks,
                articles,
                header_band_bottom=header_band_bottom,
                masthead_blocks=masthead_blocks,
            )
            if use_unclaimed_kicker_recovery
            else set()
        )

        if recovered_title_graphic_ids:

            claimed_block_ids.update(recovered_title_graphic_ids)

        # Per-language opt-in (see pipeline/languages/): enabled for
        # every language except English, matching every other repair
        # pass here.
        recovered_image_ids = (
            self._recover_unclaimed_images_via_caption(
                unclaimed_content_blocks,
                articles,
                block_lookup,
            )
            if use_unclaimed_image_recovery
            else set()
        )

        if recovered_image_ids:

            claimed_block_ids.update(recovered_image_ids)

            unclaimed_content_blocks = [
                block
                for block in unclaimed_content_blocks
                if block.id not in recovered_image_ids
            ]

        # Last-resort catch-all, after every more specific recovery
        # above has had its narrower chance: a block still unclaimed
        # at this point but already sitting inside (or barely outside)
        # one article's own established footprint. Same per-language
        # opt-in as the other unclaimed-content recovery passes above.
        recovered_footprint_ids = (
            self._recover_unclaimed_blocks_by_footprint(
                unclaimed_content_blocks,
                articles,
            )
            if use_unclaimed_footprint_recovery
            else set()
        )

        if recovered_footprint_ids:

            claimed_block_ids.update(recovered_footprint_ids)

            unclaimed_content_blocks = [
                block
                for block in unclaimed_content_blocks
                if block.id not in recovered_footprint_ids
            ]

        print()

        print("=" * 60)

        print("ARTICLE GROUPER")

        print("=" * 60)

        print(f"Articles Built : {len(articles)}")

        print(f"Ignored Blocks : {removed}")

        if unclaimed_content_blocks:

            print(
                f"WARNING: {len(unclaimed_content_blocks)} content "
                "block(s) got a real role but were not claimed by any "
                "article -- missing from the final output:"
            )

            for block in unclaimed_content_blocks:

                print(
                    f"  Block {block.id} "
                    f"(role={getattr(block, 'role', 'unknown')})"
                )

        if duplicate_conflicts:

            print(
                f"WARNING: Dropped {duplicate_conflicts} duplicate "
                "block claim(s) -- model assigned block(s) to more "
                "than one article."
            )

        if orphan_reassignments:

            print(
                f"WARNING: Reassigned {len(orphan_reassignments)} "
                "orphan block(s) -- claimed by one article but "
                "geometrically inconsistent with it, moved to a "
                "far-better-fitting article instead: "
                f"{orphan_reassignments}"
            )

        if orphan_title_reassignments:

            print(
                f"WARNING: Reassigned {len(orphan_title_reassignments)} "
                "kicker/eyebrow title(s) -- given their own single-"
                "block article by the model, merged into the "
                "headline's article that follows them instead: "
                f"{orphan_title_reassignments}"
            )

        if recovered_kicker_ids:

            print(
                f"WARNING: Recovered {len(recovered_kicker_ids)} "
                "unclaimed kicker/eyebrow title(s) -- never listed "
                "in any article by the model, merged into the "
                "headline's article that follows them instead: "
                f"{sorted(recovered_kicker_ids)}"
            )

        if recovered_title_graphic_ids:

            print(
                f"WARNING: Recovered {len(recovered_title_graphic_ids)} "
                "unclaimed title-graphic(s) -- tagged logo/decoration "
                "by the model, merged into the article directly "
                "beneath them instead: "
                f"{sorted(recovered_title_graphic_ids)}"
            )

        if recovered_image_ids:

            print(
                f"WARNING: Recovered {len(recovered_image_ids)} "
                "unclaimed image(s) -- never listed in any article "
                "by the model, merged into whichever article claimed "
                "their own caption instead: "
                f"{sorted(recovered_image_ids)}"
            )

        if recovered_footprint_ids:

            print(
                f"WARNING: Recovered {len(recovered_footprint_ids)} "
                "unclaimed block(s) -- never listed in any article by "
                "the model, merged into the one article whose own "
                "footprint already geometrically contains them: "
                f"{sorted(recovered_footprint_ids)}"
            )

        print()

        for article in articles:

            print(

                f"Article {article.article_id} "

                f"({len(article.blocks)} blocks)"

            )

        print("=" * 60)

        print()

        return articles