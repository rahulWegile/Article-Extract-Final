from dataclasses import dataclass, field
from typing import List, Tuple


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

    # Fraction of the narrower block's width that must overlap
    # horizontally for two blocks to count as sharing a column, for
    # the orphan-title-root repairs below. Matches
    # local_grouper.py's COLUMN_OVERLAP -- same geometric question
    # (does a title-only block belong to the headline below it in
    # print), just re-expressed against Block objects instead of
    # page_json dicts.
    ORPHAN_TITLE_COLUMN_OVERLAP = 0.35

    # Max vertical gap between an unclaimed article_image and its own
    # claimed caption for _recover_unclaimed_images_via_caption below
    # to treat them as one photo/caption pair. Measured over every
    # figure/figure_caption pair in the saved corpus: a real pair
    # sits within ~240px of each other; the next-nearest UNRELATED
    # caption in the same column sits 5000+px away. This leaves a
    # wide margin on the "real pair" side while staying nowhere near
    # the "no real caption exists" tail.
    IMAGE_CAPTION_MAX_GAP = 250.0

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
    def _reassign_orphan_blocks(cls, response_articles, block_lookup):
        """
        Reassign a block that is geometrically inconsistent with
        every other block in the one article that claims it, when
        some OTHER article's blocks fit it far better.

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

        Returns {block_id: new_article_id} for blocks to move. Any
        block absent from the result keeps its original article.
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
                    cls._layout_distance(
                        block,
                        block_lookup[other_id],
                    )
                    for other_id in own_reference_ids
                )

                if own_distance < ORPHAN_ISOLATION_FLOOR:
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
                        key=lambda other_id: cls._layout_distance(
                            block,
                            block_lookup[other_id],
                        ),
                    )

                    distance = cls._layout_distance(
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
                    reassignments[block_id] = best_article_id

        return reassignments

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
        Fold a still-unclaimed lone title into the nearest article
        whose OWN title-class block sits below it in the same
        column.

        Distinct from _reassign_orphan_title_roots: that repairs a
        kicker the model DID list inside some (single-block)
        article. This handles the other failure shape -- the model
        never listed the kicker in ANY article's "blocks" array at
        all, so without this it only ever shows up in the
        unclaimed-blocks warning below and vanishes from the final
        output entirely.

        Deliberately as narrow as _reassign_orphan_title_roots: only
        a title-class block with no unclaimed article_text sibling
        of its own directly below it (i.e. not a genuine standalone
        story the model dropped outright, which this must not
        swallow into an unrelated neighbor) is eligible, and it can
        only attach to another article's own title-class block. This
        cannot reproduce the nearest-ANY-geometry regression the
        unclaimed-block auto-attach was reverted for (see the
        "Report unclaimed content blocks" comment in build()) -- it
        never touches a caption or body paragraph, and never
        attaches sideways or upward.

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

            has_own_unclaimed_body = any(
                other is not block
                and getattr(other, "role", None) == "article_text"
                and other.y1 >= block.y2
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

    def build(
        self,
        parsed_response,
        blocks,
        use_contested_block_arbitration: bool = True,
        use_orphan_block_reassignment: bool = True,
        use_orphan_title_root_repair: bool = True,
        use_unclaimed_kicker_recovery: bool = True,
        use_unclaimed_image_recovery: bool = True,
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

                if has_role and role in self.IGNORE_ROLES:

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

        if recovered_image_ids:

            print(
                f"WARNING: Recovered {len(recovered_image_ids)} "
                "unclaimed image(s) -- never listed in any article "
                "by the model, merged into whichever article claimed "
                "their own caption instead: "
                f"{sorted(recovered_image_ids)}"
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