from dataclasses import dataclass
from typing import List


@dataclass
class Article:

    article_id: int

    blocks: List

    block_ids: List[int]

    confidence: float = 1.0


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

        if block_a.y2 <= block_b.y1:
            vertical_gap = float(block_b.y1 - block_a.y2)
        elif block_b.y2 <= block_a.y1:
            vertical_gap = float(block_a.y1 - block_b.y2)
        else:
            vertical_gap = 0.0

        # A different column is never the better explanation for
        # ownership, so anything poorly overlapped horizontally gets a
        # penalty larger than any plausible in-column vertical gap.
        column_penalty = (1.0 - min(1.0, overlap_ratio)) * 100000.0

        return vertical_gap + column_penalty

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

                    distance = min(
                        cls._layout_distance(
                            block,
                            block_lookup[other_id],
                        )
                        for other_id in candidate_ids
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

                if (
                    best_article_id is not None
                    and best_distance < ORPHAN_ALTERNATIVE_CEILING
                ):
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

    def build(
        self,
        parsed_response,
        blocks,
        is_hindi: bool = True,
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

        # Orphan-block reassignment runs for BOTH languages, unlike
        # the contested-block arbitration below: it corrects a block
        # that is geometrically inconsistent with the one article
        # that (uniquely) claims it, which is a raw model mistake in
        # either extraction path, not an English/Hindi policy
        # difference.
        orphan_reassignments = self._reassign_orphan_blocks(
            parsed_response["articles"],
            block_lookup,
        )

        response_articles = self._apply_orphan_reassignments(
            parsed_response["articles"],
            orphan_reassignments,
        )

        # English documents use repo B's original policy here: first
        # claim wins, no layout-distance arbitration -- an empty
        # resolved_owner map means the ownership check below never
        # overrides the natural first-claim-wins order the
        # claimed_block_ids loop already provides on its own.
        resolved_owner = (
            self._resolve_contested_blocks(
                response_articles,
                block_lookup,
            )
            if is_hindi
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
            and getattr(block, "role", "unknown") not in self.IGNORE_ROLES
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

        print()

        for article in articles:

            print(

                f"Article {article.article_id} "

                f"({len(article.blocks)} blocks)"

            )

        print("=" * 60)

        print()

        return articles