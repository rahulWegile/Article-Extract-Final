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

        overlap_x = min(block_a.x2, block_b.x2) - max(
            block_a.x1, block_b.x1
        )

        overlap_ratio = max(0.0, overlap_x) / min(a_width, b_width)

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

        for block_id in contested:

            block = block_lookup.get(block_id)

            if block is None:
                continue

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

            if best_article_id is not None:
                resolved[block_id] = best_article_id

        return resolved

    def build(
        self,
        parsed_response,
        blocks,
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

        resolved_owner = self._resolve_contested_blocks(
            parsed_response["articles"],
            block_lookup,
        )

        claimed_block_ids = set()

        #
        # Build each article
        #

        for article in parsed_response["articles"]:

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

        print()

        for article in articles:

            print(

                f"Article {article.article_id} "

                f"({len(article.blocks)} blocks)"

            )

        print("=" * 60)

        print()

        return articles