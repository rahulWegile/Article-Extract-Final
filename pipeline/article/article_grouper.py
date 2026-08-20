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
        # BoundaryBuilder. First article to claim a block (in response
        # order) keeps it; later claims are dropped.
        #

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

                block = block_lookup.get(block_id)

                if block is None:
                    continue

                role = getattr(
                    block,
                    "role",
                    "unknown",
                )

                #
                # Ignore non-article blocks
                #

                if role in self.IGNORE_ROLES:

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
        # Recover unclaimed content blocks. The model sometimes gives a
        # block a genuine content role (e.g. caption, article_image,
        # article_text) but never includes that block's id in any
        # article's "blocks" list -- the loop above only ever visits
        # blocks that appear in some article, so such blocks would
        # otherwise be silently missing from the final output.
        #
        # Attaching them to the nearest EXISTING article was tried and
        # reverted: when the model leaves a large fraction of a page
        # unclaimed (observed up to ~35% on a dense page), that merges
        # unrelated stories into one giant block instead of recovering
        # a stray caption. Instead, give each unclaimed block its own
        # standalone article -- this can never grow or alter an
        # existing article (zero risk of the same regression), it just
        # ensures nothing is left with no box at all. A distinguishing
        # article_id range (100000+) and a lower confidence mark these
        # as a fallback rather than a real model-confirmed grouping.
        #

        unclaimed_content_blocks = [
            block
            for block in blocks
            if block.id not in claimed_block_ids
            and getattr(block, "role", "unknown") not in self.IGNORE_ROLES
        ]

        for block in unclaimed_content_blocks:

            articles.append(
                Article(
                    article_id=100000 + block.id,
                    blocks=[block],
                    block_ids=[block.id],
                    confidence=0.5,
                )
            )

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