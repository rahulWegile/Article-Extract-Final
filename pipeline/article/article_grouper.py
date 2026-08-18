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

        #
        # Build each article
        #

        for article in parsed_response["articles"]:

            article_blocks = []

            article_block_ids = []

            for block_id in article["blocks"]:

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

        print()

        print("=" * 60)

        print("ARTICLE GROUPER")

        print("=" * 60)

        print(f"Articles Built : {len(articles)}")

        print(f"Ignored Blocks : {removed}")

        print()

        for article in articles:

            print(

                f"Article {article.article_id} "

                f"({len(article.blocks)} blocks)"

            )

        print("=" * 60)

        print()

        return articles