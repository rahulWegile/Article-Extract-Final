from dataclasses import dataclass
from typing import List

from pipeline.article.article_grouper import Article


@dataclass
class ArticleBoundary:
    """
    Final verified article boundary.

    The boundary is calculated from all blocks
    belonging to one Gemini-grouped article.
    """

    article_id: int

    x1: int
    y1: int
    x2: int
    y2: int

    width: int
    height: int

    area: int

    block_ids: List[int]

    confidence: float = 1.0


class BoundaryBuilder:
    """
    Build one final article boundary from each
    Gemini article.

    Geometry is calculated from the article's blocks.
    """

    def build(
        self,
        articles: List[Article],
    ):

        boundaries = []

        for article in articles:

            if not article.blocks:
                continue

            # -------------------------------------------------
            # Boundary geometry
            # -------------------------------------------------

            x1 = min(
                block.x1
                for block in article.blocks
            )

            y1 = min(
                block.y1
                for block in article.blocks
            )

            x2 = max(
                block.x2
                for block in article.blocks
            )

            y2 = max(
                block.y2
                for block in article.blocks
            )

            width = x2 - x1
            height = y2 - y1

            # -------------------------------------------------
            # Create final boundary
            # -------------------------------------------------

            boundaries.append(
                ArticleBoundary(
                    article_id=article.article_id,

                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,

                    width=width,
                    height=height,

                    area=width * height,

                    block_ids=list(
                        article.block_ids
                    ),

                    confidence=article.confidence,
                )
            )

        print()
        print("=" * 60)
        print("BOUNDARY BUILDER")
        print("=" * 60)

        print(
            f"Generated Boundaries : "
            f"{len(boundaries)}"
        )

        for boundary in boundaries:

            print(
                f"article_id={boundary.article_id}"
            )

        print("=" * 60)
        print()

        return boundaries