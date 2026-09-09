from typing import List

from pipeline.article.article_grouper import Article
from pipeline.article.article_region_reconstructor import (
    ArticleBoundary,
    reconstruct_article_regions,
)


__all__ = ["ArticleBoundary", "BoundaryBuilder"]


class BoundaryBuilder:
    """
    Build one final article boundary from each finalized article.

    Thin, backward-compatible wrapper: the actual work -- turning
    finalized article membership into exactly one outer boundary per
    article, with validation and debug output -- lives in
    article_region_reconstructor.py, the shared language-independent
    final stage. See that module's docstring for the full contract.
    """

    def build(self, articles: List[Article]):
        return reconstruct_article_regions(articles)
