"""
Sanity-checks the LLM's own article grouping (Gemini/OpenAI boundary
response, via ArticleGrouper) against the SAME geometric root-
detection algorithm local_grouper.py already uses for local mode --
and splits an article back apart when that algorithm disagrees with
the model's merge.

WHY THIS EXISTS
---------------

The boundary-detection LLM call groups blocks into articles directly
(see ARTICLE_GROUP_PROMPT) -- this file does not touch that prompt or
that call. But the model can merge multiple distinct stories into one
article, the same failure shape local_grouper.py's own gap/separator
heuristic can produce in local mode (see local_grouper.py's
ROOT_WIDTH_FACTOR history for a worked example there). Confirmed on a
real page: one Gemini-grouped "article" wrapped 34 blocks spanning 7
separate title blocks, from y1=292 to y1=2356 -- at least 4 unrelated
stories (a pellet-gun injury story, a minister's quote, a protest
story, an opposition leader's quote) glued into one crop, whose
article_text then interleaved all four stories' paragraphs together.

WHAT THIS DOES
--------------

For every LLM-grouped article that contains MORE than one title-class
block, re-run local_grouper's OWN root-detection (group_blocks) on
JUST that article's own blocks in isolation. Two outcomes:

1. group_blocks agrees it's one story (e.g. a kicker/eyebrow line
   sitting just above the main headline, or a subheadline just below
   it -- gap/separator evidence says "same story"): the article is
   left exactly as the model grouped it.
2. group_blocks finds genuinely separate roots (large gap, or no
   plausible in-column relationship between the extra title and the
   rest): the article is split into one article per root, each
   keeping only the blocks that root's own content-attachment claims.

SAFETY
------

If the split would leave ANY of the article's original blocks
unclaimed by every resulting sub-article (group_blocks' own "no root
found" case), the split is abandoned and the ORIGINAL article is kept
whole instead -- never risk silently dropping a block that was at
least included (if wrongly attributed) before this ran.

This never touches BoundaryBuilder's math (union-of-block-bboxes) or
ArticleGrouper's contested-block resolution -- it runs strictly
between them, only ever splitting one already-built Article into
multiple, never merging, never inventing new blocks.
"""

from __future__ import annotations

from typing import List

from pipeline.article.article_grouper import Article
from pipeline.article.local_grouper import group_blocks


def _block_to_dict(block) -> dict:

    return {
        "id": block.id,
        "class": block.cls,
        "bbox": {
            "x1": block.x1,
            "y1": block.y1,
            "x2": block.x2,
            "y2": block.y2,
        },
    }


def split_oversized_articles(
    articles: List[Article],
    page_image=None,
) -> List[Article]:

    result: List[Article] = []

    split_count = 0
    kept_ambiguous = 0
    aborted_unsafe = 0

    for article in articles:

        title_blocks = [
            block
            for block in article.blocks
            if getattr(block, "cls", None) == "title"
        ]

        if len(title_blocks) <= 1:
            result.append(article)
            continue

        block_dicts = [
            _block_to_dict(block)
            for block in article.blocks
        ]

        sub_response = group_blocks(block_dicts, page_image)

        sub_groups = sub_response.get("articles", [])

        if len(sub_groups) <= 1:
            # local_grouper's own root-detection agrees this is one
            # real story -- e.g. a kicker line above the headline, or
            # a subheadline below it. Keep the model's grouping as-is.
            kept_ambiguous += 1
            result.append(article)
            continue

        covered_ids = {
            block_id
            for sub_group in sub_groups
            for block_id in sub_group.get("blocks", [])
        }

        if covered_ids != set(article.block_ids):
            # group_blocks left something unclaimed on this subset
            # (its own "no plausible root" case) -- splitting would
            # silently drop that block. Keep the original, unsplit
            # article instead: worse attribution, but no data loss.
            aborted_unsafe += 1
            result.append(article)
            continue

        block_by_id = {
            block.id: block
            for block in article.blocks
        }

        for sub_group in sub_groups:

            sub_blocks = [
                block_by_id[block_id]
                for block_id in sub_group.get("blocks", [])
                if block_id in block_by_id
            ]

            if not sub_blocks:
                continue

            sub_blocks.sort(
                key=lambda block: getattr(block, "reading_order", 0)
            )

            result.append(
                Article(
                    article_id=article.article_id,
                    blocks=sub_blocks,
                    block_ids=[block.id for block in sub_blocks],
                    confidence=article.confidence,
                )
            )

        split_count += 1

    for index, article in enumerate(result, start=1):
        article.article_id = index

    if split_count or kept_ambiguous or aborted_unsafe:

        print()
        print("=" * 60)
        print("ARTICLE SPLITTER")
        print("=" * 60)
        print(f"Multi-title articles split : {split_count}")
        print(f"Multi-title articles kept (one real story) : {kept_ambiguous}")
        if aborted_unsafe:
            print(
                f"Multi-title articles left unsplit (unsafe -- "
                f"would drop a block) : {aborted_unsafe}"
            )
        print(f"Total articles after split : {len(result)}")
        print("=" * 60)
        print()

    return result
