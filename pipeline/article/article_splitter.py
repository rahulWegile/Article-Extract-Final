"""
Sanity-checks the LLM's own article grouping (Gemini/OpenAI boundary
response, via ArticleGrouper) against the SAME geometric root-
detection algorithm local_grouper.py already uses for local mode --
and splits an article back apart when that algorithm disagrees with
the model's merge.

WHY THIS EXISTS
---------------

The boundary-detection LLM call groups blocks into articles directly
(see the grouping prompts in pipeline/languages/) -- this file does
not touch that prompt or that call. But the model can merge multiple
distinct stories into one
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

NOTE ON SCOPE: an article with 0 or 1 title-class block(s) is never
re-checked here, even when a rule line visibly sits between two of
its blocks. group_blocks' root candidates are drawn EXCLUSIVELY from
title-class blocks (see local_grouper.py's "Articles are created from
HEADLINES first" principle) -- with at most one title present, it
can never produce more than one root, so re-running it would only
ever confirm "one story" regardless of pixel evidence. Fixing that
case needs the upstream layout detector to actually mark each
merged item's own heading as a title-class block; it cannot be fixed
by post-processing here without abandoning the headline-first
invariant this whole file depends on.

ORPHAN CONTENT SUB-GROUPS: group_blocks' content-attachment requires
column overlap with a title-class root (see local_grouper.py's
COLUMN_OVERLAP). Given only the narrow slice of blocks one oversized
article contains, a genuine non-title block can end up sharing no
column with ANY title in that slice -- not because it is an
independent story (it has no title of its own), but because its real
column-mate happens to be a DIFFERENT block also being split off into
its own sub-group. group_blocks' own "no root found" fallback then
strands it alone as a bogus one-block sub-group instead of attaching
it to a wrong root. Confirmed on a real page: a narrow side-column
snippet directly above a byline+continuation block, split from its
headline because a third title elsewhere in the same oversized
article was consuming all the column-overlap budget, came out as its
own one-block "article" even though group_blocks' own distance
metric puts it far closer to the byline group than to any headline.
_merge_orphan_content_subgroups folds any such single-block,
non-title sub-group into whichever OTHER sub-group is geometrically
nearest (same distance metric ArticleGrouper already uses to
arbitrate contested/orphan blocks) before the split is committed.

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

from pipeline.article.article_grouper import Article, ArticleGrouper
from pipeline.article.local_grouper import group_blocks
from pipeline.article.visual_separator import (
    has_vertical_separator_between,
)


def _block_to_dict(block, bbox_override=None) -> dict:

    block_dict = {
        "id": block.id,
        "class": block.cls,
        "bbox": {
            "x1": block.x1,
            "y1": block.y1,
            "x2": block.x2,
            "y2": block.y2,
        },
    }

    # Narrow, deliberate passthrough: the LLM's own semantic
    # role can catch a byline the raw layout detector mis-typed as
    # "title" (bold/caps byline text is a common trigger for that --
    # see local_grouper._role_for), which would otherwise make
    # group_blocks treat it as a legitimate new article root and
    # split a story's byline+continuation text away from its own
    # headline. Only "byline" is passed through -- local_grouper has
    # no pre-existing opinion about that role to conflict with,
    # unlike blindly trusting every LLM role over its own class-
    # based inference.
    if getattr(block, "role", None) == "byline":
        block_dict["role"] = "byline"

    # A banner headline's PRIMARY fragment is widened to the union
    # of every fragment's bbox before being handed to group_blocks,
    # so its column-overlap span covers every column the printed
    # line actually crosses (see BANNER HEADLINE FRAGMENTS below).
    # The other fragments are never included in the block_dicts list
    # in the first place -- see split_oversized_articles -- so this
    # is the only place a bbox gets overridden.
    if bbox_override is not None:
        block_dict["bbox"] = dict(bbox_override)

    return block_dict


def _merge_orphan_content_subgroups(sub_groups, sub_roles, block_by_id):
    """
    Fold a NON-title-rooted sub-group -- one or more blocks
    group_blocks could only place via its "no root found" content
    fallback (see the ORPHAN CONTENT SUB-GROUPS note above), never
    headed by a title of its own -- into whichever OTHER,
    title-rooted sub-group sits geometrically nearest, using the
    same column-aware distance metric ArticleGrouper already uses
    to arbitrate contested/orphan blocks.

    group_blocks' own convention: a group's root (or, for a content
    fallback, its pseudo-root) is always the FIRST id in its
    "blocks" list -- see group_blocks' root-selection and "no root
    found" fallback. A genuine independent story always has its own
    title; a sub-group group_blocks could only anchor via its
    content fallback is always a column-overlap artifact of
    re-running root-detection on a narrow subset, never a real
    second story -- confirmed on a real page where a kicker, a
    byline (also misclassified as title-class -- see the byline
    role passthrough above), and continuation body text formed a
    3-block chain this way, entirely separate from their own
    headline's group.

    This is deliberately NOT the same risk as the reverted whole-
    page nearest-ANY attach (see ArticleGrouper's "Report unclaimed
    content blocks" comment, which merged 35% of one page into a
    single blob): every block handled here already came from ONE
    Gemini-grouped article -- this only ever runs inside
    split_oversized_articles, on an already-flagged-oversized
    article's own blocks -- so an orphan fragment is choosing among
    a small set of siblings Gemini's own top-level judgment already
    said belong to the SAME story, not among every stray block on
    the page. Orphan sub-groups only ever merge INTO a title-rooted
    sub-group, never into each other, so this cannot chain multiple
    orphans together into something new.

    `sub_roles` is group_blocks' own "blocks" list (id/role pairs)
    for this subset. Returns a new list of sub-groups with every
    orphan folded into its nearest title-rooted neighbor and
    removed as a separate entry; unaffected when there is nothing
    to merge, or when no title-rooted sub-group exists to merge
    into.
    """

    role_for = {item["id"]: item["role"] for item in sub_roles}

    def is_title_rooted(group):
        blocks = group.get("blocks", [])
        return bool(blocks) and role_for.get(blocks[0]) == "article_title"

    orphan_indices = [
        index
        for index, group in enumerate(sub_groups)
        if group.get("blocks") and not is_title_rooted(group)
    ]

    if not orphan_indices:
        return sub_groups

    target_indices = [
        index
        for index, group in enumerate(sub_groups)
        if is_title_rooted(group)
    ]

    merged = [
        {**group, "blocks": list(group.get("blocks", []))}
        for group in sub_groups
    ]

    for index in orphan_indices:

        orphan_ids = sub_groups[index].get("blocks", [])

        best_target = None
        best_distance = None

        for target_index in target_indices:

            for orphan_id in orphan_ids:

                orphan_block = block_by_id.get(orphan_id)

                if orphan_block is None:
                    continue

                for other_id in sub_groups[target_index].get("blocks", []):

                    other_block = block_by_id.get(other_id)

                    if other_block is None:
                        continue

                    distance = ArticleGrouper._layout_distance(
                        orphan_block,
                        other_block,
                    )

                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_target = target_index

        if best_target is not None:
            merged[best_target]["blocks"].extend(orphan_ids)

    return [
        group
        for index, group in enumerate(merged)
        if index not in orphan_indices
    ]


def _recover_fully_dropped_blocks(sub_groups, original_ids, block_by_id):
    """
    Attach any block group_blocks left out of every sub-group
    entirely -- not merely stuck in a non-title-rooted one (see
    _merge_orphan_content_subgroups above), but absent from its
    "articles" output altogether -- to whichever sub-group sits
    geometrically nearest, so a coverage gap in group_blocks never
    forces split_oversized_articles to abandon an otherwise-good
    split and keep the whole oversized article merged instead.

    Confirmed on a real page: an image and its own caption (no role
    other than "article_text"/"article_title" is ever eligible for
    group_blocks' "no root found -> stand alone" fallback, so an
    orphaned figure or caption is silently dropped rather than kept
    standalone) and a lone kicker title that became a root, governed
    nothing, and was dropped by group_blocks' own final "single
    title-only entries are filtered out" rule -- all legitimate,
    deliberate behavior for group_blocks' PRIMARY standalone use
    case (see local_grouper.py), where dropping a truly unattachable
    orphan is an acceptable trade-off. Here, every block already
    came from one Gemini/OpenAI-grouped article, so the safer answer
    is to keep it, attached to its nearest neighbor -- same
    reasoning as the orphan-subgroup merge above, just for blocks
    group_blocks never placed anywhere at all.

    Returns a new list of sub-groups with every dropped block
    appended to its nearest sub-group; unchanged when nothing was
    dropped or no sub-group exists to attach to.
    """

    covered = {
        block_id
        for group in sub_groups
        for block_id in group.get("blocks", [])
    }

    dropped_ids = [
        block_id
        for block_id in original_ids
        if block_id not in covered
    ]

    if not dropped_ids or not sub_groups:
        return sub_groups

    merged = [
        {**group, "blocks": list(group.get("blocks", []))}
        for group in sub_groups
    ]

    for dropped_id in dropped_ids:

        dropped_block = block_by_id.get(dropped_id)

        if dropped_block is None:
            continue

        best_target = None
        best_distance = None

        for target_index, group in enumerate(sub_groups):

            for other_id in group.get("blocks", []):

                other_block = block_by_id.get(other_id)

                if other_block is None:
                    continue

                distance = ArticleGrouper._layout_distance(
                    dropped_block,
                    other_block,
                )

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_target = target_index

        if best_target is not None:
            merged[best_target]["blocks"].append(dropped_id)

    return merged


# ============================================================
# SUBORDINATE TITLES
#
# group_blocks treats EVERY title-class block that governs body text
# below it as an article root (see local_grouper.py's "A title that
# clearly governs body text below it is an article root even when the
# whitespace ABOVE the title is small"). On a newspaper package that
# is far too generous: a data box's heading, an infographic label, a
# kicker/eyebrow, a "Bhaskar Knowledge" explainer strip and a sidebar
# sub-head all govern their own few lines of body text, and each one
# then becomes a separate root -- shattering one story into five.
#
# This only became visible once the OCR fix in
# tesseract_engine._ocr_block stopped these smaller headings being
# starved to 0-2 characters and deleted by PageCleaner. They were
# always there; the splitter simply never saw them before.
#
# A newspaper's own typography already separates the two cases: the
# story's headline is set MUCH larger than anything subordinate to
# it. Measured over the nine over-split articles on one real
# 4-page Gujarati edition, every subordinate heading came in at
# 25%-61% of its own article's dominant headline, against a
# dominant headline that is 100% by definition -- so a title below
# SUBORDINATE_TITLE_RATIO of the biggest title in the SAME article is
# a sub-head, not a second story.
#
# Deliberately a WITHIN-ARTICLE comparison, never page-wide. Two
# genuinely separate stories the model wrongly merged are set at the
# same headline size as each other, so both stay dominant-class and
# the split still happens -- which is the case this whole file exists
# for (see WHY THIS EXISTS above, and the real 4-stories-in-one-crop
# Tamil and Gemini examples). All this gate removes is the split that
# had nothing behind it but a smaller sub-head.
# ============================================================

SUBORDINATE_TITLE_RATIO = 0.65


def _title_line_height(block) -> float | None:
    """
    Per-LINE height of a title block -- its box height divided by how
    many lines of text it holds, so a three-line headline is not
    mistaken for three times the type size of a one-line one.

    Returns None when the block has no text to count lines from. That
    is a title whose OCR failed but whose BOX PageCleaner deliberately
    kept for its geometry (see page_cleaner.remove_noise): it is a
    real heading, but nothing here can measure it, and a heading with
    no readable text cannot anchor a distinct story anyway -- so
    callers treat it as subordinate rather than let it force a split.
    """

    text = (getattr(block, "text", "") or "").strip()

    if not text:
        return None

    line_count = text.count("\n") + 1

    return (block.y2 - block.y1) / line_count


def _dominant_class_titles(title_blocks) -> list:
    """
    The subset of `title_blocks` set at (or near) the largest type
    size present -- i.e. the ones that could each head their own
    story. See SUBORDINATE_TITLE_RATIO above.
    """

    heights = {
        block.id: _title_line_height(block)
        for block in title_blocks
    }

    measured = [
        height
        for height in heights.values()
        if height is not None
    ]

    if not measured:
        return list(title_blocks)

    dominant_height = max(measured)

    return [
        block
        for block in title_blocks
        if heights[block.id] is not None
        and heights[block.id]
        >= SUBORDINATE_TITLE_RATIO * dominant_height
    ]


# ============================================================
# BANNER HEADLINE FRAGMENTS
#
# A headline printed across several columns is ONE printed line, but
# the layout detector routinely cuts it at a column gutter and emits
# each half as its own title block. group_blocks then makes a root
# out of each half (both have body text below them, in their own
# columns), and one story is split down the middle of its own
# headline.
#
# Confirmed on a real Punjabi page: "ਸਿੱਖ ਵੋਟ ਨੂੰ" (x 419-621) and
# "ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ ਦੋਫਾੜ : ਸੁਖਬੀਰ" (x 630-1237) -- one banner
# headline, 9px apart, on the same line -- became two articles, the
# left one taking column 1's body text and the right one taking the
# photo and the remaining three columns.
#
# The signal is that the two halves are the SAME PRINTED LINE of
# type: near-identical top and bottom edges, the same type size,
# side by side with only a gutter between them, and nothing printed
# in that gutter. Two genuinely separate stories set side by side
# are what this must not swallow, so every one of those conditions
# is required, and the gutter is probed for the rule/colour bar a
# newspaper prints when it really is dividing two items in one row
# (see visual_separator.has_vertical_separator_between).
#
# Measured over every over-split article in the saved corpus, real
# banner halves sit 0.15-0.40 of their own line height apart, while
# the tightest genuinely-separate side-by-side pair (a schedule
# box's day labels on a Gujarati page) sits 0.61 apart -- hence
# BANNER_FRAGMENT_MAX_GAP_RATIO below.
#
# Restricted to SINGLE-LINE titles on purpose. A one-line fragment
# pair is unambiguous; two multi-line headlines stacked side by side
# match every other test here while being two real stories (a
# Gujarati page has exactly that: two 2-line headlines 17px apart),
# and nothing geometric separates them from a multi-line banner cut
# at a gutter.
# ============================================================

# How far the two halves' top/bottom edges may differ, as a fraction
# of the shorter half's height. They are the same printed line, so
# this is detector/OCR slop only, not a real layout allowance.
BANNER_FRAGMENT_EDGE_TOLERANCE = 0.25

# Shorter half's height over the taller's -- i.e. the same type size.
BANNER_FRAGMENT_HEIGHT_RATIO = 0.75

# Gutter width between the halves, as a fraction of their height.
BANNER_FRAGMENT_MAX_GAP_RATIO = 0.5


def _is_single_line_title(block) -> bool:

    text = (getattr(block, "text", "") or "").strip()

    return bool(text) and "\n" not in text


def _is_banner_fragment_pair(block_a, block_b, page_image) -> bool:
    """
    Whether these two title blocks are two halves of ONE headline
    the layout detector cut at a column gutter.
    """

    if not (
        _is_single_line_title(block_a)
        and _is_single_line_title(block_b)
    ):
        return False

    height_a = block_a.y2 - block_a.y1
    height_b = block_b.y2 - block_b.y1

    shorter = min(height_a, height_b)
    taller = max(height_a, height_b)

    if shorter <= 0:
        return False

    if shorter / taller < BANNER_FRAGMENT_HEIGHT_RATIO:
        return False

    edge_tolerance = BANNER_FRAGMENT_EDGE_TOLERANCE * shorter

    if abs(block_a.y1 - block_b.y1) > edge_tolerance:
        return False

    if abs(block_a.y2 - block_b.y2) > edge_tolerance:
        return False

    left, right = (
        (block_a, block_b)
        if block_a.x1 <= block_b.x1
        else (block_b, block_a)
    )

    gap = right.x1 - left.x2

    # A negative gap means they overlap horizontally, so they are
    # stacked lines of one title, not two halves of one line.
    if gap < 0:
        return False

    if gap > BANNER_FRAGMENT_MAX_GAP_RATIO * shorter:
        return False

    return not has_vertical_separator_between(
        page_image,
        left.x2,
        right.x1,
        min(block_a.y1, block_b.y1),
        max(block_a.y2, block_b.y2),
    )


def _banner_fragment_groups(title_blocks, page_image) -> list:
    """
    Partition `title_blocks` into printed headlines: one group per
    headline, holding every fragment the layout detector cut it
    into.

    Membership is transitive, so a headline cut into three or more
    pieces collapses into one group as long as each piece is
    adjacent to the one before it.
    """

    groups = []

    for block in sorted(
        title_blocks,
        key=lambda item: (item.y1, item.x1),
    ):

        for group in groups:

            if any(
                _is_banner_fragment_pair(member, block, page_image)
                for member in group
            ):
                group.append(block)
                break

        else:
            groups.append([block])

    return groups


def _banner_primary(group):
    return min(group, key=lambda item: (item.y1, item.x1))


def _collapse_banner_fragments(title_blocks, page_image) -> list:
    """
    ONE representative (the leftmost fragment) per printed headline,
    so a banner cut at a gutter counts once rather than once per
    fragment.
    """

    return [
        _banner_primary(group)
        for group in _banner_fragment_groups(title_blocks, page_image)
    ]


def _union_bbox(blocks):
    return {
        "x1": min(block.x1 for block in blocks),
        "y1": min(block.y1 for block in blocks),
        "x2": max(block.x2 for block in blocks),
        "y2": max(block.y2 for block in blocks),
    }


def _banner_fragment_plan(title_blocks, page_image):
    """
    How to fold every banner headline in `title_blocks` down to its
    PRIMARY fragment before handing this article's blocks to
    group_blocks: which non-primary fragment ids to leave OUT of
    block_dicts entirely (group_blocks never sees them as separate
    title candidates, so it never makes a root out of a fragment),
    the union bbox each primary should be widened to (so its
    column-overlap span covers every column the printed line
    actually crosses, letting body text under EITHER fragment attach
    to the merged root), and which primary each skipped fragment
    must be reattached to afterward (see _reattach_banner_fragments
    -- a fragment's own, un-widened bbox is exactly the ambiguous
    case ArticleGrouper._layout_distance cannot be trusted with, so
    reattachment has to be deterministic, not nearest-neighbor).

    Returns (skip_ids, bbox_override_by_id, fragment_to_primary).
    """

    skip_ids = set()
    bbox_override_by_id = {}
    fragment_to_primary = {}

    for group in _banner_fragment_groups(title_blocks, page_image):

        if len(group) <= 1:
            continue

        primary = _banner_primary(group)

        bbox_override_by_id[primary.id] = _union_bbox(group)

        for block in group:

            if block.id == primary.id:
                continue

            skip_ids.add(block.id)
            fragment_to_primary[block.id] = primary.id

    return skip_ids, bbox_override_by_id, fragment_to_primary


def _reattach_banner_fragments(sub_groups, fragment_to_primary):
    """
    Put every skipped banner fragment back into the SAME sub-group as
    the primary fragment it was folded into for group_blocks (see
    _banner_fragment_plan) -- deterministic, unlike the generic
    nearest-neighbor distance _recover_fully_dropped_blocks uses for
    blocks group_blocks itself failed to place.

    A banner fragment's own, un-widened bbox is unsafe for that
    generic distance metric: confirmed on a real page where a
    fragment sitting at the top of the RIGHT column was, by column
    overlap, geometrically closer to an unrelated second headline
    starting further down that same column than to any of its own
    story's body blocks (which mostly live in the LEFT column) --
    nearest-neighbor recovery would silently hand the fragment to the
    wrong story. A banner fragment is definitionally part of its
    primary's headline, so this is never a judgment call.

    Runs BEFORE _recover_fully_dropped_blocks so a fragment placed
    here already reads as covered when that generic pass computes
    what group_blocks left unclaimed for OTHER reasons.
    """

    if not fragment_to_primary:
        return sub_groups

    primary_to_group_index = {}

    for index, group in enumerate(sub_groups):
        for block_id in group.get("blocks", []):
            primary_to_group_index[block_id] = index

    merged = [
        {**group, "blocks": list(group.get("blocks", []))}
        for group in sub_groups
    ]

    for fragment_id, primary_id in fragment_to_primary.items():

        target_index = primary_to_group_index.get(primary_id)

        if target_index is not None:
            merged[target_index]["blocks"].append(fragment_id)

    return merged


def split_oversized_articles(
    articles: List[Article],
    page_image=None,
) -> List[Article]:

    result: List[Article] = []

    split_count = 0
    kept_ambiguous = 0
    kept_subordinate = 0
    kept_banner = 0
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

        # Every extra title is a sub-head/inset heading printed
        # smaller than this article's own headline -- no evidence of a
        # second story, so never re-check it. See
        # SUBORDINATE_TITLE_RATIO above.
        dominant_titles = _dominant_class_titles(title_blocks)

        if len(dominant_titles) <= 1:
            kept_subordinate += 1
            result.append(article)
            continue

        # The extra titles are all fragments of ONE headline the
        # layout detector cut at a column gutter -- again no second
        # story to find, so never re-check it. See BANNER HEADLINE
        # FRAGMENTS above.
        if len(
            _collapse_banner_fragments(dominant_titles, page_image)
        ) <= 1:
            kept_banner += 1
            result.append(article)
            continue

        block_by_id = {
            block.id: block
            for block in article.blocks
        }

        # An article that DOES hold more than one real headline can
        # still have one of them cut at a gutter; tell group_blocks
        # which titles are two halves of one line so the split falls
        # between the real headlines instead of through a banner.
        #
        # Restricted to DOMINANT titles only -- same reasoning as the
        # SUBORDINATE_TITLE_RATIO gate above. Two subordinate sub-
        # heads (e.g. two short pull-quote/highlight boxes printed
        # side by side under a shared main headline) can independently
        # satisfy every banner-fragment geometry test against EACH
        # OTHER despite being two unrelated items, not two halves of
        # one headline -- confirmed on a real Marathi page where two
        # such highlight-box titles, each with its own self-contained
        # paragraph beneath it, were wrongly fused into one five-block
        # article. A subordinate title was never a candidate for
        # governing its own story in the first place, so it is never
        # a candidate for being a banner fragment of one either.
        skip_ids, bbox_override_by_id, fragment_to_primary = (
            _banner_fragment_plan(
                dominant_titles,
                page_image,
            )
        )

        block_dicts = [
            _block_to_dict(block, bbox_override_by_id.get(block.id))
            for block in article.blocks
            if block.id not in skip_ids
        ]

        sub_response = group_blocks(block_dicts, page_image)

        sub_groups = _reattach_banner_fragments(
            sub_response.get("articles", []),
            fragment_to_primary,
        )

        sub_groups = _recover_fully_dropped_blocks(
            sub_groups,
            article.block_ids,
            block_by_id,
        )

        sub_groups = _merge_orphan_content_subgroups(
            sub_groups,
            sub_response.get("blocks", []),
            block_by_id,
        )

        if len(sub_groups) <= 1:
            # local_grouper's own root-detection agrees this is one
            # real story -- e.g. a kicker line above the headline, or
            # a subheadline just below it. Keep the model's grouping
            # as-is.
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

    if (
        split_count
        or kept_ambiguous
        or kept_subordinate
        or kept_banner
        or aborted_unsafe
    ):

        print()
        print("=" * 60)
        print("ARTICLE SPLITTER")
        print("=" * 60)
        print(f"Multi-title articles split : {split_count}")
        print(f"Multi-title articles kept (one real story) : {kept_ambiguous}")
        if kept_subordinate:
            print(
                f"Multi-title articles kept (extra titles are "
                f"sub-heads, not headlines) : {kept_subordinate}"
            )
        if kept_banner:
            print(
                f"Multi-title articles kept (extra titles are halves "
                f"of one banner headline) : {kept_banner}"
            )
        if aborted_unsafe:
            print(
                f"Multi-title articles left unsplit (unsafe -- "
                f"would drop a block) : {aborted_unsafe}"
            )
        print(f"Total articles after split : {len(result)}")
        print("=" * 60)
        print()

    return result
