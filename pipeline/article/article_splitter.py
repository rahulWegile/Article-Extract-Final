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

NOTE ON SCOPE: an article with 0 or 1 title-class block(s) is
re-checked here ONLY when a printed rule line/border is found between
two of its own blocks (see PRINTED RULE-LINE / BORDER BREAK below,
_has_internal_rule_line_break) -- otherwise it is skipped, same as
before. Even when that trigger DOES fire, group_blocks' root
candidates are still drawn EXCLUSIVELY from title-class blocks (see
local_grouper.py's "Articles are created from HEADLINES first"
principle), so with at most one title present group_blocks usually
still confirms "one story" (kept_ambiguous) -- it can only actually
split when the blocks on either side of that printed line also land
in different columns, letting its "no root found" fallback stand
each side up as its own entry. A merged story whose own second
headline was never detected as a distinct title block AND sits in
the same column as the first is not fixable here without abandoning
the headline-first invariant this whole file depends on; that needs
the upstream layout detector to mark it as a title-class block in
the first place.

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
from pipeline.article.banner_fragments import (
    BANNER_FRAGMENT_EDGE_TOLERANCE,
    BANNER_FRAGMENT_HEIGHT_RATIO,
    BANNER_FRAGMENT_MAX_GAP_RATIO,
    BANNER_FRAGMENT_MAX_OVERLAP_RATIO,
    group_banner_fragments,
    is_banner_fragment_pair,
    is_single_line_text,
)
from pipeline.article.local_grouper import IMAGE_CLASSES, group_blocks
from pipeline.article.visual_separator import (
    SEPARATOR_MIN_BAND_HEIGHT,
    has_undetected_content,
    has_visual_separator,
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


def _merge_headless_title_subgroups(sub_groups, sub_roles, block_by_id):
    """
    Fold a title-rooted sub-group that came out of the split with NO
    body/figure content of its own -- just its lone root title, and
    nothing group_blocks attached below it -- into whichever OTHER
    sub-group sits geometrically nearest.

    _merge_orphan_content_subgroups (above) only ever catches a
    sub-group group_blocks could not root on a title at all. It
    cannot catch THIS shape: a sub-group that IS title-rooted (so
    `is_title_rooted` in that function says yes, nothing to merge)
    but whose title governs no content in this narrow slice -- e.g. a
    subordinate title rescued into `dominant_titles` purely for being
    far from the article's real headline (see SUBORDINATE_TITLE_
    MAX_GAP), which group_blocks then dutifully makes its own root
    even though it has no paragraph or photo of its own here. Left
    alone, split_oversized_articles would emit that as its own
    "article": a bare headline with no story under it -- exactly the
    "headless orphan subgroup" a split must never produce (see
    requirement (c): a split only ever counts when it yields at least
    two GENUINE, viable stories, each a title plus its own content).

    Runs AFTER _merge_orphan_content_subgroups, so every remaining
    sub-group here is already title-rooted; this only tests whether
    that root has anything besides itself. Uses the same nearest-
    neighbor `_layout_distance` metric the other merges in this file
    already use, so a headless title always folds into whichever
    surviving sub-group its own root sits closest to -- typically the
    real headline's own group, i.e. exactly "the parent headline
    group" a headless split fragment belongs back in.

    `sub_roles` is group_blocks' own "blocks" list (id/role pairs).
    Returns a new list of sub-groups with every headless title-only
    group folded into its nearest neighbor and removed as a separate
    entry; unchanged when every sub-group already has real content of
    its own, or when there is no OTHER sub-group to merge into.
    """

    role_for = {item["id"]: item["role"] for item in sub_roles}

    def has_own_content(group):
        return any(
            role_for.get(block_id) != "article_title"
            for block_id in group.get("blocks", [])
        )

    headless_indices = [
        index
        for index, group in enumerate(sub_groups)
        if group.get("blocks") and not has_own_content(group)
    ]

    if not headless_indices:
        return sub_groups

    target_indices = [
        index
        for index in range(len(sub_groups))
        if index not in headless_indices and sub_groups[index].get("blocks")
    ]

    if not target_indices:
        return sub_groups

    merged = [
        {**group, "blocks": list(group.get("blocks", []))}
        for group in sub_groups
    ]

    for index in headless_indices:

        headless_ids = sub_groups[index].get("blocks", [])

        best_target = None
        best_distance = None

        for target_index in target_indices:

            for headless_id in headless_ids:

                headless_block = block_by_id.get(headless_id)

                if headless_block is None:
                    continue

                for other_id in sub_groups[target_index].get("blocks", []):

                    other_block = block_by_id.get(other_id)

                    if other_block is None:
                        continue

                    distance = ArticleGrouper._layout_distance(
                        headless_block,
                        other_block,
                    )

                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_target = target_index

        if best_target is not None:
            merged[best_target]["blocks"].extend(headless_ids)

    return [
        group
        for index, group in enumerate(merged)
        if index not in headless_indices
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
#
# PROXIMITY, NOT JUST SIZE
# ------------------------
#
# The size-ratio measurement above only ever looked at genuine,
# ADJACENT sub-heads -- it never claimed a small title far from every
# large one must also be one of them. Confirmed on a real Urdu (THE
# INQUILAB) page: a small headline (line height 41px, "...London's
# mayor...") sitting 2026px below the page's own large banner
# headline (line height 251px, an unrelated Iran/US story) was
# dismissed as that banner's own sub-head purely because 41/251 <
# SUBORDINATE_TITLE_RATIO -- even though nothing of the banner's
# story was printed anywhere in between, and the two headlines sat on
# opposite ends of the page. article_splitter never even re-checked
# that article (group_blocks was never called), and the merged
# article's own outer boundary then stretched across virtually the
# entire page height, swallowing every story printed between the two
# headlines. A title only counts as a safely-skippable sub-head when
# it is BOTH smaller AND close to the dominant headline it supposedly
# belongs to -- see SUBORDINATE_TITLE_MAX_GAP.
#
# UPDATE: "far away" alone is not enough either. A cross-head deep
# inside one long, single-column story (e.g. a "further developments"
# section heading a few hundred pixels past SUBORDINATE_TITLE_MAX_GAP
# from the article's own top headline) used to be rescued here purely
# for being far away -- `not near_a_dominant_title(block)` short-
# circuited the `or` below before has_own_story's own
# SUBORDINATE_RESCUE_MIN_RATIO floor ever ran, so a cross-head set at
# a fraction of the dominant headline's type size (as little as ~14%
# in one real case) was promoted to its own independent story purely
# for sitting far down the page, even though it never had the type
# size a real second headline is set at. SUBORDINATE_RESCUE_MIN_RATIO
# now gates BOTH rescue paths, not just has_own_story, closing that
# bypass. Trade-off: a genuinely distant SECOND STORY whose headline
# happens to be set very small relative to the page's dominant one
# (the original real Urdu case above measured 41/251 = 0.16, below
# the 0.40 floor) no longer gets rescued by distance alone either --
# accepted deliberately, since in practice a tiny, far-away title is
# far more often an in-article cross-head than a second headline.
# ============================================================

SUBORDINATE_TITLE_RATIO = 0.65

# Vertical gap beyond which a smaller title is no longer assumed to
# be a sub-head OF a nearby dominant headline, regardless of type
# size. A genuine kicker/eyebrow/data-box heading/sidebar sub-head
# sits immediately adjacent to (or a short paragraph away from) the
# story it belongs to; this is the same order of magnitude as the
# other proven "genuinely adjacent" gap constants already used
# elsewhere in this codebase (e.g. ArticleGrouper.
# IMAGE_CAPTION_MAX_GAP = 250.0), given generous headroom since a
# sub-head can sit below a full headline-height's worth of kicker/
# byline text rather than directly against an image. The real Urdu
# gap this was measured against (2026px) is more than 3x this
# threshold, so this comfortably separates the two without needing
# to be tuned close to either number.
SUBORDINATE_TITLE_MAX_GAP = 600.0

# Floor a smaller title's own size must still clear before its OWN
# content (see _has_independent_story_content) is trusted to rescue
# it from subordinate classification at all. A real second headline
# printed smaller than the page's dominant one is still typeset as a
# genuine headline; an internal data-box/table heading or comparison
# sidebar inside one larger feature is typeset much smaller still,
# and routinely governs its own little photo/table graphic and a
# paragraph or two of its own -- exactly the content shape
# _has_independent_story_content looks for -- without being a second
# story. Confirmed on a real Gujarati page (doc_000167, page 2): an
# investigative feature's internal "NPA loan table" heading (ratio
# 0.21 to the feature's own dominant headline) and a "comparison to a
# Singapore company" sub-heading (ratio 0.14) each govern a photo/
# table graphic and real paragraphs of their own, exactly like a
# genuine second story would -- content ownership alone cannot tell
# the two apart. Confirmed on a real Gujarati page (doc_000167, page
# 3): the genuine second story this rescue exists for ("death toll
# reaches 939", ratio 0.45 to its page's own dominant headline) sits
# comfortably above both false positives, while the page's own
# genuine sub-head ("equipment shortage", ratio 0.31) sits below it.
# Set at 0.40 -- clear of the confirmed sub-head/data-box ratios
# (0.14-0.31) and clear of the confirmed real second headline (0.45),
# without being tuned close to either.
#
# Also gates the DISTANCE-based rescue (`not near_a_dominant_title`)
# in _dominant_class_titles, not just this has_own_story path -- see
# the "PROXIMITY, NOT JUST SIZE" UPDATE above for why a title has to
# clear this floor regardless of which rescue path found it.
SUBORDINATE_RESCUE_MIN_RATIO = 0.40


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


# ============================================================
# SUBORDINATE TITLE WITH ITS OWN STORY
#
# SUBORDINATE_TITLE_RATIO's size test alone cannot tell a genuine
# sub-head (a kicker/eyebrow/data-box heading that "has no
# independent body text of its own" -- see the KICKER / PRE-HEADLINE
# RULE in the grouping prompts, at most a line or two riding on the
# bigger headline beside it) apart from a second, smaller-typeset
# headline that governs a COMPLETE story of its own -- its own
# paragraph(s), sometimes its own photo -- merely printed close to a
# bigger one.
#
# Confirmed on a real Gujarati page (doc_000167, page 3): a "death
# toll reaches 939" headline (50px line height) sitting 318px above
# an unrelated "hundreds trapped in tunnels" headline (110px) was
# suppressed as that headline's own sub-head purely on the size
# ratio (50/110.5 = 0.45 < 0.65) and proximity (well under
# SUBORDINATE_TITLE_MAX_GAP) -- even though the death-toll headline
# governs its own paragraph AND its own photo, entirely separate
# from the tunnel-rescue story's own text. The two were fused into
# one article. On the SAME page, a genuine sub-head ("equipment
# shortage", 34px) sitting in the same cluster governs only one
# short paragraph and no photo of its own, and correctly stays
# classified as a sub-head -- so the discriminator below is content
# ownership (an image, or more than one paragraph, governed by the
# smaller title and by no OTHER title in the article), not size or
# distance alone.
# ============================================================


def _nearest_governing_title(block, title_blocks):
    """
    Which title in `title_blocks` most plausibly governs `block`,
    for the sole purpose of measuring a smaller title's OWN content
    below it (see _has_independent_story_content).

    A title only governs content that starts at or below it in the
    same column track (RULE_LINE_COLUMN_OVERLAP); among every title
    that qualifies, the nearest one by vertical gap wins. Gap, not
    raw column-overlap magnitude, is what has to decide this: a
    full-width banner headline overlaps a narrow column beneath it
    just as completely as that column's own, much smaller title
    does, so overlap ratio alone cannot tell them apart -- confirmed
    on the real page above, where the tunnel-rescue banner's column
    span geometrically covers the death-toll headline's own body
    paragraph too. Proximity is what a reader actually follows.
    """

    best = None
    best_gap = None

    for title in title_blocks:

        if title.id == block.id:
            continue

        if _column_overlap_ratio(title, block) < RULE_LINE_COLUMN_OVERLAP:
            continue

        if title.y1 > block.y1:
            continue

        gap = max(0.0, block.y1 - title.y2)

        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = title

    return best


def _has_independent_story_content(title, title_blocks, all_blocks) -> bool:
    """
    Whether `title` governs enough content of its own -- a photo, or
    more than one paragraph -- to plausibly be a complete second
    story rather than a sub-head riding on a bigger headline nearby.
    See SUBORDINATE TITLE WITH ITS OWN STORY above.
    """

    own_content = [
        block
        for block in all_blocks
        if (getattr(block, "cls", "") or "").strip().lower()
        in ("plain text", "figure")
        and _nearest_governing_title(block, title_blocks) is title
    ]

    has_image = any(
        (getattr(block, "cls", "") or "").strip().lower() == "figure"
        for block in own_content
    )

    text_count = sum(
        1
        for block in own_content
        if (getattr(block, "cls", "") or "").strip().lower()
        == "plain text"
    )

    return has_image or text_count >= 2


def _dominant_class_titles(title_blocks, all_blocks=None) -> list:
    """
    The subset of `title_blocks` set at (or near) the largest type
    size present, PLUS any smaller title that either sits too far
    from every one of those to plausibly be one of their own
    sub-heads, or governs a complete story of its own -- i.e. every
    title that could each head its own story. See
    SUBORDINATE_TITLE_RATIO, SUBORDINATE_TITLE_MAX_GAP, and
    SUBORDINATE TITLE WITH ITS OWN STORY above.

    `all_blocks` (optional, normally the whole article's own blocks)
    enables the independent-content rescue; omitted, this falls back
    to the original size/distance-only behavior.
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

    dominant_by_size = [
        block
        for block in title_blocks
        if heights[block.id] is not None
        and heights[block.id]
        >= SUBORDINATE_TITLE_RATIO * dominant_height
    ]

    dominant_ids = {block.id for block in dominant_by_size}

    def near_a_dominant_title(block):
        return any(
            ArticleGrouper._vertical_gap(block, other)
            <= SUBORDINATE_TITLE_MAX_GAP
            for other in dominant_by_size
        )

    def is_stacked_with_dominant(block):
        # A sub-headline or deck sitting immediately adjacent (<= 80px)
        # above or below a dominant headline with column overlap
        # is part of that headline's own multi-tier stack, never its own story.
        for other in dominant_by_size:
            if (
                ArticleGrouper._vertical_gap(block, other) <= 80.0
                and _column_overlap_ratio(block, other) >= 0.30
            ):
                return True
        return False

    def has_own_story(block):
        if all_blocks is None or is_stacked_with_dominant(block):
            return False
        height = heights.get(block.id)
        if (
            height is None
            or height < SUBORDINATE_RESCUE_MIN_RATIO * dominant_height
        ):
            return False
        return _has_independent_story_content(
            block, title_blocks, all_blocks
        )

    rescued = [
        block
        for block in title_blocks
        if block.id not in dominant_ids
        and heights.get(block.id, 0) is not None
        and heights.get(block.id, 0)
        >= SUBORDINATE_RESCUE_MIN_RATIO * dominant_height
        and (not near_a_dominant_title(block) or has_own_story(block))
    ]

    return dominant_by_size + rescued


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

# Thresholds and the actual fragment-pair/grouping geometry now live
# in banner_fragments.py, shared with local_grouper.py (which cannot
# import from this file -- article_splitter.py already imports
# group_blocks FROM local_grouper.py, so the reverse import would be
# circular). Re-imported above for anything in this file/its tests
# that still references the names under their original names here.


def _is_single_line_title(block) -> bool:
    return is_single_line_text(getattr(block, "text", ""))


def _is_banner_fragment_pair(block_a, block_b, page_image) -> bool:
    """
    Whether these two title blocks are two halves of ONE headline
    the layout detector cut at a column gutter. Thin LayoutBlock
    adapter over banner_fragments.is_banner_fragment_pair -- see that
    module for the real geometry, thresholds, and the real Urdu (The
    Siasat Daily) page that motivated BANNER_FRAGMENT_MAX_OVERLAP_RATIO.
    """

    return is_banner_fragment_pair(
        (block_a.x1, block_a.y1, block_a.x2, block_a.y2),
        getattr(block_a, "text", ""),
        (block_b.x1, block_b.y1, block_b.x2, block_b.y2),
        getattr(block_b, "text", ""),
        page_image,
        has_vertical_separator_between,
    )


def _banner_fragment_groups(title_blocks, page_image) -> list:
    """
    Partition `title_blocks` into printed headlines: one group per
    headline, holding every fragment the layout detector cut it
    into.

    Membership is transitive, so a headline cut into three or more
    pieces collapses into one group as long as each piece is
    adjacent to the one before it. Thin LayoutBlock adapter over
    banner_fragments.group_banner_fragments, keyed by `id(block)`
    since LayoutBlock objects here (unlike local_grouper.py's plain
    dicts) are hashable by identity.
    """

    ordered = sorted(title_blocks, key=lambda item: (item.y1, item.x1))

    by_key = {id(block): block for block in ordered}

    items = [
        (
            id(block),
            (block.x1, block.y1, block.x2, block.y2),
            getattr(block, "text", ""),
        )
        for block in ordered
    ]

    groups = group_banner_fragments(
        items, page_image, has_vertical_separator_between,
    )

    return [
        [by_key[key] for key in group]
        for group in groups
    ]


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


# ============================================================
# SAME-EVENT COLUMN PACKAGE
#
# Distinct from a banner fragment (one headline, cut into halves of
# the SAME sentence) and from a subordinate title (a smaller sub-head
# too close to a bigger headline to be its own story). This is two
# or more DOMINANT, differently-worded titles, each with its own real
# paragraph (and sometimes its own photo), printed side by side in
# plain newspaper columns with nothing between them but an ordinary
# gutter -- no rule line, no colour bar, no wider gap.
#
# Confirmed on a real Urdu (THE INQUILAB) page: a student-protest
# story ran as three side-by-side columns under one topical spread --
# the main report, a named attendee's quoted blame (with his photo),
# and a "Centre and Bihar government responsible" column -- 13-17px
# gutters apart, no printed divider between any of them. The grouping
# LLM correctly merged all three as one article; group_blocks' pure
# column-geometry fallback then split them straight back apart, since
# nothing here previously distinguished "plain adjacent columns" from
# "two boxes with real separation between them" (see PACKAGE_COLUMN_
# MAX_GAP_RATIO below -- deliberately the same order of magnitude as
# BANNER_FRAGMENT_MAX_GAP_RATIO, an already-validated normal-gutter
# size, not a new guess).
#
# This must NOT swallow two genuinely unrelated highlight-box titles
# sharing a page -- confirmed separately on a real Marathi page where
# exactly that was wrongly fused (see BANNER HEADLINE FRAGMENTS above
# for the sibling case this file already guards). The required
# has_vertical_separator_between check is what keeps that case
# splitting here too: a highlight/inset box prints its own visible
# border or colour fill, which this detects even in a narrow gutter,
# where BANNER_FRAGMENT_MAX_GAP_RATIO-only geometry could not tell
# the two shapes apart.
# ============================================================

PACKAGE_COLUMN_MAX_GAP_RATIO = 0.5

# How much of the shorter title's own height the two must share, so
# a title far above or below (a different row entirely) is never
# mistaken for a column neighbour just because it happens to be
# horizontally close.
PACKAGE_COLUMN_MIN_ROW_OVERLAP_RATIO = 0.5


def _row_overlap_ratio(block_a, block_b) -> float:

    top = max(block_a.y1, block_b.y1)
    bottom = min(block_a.y2, block_b.y2)

    overlap = max(0, bottom - top)

    shorter = min(
        block_a.y2 - block_a.y1,
        block_b.y2 - block_b.y1,
    )

    if shorter <= 0:
        return 0.0

    return overlap / shorter


def _is_adjacent_package_column(
    block_a, block_b, page_image, content_probe_blocks=None
) -> bool:

    if (
        _row_overlap_ratio(block_a, block_b)
        < PACKAGE_COLUMN_MIN_ROW_OVERLAP_RATIO
    ):
        return False

    left, right = (
        (block_a, block_b)
        if block_a.x1 <= block_b.x1
        else (block_b, block_a)
    )

    gap = right.x1 - left.x2

    if gap < 0:
        return False

    shorter_height = min(
        block_a.y2 - block_a.y1,
        block_b.y2 - block_b.y1,
    )

    if shorter_height <= 0:
        return False

    if gap > PACKAGE_COLUMN_MAX_GAP_RATIO * shorter_height:

        # A gap wider than an ordinary gutter is normally good
        # evidence these are two separate stories -- UNLESS the gap
        # only looks empty because the layout detector missed a
        # photo/graphic actually printed there (confirmed on a real
        # Urdu THE INQUILAB page: an undetected GDP/flag infographic
        # sat exactly in a "537px gap" between two title columns of
        # one Gemini-grouped story). Probed across this whole
        # article's own vertical extent (content_probe_blocks), not
        # just the two title rows, since such a graphic usually sits
        # BELOW the headline row, not beside it.
        if not content_probe_blocks:
            return False

        probe_y1 = min(b.y1 for b in content_probe_blocks)
        probe_y2 = max(b.y2 for b in content_probe_blocks)

        if not has_undetected_content(
            page_image,
            left.x2,
            right.x1,
            probe_y1,
            probe_y2,
        ):
            return False

    return not has_vertical_separator_between(
        page_image,
        left.x2,
        right.x1,
        min(block_a.y1, block_b.y1),
        max(block_a.y2, block_b.y2),
    )


def _forms_undivided_column_package(
    title_blocks, page_image, content_probe_blocks=None
) -> bool:
    """
    Whether every title in `title_blocks` chains to every other
    through plain, undivided column gutters (see
    _is_adjacent_package_column) -- i.e. they form ONE connected
    package rather than including some title that sits apart from
    the rest with real separation or on a different row.

    `content_probe_blocks` (optional, normally the oversized
    article's own full block set) lets an otherwise-too-wide gap
    still count as undivided when the gap itself turns out to hold
    real printed content the detector never boxed -- see
    _is_adjacent_package_column.
    """

    if len(title_blocks) < 2:
        return True

    remaining = list(title_blocks)
    connected = [remaining.pop(0)]

    changed = True

    while changed and remaining:

        changed = False

        for block in list(remaining):

            if any(
                _is_adjacent_package_column(
                    block, member, page_image, content_probe_blocks
                )
                for member in connected
            ):
                connected.append(block)
                remaining.remove(block)
                changed = True

    return not remaining


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


# ============================================================
# PRINTED RULE-LINE / BORDER BREAK
#
# A newspaper's own layout already marks most story breaks with a
# visible printed rule line, a colored box edge, or a bordered strip
# -- exactly the pixels visual_separator.has_visual_separator probes
# for (see that module; local_grouper.py's OWN root detection already
# relies on this same probe for the offline path). Everything above
# this point in the file only ever notices a merge is wrong by
# counting title-class blocks -- SUBORDINATE_TITLE_RATIO, banner
# fragments, all of it. That misses a genuine over-merge where the
# second story's own headline never made it into the layout as a
# distinct "title" block at all (a common failure on a noisy scan --
# confirmed on a real Urdu page, "THE INQUILAB": a small headline
# ("...لندن کے میئر...") did still get boxed as its own title there,
# but the newspaper had ALSO drawn a printed rule between it and the
# unrelated banner headline two-thirds of the page above it -- direct
# pixel evidence of a break that doesn't depend on title sizing at
# all).
#
# _has_internal_rule_line_break below checks the SAME thing
# local_grouper.py's root detection checks for each title candidate,
# but as an independent trigger: does a printed separator sit in an
# unusually large gap (RULE_LINE_MIN_GAP) between any two of THIS
# article's own blocks that are nearest same-column neighbours? If
# so, this article is handed to group_blocks() for a real re-check
# even when title-count evidence alone would have skipped it. The gap
# floor matters here in a way it doesn't for group_blocks' OWN root
# detection: unlike the kept_ambiguous safety net below (which only
# ever protects against an unsafe SPLIT), nothing downstream protects
# against being handed a genuinely single-story article to re-check
# in the first place -- group_blocks treats "title with body text
# below it" as an article root REGARDLESS of gap size (see SUBORDINATE
# TITLES above, "far too generous"), so a rule line alone, without
# the gap requirement, would just as happily re-split a bordered
# inline highlight/pull-quote box that was never a second story. The
# gap floor is the only thing standing between this trigger and
# reintroducing that exact over-split.
# ============================================================

# Fraction of the narrower block's width that must overlap
# horizontally for two blocks to count as sharing a column, for the
# rule-line check below. Matches local_grouper.py's COLUMN_OVERLAP --
# same geometric question, just against Block objects.
RULE_LINE_COLUMN_OVERLAP = 0.35

# Minimum vertical gap before a printed rule line between two blocks
# counts as break evidence at all. A rule line ALONE is not enough --
# a legitimate inline highlight/pull-quote box sitting inside one
# continuous story is routinely bordered too (that is what makes it
# read as a highlight box in print), typically sitting within a few
# tens of pixels of the surrounding text (confirmed elsewhere in this
# codebase: a real boxed sidebar sat just 17px above its preceding
# content). Reusing ArticleGrouper.ORPHAN_MAX_VERTICAL_GAP's own
# measurement (same-column sibling gaps: 5px median, 273px at p99,
# 419px max, over every LLM-grouped article in three full editions)
# as the floor: anything below it is ordinary same-story spacing,
# rule line or not, and anything above it is already outlier
# territory. Comfortably below the confirmed real Urdu break this
# trigger targets (2026px).
RULE_LINE_MIN_GAP = 300.0


def _column_overlap_ratio(block_a, block_b) -> float:

    width_a = max(1.0, float(block_a.x2 - block_a.x1))
    width_b = max(1.0, float(block_b.x2 - block_b.x1))

    overlap = min(block_a.x2, block_b.x2) - max(block_a.x1, block_b.x1)

    return max(0.0, overlap) / min(width_a, width_b)


def _has_internal_rule_line_break(blocks, page_image) -> bool:
    """
    True when a printed rule line or colored box edge sits in an
    unusually LARGE gap (see RULE_LINE_MIN_GAP) between two of THIS
    article's own blocks that are nearest same-column neighbours.

    The gap floor is what keeps this from re-flagging a legitimate
    inline highlight/pull-quote box inside one continuous story --
    such a box is bordered too, but sits close to the surrounding
    text; only a border sitting in an outlier-sized gap is treated as
    evidence of an actual story break.

    Also skips a gap whose upper block is an image: a caption printed
    directly under a photo is routinely set on its own colored/
    bordered background, which belongs to the photo, not to a story
    break -- the exact carve-out local_grouper.py's own root
    detection already makes for the same reason.
    """

    if page_image is None or len(blocks) < 2:
        return False

    for block in blocks:

        nearest_above = None
        nearest_gap = None

        for other in blocks:

            if other is block:
                continue

            if other.y2 > block.y1:
                continue

            if (
                _column_overlap_ratio(block, other)
                < RULE_LINE_COLUMN_OVERLAP
            ):
                continue

            gap = block.y1 - other.y2

            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest_above = other

        if nearest_above is None:
            continue

        if nearest_gap <= RULE_LINE_MIN_GAP:
            continue

        if (
            (getattr(nearest_above, "cls", "") or "").strip().lower()
            in IMAGE_CLASSES
        ):
            continue

        # A rule line directly beneath a title/subtitle is routine
        # newspaper styling separating a headline from ITS OWN body/
        # figure below it (the same "headline -> rule -> body" strip
        # every story on the page carries), never evidence that a
        # DIFFERENT, unrelated story starts here -- confirmed on a
        # real Malayalam page (doc_000166, page 2): a single-headline
        # article's own subtitle/byline block sat directly above its
        # own body paragraph with a large apparent gap between them,
        # and a printed rule/box-edge pixel band inside that gap was
        # enough to send the whole article through group_blocks,
        # which then risked stranding the lower photo/body as a
        # headless orphan sub-group (see SAFETY and requirement (c)'s
        # _merge_headless_title_subgroups for the backstop this still
        # keeps if that ever happens anyway). This carve-out mirrors
        # the IMAGE_CLASSES one immediately above it -- same
        # reasoning, just for a title's own content instead of a
        # photo's own caption.
        if (
            (getattr(nearest_above, "cls", "") or "").strip().lower()
            == "title"
        ):
            continue

        probe_height = max(nearest_gap, SEPARATOR_MIN_BAND_HEIGHT)

        if has_visual_separator(
            page_image,
            block.x1,
            block.x2,
            block.y1 - probe_height,
            block.y1 + 2,
        ):
            return True

    return False


def split_oversized_articles(
    articles: List[Article],
    page_image=None,
) -> List[Article]:

    result: List[Article] = []

    split_count = 0
    kept_ambiguous = 0
    kept_subordinate = 0
    kept_banner = 0
    kept_package = 0
    aborted_unsafe = 0
    rule_line_triggered = 0

    for article in articles:

        # An "unknown"-role title-class block (ArticleGrouper.
        # _is_uncertain_title -- the grouping model could not name a
        # semantic role for it, most often because its own OCR text is
        # too garbled to read at all) is never trusted as evidence of
        # a genuine second headline here. Confirmed on a real
        # Malayalam page (doc_000166, page 2): a tiny, semantically
        # meaningless title-class fragment ("ലമ", role "unknown") sat
        # inside a story's own logo/graphic area, was correctly folded
        # into that story by the footprint-recovery pass above
        # (ArticleGrouper._recover_unclaimed_blocks_by_footprint) since
        # nothing else could claim it, but then registered as this
        # article's SECOND title -- letting group_blocks below split
        # the story's own rightmost text column and logo away from its
        # real headline as a bogus, headless second "article". Such a
        # block was never excluded from the story it already belongs
        # to (see IGNORE_ROLES/_is_uncertain_title in article_grouper.py
        # -- it stays real, claimed content); it just cannot ALSO count
        # as a second headline candidate here with no reliable
        # evidence behind it.
        title_blocks = [
            block
            for block in article.blocks
            if getattr(block, "cls", None) == "title"
            and not ArticleGrouper._is_uncertain_title(block)
        ]

        rule_line_break = _has_internal_rule_line_break(
            article.blocks, page_image
        )

        if rule_line_break:
            rule_line_triggered += 1

        # A rule line alone is never enough to force a re-check when
        # this article has at most one real headline to begin with --
        # see requirement (b): with no second title anywhere in the
        # article, group_blocks has nothing to root a second story on,
        # so sending it through anyway only risks stranding the body/
        # photo below as a headless orphan (see SAFETY). A genuine
        # second, untitled story hiding in the remaining content is a
        # membership question for the grouping step itself (see the
        # file docstring's "headline-first invariant" note), not
        # something this geometric check can safely act on alone.
        if len(title_blocks) <= 1:
            result.append(article)
            continue

        # Every extra title is a sub-head/inset heading printed
        # smaller than this article's own headline -- no evidence of a
        # second story, so never re-check it on title evidence alone.
        # See SUBORDINATE_TITLE_RATIO above. Same reasoning as the
        # single-title gate just above: at most one DOMINANT title
        # candidate means there is still no second story for
        # group_blocks to root, so a rule line alone does not override
        # this either (requirement (b)) -- only the banner-fragment and
        # column-package checks below, which already require >=2
        # dominant titles, still let a rule line override them.
        dominant_titles = _dominant_class_titles(
            title_blocks, article.blocks
        )

        if len(dominant_titles) <= 1:
            kept_subordinate += 1
            result.append(article)
            continue

        # The extra titles are all fragments of ONE headline the
        # layout detector cut at a column gutter -- again no second
        # story to find, so never re-check it. See BANNER HEADLINE
        # FRAGMENTS above.
        if (
            len(_collapse_banner_fragments(dominant_titles, page_image)) <= 1
            and not rule_line_break
        ):
            kept_banner += 1
            result.append(article)
            continue

        # Every dominant title chains to the others through plain,
        # undivided column gutters -- a same-event package (main
        # report plus attributed reaction/blame columns), not
        # evidence of separate stories. See SAME-EVENT COLUMN
        # PACKAGE above. A detected rule line still overrides this,
        # same as the two checks above.
        if (
            _forms_undivided_column_package(
                dominant_titles, page_image, article.blocks
            )
            and not rule_line_break
        ):
            kept_package += 1
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

        sub_groups = _merge_headless_title_subgroups(
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
        or kept_package
        or aborted_unsafe
        or rule_line_triggered
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
        if kept_package:
            print(
                f"Multi-title articles kept (extra titles are "
                f"undivided same-event columns) : {kept_package}"
            )
        if aborted_unsafe:
            print(
                f"Multi-title articles left unsplit (unsafe -- "
                f"would drop a block) : {aborted_unsafe}"
            )
        if rule_line_triggered:
            print(
                f"Articles re-checked because a printed rule line/"
                f"border was found between two of their own blocks "
                f"(independent of title count) : {rule_line_triggered}"
            )
        print(f"Total articles after split : {len(result)}")
        print("=" * 60)
        print()

    return result


# ============================================================
# DISTANT ORPHAN BLOCKS
#
# A different failure from everything above: not two real headlines
# merged into one article, but a single-title article that the
# grouping LLM also handed a body-text block hundreds of pixels away,
# in a different column, with other complete unrelated articles
# printed in between. split_oversized_articles never re-examines
# this (it only re-checks articles with >1 title block; this one has
# exactly one), so it reaches the final crop untouched -- and because
# the crop is a plain rectangle over the article's own block union
# (see article_region_reconstructor.py), a block genuinely hundreds
# of pixels from the rest of its own article stretches that rectangle
# across everything printed between the two, producing a crop that
# visibly contains several other stories' text.
#
# Confirmed on a real Urdu (THE INQUILAB) page: the grouping LLM's
# own raw response put a block from a completely different, already-
# separately-grouped story (a Jharkhand voter-list report) 1008px
# below -- and in a barely-overlapping column from -- the two blocks
# of an unrelated NSA-custody story, inside the SAME article.
#
# Deliberately RAW vertical gap here, not ArticleGrouper.
# _layout_distance -- that metric's column-overlap penalty is tuned
# for a different question (does this block plausibly ROOT its own
# story) and misfires on exactly the shape a real multi-column
# article takes: a title spanning the full width above narrower,
# non-overlapping body columns beneath it. Confirmed directly: a
# genuine bus-accident story (one banner title, body text in one
# column, a photo and its caption in another) was flagged as 4
# distant orphans by an earlier version of this check that used
# _layout_distance, purely because a full-width title's column-
# overlap with any single narrow column under it is inherently low
# -- see _banner_fragment_plan's own docstring, which already
# documents this exact metric as untrustworthy for an un-widened
# title fragment's bbox. A real multi-column story's blocks stay
# within a comparable Y-band as their own title regardless of which
# column each sits in; only a genuinely misattributed block sits
# hundreds of pixels beyond where any of its own article's other
# blocks reach, in any column.
#
# Restricted to non-title blocks only -- a title anchors its own
# article, so dropping one here is never the safe move; if a title
# is genuinely misplaced that is a membership question for the
# grouping step itself, not this geometric safety net.
#
# UPDATE: raw vertical gap alone (even to the nearest same-article
# neighbor) turned out not to be a reliable break signal either --
# confirmed on a real Urdu page where THREE separate foreign stories'
# worth of content (183-390px gaps apart from each other and from
# the real article) chained onto one BCI-president story, each
# individual gap well within the range this codebase already treats
# as a normal, legitimate same-story gap elsewhere (IMAGE_CAPTION_
# MAX_GAP=250, SUBORDINATE_TITLE_MAX_GAP=600). No fixed pixel
# threshold can separate that from a real long multi-paragraph
# story without also cutting real ones.
#
# The reliable signal instead: whether ANOTHER article's own title
# is physically printed in the gap between two of THIS article's
# blocks. A real single story never has a second, independent
# headline appear in the middle of it -- that is what a headline
# means. Once a foreign title is found sitting in a gap, that block
# and everything further from the article's own title (in reading
# order) is dropped, however small the pixel gaps between those
# trailing blocks are to each other -- confirmed necessary on the
# same page: two of the three foreign clusters (12+11 together, 1px
# apart from EACH OTHER) would each individually look perfectly
# "close" and pass any single-block distance check.
#
# UPDATE: "physically printed in the gap" must be read in BOTH
# dimensions, not just Y. Confirmed on a real Gujarati page
# (doc_000167, page 2): a foreign title from the page's RIGHT-hand
# column had its vertical center land inside a tiny gap between two
# blocks of a completely unrelated LEFT-hand-column article, purely
# by coincidence of where each column's content happened to fall on
# the page -- zero horizontal overlap between the foreign title and
# the column the gap actually belongs to. See _foreign_title_in_gap's
# gap_x1/gap_x2 check.
#
# UPDATE: a foreign TITLE is sufficient evidence but was never
# necessary -- confirmed on a real Urdu (THE INQUILAB, doc_000194
# page 4) page, a "continued from page 1" tag + its body paragraph
# sat 1437px below the rest of their own article, in the same
# column as two OTHER, already fully-grouped articles with no title
# block of their own (headless single/multi-paragraph snippets).
# Neither intervening article ever registered as a "foreign title in
# the gap", so the giant span was never caught. Any real content
# block another article already OWNS is exactly as reliable a
# signal: a prior pass already gave that content a home, which is
# no weaker a claim than a headline is. See _foreign_blocks_by_
# article / _GAP_CONTENT_CLASSES.
# ============================================================


_GAP_CONTENT_CLASSES = ("title", "plain text", "figure", "figure_caption")


def _foreign_blocks_by_article(articles):

    return {
        article.article_id: [
            block
            for block in article.blocks
            if (getattr(block, "cls", None) or "") in _GAP_CONTENT_CLASSES
        ]
        for article in articles
    }


def _foreign_title_in_gap(
    gap_top, gap_bottom, gap_x1, gap_x2, foreign_blocks
):
    """
    Whether some OTHER, already-established article's own content
    physically crosses the gap between two of THIS article's own
    blocks -- both vertically (its center falls inside the gap) AND
    horizontally (its bbox overlaps the column track those two
    blocks occupy, `gap_x1` to `gap_x2`).

    `foreign_blocks` covers every real content-class block (title,
    plain text, figure, figure_caption) another article already
    owns, not just titles -- confirmed necessary on a real Urdu (THE
    INQUILAB, doc_000194 page 4) page: article_001's own blocks
    included a "بقیہ صفحہ اول" (continued from page 1) tag and its
    body paragraph sitting 1437px below the rest of the article, in
    the SAME column, with two other complete, already-grouped
    articles (headless single/multi-paragraph snippets, no title
    block of their own) physically occupying that column in between.
    Restricting this check to titles alone never caught it, because
    neither intervening article had one -- a title is sufficient
    evidence a gap is real, but for an already-OWNED foreign block
    (not merely unclaimed noise) it was never necessary: an article
    a prior pass already built and gave a home to is exactly as good
    evidence that new content starts there as its own headline would
    be.

    The horizontal check is required. A foreign block sitting in an
    entirely different column can share this article's Y-range by
    pure coincidence of newspaper layout -- confirmed on a real
    Gujarati page (doc_000167, page 2): a Messi-retirement headline
    in the RIGHT-hand column (x 1400-1728) had its vertical center
    land in a 7px gap between a leftmost-column (x 48-303) article's
    own title and photo, with zero horizontal overlap between the
    two. Without this check that unrelated title was treated as
    cutting straight through the left-column article, and the
    article's own photo and body paragraph beneath the gap were
    dropped as "distant orphans" even though nothing was ever
    printed between them in their own column.
    """

    if gap_top >= gap_bottom:
        return False

    for block in foreign_blocks:

        center = (block.y1 + block.y2) / 2.0

        if not (gap_top < center < gap_bottom):
            continue

        if block.x2 <= gap_x1 or block.x1 >= gap_x2:
            continue

        return True

    return False


def drop_distant_orphan_blocks(articles: List[Article]) -> List[Article]:

    result: List[Article] = []
    dropped_count = 0

    blocks_by_article = _foreign_blocks_by_article(articles)

    for article in articles:

        if len(article.blocks) < 2:
            result.append(article)
            continue

        foreign_blocks = [
            block
            for other_id, owned in blocks_by_article.items()
            if other_id != article.article_id
            for block in owned
        ]

        ordered = sorted(article.blocks, key=lambda block: block.y1)

        break_index = None

        for index in range(len(ordered) - 1):

            current_block = ordered[index]
            following_block = ordered[index + 1]

            # The gap's own column track is the two blocks' shared
            # (intersected) x-range, not the union of both. Sorting
            # purely by y1 routinely places two blocks that sit in
            # DIFFERENT columns of the same multi-column article next
            # to each other (e.g. a kicker line in one column next to
            # a body paragraph in another, both part of the same
            # story package) -- unioning their x-ranges then invents
            # a wide "track" that a genuinely foreign column's own
            # title can fall inside purely by coincidence. Confirmed
            # on a real Punjabi (Punjabi Jagran, doc_000188 page 2)
            # page: a column-1 kicker line and a column-0 body
            # paragraph, both kept in the same split article, unioned
            # into a track wide enough to catch a completely
            # unrelated column-1 infobox's own title sitting between
            # them by pure y-sort adjacency -- and dropped that
            # column-0 paragraph and everything below it as "distant"
            # even though nothing was printed between them in their
            # own column. When the two blocks share no column at all,
            # there is no real "track" to test either -- skip rather
            # than guess at a substitute band.
            gap_x1 = max(current_block.x1, following_block.x1)
            gap_x2 = min(current_block.x2, following_block.x2)

            if gap_x2 <= gap_x1:
                continue

            if _foreign_title_in_gap(
                current_block.y2,
                following_block.y1,
                gap_x1,
                gap_x2,
                foreign_blocks,
            ):
                break_index = index + 1
                break

        if break_index is None:
            result.append(article)
            continue

        dropped_this_article = [
            block
            for block in ordered[break_index:]
            if getattr(block, "cls", None) != "title"
        ]

        if not dropped_this_article:
            result.append(article)
            continue

        dropped_ids = {block.id for block in dropped_this_article}

        kept_blocks = [
            block for block in article.blocks if block.id not in dropped_ids
        ]

        dropped_count += len(dropped_this_article)

        print(
            f"Distant orphan block(s) detached from article "
            f"{article.article_id} into their own article: "
            f"{sorted(dropped_ids)}"
        )

        result.append(
            Article(
                article_id=article.article_id,
                blocks=kept_blocks,
                block_ids=[block.id for block in kept_blocks],
                confidence=article.confidence,
            )
        )

        # Detached into their OWN article rather than discarded --
        # confirmed necessary on the real Urdu (THE INQUILAB,
        # doc_000194 page 4) page this was generalized for: the
        # dropped group was a "continued from page 1" tag plus its
        # own 1129-character body paragraph, real substantial
        # editorial content, not noise. Silently discarding it would
        # trade "one giant duplicate-content box" for "real content
        # permanently lost", a worse failure. Placeholder id=0, then
        # renumbered below -- same convention as
        # dropped_article_recovery.recover_dropped_articles.
        result.append(
            Article(
                article_id=0,
                blocks=dropped_this_article,
                block_ids=[block.id for block in dropped_this_article],
                confidence=article.confidence,
            )
        )

    if dropped_count:

        # Placeholder article_id=0 values assigned above must never
        # collide with a real one or with each other -- renumber the
        # whole list sequentially, same convention as
        # dropped_article_recovery.recover_dropped_articles.
        for index, article in enumerate(result, start=1):
            article.article_id = index

        print()
        print("=" * 60)
        print("DISTANT ORPHAN BLOCKS")
        print("=" * 60)
        print(f"Blocks detached into their own article(s) : {dropped_count}")
        print("=" * 60)
        print()

    return result


# ============================================================
# WIDE TOP BANNER DETACHMENT
#
# A masthead/banner block that PageCleaner failed to flag as global
# page chrome (e.g. an artistic logo banner with no OCR text) can
# still get pulled into a real story by the grouping model when it
# sits directly above that story's own narrow column blocks. Once
# that happens, the article's bounding rectangle stretches to the
# banner's near-full-page width, while the actual body content
# underneath stays in one or two narrow print columns -- the final
# crop then shows a page-wide rectangle instead of just the real
# column story.
#
# Two signals mark this shape, either sufficient on its own once a
# genuinely wide top block is present:
#
#   1. The article's own topmost block is far wider than the column
#      block(s) beneath it (a banner sitting over narrow columns).
#   2. The article's block-fill ratio (how much of its own bounding
#      rectangle its blocks actually cover) is very low -- a direct
#      symptom of a wide top block dragging the rectangle far past
#      where the narrow content beneath it actually sits.
#
# Either signal detaches ONLY the wide top block into its own
# single-block article; the remaining blocks stay together as the
# original article, now with a bounding rectangle that actually
# matches their own footprint.
# ============================================================


WIDE_TOP_BLOCK_WIDTH_RATIO = 0.65
NARROW_COLUMN_WIDTH_RATIO = 0.35
MIN_BLOCK_FILL_RATIO = 0.30


def _estimate_page_width(articles: List[Article]) -> float:

    widths = [
        block.x2
        for article in articles
        for block in article.blocks
    ]

    return max(widths) if widths else 0.0


def detach_wide_top_banner_blocks(
    articles: List[Article],
    page_width: float = None,
) -> List[Article]:

    if page_width is None:
        page_width = _estimate_page_width(articles)

    if not page_width:
        return articles

    result: List[Article] = []
    detached_count = 0

    for article in articles:

        if len(article.blocks) <= 1:
            result.append(article)
            continue

        ordered = sorted(article.blocks, key=lambda block: block.y1)

        top_block = ordered[0]
        rest = ordered[1:]

        top_width = top_block.x2 - top_block.x1

        wide_top = top_width > WIDE_TOP_BLOCK_WIDTH_RATIO * page_width

        if not wide_top:
            result.append(article)
            continue

        narrow_columns_below = all(
            (block.x2 - block.x1) < NARROW_COLUMN_WIDTH_RATIO * page_width
            for block in rest
        )

        bbox_x1 = min(block.x1 for block in article.blocks)
        bbox_y1 = min(block.y1 for block in article.blocks)
        bbox_x2 = max(block.x2 for block in article.blocks)
        bbox_y2 = max(block.y2 for block in article.blocks)

        bbox_area = max(
            1.0,
            (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1),
        )

        blocks_area = sum(
            max(0.0, block.x2 - block.x1) * max(0.0, block.y2 - block.y1)
            for block in article.blocks
        )

        low_fill = (blocks_area / bbox_area) < MIN_BLOCK_FILL_RATIO

        if not (narrow_columns_below or low_fill):
            result.append(article)
            continue

        detached_count += 1

        result.append(
            Article(
                article_id=article.article_id,
                blocks=rest,
                block_ids=[block.id for block in rest],
                confidence=article.confidence,
            )
        )

        # Placeholder id=0, renumbered below -- same convention as
        # drop_distant_orphan_blocks above.
        result.append(
            Article(
                article_id=0,
                blocks=[top_block],
                block_ids=[top_block.id],
                confidence=article.confidence,
            )
        )

    if detached_count:

        for index, article in enumerate(result, start=1):
            article.article_id = index

        print()
        print("=" * 60)
        print("WIDE TOP BANNER DETACHMENT")
        print("=" * 60)
        print(f"Wide top blocks detached : {detached_count}")
        print("=" * 60)
        print()

    return result
