"""
Absorbs a small satellite article that is really a callout/sidebar box
of a larger, dominant article into that dominant article.

WHY THIS EXISTS
---------------

Confirmed on a real Marathi (Divya Marathi, doc_000186 page 1)
document: a composite feature (a banner headline, a photo, and a
continuing body paragraph -- the RSS chief's New York speech) prints
two half-column pull-quote panels ("teaching:" / "resolution:")
directly beneath its own photo, followed by a third, wider panel
("reactions from across the political spectrum") beneath both. Every
one of the three reads, in isolation, exactly like a real short news
brief -- a short headline over a short paragraph -- so the grouping
model gave each one its own separate, fully-formed article rather than
listing it in the parent's own "blocks" array. Every repair pass in
article_grouper.py only ever reattaches an UNCLAIMED block or moves a
block BETWEEN articles the model already built; none of them ever
folds one already-complete article into another, which is what a
boxed satellite needs.

WHAT THIS DOES
--------------

Two narrow, geometry-only passes, run to a fixed point so a stack of
several panels is folded in one at a time:

1. GROUP ABSORPTION: find a "row" of 2+ articles that are mutually
   side-by-side (near-identical top/bottom edges, no meaningful
   horizontal overlap with each other) and fold that whole row into
   the ONE other article that dominates it -- see _find_dominant_
   parent for the exact shape required.

2. CHAIN ABSORPTION: after a row has been folded into a parent, look
   for a single article sitting immediately beneath that (now taller)
   parent, still meeting the same dominance/containment/width test,
   and fold it in too. This is what lets a third, un-paired panel
   (the "reactions" box in the confirmed case above) attach after the
   paired row above it already has.

SAFETY -- WHY EVERY THRESHOLD BELOW IS THIS TIGHT
--------------------------------------------------

A "row of small articles sitting under a bigger one, in its own
column span" is NOT on its own evidence of a boxed satellite -- it is
also exactly the shape of two completely unrelated one-column briefs
that simply start at the same grid row underneath a longer story, or
of a big story's own next-door neighbour beginning right where the
photo above it ends. Both were confirmed as REAL pages in this
project's own corpus (a Kannada sports page pairing an unrelated rally
-racing brief with a junior-tennis brief under a cricket report; an
English world-news page pairing a Pakistan-temple brief with a NYC
-schools brief under a Nepal-flood report) and both were produced by
an earlier, looser version of this pass that lacked one or more of the
guards below -- each guard exists because removing it was measured to
reproduce one of those two false merges:

- MAX_WIDTH_RATIO: a genuine boxed panel occupies a MINORITY of its
  parent's own column width (it shares that row with the parent's own
  continuing text, or with a sibling panel) -- confirmed 0.55-0.65 in
  the real composite-panel pages above. An unrelated next brief that
  merely starts on the same grid row typically reproduces the FULL
  width of the article above it (confirmed 0.997 and 1.00 on two real,
  wrongly-merged pages) because it is not sharing that row with
  anything, it is simply the next item in the same single column.
- MAX_ABSORPTION_GAP: kept deliberately tight (an ordinary paragraph
  -to-paragraph gap, not a full inter-article gutter). The confirmed
  false merges above sat 24-96px below their false "parent"; every
  confirmed real panel in this project's corpus sits within 9px
  (touching or overlapping).
- MIN_DOMINANCE_AREA_RATIO: the parent must be a substantially bigger
  story, not a comparably-sized neighbour.
- Horizontal containment: the candidate row/satellite must fall
  inside the parent's OWN established column span, not merely share
  some pixels with it.

Even with all four guards, this remains a heuristic tuned against one
project's corpus, not a proof -- see the module-level ambiguity guard
below (more than one equally-good parent skips the merge rather than
guessing), the same conservative default
ArticleGrouper._recover_unclaimed_blocks_by_footprint already uses.
"""

from pipeline.article.article_grouper import Article
from pipeline.article.visual_separator import has_visual_separator


# Two articles' top edges (and, separately, their bottom edges) must
# be within this many px of each other to count as sharing one
# printed "row" of panels.
ROW_ALIGN_TOLERANCE = 20.0

# Two articles in a candidate row must overlap horizontally by no more
# than this fraction of the narrower one's width -- they must be
# genuinely side-by-side, not two overlapping reads of the same
# column.
MAX_GROUP_X_OVERLAP_RATIO = 0.2

# Slack allowed when checking whether a satellite/row's own horizontal
# span falls inside its candidate parent's -- a panel's own edge
# routinely sits a handful of px past its parent's outermost block
# (its own border/padding), not because it belongs to a different
# column.
HORIZONTAL_TOLERANCE = 40.0

# The parent must be at least this many times the satellite/row's own
# bbox area -- "dominant story vs. boxed excerpt of it", not two
# comparably-sized items that happen to sit close together.
MIN_DOMINANCE_AREA_RATIO = 2.0

# The satellite/row's own combined width must be no more than this
# fraction of the parent's own width. See SAFETY above -- this is
# what tells a genuine boxed panel (a MINORITY of the parent's column,
# sharing its row with the parent's own continuing content or a
# sibling panel) apart from the next unrelated brief in the same
# single column (which reproduces the parent's full width instead).
MAX_WIDTH_RATIO = 0.8

# Max vertical gap between a candidate parent's own footprint and the
# satellite/row for them to count as adjacent. Deliberately an
# ordinary paragraph-to-paragraph gap, not a full inter-article
# gutter -- see SAFETY above.
MAX_ABSORPTION_GAP = 20.0


def _rect(blocks):
    return (
        min(b.x1 for b in blocks),
        min(b.y1 for b in blocks),
        max(b.x2 for b in blocks),
        max(b.y2 for b in blocks),
    )


def _width(rect):
    return max(0.0, rect[2] - rect[0])


def _area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _x_overlap_ratio(rect_a, rect_b):
    overlap = max(
        0.0, min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0])
    )
    return overlap / min(_width(rect_a), _width(rect_b))


def _combined_rect(rects):
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


def _find_row_groups(rects_by_id):
    """
    Union-find over article ids: two articles join the same group when
    their top edges align, their bottom edges align, and they do not
    meaningfully overlap horizontally -- see ROW_ALIGN_TOLERANCE /
    MAX_GROUP_X_OVERLAP_RATIO. Returns every group of size >= 2.
    """

    ids = list(rects_by_id.keys())

    parent = {aid: aid for aid in ids}

    def find(aid):
        while parent[aid] != aid:
            parent[aid] = parent[parent[aid]]
            aid = parent[aid]
        return aid

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):

            rect_a = rects_by_id[ids[i]]
            rect_b = rects_by_id[ids[j]]

            if abs(rect_a[1] - rect_b[1]) > ROW_ALIGN_TOLERANCE:
                continue

            if abs(rect_a[3] - rect_b[3]) > ROW_ALIGN_TOLERANCE:
                continue

            if _x_overlap_ratio(rect_a, rect_b) > MAX_GROUP_X_OVERLAP_RATIO:
                continue

            union(ids[i], ids[j])

    groups = {}

    for aid in ids:
        groups.setdefault(find(aid), []).append(aid)

    return [members for members in groups.values() if len(members) >= 2]


def _find_dominant_parent(
    candidate_rect, rects_by_id, excluded_ids, page_image=None
):
    """
    The ONE article whose own footprint dominates and contains
    `candidate_rect` -- see the module docstring's SAFETY section for
    why every one of these conditions is required. Returns (None,
    None) when zero or more-than-one article qualifies -- ambiguity is
    skipped rather than guessed, same convention as
    ArticleGrouper._recover_unclaimed_blocks_by_footprint.

    A printed rule line found in the gap between the parent's own
    footprint and the candidate (has_visual_separator, the same probe
    ArticleGrouper._has_separator_between already relies on) vetoes
    that parent -- the newspaper itself marked the gap as a real
    division, which must never be overridden by geometry alone. A
    no-op whenever page_image is None.
    """

    candidate_area = _area(candidate_rect)
    candidate_width = _width(candidate_rect)

    best_id = None
    best_gap = None
    ambiguous = False

    for parent_id, parent_rect in rects_by_id.items():

        if parent_id in excluded_ids:
            continue

        if parent_rect[0] > candidate_rect[0] + HORIZONTAL_TOLERANCE:
            continue

        if parent_rect[2] < candidate_rect[2] - HORIZONTAL_TOLERANCE:
            continue

        if _area(parent_rect) < MIN_DOMINANCE_AREA_RATIO * candidate_area:
            continue

        if candidate_width > MAX_WIDTH_RATIO * _width(parent_rect):
            continue

        # The parent must already be established by the time the
        # candidate begins -- a satellite never absorbs into
        # something that starts AFTER it.
        if parent_rect[1] > candidate_rect[1] + ROW_ALIGN_TOLERANCE:
            continue

        gap = max(0.0, candidate_rect[1] - parent_rect[3])

        if gap > MAX_ABSORPTION_GAP:
            continue

        if page_image is not None and has_visual_separator(
            page_image,
            min(candidate_rect[0], parent_rect[0]),
            max(candidate_rect[2], parent_rect[2]),
            parent_rect[3],
            candidate_rect[1],
        ):
            continue

        if best_gap is None or gap < best_gap:
            best_id, best_gap, ambiguous = parent_id, gap, False
        elif gap == best_gap:
            ambiguous = True

    if ambiguous:
        return None, None

    return best_id, best_gap


def absorb_composite_sideboxes(articles, page_image=None):
    """
    Returns a new list of Article objects with any confirmed boxed
    satellite (or matched row of them) folded into its dominant
    parent. Articles this pass does not touch are returned by
    reference, unchanged -- same convention as
    headless_headline_relink.relink_headless_headlines.

    `page_image` is accepted only for a final, optional hard-separator
    veto (a printed rule line found in the gap between parent and
    satellite blocks every OTHER absorption pass here already
    respects) -- every geometric guard above runs identically with or
    without it.
    """

    if len(articles) < 2:
        return articles

    live = {article.article_id: list(article.blocks) for article in articles}

    absorbed = []
    grown_parent_ids = set()

    # STAGE 1 -- matched-row absorption, to a fixed point: each
    # absorption can change which rows/parents are available, so the
    # whole scan restarts after every successful merge.
    changed = True

    while changed:

        changed = False

        rects_by_id = {
            aid: _rect(blocks) for aid, blocks in live.items() if blocks
        }

        for group_ids in _find_row_groups(rects_by_id):

            if any(gid not in live for gid in group_ids):
                continue

            group_rect = _combined_rect(
                [rects_by_id[gid] for gid in group_ids]
            )

            parent_id, gap = _find_dominant_parent(
                group_rect,
                rects_by_id,
                excluded_ids=set(group_ids),
                page_image=page_image,
            )

            if parent_id is None:
                continue

            for gid in group_ids:
                live[parent_id] = live[parent_id] + live[gid]
                del live[gid]
                absorbed.append((gid, parent_id))

            grown_parent_ids.add(parent_id)
            changed = True
            break

    # STAGE 2 -- chain absorption: a lone satellite sitting
    # immediately beneath a parent that Stage 1 just grew. Restricted
    # to already-grown parents so this can never fire on a page with
    # no confirmed Stage 1 absorption (see the module docstring's
    # SAFETY section -- the false-merge pages found in this project's
    # own corpus never had a Stage 1 match to chain from either).
    changed = True

    while changed:

        changed = False

        rects_by_id = {
            aid: _rect(blocks) for aid, blocks in live.items() if blocks
        }

        for satellite_id in sorted(
            rects_by_id, key=lambda aid: rects_by_id[aid][1]
        ):

            if satellite_id in grown_parent_ids:
                continue

            candidates = {
                pid: rect
                for pid, rect in rects_by_id.items()
                if pid in grown_parent_ids
            }

            parent_id, gap = _find_dominant_parent(
                rects_by_id[satellite_id],
                candidates,
                excluded_ids={satellite_id},
                page_image=page_image,
            )

            if parent_id is None:
                continue

            live[parent_id] = live[parent_id] + live[satellite_id]
            del live[satellite_id]
            absorbed.append((satellite_id, parent_id))
            changed = True
            break

    if not absorbed:
        return articles

    result = []

    for article in articles:

        blocks = live.get(article.article_id)

        if blocks is None:
            # Absorbed into another article -- dropped from the
            # output entirely, its blocks now living under the
            # parent's own article_id.
            continue

        blocks = sorted(blocks, key=lambda b: getattr(b, "reading_order", 0))

        result.append(
            Article(
                article_id=article.article_id,
                blocks=blocks,
                block_ids=[block.id for block in blocks],
                confidence=article.confidence,
            )
        )

    print()
    print("=" * 60)
    print("COMPOSITE SIDEBOX ABSORPTION")
    print("=" * 60)

    for satellite_id, parent_id in absorbed:
        print(
            f"  article {satellite_id} -> absorbed into "
            f"article {parent_id}"
        )

    print(f"Sideboxes absorbed : {len(absorbed)}")
    print("=" * 60)
    print()

    return result
