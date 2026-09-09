"""
Diagnostic-only pass: flags when one article's plain outer rectangle
geometrically overlaps content owned by ANOTHER article.

WHY THIS EXISTS
---------------

BoundaryBuilder (see article_region_reconstructor.py, which now does
that work) turns an article into one axis-aligned rectangle by taking
min/max over its own blocks. Newspaper articles are frequently
L-shaped or multi-column -- a story whose body runs down the left
columns but whose photograph sits across the top-right produces a
block set whose min/max rectangle also spans the empty leg of the L,
which can happen to fall over whatever the newspaper printed there.

THIS IS EXPECTED AND NO LONGER "FIXED" HERE
--------------------------------------------

An earlier version of this module reacted to that overlap by
reshaping the article into several precise sub-rectangles (or, before
that, by splitting it into multiple articles outright). Both of those
reactions produced the exact failure this whole pipeline stage exists
to prevent: a single logical, correctly-grouped newspaper article
(one headline anchoring several columns and its own photograph) came
out as SEVERAL visible boundaries instead of ONE. A rectangle that
contains internal whitespace -- or happens to visually cross a
neighbour's empty margin -- is not evidence that the rectangle is
wrong; it is the ordinary shape of a real multi-column newspaper
story (see article_region_reconstructor.py, STEP 8).

MEMBERSHIP is what actually decides which content belongs to which
article, and membership is decided exclusively upstream (ArticleGrouper,
dropped_article_recovery, article_splitter.py) and never by this
module. This pass no longer changes geometry OR membership at all --
it only prints a diagnostic when an article's rectangle happens to
geometrically overlap another article's own blocks, purely so that
overlap stays visible in the pipeline log for a human to sanity-check
against the actual crops (which are always built from each article's
OWN blocks -- see FinalArticleCropper -- so a geometric overlap here
never means one article's TEXT was stolen by another).

This runs after article_splitter.py, which is where a genuine
model over-merge of two independent stories is caught and split
(headline-first root detection, not bare rectangle geometry).
"""

# Classes that represent real printed content. A block of any other
# class (a rule line, a colour swatch) is not treated as something
# worth flagging when it falls inside another article's rectangle.
CONTENT_CLASSES = {
    "title",
    "plain text",
    "figure",
    "figure_caption",
}

# Fraction of a foreign block's own area that must fall inside the
# article's rectangle before it is worth flagging. A block merely
# clipped by the edge is not evidence of anything.
INTRUSION_INSIDE_RATIO = 0.75

# A foreign block overlapping the article's OWN blocks by more than
# this is a duplicate/overlapping detection of content the article
# already holds, not a neighbour's content sitting in a gap.
INTRUSION_OWN_OVERLAP_MAX = 0.30

# Minimum characters before an UNOWNED (unclaimed) text block is
# worth flagging. Blocks owned by another article always count
# regardless of text, since their ownership is already established.
MIN_INTRUSION_CHARS = 40


def _rect(blocks):
    return (
        min(b.x1 for b in blocks),
        min(b.y1 for b in blocks),
        max(b.x2 for b in blocks),
        max(b.y2 for b in blocks),
    )


def _area(rect):
    return max(0, rect[2] - rect[0]) * max(0, rect[3] - rect[1])


def _intersection(a, b):
    return (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )


def _block_rect(block):
    return (block.x1, block.y1, block.x2, block.y2)


def _inside_ratio(block, rect):
    """Fraction of `block`'s own area that falls inside `rect`."""

    block_area = _area(_block_rect(block))

    if block_area <= 0:
        return 0.0

    return _area(_intersection(_block_rect(block), rect)) / block_area


def find_foreign_overlap(article, blocks, owner_of):
    """
    Foreign blocks (owned by a DIFFERENT article, or substantively
    unclaimed) whose own rectangle mostly falls inside `article`'s
    plain outer rectangle.

    Purely informational -- see module docstring. Returns [] for the
    overwhelming majority of articles.
    """

    if len(article.blocks) < 1:
        return []

    rect = _rect(article.blocks)
    own_ids = set(article.block_ids)

    intrusions = []

    for block in blocks:

        if block.id in own_ids:
            continue

        if (block.cls or "").strip().lower() not in CONTENT_CLASSES:
            continue

        is_foreign_owned = (
            block.id in owner_of and owner_of[block.id] != article.article_id
        )

        if not is_foreign_owned:
            # Unclaimed: only substantive printed text is worth
            # flagging, so a stray sliver never shows up as noise.
            if len((getattr(block, "text", "") or "").strip()) < (
                MIN_INTRUSION_CHARS
            ):
                continue

        if _inside_ratio(block, rect) < INTRUSION_INSIDE_RATIO:
            continue

        block_area = _area(_block_rect(block))

        if block_area <= 0:
            continue

        own_overlap = max(
            (
                _area(_intersection(_block_rect(block), _block_rect(own)))
                / block_area
                for own in article.blocks
            ),
            default=0.0,
        )

        if own_overlap > INTRUSION_OWN_OVERLAP_MAX:
            # Same printed thing, detected twice -- not a neighbour.
            continue

        intrusions.append(block)

    return intrusions


def decompose_overlapping_articles(articles, blocks):
    """
    Diagnostic pass: logs when an article's plain outer rectangle
    geometrically overlaps a DIFFERENT article's own content.

    Never mutates membership, never mutates geometry, never changes
    the number of articles -- returns `articles` unchanged. See the
    module docstring for why this stopped reshaping/splitting.
    """

    if not articles:
        return articles

    owner_of = {}

    for article in articles:
        for block_id in article.block_ids:
            owner_of[block_id] = article.article_id

    flagged = 0

    for article in articles:

        overlap = find_foreign_overlap(article, blocks, owner_of)

        if not overlap:
            continue

        flagged += 1

        print(
            f"NOTE: article {article.article_id}'s outer rectangle "
            f"geometrically overlaps {len(overlap)} block(s) it does "
            f"not own: {[b.id for b in overlap]}. This is expected "
            f"for an L-shaped/multi-column article and does NOT move "
            f"any content -- each article's crop is still built only "
            f"from its own blocks. Flagged for visibility only."
        )

    if flagged:

        print()
        print("=" * 60)
        print("BOUNDARY OVERLAP DIAGNOSTIC")
        print("=" * 60)
        print(
            f"Articles whose rectangle overlaps another article's "
            f"content : {flagged} / {len(articles)}"
        )
        print("=" * 60)
        print()

    return articles
