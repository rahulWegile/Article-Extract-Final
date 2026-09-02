"""
Splits an article whose bounding RECTANGLE swallows content it does
not own.

WHY THIS EXISTS
---------------

BoundaryBuilder turns an article into one axis-aligned rectangle by
taking min/max over its blocks. That is correct only when the
article's blocks actually fill that rectangle. Newspaper articles
frequently do not: a story whose body runs down the left columns but
whose photograph sits across the top-right produces an L-shaped block
set, and the min/max rectangle over an L covers the empty leg too --
along with whatever story the newspaper printed there.

The grouping prompts already state this rule for the model ("NEVER
assume that one large visual rectangle represents one article ... The
article boundary is NOT determined from an imaginary outer rectangle
around all nearby content"), but nothing enforced it on our side:
BoundaryBuilder computed exactly that imaginary outer rectangle.

Confirmed on a real Tamil page (Indhu Tamizh Thisai, page 2): a
returnees story occupying the left half plus a relief photograph in
the top-right produced the rectangle (113, 3567)-(4017, 6420) -- the
entire bottom half of the page -- which visually enclosed a separate
temperature-forecast story printed in the bottom-right. Nothing had
grouped the two stories together; one article's rectangle had simply
grown over the other.

WHAT THIS DOES
--------------

For every article, find the blocks that INTRUDE into its rectangle --
blocks it does not own that are not merely overlapping detections of
its own blocks. If there are none, the article is left untouched
(the overwhelmingly common case).

If there are intrusions, the article's own blocks are agglomerated
into the fewest groups whose individual rectangles are each
intrusion-free, always merging the closest pair first so a story's
own tightly-spaced columns bind together before any wide jump across
the page is considered.

What happens to those groups depends on whether they look like
genuinely separate page items:

- If every group looks like a standalone item on its own (a picture,
  or a headline with at least one companion block) AND splitting
  would not sever a headline from the body printed directly beneath
  it (see _splits_a_continuous_flow), each group becomes its OWN
  article -- the model over-merged two real stories into one
  rectangle, and this un-merges them.

- Otherwise (a group is just body text with no headline/picture of
  its own, or splitting would cut a story's own headline away from
  its own body), the content has to stay ONE article -- but it does
  not have to stay ONE rectangle. The article keeps its original
  article_id and block_ids, and is instead given a precise
  multi-rectangle shape (Article.sub_rects) covering exactly its own
  blocks, which avoids the neighbour without inventing any content.
  This is always safe when the groups can be geometrically separated
  at all (see _groups_are_mutually_clear) because, unlike spawning a
  new article, it asserts nothing about what any group "looks like."

- Only when the groups themselves cannot be separated without
  covering each other does this give up and leave the single,
  oversized rectangle exactly as it was -- no safe geometry exists
  either way.

This runs after article_splitter.py, which solves the different
problem of the model semantically merging two stories into one group.
Here the grouping may be perfectly correct and only the rectangle is
wrong.
"""

import math

from pipeline.article.article_grouper import Article


# Classes that represent real printed content. A block of any other
# class (a rule line, a colour swatch) is not treated as something an
# article's rectangle must avoid.
CONTENT_CLASSES = {
    "title",
    "plain text",
    "figure",
    "figure_caption",
}

# A picture stands on its own with no other block required -- a
# photo-only teaser needs no caption to be a real page item.
STANDALONE_ALONE_CLASSES = {
    "figure",
}

# A headline needs a companion block (body text, a caption, another
# title) before its group counts as standalone. A lone title block by
# itself is not enough: ArticleGrouper's own
# _reassign_orphan_title_roots treats a body-less title stranded in a
# single-block article as a raw grouping mistake to be repaired, not a
# valid article shape, and this pass must not manufacture the exact
# pattern that one exists to clean up. Confirmed on a real Gujarati
# page (Divya Bhaskar): a masthead/date-line strip mis-tagged as
# three "title" blocks by the grouping model split into three
# single-block slivers under the old any-title-counts rule -- three
# near-empty junk articles instead of one. Character-count text
# thresholds were tried and rejected: the date line ("Ahmedabad,
# Gujarat, Tuesday, September 1 2026") is 40-odd recognisable
# characters, comfortably past any plausible minimum, while carrying
# no story content at all -- length alone cannot separate a real
# headline from page furniture, but requiring company can.
STANDALONE_WITH_COMPANION_CLASSES = {
    "title",
}

# Fraction of an intruding block's own area that must fall inside the
# article's rectangle before it counts as intruding. A block merely
# clipped by the edge is not evidence of a swallowed story.
INTRUSION_INSIDE_RATIO = 0.75

# An intruder overlapping the article's OWN blocks by more than this
# is a duplicate/overlapping detection of content the article already
# holds, not a separate story sitting in a gap.
INTRUSION_OWN_OVERLAP_MAX = 0.30

# Minimum characters before an UNOWNED text block counts as an
# intrusion. Blocks owned by another article always count regardless
# of text, since their ownership is already established.
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


def _gap_distance(rect_a, rect_b):
    """Edge-to-edge distance; 0 when the two rectangles touch."""

    dx = max(0, max(rect_a[0], rect_b[0]) - min(rect_a[2], rect_b[2]))
    dy = max(0, max(rect_a[1], rect_b[1]) - min(rect_a[3], rect_b[3]))

    return math.hypot(dx, dy)


def _find_intrusions(article_blocks, foreign_blocks):
    """
    Blocks that sit inside this article's rectangle without belonging
    to it, and without being an overlapping detection of one of its
    own blocks.
    """

    rect = _rect(article_blocks)

    intrusions = []

    for block in foreign_blocks:

        if _inside_ratio(block, rect) < INTRUSION_INSIDE_RATIO:
            continue

        block_area = _area(_block_rect(block))

        if block_area <= 0:
            continue

        # Covered by content this article already owns -> the same
        # printed thing, detected twice, not an intruder.
        own_overlap = max(
            (
                _area(_intersection(_block_rect(block), _block_rect(own)))
                / block_area
                for own in article_blocks
            ),
            default=0.0,
        )

        if own_overlap > INTRUSION_OWN_OVERLAP_MAX:
            continue

        intrusions.append(block)

    return intrusions


def _rect_is_clear(rect, intrusions):
    """True when no intruding block falls inside `rect`."""

    for block in intrusions:

        if _inside_ratio(block, rect) >= INTRUSION_INSIDE_RATIO:
            return False

    return True


def _groups_are_mutually_clear(groups):
    """
    True when no group's rectangle swallows another group's blocks.

    _agglomerate only guarantees each group avoids the intrusions
    computed for the article as a WHOLE. It says nothing about the
    groups' relationship to each other, and a split that leaves one
    piece's rectangle covering another piece's blocks has simply
    moved the swallowed-content problem rather than solved it --
    measured turning one 6-intrusion page into a 20-intrusion one.

    A split that cannot separate its own pieces is abandoned, which
    leaves the article exactly as it was.
    """

    for i, group in enumerate(groups):

        rect = _rect(group)

        for j, other in enumerate(groups):

            if i == j:
                continue

            for block in other:

                if _inside_ratio(block, rect) < INTRUSION_INSIDE_RATIO:
                    continue

                block_area = _area(_block_rect(block))

                if block_area <= 0:
                    continue

                # An overlapping detection of something this group
                # already owns is the same printed thing twice, not
                # a sibling's content sitting in this rectangle.
                own_overlap = max(
                    (
                        _area(
                            _intersection(
                                _block_rect(block),
                                _block_rect(own),
                            )
                        )
                        / block_area
                        for own in group
                    ),
                    default=0.0,
                )

                if own_overlap > INTRUSION_OWN_OVERLAP_MAX:
                    continue

                return False

    return True


def _agglomerate(article_blocks, intrusions):
    """
    Merge the article's blocks into the fewest groups whose own
    rectangles are each free of every intrusion.

    Closest pair first: a story's own columns sit a few pixels apart,
    while the jump across the page to its stranded photograph is an
    order of magnitude wider, so the story binds together before any
    merge that would re-swallow the intruding content is considered.
    """

    groups = [[block] for block in article_blocks]

    while len(groups) > 1:

        best_pair = None
        best_distance = None

        for i in range(len(groups)):

            for j in range(i + 1, len(groups)):

                merged = _rect(groups[i] + groups[j])

                if not _rect_is_clear(merged, intrusions):
                    continue

                distance = _gap_distance(
                    _rect(groups[i]),
                    _rect(groups[j]),
                )

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair

        groups[i] = groups[i] + groups[j]

        del groups[j]

    return groups


def _is_standalone(group):
    """
    A group that can exist as its own page item: a picture (needs
    nothing else), or a headline WITH a companion block. A lone title
    block never qualifies on its own -- see
    STANDALONE_WITH_COMPANION_CLASSES.
    """

    classes = [(block.cls or "").strip().lower() for block in group]

    if any(cls in STANDALONE_ALONE_CLASSES for cls in classes):
        return True

    if len(group) >= 2 and any(
        cls in STANDALONE_WITH_COMPANION_CLASSES for cls in classes
    ):
        return True

    return False


def _standalone_only_via_title(group):
    """
    True when `group` passes `_is_standalone` ONLY because it holds a
    title with a companion block, not because it holds a picture in
    its own right.

    A lone photo genuinely can stand on its own regardless of what
    sits next to it. "A headline with a companion block" is a much
    weaker signal: it is equally the shape of a real standalone
    teaser AND the shape of a headline (plus byline) that is still
    attached to the body group sitting right after it. Only the
    latter needs the reading-flow check in
    `_splits_a_continuous_flow` -- a group that already stands on its
    own via a picture is left alone.
    """

    classes = [(block.cls or "").strip().lower() for block in group]

    if any(cls in STANDALONE_ALONE_CLASSES for cls in classes):
        return False

    return len(group) >= 2 and any(
        cls in STANDALONE_WITH_COMPANION_CLASSES for cls in classes
    )


def _has_substantive_body_text(group):
    """
    True when `group` holds a real running-prose block: a "plain
    text"-class block with at least MIN_INTRUSION_CHARS of text --
    the same bar this module already uses elsewhere to tell a real
    paragraph from a stray sliver. A picture's own caption (class
    "figure_caption", not "plain text") does not count, so a genuine
    photo-plus-caption teaser is never mistaken for a story's body.
    """

    for block in group:

        if (block.cls or "").strip().lower() != "plain text":
            continue

        if len((getattr(block, "text", "") or "").strip()) >= (
            MIN_INTRUSION_CHARS
        ):
            return True

    return False


def _splits_a_continuous_flow(article_blocks, groups):
    """
    True when two of the resulting groups are back-to-back in the
    ARTICLE'S OWN internal reading order -- the last block one group
    ends on (by reading_order, restricted to this article's own
    blocks) is immediately followed by the first block the next group
    starts on, with nothing else of this article between them -- AND
    one of the two groups stands alone only via a title + companion
    (see `_standalone_only_via_title`) while the OTHER holds real
    running body text (see `_has_substantive_body_text`).

    This is exactly the shape of a headline cut away from the body
    printed directly beneath it: the grouping model read the two as
    one continuous story with nothing of its own in between, and
    decompose_overlapping_articles's whole job is to fix a rectangle
    that overreaches, never to re-litigate content the grouper already
    got right (see module docstring).

    The body-text requirement on the OTHER group is what keeps this
    from also blocking the split this module was built to make (see
    the Tamil example above): a genuine standalone photo teaser split
    off from a text column holds no running prose of its own, so that
    split is untouched. A headline severed from its own elaborated
    story, by contrast, is always adjacent to a group that DOES carry
    real paragraphs -- even if that group also happens to enclose the
    story's photo, as below.

    Confirmed on a real Kannada page (Kannada Prabha): a 10-block
    article (headline, subhead, byline, image, caption, 6 body
    paragraphs) got cut into a 2-block "headline" group ({headline,
    subhead}) and an 8-block "body" group ({byline, image, caption, 6
    paragraphs}) purely because the byline block sat 4px from the
    first body paragraph vs. 5px from the subhead above it -- a 1px
    tie-break in `_agglomerate` separating a story from its own
    headline. Both groups happened to independently pass
    `_is_standalone` (each held its own title-class block -- the
    byline was itself layout-detected as class "title"), so nothing
    else caught it.
    """

    ordered = sorted(
        article_blocks,
        key=lambda block: getattr(block, "reading_order", 0),
    )

    group_of = {}

    for index, group in enumerate(groups):
        for block in group:
            group_of[block.id] = index

    title_only = [_standalone_only_via_title(group) for group in groups]
    has_body = [_has_substantive_body_text(group) for group in groups]

    for i in range(len(ordered) - 1):

        group_a = group_of[ordered[i].id]
        group_b = group_of[ordered[i + 1].id]

        if group_a == group_b:
            continue

        if title_only[group_a] and has_body[group_b]:
            return True

        if title_only[group_b] and has_body[group_a]:
            return True

    return False


def decompose_overlapping_articles(articles, blocks):
    """
    Re-cut any article whose rectangle encloses content it does not
    own. Returns a new article list, renumbered from 1.
    """

    if not articles:
        return articles

    owner_of = {}

    for article in articles:
        for block_id in article.block_ids:
            owner_of[block_id] = article.article_id

    def foreign_for(article):

        foreign = []

        own_ids = set(article.block_ids)

        for block in blocks:

            if block.id in own_ids:
                continue

            if (block.cls or "").strip().lower() not in CONTENT_CLASSES:
                continue

            if block.id in owner_of:
                # Belongs to a different article -- always an
                # intruder if it sits inside this rectangle.
                foreign.append(block)
                continue

            # Unclaimed: only substantive printed text counts, so a
            # stray sliver never triggers a re-cut.
            if len((getattr(block, "text", "") or "").strip()) >= (
                MIN_INTRUSION_CHARS
            ):
                foreign.append(block)

        return foreign

    result = []

    split_count = 0
    reshaped_count = 0
    aborted_overlap = 0
    clean_count = 0

    for article in articles:

        if len(article.blocks) < 2:
            result.append(article)
            clean_count += 1
            continue

        intrusions = _find_intrusions(
            article.blocks,
            foreign_for(article),
        )

        if not intrusions:
            result.append(article)
            clean_count += 1
            continue

        groups = _agglomerate(article.blocks, intrusions)

        if len(groups) <= 1:
            # The intrusion sits inside a single block of this
            # article (a photo printed over text, say). Nothing to
            # cut.
            result.append(article)
            clean_count += 1
            continue

        if not _groups_are_mutually_clear(groups):
            # The pieces cannot be separated without one of them
            # covering another's content -- neither splitting into
            # separate articles nor reshaping this one into precise
            # sub-rectangles would avoid that. No safe geometry
            # exists; keep the oversized rectangle. See
            # _groups_are_mutually_clear.
            aborted_overlap += 1
            result.append(article)
            continue

        splittable = all(
            _is_standalone(group) for group in groups
        ) and not _splits_a_continuous_flow(article.blocks, groups)

        if splittable:
            # Every piece looks like a real standalone page item
            # (a picture, or a headline with a companion) and none of
            # them is just this article's own headline severed from
            # its own body -- these are genuinely separate stories
            # the model over-merged into one rectangle. Split into
            # their own articles.

            groups.sort(
                key=lambda group: (_rect(group)[1], _rect(group)[0])
            )

            for group in groups:

                group.sort(
                    key=lambda block: getattr(block, "reading_order", 0)
                )

                result.append(
                    Article(
                        article_id=article.article_id,
                        blocks=group,
                        block_ids=[block.id for block in group],
                        confidence=article.confidence,
                    )
                )

            split_count += 1
            continue

        # Splitting would strand a headline-less/picture-less
        # fragment, or sever a headline from the body printed
        # directly beneath it -- this has to stay ONE article. It
        # can still be given a precise multi-rectangle shape instead
        # of a single rectangle that keeps enclosing the neighbour:
        # unlike spawning a new article, this invents no content and
        # so needs no standalone-shape safety check -- the only
        # requirement is the one already checked above, that the
        # pieces don't cover each other.
        article.sub_rects = [_rect(group) for group in groups]

        reshaped_count += 1
        result.append(article)

    for index, article in enumerate(result, start=1):
        article.article_id = index

    if split_count or reshaped_count or aborted_overlap:

        print()
        print("=" * 60)
        print("BOUNDARY DECOMPOSER")
        print("=" * 60)
        print(f"Articles with a clean rectangle : {clean_count}")
        print(
            f"Articles re-cut (rectangle enclosed foreign "
            f"content) : {split_count}"
        )
        if reshaped_count:
            print(
                f"Articles reshaped into precise multi-rectangle "
                f"boundaries (splitting would strand a fragment or "
                f"sever a headline from its own body) : "
                f"{reshaped_count}"
            )
        if aborted_overlap:
            print(
                f"Articles left as-is (no safe rectangle split "
                f"exists -- pieces would still cover each other) : "
                f"{aborted_overlap}"
            )
        print(f"Total articles after re-cut : {len(result)}")
        print("=" * 60)
        print()

    return result
