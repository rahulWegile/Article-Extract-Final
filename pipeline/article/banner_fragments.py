"""
Shared geometry for detecting when the layout detector cut ONE
printed banner headline into multiple title-class blocks at a column
gutter.

Reused by both local_grouper.py (root selection -- collapsing
fragments into one root BEFORE body-attachment runs) and
article_splitter.py (over-merge sanity-checking on an already-grouped
article's title count). Lives in its own leaf module because
article_splitter.py already imports `group_blocks` from
local_grouper.py -- local_grouper.py importing back from
article_splitter.py would be circular.

Operates on plain (x1, y1, x2, y2) bbox tuples and text, not either
caller's own block representation (LayoutBlock objects in
article_splitter.py, plain dicts in local_grouper.py) -- each caller
adapts its own blocks to this shape rather than this module knowing
about either one.
"""

# How far the two halves' top/bottom edges may differ, as a fraction
# of the shorter half's height. They are the same printed line, so
# this is detector/OCR slop only, not a real layout allowance.
BANNER_FRAGMENT_EDGE_TOLERANCE = 0.35

# Shorter half's height over the taller's -- i.e. the same type size.
BANNER_FRAGMENT_HEIGHT_RATIO = 0.75

# Gutter width between the halves, as a fraction of their height.
BANNER_FRAGMENT_MAX_GAP_RATIO = 0.75

# How much the two halves' own bboxes may OVERLAP in x, as a fraction
# of the narrower half's width, and still count as two detections of
# the SAME printed line rather than two independently stacked lines.
# See article_splitter.py's original comment (moved here) for the
# real Urdu (The Siasat Daily) page that motivated this bound.
BANNER_FRAGMENT_MAX_OVERLAP_RATIO = 0.5


def is_single_line_text(text) -> bool:
    text = (text or "").strip()
    return bool(text) and "\n" not in text


def is_banner_fragment_pair(
    box_a, text_a, box_b, text_b, page_image, vertical_separator_between,
) -> bool:
    """
    Whether these two title boxes are two halves of ONE headline the
    layout detector cut at a column gutter.

    `box_*` are (x1, y1, x2, y2) tuples. `vertical_separator_between`
    is a callable(page_image, x1, x2, y1, y2) -> bool -- injected
    (rather than imported directly here) so this stays a leaf module
    with no dependency on visual_separator.py's own import surface.
    """

    if not (is_single_line_text(text_a) and is_single_line_text(text_b)):
        return False

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    height_a = ay2 - ay1
    height_b = by2 - by1

    shorter = min(height_a, height_b)
    taller = max(height_a, height_b)

    if shorter <= 0:
        return False

    if shorter / taller < BANNER_FRAGMENT_HEIGHT_RATIO:
        return False

    edge_tolerance = BANNER_FRAGMENT_EDGE_TOLERANCE * shorter

    if abs(ay1 - by1) > edge_tolerance:
        return False

    if abs(ay2 - by2) > edge_tolerance:
        return False

    left, right = (box_a, box_b) if ax1 <= bx1 else (box_b, box_a)
    lx2 = left[2]
    rx1 = right[0]

    gap = rx1 - lx2

    if gap < 0:

        narrower_width = min(ax2 - ax1, bx2 - bx1)

        if narrower_width <= 0:
            return False

        if -gap > BANNER_FRAGMENT_MAX_OVERLAP_RATIO * narrower_width:
            return False

    elif gap > BANNER_FRAGMENT_MAX_GAP_RATIO * shorter:
        return False

    return not vertical_separator_between(
        page_image, lx2, rx1, min(ay1, by1), max(ay2, by2),
    )


def group_banner_fragments(items, page_image, vertical_separator_between):
    """
    `items` is a list of (key, box, text) tuples, already sorted by
    the caller in reading order (y1, x1) -- this function does not
    re-sort, so the first key in each returned group is the one the
    caller placed first (leftmost/topmost among the group, if the
    caller sorted as documented).

    Returns a list of groups, each a list of `key` values from
    `items`. Membership is transitive via a Disjoint-Set/Union-Find
    over ALL pairs, not just adjacent-in-list ones, so a headline cut
    into three or more pieces collapses into one group regardless of
    which order reading-order sort happens to visit its fragments in.
    A single-pass greedy walk (compare each new item only against
    already-placed groups) is NOT equivalent: if outer fragment A
    (column 2) and outer fragment C (column 4) share the same y1, a
    (y1, x1) sort visits them before bridge fragment B (column 3)
    that sits between them, A and C get placed in two different
    groups (their own gap is too wide to match directly), and B then
    only ever gets to link to WHICHEVER of the two it is compared
    against first -- stranding the other in its own separate group
    even though B is a fragment-pair match with both. Union-Find
    instead checks every pair up front and unions their groups, so
    transitive closure holds no matter what order the matching pairs
    are discovered in. Groups of size 1 (an ordinary, unfragmented
    title) are included too -- callers filter those out themselves
    where only real multi-fragment groups matter.
    """

    count = len(items)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a, index_b):
        root_a, root_b = find(index_a), find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(count):

        _, box_i, text_i = items[i]

        for j in range(i + 1, count):

            if find(i) == find(j):
                continue

            _, box_j, text_j = items[j]

            if is_banner_fragment_pair(
                box_i, text_i, box_j, text_j,
                page_image, vertical_separator_between,
            ):
                union(i, j)

    members_by_root = {}

    for index in range(count):
        members_by_root.setdefault(find(index), []).append(index)

    ordered_roots = sorted(
        members_by_root, key=lambda root: min(members_by_root[root])
    )

    return [
        [items[index][0] for index in members_by_root[root]]
        for root in ordered_roots
    ]
