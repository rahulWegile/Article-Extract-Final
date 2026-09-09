"""
Shared, language-independent final stage: reconstruct each finalized
logical article's COMPLETE outer region from its own member blocks.

WHERE THIS RUNS
---------------

    ArticleGrouper.build()                 (article grouping)
            |
    article_splitter.split_oversized_articles()   (over-merge splitter)
            |
    boundary_decomposer.decompose_overlapping_articles()  (diagnostic)
            |
    article_region_reconstructor.reconstruct_article_regions()  <- HERE
            |
    FinalArticleCropper                    (final crop generation)

ROOT PROBLEM THIS FIXES
------------------------

A newspaper article is a LOGICAL region, not necessarily one visually
connected rectangle: one story can be built from a headline, a
kicker, a subheadline, several body columns, an image, a caption and
continuation blocks -- many independent DETECTED layout blocks. A
detected block is a component, not the article; a block's own
rectangle is not the article's rectangle. Earlier pipeline stages
(now retired -- see boundary_decomposer.py's docstring) reacted to
that by reshaping or splitting an article's rectangle whenever it
looked "too big" or overlapped a neighbour, which produced the
opposite failure: ONE logical, correctly-grouped article rendered as
SEVERAL final boundaries/crops instead of one.

STEP 1 -- FINAL ARTICLE MEMBERSHIP (input contract)
----------------------------------------------------

This module does NOT decide which block belongs to which article.
That is already finalized by the time this runs: ArticleGrouper (the
model's own grouping, plus its geometric repair passes),
dropped_article_recovery, and article_splitter.py (which is where a
genuine model over-merge of two independent stories is caught, via
local_grouper's headline-first root detection -- never by this
module). Article-in-article / independent-sidebar protection already
happened there: a region keeps its own separate article_id whenever
it was grouped as carrying its own headline and its own body. This
module trusts that membership completely and never revisits it from
geometry -- it only turns already-finalized membership into geometry,
never the other way around.

STEP 2/3/4/5 -- ARTICLE EXTENT, MULTI-COLUMN, HEADLINE-ANCHORING,
COLUMN CONTINUITY
-------------------------------------------------------------------

Given membership is fixed, the article's complete visual extent is
simply the union of every one of its own blocks -- headline, kicker,
subheadline, every body column, every image, every caption, every
continuation block, regardless of how many separate columns or
detected rectangles it took to print them. Column continuity,
headline-anchoring etc. are exactly what upstream grouping already
used to decide membership (STEP 1) -- this module does not
re-derive them from raw geometry; it assembles the region THEY
already agreed on.

STEP 8 -- GEOMETRIC OUTER BOUNDARY
------------------------------------

    x1 = min(all member x1)
    y1 = min(all member y1)
    x2 = max(all member x2)
    y2 = max(all member y2)

This is the article's one and only outer boundary. It may contain
internal whitespace -- between columns, between a headline and its
body, between a story's text and its own photograph -- and that is
EXPECTED, never a reason to shrink, reshape or split it. It may even
geometrically overlap a neighbouring article's own rectangle (a
genuinely L-shaped story printed next to something else): that
overlap is flagged as a diagnostic (see boundary_decomposer.py) but
never changes geometry or membership, because the existence of one
large rectangle must never be used to decide article membership
(STEP 6/7).

STEP 9/10 -- EDGE AND SMALL-BLOCK PROTECTION
-----------------------------------------------

Every member block participates in the union regardless of its size
or position -- a block touching a page edge, or a small kicker/
byline/caption/continuation block, is included exactly like any
other member block. There is no size or position filter here at all;
filtering already happened (or didn't) upstream, in membership.

STEP 13 -- VALIDATION
------------------------

For every final article: exactly one outer boundary is produced (see
_merge_duplicate_article_ids), and every member block -- including
its headline(s), body columns and images -- is asserted to fall
inside that boundary (trivially true by construction, but checked
and self-healed rather than assumed -- see _resolve_members). A
DIAGNOSTIC (not a fix -- see STEP 8) also flags when this boundary
geometrically overlaps a DIFFERENT article's own blocks, so a human
can sanity check that nothing was accidentally absorbed.

LANGUAGE INDEPENDENCE
------------------------

Nothing here reads a language code, a script, a font size, or any
per-language prompt/config. The only inputs are the already-finalized
Article objects (block membership) and each block's own geometry
(x1/y1/x2/y2) -- the exact same four numbers regardless of whether
the page is Tamil, Kannada, Hindi, English or anything else.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from pipeline.article.article_grouper import Article
from pipeline.article.boundary_decomposer import find_foreign_overlap


# ----------------------------------------------------------------
# SUB_RECTS -- display-only L-shape refinement
#
# STEP 8 above is absolute: an article always gets exactly ONE outer
# boundary, and that boundary is always what FinalArticleCropper crops
# from (see final_article_cropper.py's _get_sub_rects: "The crop image
# ... is always the single bbox region regardless"). sub_rects changes
# NONE of that -- it is a purely cosmetic hint for the boundary
# VIEWER (frontend/src/components/viewer/BoundaryLayer.jsx,
# backend/services/boundary_editor.py), which the codebase already
# has full plumbing for and has always shipped as an empty list since
# boundary_decomposer.py's reshaping was retired (see that module's
# docstring for why RESHAPING THE ARTICLE was reverted).
#
# This is a different, narrower thing: when this article's single
# outer rectangle happens to geometrically enclose another article's
# real content in one CORNER (find_foreign_overlap, already computed
# at STEP 13 below) -- confirmed on a real Urdu (THE INQUILAB,
# doc_000182 page 3) document, where a lead flood story's banner
# +photo+right-column outer box fully enclosed three unrelated,
# correctly-grouped smaller articles sitting in the nook under its
# photo -- draw that ONE rectangle as two smaller ones instead, purely
# so the boundary viewer does not visually paint over the neighbours.
# Membership, the crop image, and the extracted article text are
# completely unaffected either way.
# ----------------------------------------------------------------

# How close (as a fraction of the outer boundary's own width/height)
# the intrusion must sit to an edge to count as "in that corner" --
# loose enough for ordinary crop/measurement noise, tight enough that
# an intrusion sitting well inside the boundary (not actually in a
# corner) is correctly left alone.
CORNER_TOUCH_TOLERANCE_RATIO = 0.08


def _l_shape_split(outer, intrusion):
    """
    Two rectangles covering `outer` minus one corner occupied by
    `intrusion`, plus that excluded corner itself (for the caller's
    own-block safety check -- see _compute_sub_rects), or None when
    the intrusion does not sit flush against one full edge in each
    direction (the only shape a plain two-rectangle split can exclude
    without also cutting into the article's OWN content).

    `outer`/`intrusion` are (x1, y1, x2, y2) tuples. The two returned
    rectangles always union back to exactly `outer` MINUS the excluded
    corner -- nothing is ever cropped away from the article's actual
    content by this, only redrawn as two boxes instead of one, and
    only once the caller has confirmed no own block reaches into the
    corner being excluded.

    Returns (band, remainder, excluded_corner).
    """

    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = intrusion

    width = max(1.0, ox2 - ox1)
    height = max(1.0, oy2 - oy1)

    tol_x = CORNER_TOUCH_TOLERANCE_RATIO * width
    tol_y = CORNER_TOUCH_TOLERANCE_RATIO * height

    touches_bottom = (oy2 - iy2) <= tol_y
    touches_top = (iy1 - oy1) <= tol_y
    touches_left = (ix1 - ox1) <= tol_x
    touches_right = (ox2 - ix2) <= tol_x

    # `band` is the full-width strip on the side OPPOSITE the
    # intrusion; `other_y1`/`other_y2` is the remaining strip on the
    # intrusion's own side, which `remainder` and `excluded_corner`
    # then split by x.
    if touches_bottom and not touches_top:
        split_y = iy1
        band = (ox1, oy1, ox2, split_y)
        other_y1, other_y2 = split_y, oy2
    elif touches_top and not touches_bottom:
        split_y = iy2
        band = (ox1, split_y, ox2, oy2)
        other_y1, other_y2 = oy1, split_y
    else:
        # Spans the full height (or touches neither edge) -- not a
        # single-corner cutout a two-rectangle split can carve out.
        return None

    if touches_left and not touches_right:
        remainder = (ix2, other_y1, ox2, other_y2)
        excluded_corner = (ox1, other_y1, ix2, other_y2)
    elif touches_right and not touches_left:
        remainder = (ox1, other_y1, ix1, other_y2)
        excluded_corner = (ix1, other_y1, ox2, other_y2)
    else:
        return None

    return band, remainder, excluded_corner


def _rects_overlap(a, b) -> bool:
    return (
        a[0] < b[2] and b[0] < a[2]
        and a[1] < b[3] and b[1] < a[3]
    )


def _compute_sub_rects(resolved_blocks, outer_bbox, foreign_overlap):
    """
    See SUB_RECTS above. Returns [] (today's behaviour, unchanged)
    unless a clean two-rectangle split both excludes every flagged
    foreign block and reaches into none of this article's OWN blocks
    -- checked explicitly rather than assumed, since a split that cut
    into so much as one member block's own rectangle would be worse
    than the single overlapping rectangle it replaced. A block that
    straddles the two rectangles (e.g. a right-column story block
    running from above the split down past it) is fine, and expected
    -- see _l_shape_split's own docstring -- as long as it never
    enters the excluded corner itself.
    """

    if not foreign_overlap:
        return []

    intrusion = (
        min(block.x1 for block in foreign_overlap),
        min(block.y1 for block in foreign_overlap),
        max(block.x2 for block in foreign_overlap),
        max(block.y2 for block in foreign_overlap),
    )

    split = _l_shape_split(outer_bbox, intrusion)

    if split is None:
        return []

    band, remainder, excluded_corner = split

    for block in resolved_blocks:

        block_rect = (block.x1, block.y1, block.x2, block.y2)

        if _rects_overlap(block_rect, excluded_corner):
            # This own block reaches into the corner the split would
            # exclude -- abandon rather than draw a sub-rect that
            # visually cuts into the article's own content.
            return []

    return [
        (
            int(round(rect[0])), int(round(rect[1])),
            int(round(rect[2])), int(round(rect[3])),
        )
        for rect in (band, remainder)
    ]


# Roles treated as this article's headline for the STEP 12 debug
# breakdown. "article_title" is the role ArticleGrouper/GeminiParser
# assign to any title-class block folded into an article (main
# headline, kicker, or subheadline alike -- STEP 12 only asks for a
# breakdown by kind, not a further distinction among them).
HEADLINE_ROLES = {"article_title"}

# Roles treated as this article's body for the STEP 12 debug
# breakdown.
BODY_ROLES = {"article_text", "byline"}

# Roles treated as this article's image content for the STEP 12
# debug breakdown.
IMAGE_ROLES = {"article_image", "caption"}

# Every role that marks a block as this article's own recognised
# content -- headline-ish, body-ish or image-ish alike. STEP 8's own
# defensive masthead/header exclusion (below) never touches a block
# carrying any of these, satisfying the contract that the headline/
# kicker/subheadline always survives into the outer boundary.
CONTENT_ROLES = HEADLINE_ROLES | BODY_ROLES | IMAGE_ROLES

# Direct role signal for page-level newspaper chrome that sits above
# a story rather than inside it. ArticleGrouper.IGNORE_ROLES already
# drops a block classified this way before an Article object even
# exists (STEP 1), so this match is normally a no-op here -- kept
# purely as a defensive, language-independent backstop.
PAGE_CHROME_ROLES = {"masthead", "page_header"}

# How far a role-less, textless graphic block may extend past the
# left/right edge of an article's own recognised content, as a
# fraction of that content's own width, before it counts as page-
# level chrome rather than the story's own graphic headline banner.
#
# A story's own stylized banner (see article_grouper.py's
# _recover_unclaimed_title_graphics -- "some newspapers render a
# story's own headline as a colored graphic banner") is sized to that
# same story's own column and so sits within (or very close to) its
# content's own left/right edges. A newspaper's masthead/section
# nameplate, by contrast, is laid out independently of any one
# story's column grid and typically spans much wider -- confirmed on
# a real Tamil business-page: a masthead nameplate (x-span 1213-2918)
# sitting directly above a story whose own content spans only
# 1192-2087 got folded into that story by
# _recover_unclaimed_title_graphics (its column-overlap check
# normalizes by the NARROWER block's width, so a wide banner over a
# narrow column below always scores as "overlapping"), stretching the
# story's own final crop upward to swallow the masthead. The
# overhang here (831px, 93% of the story's own 895px content width)
# is what a same-story banner never produces.
MASTHEAD_OVERHANG_RATIO = 0.25


def _block_role(block):
    return (getattr(block, "role", None) or "").strip().lower()


def _block_text(block):
    return (getattr(block, "text", "") or "").strip()


def _split_page_chrome(resolved_blocks):
    """
    Partition one article's own resolved blocks into (kept, excluded)
    ahead of STEP 8's union bbox -- excluded holds only page-level
    masthead/header chrome that ended up as one of this article's
    "own" blocks despite never being part of the printed story,
    the failure this function exists to catch (see module docstring's
    ROOT PROBLEM and MASTHEAD_OVERHANG_RATIO's comment above for the
    confirmed real-page shape).

    A block is never excluded, regardless of position, when it
    carries a recognised content role (CONTENT_ROLES -- headline,
    kicker and subheadline included) or any OCR'd text of its own:
    every real headline/kicker/subheadline/body/image/caption block
    in this pipeline has one or the other, so this can never strip an
    article's own content. Only a block with neither -- a pure
    graphic bearing no role the grouping model recognised as this
    story's own -- is even considered, and then only when it also
    sits above every one of this article's own content blocks and
    spans well past that content's own left/right edges (STEP 8 must
    never use a fixed page-Y threshold -- this compares a candidate
    only against this article's OWN other blocks, never against page
    dimensions).
    """

    content_blocks = [
        block
        for block in resolved_blocks
        if _block_role(block) not in PAGE_CHROME_ROLES
        and (_block_role(block) in CONTENT_ROLES or _block_text(block))
    ]

    if not content_blocks:
        # No block here reads as a real headline or body of any kind
        # -- this "article" is entirely graphic/chrome, not a story
        # this stage should be drawing a boundary around at all.
        return [], list(resolved_blocks)

    content_x1 = min(block.x1 for block in content_blocks)
    content_x2 = max(block.x2 for block in content_blocks)
    content_y1 = min(block.y1 for block in content_blocks)
    content_width = max(1.0, float(content_x2 - content_x1))

    kept = []
    excluded = []

    for block in resolved_blocks:

        role = _block_role(block)

        # The explicit role signal always wins, even over a chrome
        # block that happens to carry OCR'd text of its own (e.g. a
        # page_header date/edition line) -- see PAGE_CHROME_ROLES.
        if role in PAGE_CHROME_ROLES:
            excluded.append(block)
            continue

        if role in CONTENT_ROLES or _block_text(block):
            kept.append(block)
            continue

        overhang = max(
            content_x1 - block.x1,
            block.x2 - content_x2,
            0.0,
        )

        if (
            block.y2 <= content_y1
            and overhang > MASTHEAD_OVERHANG_RATIO * content_width
        ):
            excluded.append(block)
        else:
            kept.append(block)

    if not kept:
        return [], list(resolved_blocks)

    return kept, excluded


@dataclass
class ArticleBoundary:
    """
    Final verified article boundary: the union of every block
    belonging to ONE finalized logical article (see module
    docstring). Always exactly one per article_id.
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

    # Retained for backward compatibility with callers that still
    # read it defensively (FinalArticleCropper, FinalBoundaryVisualizer,
    # the frontend viewer). Always empty now -- see
    # boundary_decomposer.py for why this module never reshapes an
    # article into multiple display rectangles: STEP 8 requires
    # exactly ONE outer boundary per article, full stop.
    sub_rects: List[Tuple[int, int, int, int]] = field(
        default_factory=list
    )


def _merge_duplicate_article_ids(articles: List[Article]) -> List[Article]:
    """
    Collapse any Article objects that share the same article_id into
    ONE article holding the union of their blocks.

    FINAL ARTICLE MEMBERSHIP must be fully settled before a region is
    reconstructed (STEP 1): every earlier stage (ArticleGrouper,
    dropped_article_recovery, article_splitter, boundary_decomposer)
    is expected to keep article_id unique by construction, but
    nothing upstream is in a position to notice if it ever doesn't --
    only here, where every article for the page is finally in one
    list, can that be checked. Without this, two Article objects
    sharing one article_id would each get their own ArticleBoundary,
    and FinalArticleCropper names crops by their position in that
    list rather than by article_id -- so one logical article_id would
    silently turn into two separate numbered crop folders, exactly
    the "several boundaries instead of one" failure this whole module
    exists to prevent.
    """

    grouped: dict = {}
    order = []

    for article in articles:

        key = article.article_id

        if key not in grouped:
            grouped[key] = []
            order.append(key)

        grouped[key].append(article)

    merged = []

    for key in order:

        group = grouped[key]

        if len(group) == 1:
            merged.append(group[0])
            continue

        blocks_by_id = {}

        for article in group:
            for block in article.blocks:
                blocks_by_id[block.id] = block

        blocks = sorted(
            blocks_by_id.values(),
            key=lambda b: getattr(b, "reading_order", 0),
        )

        merged.append(
            Article(
                article_id=key,
                blocks=blocks,
                block_ids=[block.id for block in blocks],
                confidence=min(article.confidence for article in group),
            )
        )

        print(
            f"WARNING: article_id={key} was produced by "
            f"{len(group)} separate group(s) upstream -- merged into "
            f"one article ({len(blocks)} block(s) total) before "
            f"region reconstruction so it yields exactly ONE final "
            f"boundary/crop."
        )

    return merged


def _resolve_members(articles: List[Article]):
    """
    De-duplicate each article's own blocks by id, and exclude any
    block a DIFFERENT final article actually owns (STEP 1: trust
    finalized membership, but resolve the rare case of a block
    accidentally double-claimed across two different article_ids --
    first claim wins, deterministically, so it is never silently
    dropped from every article nor double-counted in two).

    Returns (owner_article_of, duplicate_owner_blocks, resolved_by_article)
    where resolved_by_article maps article_id -> ordered list of
    that article's own resolved blocks.
    """

    owner_article_of = {}
    duplicate_owner_blocks = set()

    for article in articles:
        for block_id in article.block_ids:

            if (
                block_id in owner_article_of
                and owner_article_of[block_id] != article.article_id
            ):
                duplicate_owner_blocks.add(block_id)

            owner_article_of.setdefault(block_id, article.article_id)

    resolved_by_article = {}

    for article in articles:

        own_blocks_by_id = {}

        for block in article.blocks:

            if block.id in duplicate_owner_blocks:
                owner = owner_article_of.get(block.id)
                if owner is not None and owner != article.article_id:
                    continue

            own_blocks_by_id[block.id] = block

        resolved_by_article[article.article_id] = list(
            own_blocks_by_id.values()
        )

    return owner_article_of, duplicate_owner_blocks, resolved_by_article


def _role_breakdown(blocks):

    headline_ids = []
    body_ids = []
    image_ids = []

    for block in blocks:

        role = (getattr(block, "role", None) or "").strip().lower()

        if role in HEADLINE_ROLES:
            headline_ids.append(block.id)
        elif role in BODY_ROLES:
            body_ids.append(block.id)
        elif role in IMAGE_ROLES:
            image_ids.append(block.id)

    return headline_ids, body_ids, image_ids


def reconstruct_article_regions(
    articles: List[Article],
    blocks: Optional[list] = None,
) -> List[ArticleBoundary]:
    """
    Build exactly ONE outer ArticleBoundary per finalized article_id.

    `blocks` is optional and used only for the STEP 13 cross-article
    overlap diagnostic (purely informational -- see
    boundary_decomposer.find_foreign_overlap); when omitted, the
    union of every article's own blocks is used instead, which covers
    the same ground for blocks that made it into some article.
    """

    articles = _merge_duplicate_article_ids(articles)

    if blocks is None:
        blocks = [block for article in articles for block in article.blocks]

    owner_article_of, duplicate_owner_blocks, resolved_by_article = (
        _resolve_members(articles)
    )

    # STEP 13: exactly one final boundary per article_id -- guaranteed
    # by the merge above, asserted here rather than assumed.
    seen_article_ids = set()
    duplicate_article_ids = set()

    for article in articles:
        if article.article_id in seen_article_ids:
            duplicate_article_ids.add(article.article_id)
        seen_article_ids.add(article.article_id)

    assert not duplicate_article_ids, (
        f"article_id(s) {sorted(duplicate_article_ids)} produced more "
        f"than one final boundary after merging -- this should be "
        f"structurally impossible"
    )

    boundaries = []

    # Own resolved blocks (post-chrome-split) per article_id, kept for
    # the STEP 13 sub_rects computation below -- see SUB_RECTS.
    own_blocks_by_article_id = {}

    for article in articles:

        resolved_blocks = resolved_by_article[article.article_id]

        if not resolved_blocks:
            continue

        resolved_blocks, excluded_chrome_blocks = _split_page_chrome(
            resolved_blocks
        )

        if not resolved_blocks:
            continue

        if excluded_chrome_blocks:
            print(
                f"NOTE: article {article.article_id} excluded "
                f"{len(excluded_chrome_blocks)} page-level masthead/"
                f"header block(s) from its outer boundary -- kept out "
                f"of the union rather than stretching the crop to "
                f"include them: "
                f"{[block.id for block in excluded_chrome_blocks]}"
            )

        # STEP 8: the outer boundary is the plain union of every
        # member block. No reshaping, no splitting, internal
        # whitespace and incidental overlap with a neighbour are both
        # expected.
        x1 = min(block.x1 for block in resolved_blocks)
        y1 = min(block.y1 for block in resolved_blocks)
        x2 = max(block.x2 for block in resolved_blocks)
        y2 = max(block.y2 for block in resolved_blocks)

        # STEP 13 self-consistency check: every member block must
        # fall inside the boundary just computed FROM those same
        # blocks. Can only fail on inconsistent block coordinates
        # (e.g. mutated after being counted); self-heal by rebuilding
        # from the full set rather than emitting a boundary narrower
        # than a block it claims to contain.
        out_of_bounds = [
            block
            for block in resolved_blocks
            if block.x1 < x1 or block.y1 < y1
            or block.x2 > x2 or block.y2 > y2
        ]

        if out_of_bounds:

            print(
                f"WARNING: article {article.article_id} had "
                f"{len(out_of_bounds)} block(s) outside its own "
                f"computed boundary -- rebuilding union bbox from the "
                f"full assigned block set: {[b.id for b in out_of_bounds]}"
            )

            x1 = min(x1, *(b.x1 for b in out_of_bounds))
            y1 = min(y1, *(b.y1 for b in out_of_bounds))
            x2 = max(x2, *(b.x2 for b in out_of_bounds))
            y2 = max(y2, *(b.y2 for b in out_of_bounds))

        headline_ids, body_ids, image_ids = _role_breakdown(resolved_blocks)

        # STEP 12 debug output.
        print(f"ARTICLE_{article.article_id:03d}")
        print()
        print("members:")
        print(",".join(str(b.id) for b in resolved_blocks))
        if headline_ids:
            print(f"headline_block_ids: {headline_ids}")
        if body_ids:
            print(f"body_block_ids: {body_ids}")
        if image_ids:
            print(f"image_block_ids: {image_ids}")
        print()
        print("outer_bbox:")
        print(f"[{x1}, {y1}, {x2}, {y2}]")
        print()

        own_blocks_by_article_id[article.article_id] = resolved_blocks

        boundaries.append(
            ArticleBoundary(
                article_id=article.article_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                width=x2 - x1,
                height=y2 - y1,
                area=(x2 - x1) * (y2 - y1),
                block_ids=[block.id for block in resolved_blocks],
                confidence=article.confidence,
            )
        )

    boundary_by_article_id = {
        boundary.article_id: boundary for boundary in boundaries
    }

    # STEP 13: does this article's boundary geometrically overlap
    # another article's own content? Diagnostic (STEP 6/8: overlap is
    # expected and this never changes membership or the crop image);
    # logged here too so it is visible right next to the final
    # boundary it concerns. Also feeds sub_rects -- see SUB_RECTS --
    # a purely cosmetic, non-mutating refinement of how this SAME
    # boundary is drawn in the viewer, never of what it crops.
    for article in articles:

        boundary = boundary_by_article_id.get(article.article_id)

        if boundary is None:
            continue

        overlap = find_foreign_overlap(article, blocks, owner_article_of)

        if not overlap:
            continue

        print(
            f"NOTE: article {article.article_id}'s outer boundary "
            f"geometrically overlaps {len(overlap)} block(s) "
            f"belonging to another article: "
            f"{[b.id for b in overlap]}. No content was moved -- "
            f"verify the crops independently if this is "
            f"unexpected."
        )

        sub_rects = _compute_sub_rects(
            own_blocks_by_article_id.get(article.article_id, []),
            (boundary.x1, boundary.y1, boundary.x2, boundary.y2),
            overlap,
        )

        if sub_rects:

            boundary.sub_rects = sub_rects

            print(
                f"NOTE: article {article.article_id}'s outer boundary "
                f"redrawn as {len(sub_rects)} sub-rectangles for "
                f"display, to avoid visually covering the block(s) "
                f"above -- crop and membership unchanged: {sub_rects}"
            )

    print()
    print("=" * 60)
    print("ARTICLE REGION RECONSTRUCTOR")
    print("=" * 60)
    print(f"Generated Boundaries : {len(boundaries)}")

    if duplicate_owner_blocks:
        print(
            f"WARNING: {len(duplicate_owner_blocks)} block(s) were "
            f"claimed by more than one final article -- resolved to a "
            f"single owner: {sorted(duplicate_owner_blocks)}"
        )

    for boundary in boundaries:
        print(f"article_id={boundary.article_id}")

    print("=" * 60)
    print()

    return boundaries
