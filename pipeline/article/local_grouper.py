"""
Deterministic, offline article grouping.

Produces exactly the same response shape the page-level LLM returns:

    {
        "blocks":   [{"id": int, "role": str}, ...],
        "articles": [{"article_id": int, "blocks": [int, ...]}, ...]
    }

so every downstream stage (GeminiParser -> ArticleGrouper ->
BoundaryBuilder -> cropper) works against it unchanged.

No network calls, no model, no API key.

HOW IT SEPARATES ARTICLES
-------------------------

Newspapers print noticeably more whitespace between two stories than
between paragraphs of one story, so the gap above a block is the
primary signal here:

1. Every "title" block is a CANDIDATE article root.
2. A candidate becomes a real root when it clearly governs body text
   below it, OR when gap/separator evidence indicates a genuine story
   break.
3. Caption-like titles immediately under images are protected from
   separator-based false positives.
4. Every remaining content block joins the latest eligible root ABOVE
   it that shares its column span, so a later headline acts as an
   ownership boundary for that visual lane.

KNOWN LIMITATIONS vs the LLM path
---------------------------------

This grouper is geometric. It cannot read meaning, so it does NOT
detect advertisements, teaser/digest boxes, or utility panels as
distinct kinds of content -- they are grouped like any other blocks
and will appear as articles. It also cannot resolve cross-page
continuations. Use it when API calls must be avoided; the LLM path
remains more accurate on dense pages.
"""

import json
from pathlib import Path

import cv2

from pipeline.article.banner_fragments import group_banner_fragments
from pipeline.article.visual_separator import (
    SEPARATOR_MIN_BAND_HEIGHT,
    has_visual_separator,
    has_vertical_separator,
    has_vertical_separator_between,
)


# Layout-detector classes that carry article text.
TEXT_CLASSES = {"plain text", "title", "figure_caption"}

# Layout-detector classes that are pictures.
IMAGE_CLASSES = {"figure", "image", "object", "photo", "picture"}

# A title with its own body text below it is a strong article-root candidate,
# so root detection must not depend on a large blank gap above the title.
# Gap/separator evidence remains useful for distinguishing captions and
# inline crossheads that the detector may label as titles.
ROOT_GAP_FACTOR = 2.0

# Floor for the gap-based signal.
ROOT_GAP_MINIMUM = 25.0

# Minimum vertical distance below a title before a text block can be
# considered its supporting body text. This prevents a title from
# claiming text that overlaps its own bbox due to detector slop.
TITLE_BODY_MIN_GAP = 2.0

# Fraction of the narrower block's width that must overlap
# horizontally for two blocks to count as sharing a column.
COLUMN_OVERLAP = 0.35


def _bbox(block):
    box = block.get("bbox") or {}
    if None in (box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")):
        return None
    return box


def _overlap_ratio(a, b):
    """
    IoU-style overlap over the UNION of both widths, not the
    narrower block's width alone -- normalizing by the narrower
    width gives a wide block near-total "overlap" against almost any
    narrower block it happens to contain horizontally, even when
    that narrower block is a different, differently-sized region
    entirely (a full-width caption sitting under a narrow-column
    headline, for instance). Matches the fix already validated in
    ArticleGrouper._layout_distance for the identical normalization
    problem -- see that function's own comment for the measured
    reasoning.
    """
    width_a = max(1.0, float(a["x2"] - a["x1"]))
    width_b = max(1.0, float(b["x2"] - b["x1"]))
    overlap = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    union_width = width_a + width_b - overlap
    return overlap / union_width if union_width > 0 else 0.0


def _containment_ratio(candidate, root):
    """
    Fraction of `candidate`'s OWN width that falls inside `root`'s
    horizontal span -- directional, unlike _overlap_ratio's union
    normalization.

    Column ELIGIBILITY (does this candidate sit in the root's
    territory) turns out to be a different question from "do these
    two blocks occupy comparable visual lanes" (_overlap_ratio's job
    everywhere else in this file). A wide multi-column title's root
    bbox spans all of its own sub-columns; the union test punishes
    every one of those narrower sub-columns for not being as wide as
    the title itself -- confirmed on a real Urdu page: a title
    spanning 3 columns failed the union test against each of its own
    3 body sub-columns by the same ~0.32 margin, just under
    COLUMN_OVERLAP, so all 3 were left unclaimed even though each
    sits almost entirely under it. Normalizing by the candidate's own
    width fixes that, while STILL rejecting the opposite, separately
    confirmed failure (a wide, unrelated image spuriously
    "overlapping" a narrow title merely by spanning across its
    column) since that image's own width is what the ratio divides by
    in that case too.
    """
    width_candidate = max(1.0, float(candidate["x2"] - candidate["x1"]))
    overlap = max(0.0, min(candidate["x2"], root["x2"]) - max(candidate["x1"], root["x1"]))
    return overlap / width_candidate


def _row_overlap_ratio(a, b):
    """
    Fraction of the SHORTER block's own height that the two share
    vertically -- i.e. whether they sit in the same horizontal
    row/band, the way two side-by-side columns of one story do.
    Mirrors article_splitter.py's _row_overlap_ratio (same reasoning,
    LayoutBlock attributes there vs bbox dicts here).
    """
    top = max(a["y1"], b["y1"])
    bottom = min(a["y2"], b["y2"])
    overlap = max(0.0, bottom - top)
    shorter = min(a["y2"] - a["y1"], b["y2"] - b["y1"])
    return overlap / shorter if shorter > 0 else 0.0


def _find_next_root_below(box, candidate_roots):
    """
    The NEAREST root BELOW `box` (smallest y1 among roots that start
    at or after this box ends) that shares its column.

    Used only for orphan-root cleanup, never for ordinary content
    attachment: a kicker/eyebrow line that ends up governing nothing
    of its own naturally belongs to the main headline that FOLLOWS
    it in print, not to whatever happens to sit above it (there may
    be nothing above it at all -- it can be the very first element
    in its column, as a kicker line always is).
    """

    best_root = None
    best_distance = None

    for root in candidate_roots:

        root_box = _bbox(root)

        if root_box["y1"] < box["y2"]:
            continue

        if _overlap_ratio(box, root_box) < COLUMN_OVERLAP:
            continue

        distance = float(root_box["y1"] - box["y2"])

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_root = root

    return best_root


def _find_owning_root(box, candidate_roots):
    """
    The LATEST root ABOVE `box` (largest y1, i.e. nearest above) that
    shares its column, or None if no candidate root qualifies.

    Shared by the main content-attachment pass and the orphan-root
    cleanup pass below, so both use exactly the same "which root does
    this block belong to" rule -- see build's own inline version this
    was extracted from for the ownership-boundary rationale.
    """

    eligible_roots = []

    for root in candidate_roots:

        root_box = _bbox(root)

        if root_box["y1"] > box["y1"]:
            continue

        if _overlap_ratio(box, root_box) < COLUMN_OVERLAP:
            continue

        eligible_roots.append(root)

    if not eligible_roots:
        return None

    return max(eligible_roots, key=lambda root: _bbox(root)["y1"])


def _role_for(block):
    """
    Map a detector class / cleaner type onto the semantic role
    vocabulary the downstream stages expect.

    A caller that already knows a block is a byline (e.g.
    article_splitter.py re-running this on a subset the LLM already
    semantically classified) can set block["role"] = "byline" to
    short-circuit the class-based inference below. Confirmed on a
    real page: a byline ("एजेंसी नई दिल्ली") detected with raw class
    "title" -- a common upstream layout-detector mistake for bold/
    caps byline text -- was otherwise treated as a legitimate new
    article root by the class-only inference, splitting a story's
    byline+continuation text away from its own headline. Offline
    mode never sets this key, so this is a no-op there.
    """

    if block.get("role") == "byline":
        return "byline"

    block_type = (block.get("type") or "").strip().lower()

    # PageCleaner already identified page furniture.
    if block_type in {
        "masthead",
        "page_header",
        "page_footer",
        "page_number",
        "section_header",
    }:
        return block_type

    if block_type == "caption":
        return "caption"

    block_class = (block.get("class") or "").strip().lower()

    if block_class == "abandon":
        return "decoration"

    if block_class in IMAGE_CLASSES:
        return "article_image"

    if block_class == "figure_caption":
        return "caption"

    if block_class == "title":
        return "article_title"

    if block_class == "plain text":
        return "article_text"

    return "unknown"


def _annotate_gaps(blocks):
    """
    Attach the blank vertical distance to each block's nearest
    same-column neighbour above (and that neighbour's own class, so
    a caption sitting just under a photo can be told apart from a
    genuine article break -- see _has_visual_separator's caller),
    and return the page's median gap.
    """

    boxes = [_bbox(b) for b in blocks]
    gaps = []

    for index, box in enumerate(boxes):

        if box is None:
            blocks[index]["_gap_above"] = None
            blocks[index]["_neighbor_above_class"] = None
            continue

        nearest = None
        nearest_class = None

        for other_index, other in enumerate(boxes):

            if other is None or other_index == index:
                continue

            if other["y2"] > box["y1"]:
                continue

            if _overlap_ratio(box, other) < COLUMN_OVERLAP:
                continue

            distance = float(box["y1"] - other["y2"])

            if nearest is None or distance < nearest:
                nearest = distance
                nearest_class = blocks[other_index].get("class")

        blocks[index]["_gap_above"] = nearest
        blocks[index]["_neighbor_above_class"] = nearest_class

        if nearest is not None and nearest > 0:
            gaps.append(nearest)

    if not gaps:
        return 0.0

    gaps.sort()
    middle = len(gaps) // 2

    if len(gaps) % 2:
        return float(gaps[middle])

    return (gaps[middle - 1] + gaps[middle]) / 2.0


def _has_body_below(title, content, roles):
    """
    Return True when a title has at least one article-text block
    immediately below it in the same visual column/lane.

    This is intentionally a conservative local test. A title should
    become a root when it clearly governs real body text below it, even
    when the newspaper leaves very little whitespace above the title.

    Uses _containment_ratio (normalized by the CANDIDATE body block's
    own width), not _overlap_ratio -- same reasoning as
    _containment_ratio's own docstring: a wide multi-column title (or
    a banner headline collapsed from several column-cut fragments,
    see group_blocks' fragment-collapsing pass) would otherwise fail
    this test against each of its own narrower body sub-columns under
    the union-of-both-widths normalization, even though each sits
    almost entirely beneath it.
    """

    title_box = _bbox(title)
    if title_box is None:
        return False

    candidates = []

    for block in content:
        if block is title:
            continue

        if roles.get(block["id"]) != "article_text":
            continue

        box = _bbox(block)
        if box is None:
            continue

        # Body must begin below the title, with only a small allowance
        # for detector/rendering alignment noise.
        if box["y1"] < title_box["y2"] + TITLE_BODY_MIN_GAP:
            continue

        # It must occupy the same column/lane as the headline.
        overlap = _containment_ratio(box, title_box)
        if overlap < COLUMN_OVERLAP:
            continue

        distance = float(box["y1"] - title_box["y2"])
        candidates.append((distance, box["y1"], block))

    if not candidates:
        return False

    candidates.sort(key=lambda item: (item[0], item[1]))
    return True


def build_local_response(page_json_path, page_image_path=None, debug=False):
    """
    Group one page's blocks into articles without any model call.

    Thin file-loading wrapper -- the actual grouping logic lives in
    group_blocks() so it can also be reused as a sanity-check/splitter
    over an ALREADY-GROUPED set of blocks (see article_splitter.py),
    not just as the primary local-mode grouper.
    """

    page = json.loads(
        Path(page_json_path).read_text(encoding="utf-8")
    )

    blocks = page.get("blocks", []) if isinstance(page, dict) else page

    page_image = None

    if page_image_path is not None:

        try:
            page_image = cv2.imread(str(page_image_path))
        except Exception:
            page_image = None

    return group_blocks(blocks, page_image, debug=debug)


def group_blocks(blocks, page_image=None, debug=False):
    """
    Group an already-loaded list of block dicts into articles.

    Each block dict must have "id", "class", "bbox" -- the same
    shape page_json blocks already have. `page_image` is an already-
    loaded (cv2.imread) BGR array, or None to skip the visual-
    separator checks (gap/width signals still work without it).
    """

    if not blocks:
        return {"blocks": [], "articles": []}

    roles = {b["id"]: _role_for(b) for b in blocks if "id" in b}

    # ----------------------------------------------------------
    # Collapse banner-headline fragments BEFORE root selection.
    #
    # A headline printed across several columns is one printed line,
    # but the layout detector routinely cuts it at a column gutter
    # and emits each half as its own title block. Root selection
    # below has no notion of "these two titles are the same
    # headline" -- each independently has its own body column below
    # it, so each independently becomes its own root, and one story
    # is split down the middle of its own headline. article_splitter
    # already fixes this shape, but only for an article the LLM
    # already merged into one (>1 title block); when local grouping
    # produces the split in the first place, each fragment starts
    # out as its own single-title article and nothing downstream
    # merges separate articles back together. See
    # banner_fragments.py for the real geometry this reuses (shared
    # with article_splitter.py, which cannot be imported back from
    # here -- it already imports group_blocks from this module).
    #
    # The primary fragment's own bbox is widened in place to the
    # union of every fragment's bbox, so root selection and body-
    # attachment (both against _containment_ratio) see the headline's
    # true extent and correctly claim body text under EVERY fragment's
    # original column, not just the primary's own. Secondary
    # fragments are excluded from `content` below (never considered
    # as independent root candidates) and re-attached to whichever
    # article the primary ends up in, once that's known -- see the
    # end of this function.
    # ----------------------------------------------------------

    by_id = {b["id"]: b for b in blocks if "id" in b}

    title_items = sorted(
        (
            (b["id"], (box["x1"], box["y1"], box["x2"], box["y2"]), b.get("text", ""))
            for b in blocks
            if roles.get(b.get("id")) == "article_title"
            and (box := _bbox(b)) is not None
        ),
        key=lambda item: (item[1][1], item[1][0]),
    )

    fragment_secondary_ids = set()
    fragment_members_of_primary = {}

    for group in group_banner_fragments(
        title_items, page_image, has_vertical_separator_between,
    ):

        if len(group) < 2:
            continue

        primary_id, *secondary_ids = group

        member_boxes = [_bbox(by_id[block_id]) for block_id in group]

        by_id[primary_id]["bbox"] = {
            "x1": min(box["x1"] for box in member_boxes),
            "y1": min(box["y1"] for box in member_boxes),
            "x2": max(box["x2"] for box in member_boxes),
            "y2": max(box["y2"] for box in member_boxes),
        }

        fragment_secondary_ids.update(secondary_ids)
        fragment_members_of_primary[primary_id] = secondary_ids

    median_gap = _annotate_gaps(blocks)

    root_threshold = max(
        ROOT_GAP_MINIMUM,
        median_gap * ROOT_GAP_FACTOR,
    )

    # ----------------------------------------------------------
    # Choose article roots
    # ----------------------------------------------------------

    content = [
        b
        for b in blocks
        if _bbox(b) is not None
        and b["id"] not in fragment_secondary_ids
        and roles.get(b["id"])
        in {"article_title", "article_text", "article_image", "caption", "byline"}
    ]

    content.sort(key=lambda b: (_bbox(b)["y1"], _bbox(b)["x1"]))

    roots = []

    for block in content:

        if roles.get(block["id"]) != "article_title":
            continue

        gap = block.get("_gap_above")
        has_own_body = _has_body_below(block, content, roles)

        # A title that clearly governs body text below it is an article
        # root even when the whitespace ABOVE the title is small.
        # This is the primary fix for small/stacked newspaper stories.
        if has_own_body:
            roots.append(block)
            continue

        # No neighbour above means this is the first thing in its
        # column, which is where a headline normally sits.
        if gap is None or gap >= root_threshold:
            roots.append(block)
            continue

        # A photo is inherently non-uniform and often colorful, so a
        # caption sitting directly under one would otherwise look
        # exactly like a colored separator box to the pixel check
        # below. A caption's relationship is to the image above it,
        # never to a rule line, so skip the visual separator test here.
        if block.get("_neighbor_above_class") in IMAGE_CLASSES:
            continue

        # The gap alone did not clear the threshold, but a boxed
        # sidebar can be printed closer than that because the
        # newspaper uses a border or colored background to mark the
        # split. Check the pixels directly above this title.
        box = _bbox(block)

        probe_height = max(gap, SEPARATOR_MIN_BAND_HEIGHT)

        has_horizontal_separator = has_visual_separator(
            page_image,
            box["x1"],
            box["x2"],
            box["y1"] - probe_height,
            box["y1"] + 2,
        )

        # Some layouts print several short items in one row,
        # divided by a vertical bar instead of a horizontal gap.
        has_vertical_sep_line = has_vertical_separator(
            page_image,
            box["x1"],
            box["x2"],
            box["y1"],
            box["y2"],
        )

        if has_horizontal_separator or has_vertical_sep_line:
            roots.append(block)

    # A page of pure body text still needs somewhere to put it.
    if not roots and content:
        roots = [content[0]]

    # ----------------------------------------------------------
    # Attach every content block to a root
    #
    # UPDATE -- CONTINUITY CHAIN, not just column overlap with the
    # root's own original bbox.
    #
    # Confirmed on a real Urdu page: a story with no detected title
    # at all (the layout detector never classified its headline as
    # "title") had its body text sit ~465px below an unrelated,
    # already-fully-claimed story's last paragraph -- with two photo
    # blocks in between -- yet still numerically "shared a column"
    # with that unrelated title (a narrow block sitting entirely
    # inside a much wider one). The old rule only ever measured
    # column overlap and distance against the root's ORIGINAL bbox,
    # so it had no way to notice that this candidate was nowhere
    # near where that root's own content actually reaches.
    #
    # Now each root's continuity is judged against its own MOST
    # RECENTLY ATTACHED block first (falling back to the root's own
    # bbox only when nothing already attached shares this candidate's
    # column -- the legitimate case of a wide title's story entering
    # a fresh column for the first time). Two additional guards catch
    # what column overlap alone cannot:
    #
    # 1. MAX_CONTINUATION_GAP -- a hard backstop; nothing this far
    #    from the nearest relevant chain point is worth attaching
    #    blindly.
    # 2. A section-break image -- a real photo accompanying a
    #    narrow-column story is ordinarily close to that column's own
    #    width; one MUCH wider sitting between the chain and the
    #    candidate is what a NEW section's own banner photo looks
    #    like, not a continuation of the story above it.
    # ----------------------------------------------------------

    MAX_CONTINUATION_GAP = 700.0
    WIDE_IMAGE_SECTION_BREAK_RATIO = 2.0

    def _vgap(top_box, bottom_box):
        return float(bottom_box["y1"] - top_box["y2"])

    def _section_break_image_between(top_y, bottom_y, established_width, image_boxes):

        for img_box in image_boxes:

            if img_box["y1"] < top_y or img_box["y2"] > bottom_y:
                continue

            width = float(img_box["x2"] - img_box["x1"])

            if width > WIDE_IMAGE_SECTION_BREAK_RATIO * established_width:
                return True

        return False

    articles = {}
    root_ids = {id(r) for r in roots}
    root_chain_tips = {}
    root_established_width = {}

    # Multi-tip chaining (below) is scoped to ONLY the roots this
    # pass's fragment-collapsing actually widened -- an ordinary,
    # single-fragment title keeps the original single shared "last
    # attached block" pointer (a second tip is never appended for
    # it). Confirmed necessary on a real page: letting an ordinary
    # (non-collapsed) wide-ish title track multiple column tips
    # changed which of TWO separate, unrelated headlines won a wide
    # bridging figure+caption in the tiebreak -- both were already
    # geometrically eligible for either story before this change, so
    # that reassignment cannot be verified correct from geometry
    # alone. Multi-tip tracking is only PROVEN necessary for a
    # fragment-collapsed root (see the block-20 case in this
    # function's own history/tests), so it is kept scoped there.
    fragment_root_ids = {
        id(by_id[primary_id]) for primary_id in fragment_members_of_primary
    }

    for order, root in enumerate(roots, start=1):
        root_box = _bbox(root)
        articles[id(root)] = {
            "article_id": order,
            "blocks": [root["id"]],
        }
        # One chain tip per COLUMN this root currently governs, not
        # one shared pointer -- a wide root (a multi-column title, or
        # a banner collapsed from several fragments, see the
        # fragment-collapsing pass above) can have several columns of
        # body text attaching to it in (y1, x1) order, interleaved
        # with each other. A single shared "last attached block"
        # pointer lets a later column's attachment silently steal the
        # reference point away from an earlier column that still has
        # more content below it. Confirmed on a real Urdu page: a
        # title collapsed from two column-cut fragments had its RIGHT
        # column's last block (5) overwritten by its LEFT column's
        # next block (46, processed in between in y-order, sharing no
        # column with 5 at all) before the right column's own next
        # block (20) was reached -- 20 then measured its gap against
        # 46 (a different column entirely, 1063px away) instead of 5
        # (603px away, its real predecessor), and was rejected as
        # exceeding MAX_CONTINUATION_GAP.
        root_chain_tips[id(root)] = [root_box]
        root_established_width[id(root)] = max(
            1.0, float(root_box["x2"] - root_box["x1"])
        )

    image_boxes = [
        box
        for b in content
        if roles.get(b["id"]) == "article_image"
        and (box := _bbox(b)) is not None
    ]

    merge_log = []
    orphans = []

    for block in content:

        if id(block) in root_ids:
            continue

        box = _bbox(block)

        best_root = None
        best_gap = None
        best_reason = None
        best_chain_index = None
        blocking_reason = None

        for root in roots:

            root_box = _bbox(root)

            if root_box["y1"] > box["y1"]:
                continue

            if _containment_ratio(box, root_box) < COLUMN_OVERLAP:
                continue

            # Which of this root's current chain tips (one per column
            # it already governs) does this candidate continue, if
            # any -- the NEAREST one it column-overlaps with, not
            # just the most recently attached tip overall. See the
            # comment where root_chain_tips is built.
            tips = root_chain_tips[id(root)]
            chain_index = None
            chain_gap = None

            for index, tip in enumerate(tips):

                if _overlap_ratio(box, tip) < COLUMN_OVERLAP:
                    continue

                tip_gap = _vgap(tip, box)

                if tip_gap < 0:
                    tip_gap = 0.0

                if chain_gap is None or tip_gap < chain_gap:
                    chain_gap = tip_gap
                    chain_index = index

            chained = chain_index is not None

            reference_box = tips[chain_index] if chained else root_box
            gap = _vgap(reference_box, box)

            if gap < 0:
                gap = 0.0

            if _section_break_image_between(
                reference_box["y2"], box["y1"],
                root_established_width[id(root)],
                image_boxes,
            ):
                blocking_reason = (
                    f"root {root['id']}: wide section-break image "
                    f"between last content and candidate"
                )
                continue

            if gap > MAX_CONTINUATION_GAP:
                blocking_reason = (
                    f"root {root['id']}: gap {gap:.0f}px exceeds "
                    f"MAX_CONTINUATION_GAP"
                )
                continue

            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_root = root
                best_chain_index = chain_index
                best_reason = (
                    f"chained to last block of root {root['id']}"
                    if chained
                    else f"new column under root {root['id']}"
                )

        if best_root is None:
            # No headline governs this block in its own column.
            # Attaching it to the vertically nearest root anywhere
            # was tried and produced one 34-block blob that swallowed
            # the masthead strip and two unrelated stories, because
            # every orphan on the page drained into whichever article
            # happened to be closest. Standing on its own is the
            # lesser error: it keeps the text and keeps it out of an
            # article it does not belong to.
            merge_log.append(
                (block["id"], None, blocking_reason or "no column-eligible root")
            )

            if roles.get(block["id"]) == "article_title":
                # An independent headline with no body of its own
                # below it still normally starts its own story rather
                # than being folded into another -- it becomes its
                # own root so anything genuinely printed under IT can
                # still attach in a later iteration of this same loop.
                articles[id(block)] = {
                    "article_id": len(articles) + 1,
                    "blocks": [block["id"]],
                }
                roots.append(block)
                root_ids.add(id(block))
                root_chain_tips[id(block)] = [box]
                root_established_width[id(block)] = max(
                    1.0, float(box["x2"] - box["x1"])
                )
            else:
                # No title anywhere governs this block -- but that
                # does not mean it is its own whole story. It is
                # deferred to the orphan-continuity clustering pass
                # below, which can still reconnect it with OTHER
                # title-less blocks that are really the same story
                # (see that pass for why title roots can't do this
                # job themselves).
                orphans.append(block)

            continue

        merge_log.append((block["id"], best_root["id"], best_reason))
        articles[id(best_root)]["blocks"].append(block["id"])

        tips = root_chain_tips[id(best_root)]

        if best_chain_index is not None:
            # Continues an existing column's chain -- advance THAT
            # column's tip only, leaving every other column's own
            # chain tip untouched.
            tips[best_chain_index] = box
        elif id(best_root) in fragment_root_ids:
            # A genuinely new column entering this FRAGMENT-COLLAPSED
            # root's story for the first time -- gets its own chain
            # tip going forward. Scoped to fragment roots only; see
            # fragment_root_ids above.
            tips.append(box)
        else:
            # Ordinary root: preserve the original single shared
            # "last attached block" pointer behaviour exactly.
            tips[0] = box

    # ----------------------------------------------------------
    # Orphan continuity clustering
    #
    # A block with no column-eligible title root is not necessarily
    # its own whole story -- it may be one column (or one image) of a
    # story whose real headline the layout detector never classified
    # as "title" at all, a genuine DETECTION gap rather than a
    # grouping one. Confirmed on a real Urdu page: a story printed
    # across 3 side-by-side columns under one banner photo, with no
    # title-class block anywhere near it, left that photo and all 3
    # text columns as 4 separate singletons -- the root-attachment
    # loop above only ever measures a candidate against TITLE roots,
    # so it has no way to see that these 4 orphans are each other's
    # story.
    #
    # Two blocks that neither attached to any title root are still
    # the same story when either:
    #
    # 1. SIDE BY SIDE -- they share a horizontal row/band (not a
    #    column) with only an ordinary gutter between them and no
    #    printed rule line inside that gutter. This is exactly what
    #    lets 2 or 3 text columns of one headline-less story rejoin --
    #    the same test already validated for title "packages" in
    #    article_splitter.py's _is_adjacent_package_column, adapted to
    #    bbox dicts here.
    #
    # 2. STACKED -- one sits directly below the other in the same
    #    visual lane by CONTAINMENT rather than union overlap, so a
    #    wide banner photo can anchor the narrower text below it (see
    #    _containment_ratio), within MAX_CONTINUATION_GAP, with no
    #    OTHER story's own title printed between them and no wide
    #    section-break image between them that isn't one of the two
    #    endpoints themselves.
    #
    # Anything that qualifies for neither is left standing alone,
    # exactly like the pre-existing behaviour for a genuinely isolated
    # block.
    # ----------------------------------------------------------

    ORPHAN_ROW_MIN_OVERLAP = 0.5
    ORPHAN_SIDE_GAP_MAX_RATIO = 0.5
    ORPHAN_STACK_CONTAINMENT_MIN = 0.5

    title_root_boxes = [
        _bbox(r) for r in roots if roles.get(r["id"]) == "article_title"
    ]

    def _title_between(top_y, bottom_y):

        if top_y >= bottom_y:
            return False

        for title_box in title_root_boxes:

            center = (title_box["y1"] + title_box["y2"]) / 2.0

            if top_y < center < bottom_y:
                return True

        return False

    def _orphans_connect(block_a, block_b):

        box_a = _bbox(block_a)
        box_b = _bbox(block_b)

        if _row_overlap_ratio(box_a, box_b) >= ORPHAN_ROW_MIN_OVERLAP:

            left, right = (
                (box_a, box_b) if box_a["x1"] <= box_b["x1"] else (box_b, box_a)
            )

            gap = right["x1"] - left["x2"]

            if gap >= 0:

                shorter_height = min(
                    box_a["y2"] - box_a["y1"], box_b["y2"] - box_b["y1"]
                )

                if (
                    shorter_height > 0
                    and gap <= ORPHAN_SIDE_GAP_MAX_RATIO * shorter_height
                    and not has_vertical_separator_between(
                        page_image,
                        left["x2"], right["x1"],
                        min(box_a["y1"], box_b["y1"]),
                        max(box_a["y2"], box_b["y2"]),
                    )
                ):
                    return True

            # A negative gap here means the two boxes actually
            # overlap horizontally too, not just share a row -- this
            # is not a side-by-side column pair at all, but usually a
            # duplicate/overlapping detection of the same region (a
            # known layout-detector artifact elsewhere in this
            # codebase, see article_splitter.py's
            # _collapse_banner_fragments). Fall through to the
            # containment test below instead of rejecting outright.

        top, bottom = (
            (box_a, box_b) if box_a["y1"] <= box_b["y1"] else (box_b, box_a)
        )

        containment = max(
            _containment_ratio(top, bottom),
            _containment_ratio(bottom, top),
        )

        if containment < ORPHAN_STACK_CONTAINMENT_MIN:
            return False

        gap = float(bottom["y1"] - top["y2"])

        if gap < 0:
            gap = 0.0

        if gap > MAX_CONTINUATION_GAP:
            return False

        if _title_between(top["y2"], bottom["y1"]):
            return False

        narrower_width = min(
            top["x2"] - top["x1"], bottom["x2"] - bottom["x1"]
        )

        other_images = [
            img for img in image_boxes
            if img is not box_a and img is not box_b
        ]

        return not _section_break_image_between(
            top["y2"], bottom["y1"], narrower_width, other_images,
        )

    parent = list(range(len(orphans)))

    def _find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(index_a, index_b):
        root_a, root_b = _find(index_a), _find(index_b)
        if root_a != root_b:
            parent[root_a] = root_b

    for i in range(len(orphans)):
        for j in range(i + 1, len(orphans)):
            if _orphans_connect(orphans[i], orphans[j]):
                _union(i, j)

    clusters = {}

    for index, block in enumerate(orphans):
        clusters.setdefault(_find(index), []).append(block)

    for member_blocks in clusters.values():

        member_blocks.sort(key=lambda b: (_bbox(b)["y1"], _bbox(b)["x1"]))
        representative = member_blocks[0]

        articles[id(representative)] = {
            "article_id": len(articles) + 1,
            "blocks": [b["id"] for b in member_blocks],
        }

        for member in member_blocks[1:]:
            merge_log.append((
                member["id"],
                representative["id"],
                f"orphan cluster: reading-order continuity with "
                f"block {representative['id']}",
            ))

    # ----------------------------------------------------------
    # Orphan-root cleanup
    #
    # A title can become a root (has_own_body / gap / separator) and
    # still end up governing nothing: the content directly below it
    # can independently qualify as ITS OWN root instead of attaching
    # as this one's body (e.g. a kicker/eyebrow line immediately
    # above a main headline that itself has real body text -- the
    # main headline becomes its own root, leaving the kicker with
    # nothing attached). That leaves a lone title-only entry, which
    # the final filter below drops -- silently losing a real,
    # genuine heading rather than just mis-attributing it.
    #
    # So: any root that ended up governing nothing is offered back to
    # the SAME nearest-owning-root search ordinary content already
    # uses (excluding itself) -- exactly what would have happened had
    # it never become a root. Only if no OTHER root can plausibly own
    # it either does it fall back to standing alone (and still get
    # dropped by the filter below if it's a lone title) -- matching
    # the pre-existing "no root found" behaviour precisely, just
    # tried as a last resort instead of a first one.
    # ----------------------------------------------------------

    orphan_roots = [
        root
        for root in roots
        if (
            roles.get(root["id"]) == "article_title"
            and len(articles[id(root)]["blocks"]) == 1
        )
    ]

    for root in orphan_roots:

        other_roots = [
            candidate
            for candidate in roots
            if id(candidate) != id(root)
        ]

        box = _bbox(root)

        fallback_root = _find_owning_root(
            box, other_roots,
        )

        if fallback_root is None:
            # Nothing above it (it may be the very first element in
            # its column, as a kicker/eyebrow line always is) -- try
            # the headline that follows it instead.
            fallback_root = _find_next_root_below(
                box, other_roots,
            )

        if fallback_root is None:
            continue

        del articles[id(root)]
        roots.remove(root)
        root_ids.discard(id(root))

        articles[id(fallback_root)]["blocks"].append(root["id"])

    # ----------------------------------------------------------
    # Re-attach banner-fragment secondaries to their primary's
    # article, now that every root's final article is known.
    #
    # Done as a search-and-append over the finished `articles` dict,
    # not at root-creation time, because a primary fragment can end
    # up governing its article via ANY of three different paths
    # (established as a root directly above; falls through to the
    # content-attachment loop's own "title with no eligible root
    # becomes its own root" branch; or gets folded into a DIFFERENT
    # root by the orphan-root cleanup pass just above, if it ended up
    # governing nothing of its own) -- searching after the fact is
    # correct regardless of which path the primary actually took.
    #
    # Known imprecision: the orphan-root cleanup pass above decides
    # whether a lone-title root is "governing nothing" using its
    # block count BEFORE this injection runs, so a banner headline
    # with real fragments but no body text of its own can still be
    # folded into a nearby root there instead of surviving as its
    # own (now multi-block) entity. Rare in practice -- a genuine
    # section-divider banner with zero body text under any of its
    # fragments -- and not clearly worse than the pre-existing
    # single-fragment behaviour, so left as a known limitation
    # rather than reordering this function's passes for it.
    # ----------------------------------------------------------

    for primary_id, secondary_ids in fragment_members_of_primary.items():

        for article in articles.values():

            if primary_id not in article["blocks"]:
                continue

            article["blocks"].extend(secondary_ids)

            for secondary_id in secondary_ids:
                merge_log.append((
                    secondary_id,
                    primary_id,
                    f"banner headline fragment of block {primary_id}",
                ))

            break

    # ----------------------------------------------------------
    # Emit the LLM-compatible response
    # ----------------------------------------------------------

    # A lone TITLE with nothing ever attached to it is just a
    # headline that governs no content -- correctly dropped. A lone
    # block of any OTHER content role (text, image, caption, byline)
    # is real page content that belongs to no other story; silently
    # dropping it loses real photos and paragraphs off the page
    # entirely, which is strictly worse than showing it as its own
    # single-block story.
    grouped = [
        entry
        for entry in articles.values()
        if len(entry["blocks"]) > 1
        or roles.get(entry["blocks"][0]) != "article_title"
    ]

    for index, entry in enumerate(grouped, start=1):
        entry["article_id"] = index

    if debug:

        print()
        print("=" * 60)
        print("LOCAL GROUPER DIAGNOSTIC")
        print("=" * 60)

        for entry in grouped:

            root_id = entry["blocks"][0]

            headline_ids = [
                bid for bid in entry["blocks"]
                if roles.get(bid) == "article_title"
            ]
            body_ids = [
                bid for bid in entry["blocks"]
                if roles.get(bid) == "article_text"
            ]
            image_ids = [
                bid for bid in entry["blocks"]
                if roles.get(bid) == "article_image"
            ]
            caption_ids = [
                bid for bid in entry["blocks"]
                if roles.get(bid) == "caption"
            ]

            print(f"Story {entry['article_id']}:")
            print(f"  headline : {headline_ids or '(none)'}")
            print(f"  body     : {body_ids}")
            print(f"  images   : {image_ids}")
            print(f"  captions : {caption_ids}")

            reasons = [
                f"    block {bid} <- {reason}"
                for bid, target, reason in merge_log
                if target == root_id
            ]

            if reasons:
                print("  merge reasons:")
                for line in reasons:
                    print(line)

        rejected = [
            (bid, reason)
            for bid, target, reason in merge_log
            if target is None
        ]

        if rejected:
            print()
            print("Blocks left standing alone (no root accepted them):")
            for bid, reason in rejected:
                print(f"  block {bid}: {reason}")

        print("=" * 60)
        print()

    return {
        "blocks": [
            {"id": block_id, "role": role}
            for block_id, role in roles.items()
        ],
        "articles": grouped,
    }
