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
2. A candidate becomes a real root when EITHER:
   - the blank space above it clearly exceeds the page's normal
     block spacing, which is what stops a bold photo caption or an
     inline crosshead -- both of which the layout detector happily
     labels "title" -- from splitting one story into several, OR
   - the pixels directly above it show a printed rule line or a
     colored box background, which is how a newspaper marks a
     boxed sidebar story that sits closer than the whitespace rule
     alone would accept (see _has_visual_separator).
3. Every remaining content block joins the nearest root ABOVE it that
   shares its column span, which is how newspaper columns actually
   read.

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


# Layout-detector classes that carry article text.
TEXT_CLASSES = {"plain text", "title", "figure_caption"}

# Layout-detector classes that are pictures.
IMAGE_CLASSES = {"figure", "image", "object", "photo", "picture"}

# A title needs at least this much more blank space above it than the
# page's median block gap before it is accepted as a new article root.
ROOT_GAP_FACTOR = 2.0

# Floor for the above, so a page whose median gap is tiny does not turn
# every stray title into a root.
ROOT_GAP_MINIMUM = 25.0

# Fraction of the narrower block's width that must overlap
# horizontally for two blocks to count as sharing a column.
COLUMN_OVERLAP = 0.35


def _bbox(block):
    box = block.get("bbox") or {}
    if None in (box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")):
        return None
    return box


def _overlap_ratio(a, b):
    width_a = max(1.0, float(a["x2"] - a["x1"]))
    width_b = max(1.0, float(b["x2"] - b["x1"]))
    overlap = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
    return max(0.0, overlap) / min(width_a, width_b)


def _role_for(block):
    """
    Map a detector class / cleaner type onto the semantic role
    vocabulary the downstream stages expect.
    """

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


# ----------------------------------------------------------------
# VISUAL SEPARATOR (rule line / colored box edge)
# ----------------------------------------------------------------
#
# Confirmed against a real page: a boxed sidebar story (a title
# printed inside a bordered or colored-background box) can sit as
# close as 17px above the preceding content -- BELOW the page's own
# 23px median paragraph gap. No gap threshold can distinguish that
# from an ordinary paragraph break; the newspaper is using a printed
# border instead of whitespace to mark the split, so the pixels
# themselves have to be checked for that border.

# A printed rule line or a colored box background is a MOSTLY
# UNIFORM row of pixels that is not blank paper: low variation across
# the row (unlike a row of body text, which mixes dark glyphs and
# white background and so has high variation), but not lit up close
# to white either.
SEPARATOR_ROW_STD_MAX = 15.0
SEPARATOR_ROW_BRIGHTNESS_MAX = 205.0

# A colored tag/box background (the common "अवसर" / "आदेश" style
# orange or yellow label boxes) can be almost as bright as paper in
# plain grayscale, so it needs its own check: a saturated row is
# color, which blank paper and plain body text never are.
SEPARATOR_ROW_SATURATION_MIN = 45.0

# The probe band is at least this tall even when the measured gap is
# ~0, so a border drawn flush against the block is still inspected.
SEPARATOR_MIN_BAND_HEIGHT = 10


def _has_visual_separator(page_image, x1, x2, y1, y2):
    """
    Look for a printed rule line or colored box edge in the page
    image, within columns [x1, x2) and rows [y1, y2).
    """

    if page_image is None:
        return False

    height, width = page_image.shape[:2]

    x1 = max(0, int(x1))
    x2 = min(width, int(x2))
    y1 = max(0, int(y1))
    y2 = min(height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return False

    band = page_image[y1:y2, x1:x2]

    if band.size == 0:
        return False

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)

    row_std = gray.std(axis=1)
    row_mean = gray.mean(axis=1)
    row_saturation = hsv[:, :, 1].mean(axis=1)

    is_rule_line = (
        (row_std < SEPARATOR_ROW_STD_MAX)
        & (row_mean < SEPARATOR_ROW_BRIGHTNESS_MAX)
    )

    is_colored_band = (
        row_saturation > SEPARATOR_ROW_SATURATION_MIN
    )

    return bool((is_rule_line | is_colored_band).any())


# Width of the strip probed just outside a title's left/right edge.
VERTICAL_SEPARATOR_PROBE_WIDTH = 20

# A vertical divider bar only needs to be dark/colored for a solid
# majority of the title's height, not the full height, since OCR
# bounding boxes are usually a few px tighter than the printed bar.
VERTICAL_SEPARATOR_MIN_COVERAGE = 0.6


def _has_vertical_separator(page_image, x1, x2, y1, y2):
    """
    Look for a vertical divider bar immediately to the LEFT or RIGHT
    of a block -- the pattern used by side-by-side tag/label boxes
    printed in one row (e.g. three short items separated by thin red
    bars rather than being stacked with a gap between them, which
    _has_visual_separator's row-wise horizontal check cannot see at
    all since there is no horizontal band between such blocks).
    """

    if page_image is None:
        return False

    height, width = page_image.shape[:2]

    y1 = max(0, int(y1))
    y2 = min(height, int(y2))

    if y2 <= y1:
        return False

    def probe_strip(strip_x1, strip_x2):

        strip_x1 = max(0, int(strip_x1))
        strip_x2 = min(width, int(strip_x2))

        if strip_x2 <= strip_x1:
            return False

        strip = page_image[y1:y2, strip_x1:strip_x2]

        if strip.size == 0:
            return False

        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        # Per-pixel ink test (dark OR saturated), then per-COLUMN
        # coverage of that test down the probe height. A mean/std
        # summary taken across the whole column was tried first and
        # rejected: a bar only 1-3px wide sits among ~20px of blank
        # margin, and a bar that does not run the full probe height
        # (rendering/alignment slop) mixes with blank rows in the
        # same column -- both dilute a plain average toward "blank"
        # even directly on top of a real, visible bar. Coverage
        # fraction is not fooled by either: the bar's OWN column
        # still shows high ink coverage regardless of how much blank
        # margin surrounds it or how much of the full height it
        # covers.
        ink = (
            (gray < SEPARATOR_ROW_BRIGHTNESS_MAX)
            | (hsv[:, :, 1] > SEPARATOR_ROW_SATURATION_MIN)
        )

        coverage = ink.mean(axis=0)

        return bool(
            (coverage >= VERTICAL_SEPARATOR_MIN_COVERAGE).any()
        )

    left = probe_strip(
        x1 - VERTICAL_SEPARATOR_PROBE_WIDTH,
        x1,
    )

    right = probe_strip(
        x2,
        x2 + VERTICAL_SEPARATOR_PROBE_WIDTH,
    )

    return bool(left or right)


def build_local_response(page_json_path, page_image_path=None):
    """
    Group one page's blocks into articles without any model call.
    """

    page = json.loads(
        Path(page_json_path).read_text(encoding="utf-8")
    )

    blocks = page.get("blocks", []) if isinstance(page, dict) else page

    if not blocks:
        return {"blocks": [], "articles": []}

    roles = {b["id"]: _role_for(b) for b in blocks if "id" in b}

    median_gap = _annotate_gaps(blocks)

    root_threshold = max(
        ROOT_GAP_MINIMUM,
        median_gap * ROOT_GAP_FACTOR,
    )

    page_image = None

    if page_image_path is not None:

        try:
            page_image = cv2.imread(str(page_image_path))
        except Exception:
            page_image = None

    # ----------------------------------------------------------
    # Choose article roots
    # ----------------------------------------------------------

    content = [
        b
        for b in blocks
        if _bbox(b) is not None
        and roles.get(b["id"])
        in {"article_title", "article_text", "article_image", "caption"}
    ]

    content.sort(key=lambda b: (_bbox(b)["y1"], _bbox(b)["x1"]))

    roots = []

    for block in content:

        if roles.get(block["id"]) != "article_title":
            continue

        gap = block.get("_gap_above")

        # No neighbour above means this is the first thing in its
        # column, which is where a headline normally sits.
        if gap is None or gap >= root_threshold:
            roots.append(block)
            continue

        # A photo is inherently non-uniform and often colorful, so a
        # caption sitting directly under one would otherwise look
        # exactly like a colored separator box to the pixel check
        # below (confirmed: this false-positived on a real photo
        # caption, sampling the photo's own bottom edge as if it
        # were a box border). A caption's relationship is to the
        # image above it, never to a rule line, so the visual check
        # is skipped entirely in that situation rather than tuned
        # around it.

        if block.get("_neighbor_above_class") in IMAGE_CLASSES:
            continue

        # The gap alone did not clear the threshold, but a boxed
        # sidebar can be printed closer than that (confirmed as low
        # as 17px on a page with a 23px median gap) because the
        # newspaper is using a border or a colored background to
        # mark the split instead of blank space. Check the pixels
        # directly above this title for that border before deciding
        # it is not a new article.

        box = _bbox(block)

        probe_height = max(gap, SEPARATOR_MIN_BAND_HEIGHT)

        has_horizontal_separator = _has_visual_separator(
            page_image,
            box["x1"],
            box["x2"],
            box["y1"] - probe_height,
            box["y1"] + 2,
        )

        # Some layouts print several short items in one row,
        # divided from each other by a vertical bar instead of
        # being stacked with a gap -- confirmed on a real page: a
        # row of three tag-style items ("अवसर" | title | "आदेश")
        # where only the outer two have a colored box background,
        # so the middle one is invisible to the horizontal check
        # even though it is printed as its own bounded item.
        has_vertical_separator = _has_vertical_separator(
            page_image,
            box["x1"],
            box["x2"],
            box["y1"],
            box["y2"],
        )

        if has_horizontal_separator or has_vertical_separator:
            roots.append(block)

    # A page of pure body text still needs somewhere to put it.
    if not roots and content:
        roots = [content[0]]

    # ----------------------------------------------------------
    # Attach every content block to a root
    # ----------------------------------------------------------

    articles = {}
    root_ids = {id(r) for r in roots}

    for order, root in enumerate(roots, start=1):
        articles[id(root)] = {
            "article_id": order,
            "blocks": [root["id"]],
        }

    for block in content:

        if id(block) in root_ids:
            continue

        box = _bbox(block)

        best_root = None
        best_distance = None

        for root in roots:

            root_box = _bbox(root)

            # A headline governs what reads BELOW it.
            if root_box["y1"] > box["y1"]:
                continue

            if _overlap_ratio(box, root_box) < COLUMN_OVERLAP:
                continue

            distance = float(box["y1"] - root_box["y1"])

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_root = root

        if best_root is None:
            # No headline governs this block in its own column.
            # Attaching it to the vertically nearest root anywhere
            # was tried and produced one 34-block blob that swallowed
            # the masthead strip and two unrelated stories, because
            # every orphan on the page drained into whichever article
            # happened to be closest. Standing on its own is the
            # lesser error: it keeps the text and keeps it out of an
            # article it does not belong to.
            if roles.get(block["id"]) in {
                "article_text",
                "article_title",
            }:
                articles[id(block)] = {
                    "article_id": len(articles) + 1,
                    "blocks": [block["id"]],
                }
                roots.append(block)
                root_ids.add(id(block))

            continue

        articles[id(best_root)]["blocks"].append(block["id"])

    # ----------------------------------------------------------
    # Emit the LLM-compatible response
    # ----------------------------------------------------------

    grouped = [
        entry
        for entry in articles.values()
        if len(entry["blocks"]) > 1
        or roles.get(entry["blocks"][0]) == "article_text"
    ]

    for index, entry in enumerate(grouped, start=1):
        entry["article_id"] = index

    return {
        "blocks": [
            {"id": block_id, "role": role}
            for block_id, role in roles.items()
        ],
        "articles": grouped,
    }
