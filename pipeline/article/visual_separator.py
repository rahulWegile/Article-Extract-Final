"""
Pixel-level detection of a printed rule line or colored box edge in a
page image.

Extracted out of local_grouper.py so both the offline geometric
grouper AND the Gemini LLM path (gemini_service.py, when it annotates
each block with a separator signal for the model) can share the same
tuned probe instead of maintaining two copies of this cv2 logic.
"""

from typing import Any

import cv2

from pipeline.block_parser import LayoutBlock


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
SEPARATOR_ROW_BRIGHTNESS_MAX = 235.0

# A colored tag/box background (the common "अवसर" / "आदेश" style
# orange or yellow label boxes) can be almost as bright as paper in
# plain grayscale, so it needs its own check: a saturated row is
# color, which blank paper and plain body text never are.
SEPARATOR_ROW_SATURATION_MIN = 45.0

# The probe band is at least this tall even when the measured gap is
# ~0, so a border drawn flush against the block is still inspected.
SEPARATOR_MIN_BAND_HEIGHT = 10


def has_visual_separator(page_image, x1, x2, y1, y2):
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


def _has_bar_in_strip(page_image, x1, x2, y1, y2):
    """
    Whether the strip [x1, x2) x [y1, y2) contains a vertical bar:
    at least one COLUMN of pixels that is mostly ink.

    Per-pixel ink test (dark OR saturated), then per-COLUMN coverage
    of that test down the probe height. A mean/std summary taken
    across the whole column was tried first and rejected: a bar only
    1-3px wide sits among ~20px of blank margin, and a bar that does
    not run the full probe height (rendering/alignment slop) mixes
    with blank rows in the same column -- both dilute a plain average
    toward "blank" even directly on top of a real, visible bar.
    Coverage fraction is not fooled by either: the bar's OWN column
    still shows high ink coverage regardless of how much blank margin
    surrounds it or how much of the full height it covers.
    """

    height, width = page_image.shape[:2]

    x1 = max(0, int(x1))
    x2 = min(width, int(x2))
    y1 = max(0, int(y1))
    y2 = min(height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return False

    strip = page_image[y1:y2, x1:x2]

    if strip.size == 0:
        return False

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    ink = (
        (gray < SEPARATOR_ROW_BRIGHTNESS_MAX)
        | (hsv[:, :, 1] > SEPARATOR_ROW_SATURATION_MIN)
    )

    coverage = ink.mean(axis=0)

    return bool(
        (coverage >= VERTICAL_SEPARATOR_MIN_COVERAGE).any()
    )


def has_vertical_separator(page_image, x1, x2, y1, y2):
    """
    Look for a vertical divider bar immediately to the LEFT or RIGHT
    of a block -- the pattern used by side-by-side tag/label boxes
    printed in one row (e.g. three short items separated by thin red
    bars rather than being stacked with a gap between them, which
    has_visual_separator's row-wise horizontal check cannot see at
    all since there is no horizontal band between such blocks).
    """

    if page_image is None:
        return False

    left = _has_bar_in_strip(
        page_image,
        x1 - VERTICAL_SEPARATOR_PROBE_WIDTH,
        x1,
        y1,
        y2,
    )

    right = _has_bar_in_strip(
        page_image,
        x2,
        x2 + VERTICAL_SEPARATOR_PROBE_WIDTH,
        y1,
        y2,
    )

    return bool(left or right)


def has_vertical_separator_between(page_image, x1, x2, y1, y2):
    """
    Look for a vertical divider bar INSIDE the strip [x1, x2) --
    i.e. in the blank channel BETWEEN two side-by-side blocks,
    rather than just outside one block's own edge.

    has_vertical_separator cannot answer this: its probe strips are
    anchored 20px outside the block edges it is given, so on a
    narrow inter-block channel they land on the neighbouring blocks'
    own glyphs and report ink that belongs to text, not to a rule.
    """

    if page_image is None:
        return False

    return _has_bar_in_strip(page_image, x1, x2, y1, y2)


# ----------------------------------------------------------------
# UNDETECTED CONTENT IN A "GAP"
#
# The opposite question from has_visual_separator: not "is there a
# printed rule/border here", but "is this supposedly blank gutter
# actually blank paper at all -- or does the newspaper print real
# content there that the layout detector simply never boxed?"
#
# Confirmed on a real Urdu (THE INQUILAB) page: a colourful GDP/flag/
# gold-bars infographic sat between two story columns and was never
# detected as a block at all (the detector boxed the photo beside it
# but not this one). article_splitter.py's column-package check,
# reasoning only about DETECTED blocks, measured a 537px "gap"
# between the two title columns either side of it -- far past its
# own gutter-width allowance -- and concluded they were two separate
# stories, splitting a single Gemini-grouped article back apart even
# though Gemini's own read of the full page image had it right.
#
# Reuses the same ink test has_visual_separator already applies per
# ROW (dark OR saturated pixels), just averaged over the whole probed
# rectangle instead of tested row-by-row: a real photo/graphic lights
# up a meaningful fraction of its own pixels either dark or coloured,
# while genuinely blank newsprint does not.
# ----------------------------------------------------------------

# Fraction of ink pixels (dark or saturated) a rectangle must contain
# before it counts as "real printed content", not blank paper. Set
# low relative to SEPARATOR_ROW_STD_MAX's per-row test: a photo mixes
# plenty of near-white sky/background/margin pixels with its actual
# subject, so this only needs to catch a photo/graphic's presence, not
# characterise its whole area as covered.
UNDETECTED_CONTENT_INK_RATIO_MIN = 0.12


def has_undetected_content(page_image, x1, x2, y1, y2) -> bool:
    """
    Whether the rectangle [x1, x2) x [y1, y2) contains substantial
    printed ink -- i.e. is NOT genuinely blank paper -- regardless of
    whether any layout-detector block covers it.

    Used to tell a real editorial gutter (blank paper the newspaper
    deliberately left empty between two separate stories) apart from
    a gap that only LOOKS empty because the layout detector missed a
    photo/graphic sitting in it (see module docstring above).
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

    ink = (
        (gray < SEPARATOR_ROW_BRIGHTNESS_MAX)
        | (hsv[:, :, 1] > SEPARATOR_ROW_SATURATION_MIN)
    )

    return float(ink.mean()) >= UNDETECTED_CONTENT_INK_RATIO_MIN


# ----------------------------------------------------------------
# GAP / RULE-LINE ANNOTATION
#
# Shared by both LLM-provider payload builders (gemini_service.py,
# openai_service.py) so a document's boundary-detection call gets
# the SAME gap and rule-line evidence regardless of which provider
# its language routes to. Originally lived only in gemini_service.py
# (Hindi's provider); an Urdu document (openai_service.py) never
# received gap_above/gap_below/median_block_gap/has_rule_line_above/
# has_rule_line_below at all, even though Urdu newspapers box every
# story in a printed rule at least as often as the Hindi papers this
# was built for -- confirmed on a real Urdu (THE INQUILAB) page: the
# printed red/green boxes visible around each story are exactly this
# signal, and it was silently unavailable to the model grouping that
# page.
# ----------------------------------------------------------------

def has_intervening_block(
    boxes: list[dict[str, Any] | None],
    self_index: int,
    y_start: float,
    y_end: float,
    bbox: dict[str, Any],
) -> bool:
    """
    True when some OTHER block's y-range falls inside (y_start,
    y_end) and it has ANY horizontal overlap with `bbox` at all --
    not just the stricter 50% same-column overlap used to find a
    gap neighbour. Such a block sits visually inside the measured
    gap, so the gap is not clean whitespace.
    """

    lo, hi = min(y_start, y_end), max(y_start, y_end)

    for other_index, other in enumerate(boxes):

        if other is None or other_index == self_index:
            continue

        if other["y1"] >= hi or other["y2"] <= lo:
            continue

        overlap = min(bbox["x2"], other["x2"]) - max(
            bbox["x1"], other["x1"]
        )

        if overlap > 0:
            return True

    return False

def annotate_gaps(
    compact_blocks: list[dict[str, Any]],
    page_image=None,
) -> float | None:
    """
    Annotate each block with the whitespace gap to its nearest
    neighbour above and below within the same print column, plus
    return the page's median gap for scale.

    Newspapers separate stories with visibly more whitespace than
    they put between paragraphs of the same story, so the size of
    a gap relative to the page's normal gap is one of the
    strongest available separation signals. The model previously
    received only raw bounding boxes and had to re-derive this
    itself for every block, which it did inconsistently on dense
    pages.

    Column neighbours are found by horizontal overlap rather than
    by the 6-slot column index, because a wide block (a spanning
    headline, a photo) belongs to several index slots at once.

    When `page_image` is given, each block also gets
    `has_rule_line_above`/`has_rule_line_below`: a printed rule
    line or colored box edge found by probing the page's own
    pixels (see visual_separator.has_visual_separator), the same
    check local_grouper.py already relies on to catch a boxed
    sidebar story printed closer to its neighbour than any gap
    threshold could distinguish from a paragraph break. Gaps
    alone leave the model to spot such a line purely by eye in
    the uploaded image; this hands it the answer directly.
    """

    boxes = []

    for block in compact_blocks:

        bbox = block.get("bbox") or {}

        if None in (
            bbox.get("x1"),
            bbox.get("y1"),
            bbox.get("x2"),
            bbox.get("y2"),
        ):
            boxes.append(None)
            continue

        boxes.append(bbox)

    gaps: list[float] = []

    for index, bbox in enumerate(boxes):

        if bbox is None:
            continue

        width = max(1.0, float(bbox["x2"] - bbox["x1"]))

        gap_above = None
        gap_below = None
        above_neighbor_y = None
        below_neighbor_y = None

        for other_index, other in enumerate(boxes):

            if other is None or other_index == index:
                continue

            other_width = max(1.0, float(other["x2"] - other["x1"]))

            overlap = min(bbox["x2"], other["x2"]) - max(
                bbox["x1"], other["x1"]
            )

            # Same print column only.
            if max(0.0, overlap) / min(width, other_width) < 0.5:
                continue

            if other["y2"] <= bbox["y1"]:
                distance = float(bbox["y1"] - other["y2"])
                if gap_above is None or distance < gap_above:
                    gap_above = distance
                    above_neighbor_y = other["y2"]

            elif bbox["y2"] <= other["y1"]:
                distance = float(other["y1"] - bbox["y2"])
                if gap_below is None or distance < gap_below:
                    gap_below = distance
                    below_neighbor_y = other["y1"]

        # The same-column search above can skip straight past some
        # OTHER block sitting visually in the gap (e.g. an inset
        # photo of a different width, which fails the 50%
        # column-overlap test and gets silently passed over) and
        # report the distance across it as if it were plain
        # whitespace. That number then gets handed to the model as
        # a "this looks like a new article" signal even though the
        # gap is actually occupied by unrelated content, not empty.
        # When any block -- regardless of column overlap -- sits
        # inside the measured span, the measurement isn't trustworthy
        # as a whitespace gap, so drop it rather than report a
        # number that can be silently wrong.
        # Rule-line probing uses the RAW (pre-intervening-block-
        # check) gap distance for its band height: whether a
        # printed line sits right at this block's own edge is a
        # separate question from whether the measured span counts
        # as trustworthy blank whitespace.
        probe_height_above = max(
            gap_above if gap_above is not None else 0.0,
            SEPARATOR_MIN_BAND_HEIGHT,
        )

        probe_height_below = max(
            gap_below if gap_below is not None else 0.0,
            SEPARATOR_MIN_BAND_HEIGHT,
        )

        compact_blocks[index]["has_rule_line_above"] = has_visual_separator(
            page_image,
            bbox["x1"],
            bbox["x2"],
            bbox["y1"] - probe_height_above,
            bbox["y1"] + 2,
        )

        compact_blocks[index]["has_rule_line_below"] = has_visual_separator(
            page_image,
            bbox["x1"],
            bbox["x2"],
            bbox["y2"] - 2,
            bbox["y2"] + probe_height_below,
        )

        if gap_above is not None and has_intervening_block(
            boxes, index, above_neighbor_y, bbox["y1"], bbox
        ):
            gap_above = None

        if gap_below is not None and has_intervening_block(
            boxes, index, bbox["y2"], below_neighbor_y, bbox
        ):
            gap_below = None

        compact_blocks[index]["gap_above"] = (
            None if gap_above is None else round(gap_above)
        )

        compact_blocks[index]["gap_below"] = (
            None if gap_below is None else round(gap_below)
        )

        for value in (gap_above, gap_below):
            if value is not None and value > 0:
                gaps.append(value)

    if not gaps:
        return None

    gaps.sort()

    middle = len(gaps) // 2

    if len(gaps) % 2:
        median = gaps[middle]
    else:
        median = (gaps[middle - 1] + gaps[middle]) / 2.0

    return round(median, 1)


TITLE_INK_GRAY_MAX = 150
TITLE_ROW_INK_MIN = 0.03
TITLE_BAND_GAP_MAX = 3
TITLE_HEIGHT_FACTOR = 1.3
TITLE_PROBE_FACTOR = 6.0
TITLE_ATTACH_GAP_FACTOR = 2.0
TITLE_HAS_ABOVE_FACTOR = 2.0
TITLE_CONTRAST_TRIM = 0.12
TITLE_CONTRAST_MAX = 0.15
TITLE_COLUMN_OVERLAP_MIN = 0.35
TITLE_RULE_LINE_COVERAGE_MIN = 0.85


def _title_ink_rows(band):
    if band is None or band.size == 0:
        return None
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    ink = (gray < TITLE_INK_GRAY_MAX) | (
        hsv[:, :, 1] > SEPARATOR_ROW_SATURATION_MIN
    )
    return ink.mean(axis=1)


def _title_text_bands(page_image, x1, x2, y1, y2):
    height, width = page_image.shape[:2]
    x1 = max(0, int(x1))
    x2 = min(width, int(x2))
    y1 = max(0, int(y1))
    y2 = min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return []
    row_ink = _title_ink_rows(page_image[y1:y2, x1:x2])
    if row_ink is None:
        return []
    bands = []
    start = None
    gap = 0
    for row, coverage in enumerate(row_ink):
        if coverage >= TITLE_ROW_INK_MIN:
            if start is None:
                start = row
            gap = 0
        elif start is not None:
            gap += 1
            if gap > TITLE_BAND_GAP_MAX:
                bands.append((start, row - gap))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, len(row_ink) - 1 - gap))
    return [(y1 + s, y1 + e) for s, e in bands if e >= s]


def _title_column_overlap(a_x1, a_x2, block):
    width_a = max(1.0, float(a_x2 - a_x1))
    width_b = max(1.0, float(block.x2 - block.x1))
    overlap = min(a_x2, block.x2) - max(a_x1, block.x1)
    return max(0.0, overlap) / min(width_a, width_b)


def _title_body_line_height(page_image, blocks):
    heights = []
    for block in blocks:
        if (block.cls or "").strip().lower() != "plain text":
            continue
        for start, end in _title_text_bands(
            page_image, block.x1, block.x2, block.y1, block.y2
        ):
            heights.append(end - start + 1)
    if not heights:
        return None
    heights.sort()
    middle = len(heights) // 2
    if len(heights) % 2:
        return heights[middle]
    return (heights[middle - 1] + heights[middle]) / 2.0


def _title_is_rule_line(page_image, x1, x2, band):
    row_cov = _title_ink_rows(
        page_image[int(band[0]):int(band[1]) + 1, int(x1):int(x2)]
    )
    if row_cov is None or len(row_cov) == 0:
        return False
    return float(row_cov.mean()) >= TITLE_RULE_LINE_COVERAGE_MIN


def recover_missing_titles(page_image, blocks):
    if page_image is None:
        return []

    body_line_height = _title_body_line_height(page_image, blocks)
    if not body_line_height:
        return []

    probe_height = body_line_height * TITLE_PROBE_FACTOR
    attach_gap_max = body_line_height * TITLE_ATTACH_GAP_FACTOR
    has_above_gap_max = body_line_height * TITLE_HAS_ABOVE_FACTOR

    all_blocks = list(blocks)
    recovered = []
    next_id = max((b.id for b in blocks), default=-1) + 1

    for target in blocks:

        cls = (target.cls or "").strip().lower()
        if cls not in ("plain text", "figure"):
            continue

        has_above = any(
            other is not target
            and other.y2 <= target.y1
            and (target.y1 - other.y2) <= has_above_gap_max
            and _title_column_overlap(target.x1, target.x2, other)
            >= TITLE_COLUMN_OVERLAP_MIN
            for other in blocks
        )
        if has_above:
            continue

        probe_y1 = max(0.0, target.y1 - probe_height)
        bands = _title_text_bands(
            page_image, target.x1, target.x2, probe_y1, target.y1
        )
        if not bands:
            continue

        rule_lines = [
            b for b in bands
            if _title_is_rule_line(page_image, target.x1, target.x2, b)
        ]
        if rule_lines:
            boundary = max(b[1] for b in rule_lines)
            bands = [b for b in bands if b[0] > boundary]
        if not bands:
            continue

        strong = [
            b for b in bands
            if (b[1] - b[0] + 1) >= body_line_height * TITLE_HEIGHT_FACTOR
        ]
        if not strong:
            continue

        if (target.y1 - strong[-1][1]) > attach_gap_max:
            continue

        cluster = [b for b in bands if b[0] >= strong[0][0]]
        y1 = cluster[0][0]
        y2 = cluster[-1][1]

        row_cov = _title_ink_rows(
            page_image[int(y1):int(y2) + 1, int(target.x1):int(target.x2)]
        )
        if row_cov is None or len(row_cov) == 0:
            continue

        trim = max(1, int(len(row_cov) * TITLE_CONTRAST_TRIM))
        inner = row_cov[trim:len(row_cov) - trim]
        if len(inner) == 0:
            continue

        contrast = float(inner.min()) / float(row_cov.max() + 1e-6)
        if contrast > TITLE_CONTRAST_MAX:
            continue

        crop = page_image[
            int(y1):int(y2) + 1, int(target.x1):int(target.x2)
        ]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ink = (gray_crop < TITLE_INK_GRAY_MAX) | (
            hsv_crop[:, :, 1] > SEPARATOR_ROW_SATURATION_MIN
        )
        cols = ink.any(axis=0)
        if not cols.any():
            continue

        col_indices = cols.nonzero()[0]
        x1 = target.x1 + int(col_indices[0])
        x2 = target.x1 + int(col_indices[-1]) + 1

        overlaps_existing = any(
            not (x2 <= b.x1 or x1 >= b.x2 or (y2 + 1) <= b.y1 or y1 >= b.y2)
            for b in all_blocks
        )
        if overlaps_existing:
            continue

        new_block = LayoutBlock(
            id=next_id,
            cls="title",
            x1=x1,
            y1=int(y1),
            x2=x2,
            y2=int(y2) + 1,
            confidence=1.0,
        )
        recovered.append(new_block)
        all_blocks.append(new_block)
        next_id += 1

    return recovered
