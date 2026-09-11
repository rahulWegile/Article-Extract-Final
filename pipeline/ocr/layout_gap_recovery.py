"""
Recovers real text the LAYOUT DETECTOR missed entirely -- confirmed
on a real page: two bold headlines sitting in a 274px gap between two
detected blocks in the same column, with zero layout box drawn over
them at all (not misclassified, not merged elsewhere -- genuinely
never detected).

DESIGN HISTORY -- why this looks the way it does
-------------------------------------------------

The first version of this tried to recover missed text from OCR
lines the full-page OCR pass had already read but no block claimed,
clustering them and comparing against the page's median inter-block
gap alone. Validated against the SAME real page that motivated this
feature, that version failed in both directions at once:

1. It fabricated 29 bogus "title" blocks out of a dense government
   tender/e-auction notice box (survey coordinates, deposit amounts,
   legal boilerplate) elsewhere on the page. That box had almost no
   layout detection at all (3 blocks across a ~1900px span), so its
   internal "gaps" looked anomalous purely because there was nothing
   nearby to compare against -- a fundamentally different failure
   (the detector barely touched that whole box) from the one clean
   274px gap this feature targets.
2. It still missed the actual target headlines: the full-page OCR
   pass's own text-line detector, not just the layout model, failed
   to find them at all when reading the whole 4072x6368 page in one
   call -- large bold headline text can lose out to scale exactly
   the same way small print does, just for the opposite reason. If
   the text was never read in the first place, no amount of
   after-the-fact line recovery can produce it.

This version fixes both, directly:

- Gap candidates are found purely from EXISTING, already-detected
  blocks (never from OCR output), and must clear an ABSOLUTE size
  cap (MAX_GAP_HEIGHT) in addition to being large relative to the
  page's own median -- a 1900px span with almost nothing in it is
  rejected outright, where the real 274px headline gap is not.
- A candidate additionally needs a minimum LOCAL BLOCK DENSITY
  nearby (MIN_LOCAL_DENSITY within DENSITY_WINDOW) -- requiring
  "this sits inside an otherwise normally-populated newspaper
  column", which the sparse notice box fails and a real article
  column passes.
- Only for a gap that survives BOTH filters is a crop of just that
  region taken (padded, from the full-resolution source image) and
  read -- isolated from the rest of the page, so a large headline
  gets full attention instead of competing with everything else on
  the page at once.

READING A VERIFIED GAP -- detection bypassed entirely
-------------------------------------------------------

Confirmed directly on the real 274px target gap: RapidOCR's own
text-DETECTOR (not just the layout model) returns ZERO text regions
for this crop, even at full resolution, even enhanced -- the
recognizer is never even reached. This is not a recognition-quality
problem; detection itself fails on this bold/large headline style.

So detection is bypassed entirely for a verified gap. Since the
region is already known (from the layout gap, not from any OCR
detection), the crop is split into individual line-bands with a
plain row-wise ink-projection (find the blank rows between printed
lines), and each line-band is sent straight to the recognizer via
RapidOCR's own `use_det=False` mode. Confirmed on the same real gap:
this reads both headline lines correctly at 0.94+ confidence, where
detection-based reading returned nothing at all for one target
headline and a garbled, incomplete read for the other.

SAFETY
------

Never touches, merges, or reclassifies any EXISTING block -- only
ever adds new ones, and only into a gap that passes every check
above AND whose recognized line(s) clear a confidence floor.
"""

from __future__ import annotations

import difflib
import re
import statistics
from typing import Any, Callable

import cv2
import numpy as np

from pipeline.block_parser import LayoutBlock

# Fraction of the narrower block's width that must overlap for two
# boxes to count as sharing a column -- same definition used
# throughout the rest of the codebase (local_grouper.py,
# heading_repair.py) so "same column" means the same thing
# everywhere.
COLUMN_OVERLAP = 0.35

# A candidate gap must be at least this many times the page's own
# median same-column inter-block gap, AND at least GAP_FLOOR pixels,
# before it's even considered "anomalous" -- ordinary paragraph/
# story spacing must never qualify.
GAP_FACTOR = 3.0
GAP_FLOOR = 150.0

# Absolute ceiling on a recoverable gap. A real missed headline/
# paragraph is on the order of one or two lines -- a gap in the
# thousands of pixels is far more likely a whole different, sparsely
# -detected content region (an ad, a notice box) than one clean miss,
# and attempting to OCR-and-classify that much area at once is how
# the first version fabricated 29 blocks from one such box.
MAX_GAP_HEIGHT = 600.0

# The recovered crop's width, computed as the union of the two
# bracketing blocks' x-ranges, is capped at this multiple of whichever
# of the two is wider -- keeps a badge/label-vs-real-column pairing
# from clipping the real content (union fixes that) while still
# stopping an irregular pairing from ballooning across several
# unrelated columns at once.
MAX_WIDTH_RATIO = 2.5

# A gap only qualifies when at least this many OTHER existing blocks
# sit within DENSITY_WINDOW px above/below it (roughly overlapping
# column) -- i.e. it's sitting inside an otherwise normally-detected
# newspaper column, not a region the detector barely touched at all.
MIN_LOCAL_DENSITY = 3
DENSITY_WINDOW = 900.0

# Padding added around a candidate gap before cropping for the
# isolated OCR read, so a headline's own ascenders/descenders right
# at the gap's edge aren't clipped.
CROP_PADDING = 15

# A pixel darker than this (0-255 grayscale) counts as "ink" for the
# row-wise projection used to split a verified gap's crop into
# individual printed lines.
LINE_INK_DARKNESS_THRESHOLD = 180

# A row whose ink-pixel count is below this fraction of the crop's
# width counts as "blank" -- i.e. the gap between two printed lines.
LINE_BLANK_ROW_RATIO = 0.01

# A run of non-blank rows shorter than this is noise (a stray mark,
# an anti-aliasing artifact at the crop's own edge), not a real
# printed line, and is discarded.
MIN_LINE_HEIGHT = 20

# Padding added around each individual detected line before it's
# handed to the recognizer, so ascenders/descenders/matras right at
# the line's own edge aren't clipped.
LINE_CROP_PADDING = 5

# A recognized line must clear this confidence to be kept -- below
# it is more likely noise (a rule line, a stray mark) than real text.
MIN_RECOVERED_CONFIDENCE = 0.5


# ----------------------------------------------------------------
# SCRIPT-RELATIVE CONFIDENCE FLOORS
#
# Both recovery passes below gate on an ABSOLUTE OCR confidence.
# Those two numbers (0.5 here, 0.6 for the figure pass) were tuned
# on scripts whose engines report 0.8+ on a clean reading, which
# made them look like generous floors. They are not generous for
# every script: an OCR engine's confidence is a per-glyph certainty,
# and a cursive, ligature-dense script drags it down across the
# board even when the reading itself is correct.
#
# Confirmed on a real Urdu (THE INQUILAB, Nastaliq) document read
# with tessdata_best `urd`: the HIGHEST confidence Tesseract
# reported for ANY block on either page was 0.576, with a mean of
# 0.39 over 90 body-text blocks that all read back as real,
# well-formed Urdu. Both floors therefore sat ABOVE the engine's own
# ceiling for that script, so neither recovery pass could fire even
# once -- not because a check failed on the merits, but because the
# threshold was unreachable by construction. Three Nastaliq banner
# headlines detected as "figure" (766x77 aspect 9.9, 1037x121
# aspect 8.6, 765x82 aspect 9.3) were rejected on confidence alone,
# and the stories under them were then grouped with no headline at
# all.
#
# So the floor is now calibrated against the SAME page's own
# measured confidence distribution -- the median confidence of the
# blocks the primary OCR pass already read successfully, which are
# known-good text by definition. This is the same approach the rest
# of the pipeline already takes for page-varying quantities rather
# than hardcoding one absolute number (see local_grouper's
# `median_gap * ROOT_GAP_FACTOR` and gemini_service's
# `median_block_gap`).
#
# The absolute floor stays the CEILING of what can be asked, so no
# script that already clears it today has its bar lowered -- a
# Gujarati page whose reference is 0.85 still gets asked for the
# full 0.6 (0.85 * 0.75 = 0.64, capped back to 0.6) and behaves
# exactly as before. Only a script whose engine cannot physically
# reach the absolute floor gets a reachable one instead.
# ----------------------------------------------------------------

# Fraction of the page's own median OCR confidence a recovered
# reading must reach. Below 1.0 because what gets recovered here is
# almost always DISPLAY type (a banner headline), and display type
# reads lower than body type on the same page: it is read in
# isolation line by line, and large tightly-kerned headline
# lettering is harder for the recognizer than the body face it was
# calibrated on.
#
# 0.70 is that measured ratio, not a free knob. On the real Urdu
# document above, the blocks the layout detector DID label "title"
# read at a median of 0.315 (page 1) and 0.215 (page 2) against
# body-text medians of 0.385 and 0.403 -- display type running at
# roughly 0.55-0.80x body on the same page. The two banner
# headlines still being lost at 0.75 (0.295 and 0.290, against a
# floor of 0.302) sit inside exactly that band.
#
# Taking the median over title blocks alone would be the more
# direct comparison, but is not usable as the reference: a page
# where the detector found almost no titles is precisely the page
# this recovery exists for (page 2 above yielded two), so the
# denominator would be least reliable exactly when it matters most.
# The all-block median is stable, and this factor carries the
# display-vs-body difference instead.
PAGE_CONFIDENCE_REFERENCE_FACTOR = 0.70

# Absolute floor beneath which a reading is never trusted, whatever
# the page's own distribution says. This is what stops the
# page-relative floor from collapsing toward zero on a page the OCR
# engine failed at wholesale -- if the reference itself is near
# zero, "0.75x the reference" would accept pure noise. Set below the
# 0.39 mean measured on real Nastaliq (which must stay recoverable)
# and well above the confidence a rule line or a photo's texture
# reads back at.
CONFIDENCE_HARD_FLOOR = 0.25


def _effective_confidence_floor(absolute_floor, confidence_reference):
    """
    The confidence a recovered reading must clear on THIS page.

    `confidence_reference` is the median confidence of the blocks the
    primary OCR pass already read successfully on this page, or None
    when the caller has no such measurement -- in which case the
    absolute floor is used unchanged, so a caller that does not
    supply a reference behaves exactly as it did before.

    Never above `absolute_floor` (no script's bar is raised) and
    never below CONFIDENCE_HARD_FLOOR (no page's bar collapses).
    """

    if confidence_reference is None:
        return absolute_floor

    page_relative = (
        float(confidence_reference)
        * PAGE_CONFIDENCE_REFERENCE_FACTOR
    )

    return max(
        CONFIDENCE_HARD_FLOOR,
        min(absolute_floor, page_relative),
    )


def page_confidence_reference(confidences):
    """
    Median of the OCR confidences the primary pass reported for
    blocks it read real text off, or None when there are none.

    Lives here (rather than in each OCR engine) so every engine
    feeds the recovery passes a reference computed the same way.
    Callers pass only the confidences of blocks that actually
    produced text -- an empty/skipped block's 0.0 is not a
    measurement of anything and would drag the reference down.
    """

    values = [
        float(value)
        for value in confidences
        if value is not None and float(value) > 0.0
    ]

    if not values:
        return None

    values.sort()

    middle = len(values) // 2

    if len(values) % 2:
        return values[middle]

    return (values[middle - 1] + values[middle]) / 2.0

# Two candidate boxes count as "the same physical region" when their
# IoU clears this. Deliberately high -- two overlapping-but-distinct
# gap candidates (e.g. one pairing (block20, block119), another
# (block15, block82)) both bracketing the SAME headline produce
# near-identical recovered boxes in practice (confirmed on a real
# page: two candidates 3px apart on all four edges), while two
# genuinely different headlines sitting in the same column never
# come close to this.
DUPLICATE_BBOX_IOU_THRESHOLD = 0.5

# Two recognized texts count as "substantially identical" at or above
# this similarity ratio (difflib, on whitespace-normalized text) --
# high enough that two different headlines of similar length don't
# false-positive, low enough to still match the same headline read
# with a few characters' difference between two overlapping crops.
DUPLICATE_TEXT_SIMILARITY_THRESHOLD = 0.85


def _bbox_from_block(block) -> dict:
    return {
        "x1": float(block.x1), "y1": float(block.y1),
        "x2": float(block.x2), "y2": float(block.y2),
    }


def _bbox_iou(a: dict, b: dict) -> float:

    inter_x1 = max(a["x1"], b["x1"])
    inter_y1 = max(a["y1"], b["y1"])
    inter_x2 = min(a["x2"], b["x2"])
    inter_y2 = min(a["y2"], b["y2"])

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _normalize_recovered_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _texts_substantially_identical(a: str, b: str) -> bool:

    norm_a = _normalize_recovered_text(a)
    norm_b = _normalize_recovered_text(b)

    if not norm_a or not norm_b:
        return False

    if norm_a == norm_b:
        return True

    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    return ratio >= DUPLICATE_TEXT_SIMILARITY_THRESHOLD


def _overlap_ratio(ax1, ax2, bx1, bx2) -> float:

    width_a = max(1.0, float(ax2 - ax1))
    width_b = max(1.0, float(bx2 - bx1))

    overlap = min(ax2, bx2) - max(ax1, bx1)

    return max(0.0, overlap) / min(width_a, width_b)


def _page_median_column_gap(blocks) -> float:

    gaps = []

    for block in blocks:

        nearest_gap = None

        for other in blocks:

            if other is block:
                continue

            if other.y2 > block.y1:
                continue

            if (
                _overlap_ratio(block.x1, block.x2, other.x1, other.x2)
                < COLUMN_OVERLAP
            ):
                continue

            gap = float(block.y1 - other.y2)

            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap

        if nearest_gap is not None and nearest_gap > 0:
            gaps.append(nearest_gap)

    return statistics.median(gaps) if gaps else 0.0


def _find_gap_candidates(blocks):
    """
    (gap, above, below) for every pair of vertically-adjacent
    EXISTING blocks in the same column -- purely from already-
    detected blocks, never from OCR output.
    """

    pairs = []

    for block in blocks:

        nearest = None
        nearest_gap = None

        for other in blocks:

            if other is block:
                continue

            if other.y2 > block.y1:
                continue

            if (
                _overlap_ratio(block.x1, block.x2, other.x1, other.x2)
                < COLUMN_OVERLAP
            ):
                continue

            gap = float(block.y1 - other.y2)

            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                nearest = other

        if nearest is not None and nearest_gap is not None:
            pairs.append((nearest_gap, nearest, block))

    return pairs


def _local_block_density(bbox, blocks) -> int:

    count = 0

    for block in blocks:

        if (
            _overlap_ratio(bbox["x1"], bbox["x2"], block.x1, block.x2)
            < COLUMN_OVERLAP
        ):
            continue

        # Within DENSITY_WINDOW px above or below the gap.
        if block.y2 < bbox["y1"] - DENSITY_WINDOW:
            continue

        if block.y1 > bbox["y2"] + DENSITY_WINDOW:
            continue

        count += 1

    return count


def _bbox_overlaps_other_block(bbox, above, below, blocks) -> bool:
    """
    True if the candidate gap box `bbox` -- computed from `above`/
    `below`'s UNION x-range (see the comment at its call site for why
    union, not intersection) -- intersects any EXISTING block other
    than the two that bracket it.

    A genuine missed-content gap has nothing else already detected
    inside it, by definition. Confirmed on a real Urdu (THE INQUILAB,
    doc_000172 page 3) document where this was NOT checked: a gap
    candidate's union bbox (0,850)-(1055,1192) bridged clean across a
    column boundary and fully enclosed four already-detected blocks
    from a DIFFERENT column (a title and three body/plain-text
    blocks), none of which is `above` or `below` for this candidate.
    Nothing before this caught it -- `_local_block_density` only
    checks that blocks exist somewhere NEAR the box, not that the box
    itself is empty, and the final dedup check compares IoU against
    ONE block at a time, which stays low when a big box encloses
    several small ones (the union's area swamps any single overlap).
    Rejecting the whole candidate here is what the fix in
    recover_missed_blocks's docstring calls preventing a recovered
    box from "spanning across or enclosing" other columns' content --
    a genuine one-column gap is never affected, since by construction
    nothing already occupies it.
    """

    for block in blocks:

        if block is above or block is below:
            continue

        other = _bbox_from_block(block)

        inter_x1 = max(bbox["x1"], other["x1"])
        inter_y1 = max(bbox["y1"], other["y1"])
        inter_x2 = min(bbox["x2"], other["x2"])
        inter_y2 = min(bbox["y2"], other["y2"])

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            return True

    return False


def _split_into_line_bands(crop) -> list[tuple[int, int]]:
    """
    Row-wise ink projection: find the (y1, y2) pixel ranges of each
    printed line inside `crop`, using the blank rows between lines as
    separators. No text detection involved -- this only looks at
    which rows have ink at all, which is enough to isolate lines even
    when RapidOCR's own detector finds nothing in the crop.
    """

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ink = (gray < LINE_INK_DARKNESS_THRESHOLD).astype(np.uint8)

    row_ink = ink.sum(axis=1)

    is_blank = row_ink < (crop.shape[1] * LINE_BLANK_ROW_RATIO)

    bands = []

    start = None

    for row, blank in enumerate(is_blank):

        if not blank and start is None:
            start = row

        if blank and start is not None:

            if row - start >= MIN_LINE_HEIGHT:
                bands.append((start, row))

            start = None

    if start is not None and len(is_blank) - start >= MIN_LINE_HEIGHT:
        bands.append((start, len(is_blank)))

    return bands


def _recognize_lines(reader, crop, crop_x1, crop_y1):
    """
    Split `crop` into individual printed lines (see
    _split_into_line_bands) and recognize each one directly, bypassing
    detection entirely (RapidOCR's `use_det=False`) -- confirmed
    necessary on a real page: the detector itself returns zero text
    regions for this kind of large/bold headline crop, so recognition
    never even runs unless it's invoked directly like this. Returns
    lines in page-space coordinates, or [] if nothing usable came
    back.
    """

    height = crop.shape[0]

    lines = []

    for band_y1, band_y2 in _split_into_line_bands(crop):

        pad_y1 = max(0, band_y1 - LINE_CROP_PADDING)
        pad_y2 = min(height, band_y2 + LINE_CROP_PADDING)

        line_crop = crop[pad_y1:pad_y2, :]

        if line_crop.size == 0:
            continue

        try:
            result = reader(line_crop, use_det=False, use_cls=False, use_rec=True)
        except Exception:
            continue

        if result is None:
            continue

        txts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)

        if not txts:
            continue

        text = str(txts[0]).strip()

        if not text:
            continue

        try:
            confidence = float(scores[0]) if scores else 0.0
        except Exception:
            confidence = 0.0

        lines.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": {
                    "x1": float(crop_x1),
                    "y1": float(crop_y1 + pad_y1),
                    "x2": float(crop_x1 + crop.shape[1]),
                    "y2": float(crop_y1 + pad_y2),
                },
            }
        )

    return lines


def recovered_line_height(lines) -> float:
    """
    Median per-line ink height of a recovered region.

    The line boxes handed back by _recognize_lines carry
    LINE_CROP_PADDING on each side (added so ascenders/descenders
    are not clipped before recognition), so that padding is removed
    here -- this must be comparable with a per-line height measured
    off an ordinary OCR'd block, which has no such padding.
    """

    heights = [
        float(line["bbox"]["y2"] - line["bbox"]["y1"])
        - 2 * LINE_CROP_PADDING
        for line in lines
    ]

    heights = [height for height in heights if height > 0]

    if not heights:
        return 0.0

    return statistics.median(heights)


def page_line_heights(samples) -> dict:
    """
    The page's own median per-line height for display type vs body
    type, measured from the line boxes the primary OCR pass already
    reported.

    `samples` is an iterable of (block_class, lines) pairs, where
    `lines` is that block's OCR line list. Returns
    {"title": float|None, "body": float|None}.

    Lives here (rather than in each OCR engine) so every engine
    measures the reference the same way -- see
    _classify_recovered_block for what it is compared against and
    why a per-BLOCK height cannot stand in for it.
    """

    per_class = {"title": [], "body": []}

    for block_class, lines in samples:

        if not lines:
            continue

        if block_class == "title":
            key = "title"
        elif block_class == "plain text":
            key = "body"
        else:
            continue

        heights = [
            float(line["bbox"]["y2"] - line["bbox"]["y1"])
            for line in lines
        ]

        heights = [height for height in heights if height > 0]

        if heights:
            per_class[key].append(statistics.median(heights))

    return {
        key: (statistics.median(values) if values else None)
        for key, values in per_class.items()
    }


def _classify_recovered_block(
    line_height: float,
    blocks,
    line_heights=None,
) -> str:
    """
    Title vs body for a recovered region, from how tall its type is.

    WHY THIS NEEDS `line_heights`
    -----------------------------

    `line_height` is a PER-LINE measurement. The fallback path below
    compares it against the page's existing title and "plain text"
    BLOCK heights, which is a unit mismatch: a title block is one or
    two lines tall, so its block height is close to its line height,
    but a body block is a whole multi-line paragraph and its block
    height is many times its line height. A big display headline
    therefore lands nearer the body-BLOCK median than the
    title-BLOCK median purely by arithmetic, and the taller the
    headline is, the more confidently it is called body text.

    Confirmed on a real Urdu (THE INQUILAB) page: banner headlines
    recovered at a per-line height of 127-157px were classified
    "plain text" against a title-block median of 49 and a body-block
    median of 167 -- so they were recovered from "figure" only to
    land as body text, still leaving their stories with no title to
    start at. Measured per-LINE on the same pages, the three
    populations separate cleanly and in the right order: body 35,
    title 48, recovered banners 127-157.

    So when the caller supplies the page's own per-line reference,
    the test is like-for-like AND monotonic -- taller type is more
    headline-like, never less. The fallback is kept unchanged for
    callers that have no such measurement.
    """

    if line_heights:

        title_line_height = line_heights.get("title")
        body_line_height = line_heights.get("body")

        if title_line_height and body_line_height:

            # Midway between the page's body and display line
            # heights. Anything at or above display height is a
            # headline however far above it sits, which is what the
            # absolute-distance test below gets wrong.
            boundary = (
                float(title_line_height)
                + float(body_line_height)
            ) / 2.0

            if line_height >= boundary:
                return "title"

            return "plain text"

    # ------------------------------------------------------------
    # Fallback: no per-line reference available. Compares against
    # per-BLOCK heights (see the unit mismatch above) and so is only
    # reliable when the recovered region is itself about one line
    # tall. Defaults to "plain text" when the page has no title
    # blocks to compare against, since under-classifying a real
    # headline as body text is a smaller, safer error than the
    # reverse.
    # ------------------------------------------------------------

    title_heights = [
        float(block.y2 - block.y1)
        for block in blocks
        if block.cls == "title"
    ]

    body_heights = [
        float(block.y2 - block.y1)
        for block in blocks
        if block.cls == "plain text"
    ]

    if not title_heights or not body_heights:
        return "plain text"

    title_median = statistics.median(title_heights)
    body_median = statistics.median(body_heights)

    if abs(line_height - title_median) < abs(
        line_height - body_median
    ):
        return "title"

    return "plain text"


def recover_missed_blocks(
    blocks: list,
    page_image,
    reader: Callable[[Any], Any],
    next_id: int,
    confidence_reference=None,
    line_heights=None,
) -> list[tuple]:
    """
    Returns a list of (LayoutBlock, text, confidence, lines) tuples
    for every genuinely-recovered block. Never mutates `blocks`; the
    caller decides how to fold the result in (see rapidocr_engine.py).

    `reader` is the same RapidOCR callable used for the primary
    full-page pass (self.reader) -- called here on a small, isolated
    crop of just one candidate gap at a time, never on the whole
    page.

    `confidence_reference` is this page's own median OCR confidence
    (see page_confidence_reference) and makes the confidence gate
    reachable on scripts whose engine never reports high absolute
    confidence -- see SCRIPT-RELATIVE CONFIDENCE FLOORS above. None
    keeps the previous absolute-only behaviour.
    """

    if page_image is None or not blocks:
        return []

    height, width = page_image.shape[:2]

    min_confidence = _effective_confidence_floor(
        MIN_RECOVERED_CONFIDENCE,
        confidence_reference,
    )

    median_gap = _page_median_column_gap(blocks)

    gap_threshold = max(GAP_FLOOR, median_gap * GAP_FACTOR)

    recovered = []

    for gap, above, below in _find_gap_candidates(blocks):

        if gap < gap_threshold:
            continue

        if gap > MAX_GAP_HEIGHT:
            continue

        # UNION of the two neighbours' x-ranges, not their
        # intersection. Confirmed on a real page why this matters: a
        # small badge/tag block (e.g. a "1"/"2" item-number label,
        # ~190px wide) sitting right next to the real, much wider
        # headline column (~400px) was picked as one of the two
        # neighbours -- taking the INTERSECTION clipped the crop down
        # to the badge's own narrow width, cutting off most of the
        # actual headline before OCR ever saw it, and reading
        # whatever sliver remained as garbage. The union is capped
        # (MAX_WIDTH_RATIO) against the WIDER of the two neighbours'
        # own widths so an irregular pairing still can't balloon into
        # spanning several unrelated columns at once.
        x1 = min(above.x1, below.x1)
        x2 = max(above.x2, below.x2)

        above_width = max(1.0, float(above.x2 - above.x1))
        below_width = max(1.0, float(below.x2 - below.x1))
        wider_neighbor_width = max(above_width, below_width)

        max_crop_width = wider_neighbor_width * MAX_WIDTH_RATIO

        if (x2 - x1) > max_crop_width:

            center = (x1 + x2) / 2.0

            x1 = center - max_crop_width / 2.0
            x2 = center + max_crop_width / 2.0

        bbox = {
            "x1": float(x1), "y1": float(above.y2),
            "x2": float(x2), "y2": float(below.y1),
        }

        # Reject a candidate whose box already has other detected
        # content inside it -- see _bbox_overlaps_other_block. Checked
        # before the density check below since an enclosing candidate
        # is disqualified outright regardless of how dense its
        # surroundings are.
        if _bbox_overlaps_other_block(bbox, above, below, blocks):
            continue

        if _local_block_density(bbox, blocks) < MIN_LOCAL_DENSITY:
            continue

        crop_x1 = max(0, int(bbox["x1"]) - CROP_PADDING)
        crop_y1 = max(0, int(bbox["y1"]) - CROP_PADDING)
        crop_x2 = min(width, int(bbox["x2"]) + CROP_PADDING)
        crop_y2 = min(height, int(bbox["y2"]) + CROP_PADDING)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        crop = page_image[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            continue

        # Detection is bypassed entirely -- see _recognize_lines.
        # The region is already known from the layout gap itself, so
        # the crop is split into individual printed lines by a plain
        # ink projection and each is recognized directly.
        lines = _recognize_lines(reader, crop, crop_x1, crop_y1)

        if not lines:
            continue

        text_parts = [line["text"] for line in lines]
        confidences = [line["confidence"] for line in lines]

        avg_confidence = sum(confidences) / len(confidences)

        if avg_confidence < min_confidence:
            continue

        recovered_x1 = min(line["bbox"]["x1"] for line in lines)
        recovered_y1 = min(line["bbox"]["y1"] for line in lines)
        recovered_x2 = max(line["bbox"]["x2"] for line in lines)
        recovered_y2 = max(line["bbox"]["y2"] for line in lines)

        new_bbox = {
            "x1": recovered_x1, "y1": recovered_y1,
            "x2": recovered_x2, "y2": recovered_y2,
        }

        text = " ".join(text_parts)

        # Never duplicate an EXISTING (already layout-detected) block.
        # A genuine gap sits strictly between two existing blocks, so
        # this only fires if a gap candidate was computed from a
        # malformed/overlapping pair -- a safety net, not the primary
        # dedup path.
        if any(
            _bbox_iou(new_bbox, _bbox_from_block(block)) >= DUPLICATE_BBOX_IOU_THRESHOLD
            for block in blocks
        ):
            continue

        # Two overlapping gap candidates (e.g. (block20, block119) and
        # (block15, block82)) can both bracket the SAME missed
        # headline and each independently recover it. Confirmed on a
        # real page: this produced two near-identical blocks, 2-3px
        # apart on every edge, both reading the same headline. When a
        # previously recovered block in THIS SAME call is both a
        # near-exact bbox match and a near-exact text match, keep only
        # the higher-confidence one instead of inserting a duplicate.
        duplicate_index = None

        for i, (existing_block, existing_text, existing_confidence, _existing_lines) in enumerate(recovered):

            if _bbox_iou(new_bbox, _bbox_from_block(existing_block)) < DUPLICATE_BBOX_IOU_THRESHOLD:
                continue

            if not _texts_substantially_identical(text, existing_text):
                continue

            duplicate_index = i
            break

        if duplicate_index is not None:

            if avg_confidence > recovered[duplicate_index][2]:

                kept_id = recovered[duplicate_index][0].id

                block_class = _classify_recovered_block(
                    recovered_line_height(lines),
                    blocks,
                    line_heights=line_heights,
                )

                replacement_block = LayoutBlock(
                    id=kept_id,
                    cls=block_class,
                    x1=int(recovered_x1),
                    y1=int(recovered_y1),
                    x2=int(recovered_x2),
                    y2=int(recovered_y2),
                    confidence=1.0,
                )

                recovered[duplicate_index] = (replacement_block, text, avg_confidence, lines)

            continue

        block_class = _classify_recovered_block(
            recovered_line_height(lines),
            blocks,
            line_heights=line_heights,
        )

        new_block = LayoutBlock(
            id=next_id,
            cls=block_class,
            x1=int(recovered_x1),
            y1=int(recovered_y1),
            x2=int(recovered_x2),
            y2=int(recovered_y2),
            confidence=1.0,
        )

        recovered.append((new_block, text, avg_confidence, lines))

        next_id += 1

    return recovered


# ============================================================
# MISCLASSIFIED FIGURE RECOVERY -- layout detector labeled real
# text as an image
# ============================================================
#
# A DIFFERENT failure from the gap recovery above: here the layout
# detector DID draw a box, but classified it "figure" (image) when
# the region is actually text -- confirmed on a real page: a kicker
# line plus a large bold headline ("H1N1થી બાળકનું હાર્ટ પમ્પિંગ 10
# ટકા થયું...") sitting entirely inside one "figure"-class box, no
# photograph anywhere in it. Because OCR_CLASSES (rapidocr_engine.py)
# excludes "figure", this block is never sent through recognition at
# all -- the full-page OCR pass's own line-to-block mapping only
# considers OCR-eligible blocks as candidates, so even a correctly
# -detected line overlapping a "figure" block's region is discarded
# rather than attached to it. The story's real headline then simply
# never exists as a block, and grouping falls back to whatever
# nearby title-class fragment happens to exist instead (in the
# confirmed case, a small explainer caption for an unrelated
# diagram).
#
# NOT scoped to any one language (unlike the gap recovery above,
# which is Hindi/Marathi-only for OCR-detection-quality reasons): a
# layout model calling bold/stylized headline text an "image" is a
# visual/geometric misclassification, not a script-specific OCR
# problem, so this runs for every document.
#
# SAFETY: must never turn a genuine photograph into a bogus text
# block. Two independent checks before ANY reclassification:
# 1. The pixel-only line-band projection (_split_into_line_bands,
#    reused as-is from the gap recovery above) must find a plausible
#    MULTI-LINE pattern first -- a real photo's continuous tonal
#    gradients essentially never decompose into the same blank-row
#    -separated horizontal bands printed text does. Checked BEFORE
#    any OCR call, so a block that doesn't look like text costs
#    nothing extra.
# 2. The recognized text must clear both a confidence floor AND a
#    minimum character count -- a couple of garbled characters
#    misread from a photo's texture must never be enough on their
#    own to relabel it.
# ------------------------------------------------------------

# A real headline/caption is always more than one printed line in
# practice; requiring at least this many detected line-bands is what
# keeps a photo with one incidental horizontal edge (a horizon line,
# a shelf) from ever reaching the OCR call at all.
MIN_FIGURE_TEXT_LINES = 2

# ...unless the block is far too wide and flat to be anything BUT a
# single line of type.
#
# "A real headline is always more than one printed line" turned out
# not to hold: confirmed on a real Bengali page, the banner headline
# "ছুটির খবর দিতে ২৮ কিলোমিটার পাড়ি শিক্ষকের" runs the full width of
# its story as ONE line -- 1959x136px, aspect 14.4 -- was detected as
# a figure, and was rejected here before the OCR call because it
# projects to exactly one ink band. Its story was then cropped with no
# headline at all.
#
# What the two-line rule is really protecting against is a photo with
# one incidental horizontal edge (a horizon, a shelf). Such a photo is
# a normally-proportioned rectangle; a one-line banner headline is a
# thin ribbon. Requiring that ribbon shape lets the real case through
# while a squarish figure still needs the full two bands, and the
# confidence, character-count, size and top-of-page checks all still
# apply either way.
SINGLE_LINE_MIN_ASPECT = 8.0

# Below this many recognized (non-space) characters, "reads as text"
# is not a confident enough claim to override a layout-model image
# classification.
MIN_FIGURE_TEXT_CHARS = 8

# A FIFTH safety check -- added after a confirmed false positive on a
# real Malayalam (ജന്മഭൂമി) page: a lottery/classified results grid
# detected as a figure read back as
# "1 പു 0182 0191 01990362 0516 ഷ്യം തയാ? 211 0792 1118 1325 13"
# -- high enough confidence and long enough to clear every check
# above, and duly relabeled a "title".
#
# What separates it from a real headline is not confidence or length
# but CONTENT: a headline is language, a results grid is numbers. A
# genuine headline does carry the odd figure ("7.5 કરોડથી વધુ લોકોએ
# રિટર્ન ભર્યું" is ~7% digits), so this only rejects text that is
# mostly digits -- which no headline in any of these scripts is,
# since Indic digits are not ASCII either.
MAX_FIGURE_TEXT_DIGIT_RATIO = 0.5

# Deliberately higher than MIN_RECOVERED_CONFIDENCE above: reclassifying
# an EXISTING block's type is a stronger claim than filling a gap that
# had no block at all, so this asks for more certainty before doing it.
MIN_FIGURE_TEXT_CONFIDENCE = 0.6

# A THIRD safety check, on top of the two in the module docstring above
# -- added after a confirmed false positive on a real page: the small
# CMYK printer color-registration mark that appears near the bottom of
# virtually every newspaper page (four tiny "C"/"M"/"Y"/"K" swatches,
# printed in two stacked rows) cleared both the line-band and OCR-
# confidence checks and was wrongly relabeled from "figure" to a text
# block containing nonsense ("cYanmaallowbl..."). That mark measured
# 65x186px; the confirmed genuine recovery case (a real bold headline)
# measured 235x795px -- a real headline/caption is always a
# substantially sized block, never a small decorative mark, so a
# minimum size floor (checked before the cheap line-band pre-filter,
# so a tiny mark never even reaches it) closes this off without
# touching the confidence/line-count checks that the real case relies
# on.
#
# The HEIGHT floor is skipped for a block already shaped like one
# line of type (see SINGLE_LINE_MIN_ASPECT), because being short is
# precisely what makes it that shape -- flooring its height asks a
# banner headline not to be a banner headline.
#
# Confirmed on a real Punjabi (ਪੰਜਾਬੀ ਜਾਗਰਣ) page: the headline
# "'ਅਣਖ' ਖ਼ਾਤਰ ਨੌਜਵਾਨ ਦਾ ਕਤਲ" was detected as a figure at 607x58px,
# aspect 10.5, and reads back at 0.85 confidence -- it cleared every
# other check here and was rejected on height alone, leaving the
# story beneath it with no headline.
#
# These are ABSOLUTE pixel floors while page renders vary about 3x
# across real documents (1607x2577 to 4923x7314), so the same
# headline passes or fails depending only on how large its PDF
# rasterised: 100px is 1.4% of a Malayalam page's height but 3.7% of
# a Punjabi one's. Making both floors relative to page size would fix
# that class properly; the aspect exemption below fixes the shape
# that actually breaks, without changing what any other document
# already does.
#
# Dropping the height floor for ribbons cannot readmit the CMYK mark:
# at 186px wide it fails MIN_FIGURE_BLOCK_WIDTH, and at aspect 2.9 it
# is not a ribbon in the first place.
MIN_FIGURE_BLOCK_HEIGHT = 100.0
MIN_FIGURE_BLOCK_WIDTH = 250.0

# A FOURTH safety check -- added after a second confirmed false
# positive on a real page: a newspaper section masthead/banner (a
# large graphic combining a section logo, an icon, and small text --
# a section name, a date, an email address) sat at the very top of
# the page, was large enough to pass the size floor above, genuinely
# does contain real text, and got relabeled "title" -- semantically
# wrong, since masthead/section-header content must never become an
# article headline (see ArticleGrouper.IGNORE_ROLES). This is not
# something the line-band/confidence/size checks above can catch --
# it IS real, substantial text, just not article text.
#
# PageCleaner (pipeline/preprocess/page_cleaner.py) already treats
# "near the top of the page" as the deciding signal for masthead vs.
# page-header detection (top_limit = page_height * 0.12-0.15), but
# only for blocks matching a hardcoded list of known English/Hindi
# newspaper names and keywords -- it has no non-text, position-only
# fallback, and no Tamil/other-script coverage at all, so a Tamil
# masthead was never going to be caught downstream either way. This
# reuses the SAME top-of-page convention directly here instead,
# independent of what text is actually recognized: a genuine missed
# ARTICLE headline is never printed inside the masthead/header strip
# in the first place (confirmed real case sat at 83% down its page;
# this false positive sat at 10% down its own) -- so a "figure" block
# up there is far more likely to be a banner than a story.
#
# PAGE 1 ONLY -- see the `page_number` argument below. A masthead
# exists on page 1 and nowhere else, which PageCleaner already
# encodes directly (pipeline/preprocess/page_cleaner.py: masthead
# detection runs `if page_number is None or page_number == 1`, and
# prints "Mastheads Detected : 0 (not page 1)" otherwise). Applied
# to every page, this guard costs real headlines: confirmed on page
# 2 of a real Urdu (THE INQUILAB) document, the two lead banner
# headlines of the page -- 1031x164 at y=103 and 766x77 at y=114,
# both detected as "figure" -- were rejected on position alone,
# before any OCR ran, because they sit where page 1 would have its
# masthead. Nothing was up there to protect: page 2 has no masthead.
TOP_OF_PAGE_FRACTION = 0.15

# TOP_OF_PAGE_FRACTION is a crude upper CAP, not the real masthead
# edge -- confirmed on a different real Urdu (THE INQUILAB,
# doc_000194 page 1) document: the actual masthead + edition/price
# banner underneath it (a near-full-page-width block at y=18-295,
# followed by a narrower publication-info line ending at y=324) is
# real, but the fixed 15% fraction of this page's 3004px height
# (450px) reaches 126px past where it actually ends -- rejecting a
# genuine lead headline ("دہلی عمارت حادثہ...", a "figure"-
# misclassified banner at y=345-428, sitting in the exact same row
# as an adjacent block the layout detector DID correctly label
# "title") purely because 428 < 450.
#
# A real masthead/banner is reliably near-full-page-width (this is
# also PageCleaner's own assumption for its keyword-based masthead
# detector); an article headline below it is virtually never that
# wide, even a two-or-three-column one. So: look for a block within
# the fixed-fraction cap that spans at least this fraction of the
# page's width, and if one exists, use ITS bottom edge (plus a small
# buffer for a thin info-strip directly beneath it) as the real
# limit instead -- strictly SHRINKING the guard's rejection zone,
# never growing it. No wide block found (or none available -- older
# callers that never pass `blocks` still work) falls back to the
# original fixed fraction exactly as before.
MASTHEAD_WIDTH_RATIO = 0.85
MASTHEAD_LIMIT_BUFFER = 50.0


def recover_misclassified_text_blocks(
    blocks: list,
    page_image,
    reader: Callable[[Any], Any],
    confidence_reference=None,
    page_number=None,
    line_heights=None,
    min_multiline_bands: int = MIN_FIGURE_TEXT_LINES,
) -> list[tuple]:
    """
    Check every "figure"-class block for real text the layout
    detector missed by mislabeling it an image instead of skipping
    it entirely, rather than a genuine photograph.

    Returns a list of (block, text, confidence, lines) tuples for
    blocks that were RECLASSIFIED. `block` is the SAME LayoutBlock
    object passed in `blocks`, mutated in place (its `cls` updated)
    -- never removed, replaced, or duplicated. A genuine photograph
    elsewhere on the page is completely unaffected: this only ever
    touches a "figure" block that passes every safety check above.

    `confidence_reference` is this page's own median OCR confidence
    (see page_confidence_reference). Without it, this pass cannot
    fire at all on a script whose engine never reaches
    MIN_FIGURE_TEXT_CONFIDENCE -- see SCRIPT-RELATIVE CONFIDENCE
    FLOORS above for the confirmed Nastaliq case. None keeps the
    previous absolute-only behaviour.

    `page_number` gates the masthead/banner position guard to page 1,
    where a masthead actually is -- see TOP_OF_PAGE_FRACTION and
    MASTHEAD_WIDTH_RATIO. None (a caller that does not know its page
    number) applies the guard, exactly as before.

    `min_multiline_bands` overrides MIN_FIGURE_TEXT_LINES for a
    below-ribbon-aspect block -- see its call site in
    _split_into_line_bands's usage below. Confirmed necessary for
    Urdu: real headlines at aspect 4.0-6.3 (below SINGLE_LINE_MIN_
    ASPECT=8.0) read back perfectly (0.92-0.95 confidence, correct
    text) once actually OCR'd, but Nastaliq's connected diacritics
    and ligatures routinely fill in the blank row a Latin/Devanagari
    line-break would leave, so the OCR-free ink-projection band
    splitter (_split_into_line_bands) undercounts them to a single
    band and the default 2-band floor rejects them before OCR ever
    runs. UTRNetOCREngine passes 1 here for exactly this reason; the
    confidence/char-count/digit-ratio checks below are what actually
    guards against a genuine photograph either way, same reasoning
    SINGLE_LINE_MIN_ASPECT's own docstring already gives.
    """

    if page_image is None:
        return []

    height, width = page_image.shape[:2]

    # See TOP_OF_PAGE_FRACTION: a masthead/section banner is a page-1
    # phenomenon, so on later pages a "figure" near the top edge is
    # just a story's headline and the guard is skipped.
    apply_top_of_page_guard = (
        page_number is None or int(page_number) == 1
    )

    fixed_top_of_page_limit = height * TOP_OF_PAGE_FRACTION

    top_of_page_limit = 0.0

    if apply_top_of_page_guard:

        top_of_page_limit = fixed_top_of_page_limit

        # See MASTHEAD_WIDTH_RATIO: shrink the guard to the real
        # masthead's own measured extent when one is identifiable,
        # rather than trusting the fixed fraction blindly.
        masthead_bottoms = [
            float(candidate.y2)
            for candidate in blocks
            if candidate.y1 < fixed_top_of_page_limit
            and (candidate.x2 - candidate.x1) >= MASTHEAD_WIDTH_RATIO * width
        ]

        if masthead_bottoms:

            top_of_page_limit = min(
                fixed_top_of_page_limit,
                max(masthead_bottoms) + MASTHEAD_LIMIT_BUFFER,
            )

    min_confidence = _effective_confidence_floor(
        MIN_FIGURE_TEXT_CONFIDENCE,
        confidence_reference,
    )

    recovered = []

    for block in blocks:

        if block.cls != "figure":
            continue

        block_aspect = (
            float(block.x2 - block.x1)
            / max(1.0, float(block.y2 - block.y1))
        )

        # A ribbon is exempt from the HEIGHT floor -- being short is
        # what makes it a ribbon. See SINGLE_LINE_MIN_ASPECT; the
        # width floor and the aspect test below both still apply, and
        # each of them independently excludes the CMYK mark this
        # height floor was added for (186px wide, aspect 2.9).
        if (
            block_aspect < SINGLE_LINE_MIN_ASPECT
            and float(block.y2 - block.y1) < MIN_FIGURE_BLOCK_HEIGHT
        ):
            continue

        if float(block.x2 - block.x1) < MIN_FIGURE_BLOCK_WIDTH:
            continue

        if float(block.y2) <= top_of_page_limit:
            continue

        x1 = max(0, int(block.x1) - CROP_PADDING)
        y1 = max(0, int(block.y1) - CROP_PADDING)
        x2 = min(width, int(block.x2) + CROP_PADDING)
        y2 = min(height, int(block.y2) + CROP_PADDING)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = page_image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # A ribbon-shaped block can only hold one line of type, so
        # one band is all there is to find. See
        # SINGLE_LINE_MIN_ASPECT.
        min_lines = (
            1
            if block_aspect >= SINGLE_LINE_MIN_ASPECT
            else min_multiline_bands
        )

        # Cheap, OCR-free pre-filter -- see SAFETY check 1 above.
        bands = _split_into_line_bands(crop)

        if len(bands) < min_lines:
            continue

        lines = _recognize_lines(reader, crop, x1, y1)

        if len(lines) < min_lines:
            continue

        text = " ".join(line["text"] for line in lines)

        compact = text.replace(" ", "")

        if len(compact) < MIN_FIGURE_TEXT_CHARS:
            continue

        # Mostly-numeric content is a table, not a headline. See
        # MAX_FIGURE_TEXT_DIGIT_RATIO.
        digit_count = sum(1 for char in compact if char.isascii() and char.isdigit())

        if digit_count / len(compact) > MAX_FIGURE_TEXT_DIGIT_RATIO:
            continue

        avg_confidence = sum(
            line["confidence"] for line in lines
        ) / len(lines)

        if avg_confidence < min_confidence:
            continue

        recovered_y1 = min(line["bbox"]["y1"] for line in lines)
        recovered_y2 = max(line["bbox"]["y2"] for line in lines)

        block.cls = _classify_recovered_block(
            recovered_line_height(lines),
            blocks,
            line_heights=line_heights,
        )

        recovered.append((block, text, avg_confidence, lines))

    return recovered
