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


def _classify_recovered_block(
    cluster_height: float,
    line_count: int,
    blocks,
) -> str:
    """
    Title vs body, by comparing this region's own per-line height
    against the page's EXISTING title/body per-line heights.
    Defaults to "plain text" when the page has no title blocks to
    compare against, since under-classifying a real headline as body
    text is a smaller, safer error than the reverse.
    """

    recovered_line_height = cluster_height / max(1, line_count)

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

    if abs(recovered_line_height - title_median) < abs(
        recovered_line_height - body_median
    ):
        return "title"

    return "plain text"


def recover_missed_blocks(
    blocks: list,
    page_image,
    reader: Callable[[Any], Any],
    next_id: int,
) -> list[tuple]:
    """
    Returns a list of (LayoutBlock, text, confidence, lines) tuples
    for every genuinely-recovered block. Never mutates `blocks`; the
    caller decides how to fold the result in (see rapidocr_engine.py).

    `reader` is the same RapidOCR callable used for the primary
    full-page pass (self.reader) -- called here on a small, isolated
    crop of just one candidate gap at a time, never on the whole
    page.
    """

    if page_image is None or not blocks:
        return []

    height, width = page_image.shape[:2]

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

        if avg_confidence < MIN_RECOVERED_CONFIDENCE:
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
                    recovered_y2 - recovered_y1, len(lines), blocks,
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
            recovered_y2 - recovered_y1, len(lines), blocks,
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
