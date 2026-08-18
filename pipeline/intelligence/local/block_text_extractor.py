"""
Reconstruct an article's text from the OCR already computed at the
page level (page_json/page_NNN.json), instead of re-running RapidOCR a
second time on the article's cropped image.

The pipeline already runs one full-page RapidOCR pass per page to
build page_json (see pipeline/ocr/rapidocr_engine.py + page_processor_gemini.py).
LocalArticleExtractor used to re-OCR every article crop on top of that,
duplicating work. Since crop.json now records which page-level block
ids belong to each article (see FinalArticleCropper), we can just look
those blocks up and reuse their already-computed `text` /
`ocr_confidence`.

Returns the same shape as RapidOCREngine.process_article_crop() --
{text, confidence, lines, line_count} -- so it's a drop-in replacement
wherever that was used; each line also carries the source block's
`class` (e.g. "title") so downstream heading detection can use the
layout model's own classification instead of re-guessing from text.

Two corrections are applied on top of the raw page_json blocks, both
invisible when Gemini's vision model read crop images directly (it
inferred correct reading order and ignored duplicate detections itself)
but very visible once block text is concatenated mechanically:

1. Overlapping/duplicate block detection: DocLayout-YOLO sometimes
   emits more than one bounding box for the same text region (e.g. one
   coarse box spanning two lines plus two precise per-line boxes for
   the same two lines) -- naively concatenating every block's text
   reproduces that text 2-3 times. See `_deduplicate_overlapping_blocks`.

2. Column-unaware ordering: `pipeline/sort_blocks.py` (the source of
   each block's `reading_order`) sorts purely by (y1, x1) -- top-to-
   bottom then left-to-right across the ENTIRE page width, with no
   concept of newspaper columns. For a multi-column article this
   interleaves paragraphs from different columns instead of reading
   one column fully before the next. See `_order_body_blocks_by_column`.
"""

from __future__ import annotations

from typing import Any

# Matches OCR_CLASSES in pipeline/ocr/rapidocr_engine.py -- the block
# classes that actually carry OCR'd text.
TEXT_BLOCK_CLASSES = {"plain text", "title", "figure_caption"}

# A candidate block is dropped as a duplicate/superset of already-kept
# blocks once this fraction of its own area is covered by them.
_DUPLICATE_COVERAGE_THRESHOLD = 0.6

# Two blocks are considered to be in the same column if the horizontal
# overlap between them covers at least this fraction of the narrower
# block's width.
_COLUMN_OVERLAP_RATIO = 0.3


def _bbox_area(bbox: dict[str, float]) -> float:
    width = max(0.0, bbox.get("x2", 0.0) - bbox.get("x1", 0.0))
    height = max(0.0, bbox.get("y2", 0.0) - bbox.get("y1", 0.0))
    return width * height


def _intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1 = max(a.get("x1", 0.0), b.get("x1", 0.0))
    y1 = max(a.get("y1", 0.0), b.get("y1", 0.0))
    x2 = min(a.get("x2", 0.0), b.get("x2", 0.0))
    y2 = min(a.get("y2", 0.0), b.get("y2", 0.0))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _deduplicate_overlapping_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Drop blocks whose bounding box is already substantially covered by
    other (smaller, more specific) blocks already kept -- the layout
    detector's own duplicate/overlapping-box artifacts, not genuine
    additional text.

    Processes smallest-area-first so precise, narrow boxes are kept
    and the coarse box that duplicates their combined area is dropped.
    """
    ordered = sorted(blocks, key=lambda b: _bbox_area(b.get("bbox", {})))

    kept: list[dict[str, Any]] = []

    for block in ordered:

        bbox = block.get("bbox", {})
        area = _bbox_area(bbox)

        if area <= 0:
            kept.append(block)
            continue

        covered = sum(
            _intersection_area(kept_block.get("bbox", {}), bbox)
            for kept_block in kept
        )

        if covered / area < _DUPLICATE_COVERAGE_THRESHOLD:
            kept.append(block)

    return kept


def _deduplicate_by_text_containment(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Catch duplicate-detection boxes the geometric coverage check
    misses: two boxes with the same top-left corner but different
    heights, where the shorter box's text is a truncated prefix of the
    taller box's (e.g. one detection stopped mid-paragraph while
    another correctly captured the whole paragraph). Unlike the
    coverage check, this works even when the smaller box's area is a
    minority of the larger one's.
    """
    normalized = [
        " ".join(str(block.get("text", "") or "").split())
        for block in blocks
    ]

    drop_ids = set()

    for i, text_a in enumerate(normalized):

        if not text_a or id(blocks[i]) in drop_ids:
            continue

        probe = text_a[:30]

        if not probe:
            continue

        for j, text_b in enumerate(normalized):

            if i == j or len(text_a) >= len(text_b):
                continue

            if text_b.startswith(probe):
                drop_ids.add(id(blocks[i]))
                break

    return [block for block in blocks if id(block) not in drop_ids]


def _order_body_blocks_by_column(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Group blocks into left-to-right column bands by horizontal overlap,
    then read each column fully top-to-bottom before moving to the
    next -- the way a person actually reads a multi-column newspaper
    article, instead of a single top-to-bottom-then-left-to-right pass
    across the whole width.
    """
    ordered_by_x = sorted(blocks, key=lambda b: b.get("bbox", {}).get("x1", 0.0))

    columns: list[dict[str, Any]] = []

    for block in ordered_by_x:

        bbox = block.get("bbox", {})
        x1, x2 = bbox.get("x1", 0.0), bbox.get("x2", 0.0)
        width = max(1.0, x2 - x1)

        placed = False

        for column in columns:

            overlap = min(x2, column["x2"]) - max(x1, column["x1"])
            column_width = max(1.0, column["x2"] - column["x1"])
            narrower_width = min(width, column_width)

            if overlap > 0 and overlap / narrower_width >= _COLUMN_OVERLAP_RATIO:
                column["x1"] = min(column["x1"], x1)
                column["x2"] = max(column["x2"], x2)
                column["blocks"].append(block)
                placed = True
                break

        if not placed:
            columns.append({"x1": x1, "x2": x2, "blocks": [block]})

    columns.sort(key=lambda c: c["x1"])

    result = []

    for column in columns:
        result.extend(
            sorted(column["blocks"], key=lambda b: b.get("bbox", {}).get("y1", 0.0))
        )

    return result


def extract_article_text_from_blocks(
    blocks: list[dict[str, Any]],
    block_ids: list[int],
) -> dict[str, Any]:

    if not blocks or not block_ids:
        return {"text": "", "confidence": 0.0, "lines": [], "line_count": 0}

    id_set = set(block_ids)

    selected = [
        block for block in blocks
        if block.get("id") in id_set
        and block.get("class") in TEXT_BLOCK_CLASSES
        and str(block.get("text", "") or "").strip()
    ]

    selected = _deduplicate_overlapping_blocks(selected)
    selected = _deduplicate_by_text_containment(selected)

    # Headings sit above the columns and read top-to-bottom among
    # themselves; body text (and captions) need column-aware ordering.
    heading_blocks = [b for b in selected if b.get("class") == "title"]
    body_blocks = [b for b in selected if b.get("class") != "title"]

    heading_blocks.sort(key=lambda b: b.get("bbox", {}).get("y1", 0.0))
    ordered_body = _order_body_blocks_by_column(body_blocks)

    ordered = heading_blocks + ordered_body

    lines = []
    confidences = []

    for block in ordered:

        text = str(block.get("text", "") or "").strip()
        confidence = float(block.get("ocr_confidence", 0.0) or 0.0)

        lines.append({
            "text": text,
            "confidence": confidence,
            "bbox": block.get("bbox", {}) or {},
            "block_class": block.get("class"),
        })

        confidences.append(confidence)

    full_text = "\n".join(line["text"] for line in lines)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": full_text,
        "confidence": avg_confidence,
        "lines": lines,
        "line_count": len(lines),
    }
