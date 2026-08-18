"""
Map DocLayout-YOLO `figure` / `figure_caption` page-level blocks onto
the `images[]` schema field of an article, using the `block_ids` each
article crop now carries (see `FinalArticleCropper._get_block_ids`).

This is geometry-derived, not vision-verified: it can say a figure
block belongs to this article's block set and, if present, which
`figure_caption` block sits nearest to it, but it cannot describe image
content the way Gemini's vision model could.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_GEOMETRY_CONFIDENCE = 0.6
_COORDINATE_MAX = 1000.0

# A caption is only paired with a figure if it starts within this many
# pixels of the figure's bottom edge (or a small negative overlap, for
# bbox imprecision) -- without this, an article with more figures than
# captions (or vice versa) can pair a caption with a figure hundreds of
# pixels away just because it's the only one left, rather than leaving
# the figure uncaptioned as the honest answer.
_MAX_CAPTION_GAP_RATIO = 0.5
_MAX_CAPTION_GAP_FLOOR = 150.0
_MAX_CAPTION_OVERLAP = 20.0


def _load_page_json(document_dir: Path, page: int) -> dict[str, Any] | None:

    path = document_dir / "page_json" / f"page_{page:03d}.json"

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _caption_gap(figure: dict[str, Any], caption: dict[str, Any]) -> float | None:
    """
    Vertical gap from the bottom of `figure` to the top of `caption`.
    A newspaper caption sits directly below its image, so this should
    be small and non-negative; a small negative value is tolerated for
    bbox imprecision. Returns None when the caption clearly isn't
    below this figure at all (e.g. it's above it, or belongs to a
    different image entirely).
    """
    gap = caption["bbox"]["y1"] - figure["bbox"]["y2"]

    if gap < -_MAX_CAPTION_OVERLAP:
        return None

    return gap


def _pair_figures_with_captions(
    figures: list[dict[str, Any]],
    captions: list[dict[str, Any]],
) -> dict[int, int]:
    """
    Global nearest-pair assignment (closest gap first) instead of
    processing figures in arbitrary order and grabbing whatever
    caption is "nearest among what's left" -- that greedy-per-figure
    approach can assign a caption to the wrong figure whenever there
    are more figures than captions (or the reverse) on the same
    article. A pairing is only made when the gap is within a
    reasonable distance of the figure itself; otherwise the figure is
    left without a caption rather than being given a wrong one.

    Returns: {figure_id: caption_id} for confidently-paired figures.
    """
    candidates = []

    for figure in figures:

        fig_height = max(1.0, figure["bbox"]["y2"] - figure["bbox"]["y1"])
        threshold = max(_MAX_CAPTION_GAP_FLOOR, fig_height * _MAX_CAPTION_GAP_RATIO)

        for caption in captions:

            gap = _caption_gap(figure, caption)

            if gap is None or gap > threshold:
                continue

            candidates.append((gap, figure["id"], caption["id"]))

    candidates.sort(key=lambda item: item[0])

    figure_to_caption: dict[int, int] = {}
    used_figures: set[int] = set()
    used_captions: set[int] = set()

    for gap, figure_id, caption_id in candidates:

        if figure_id in used_figures or caption_id in used_captions:
            continue

        figure_to_caption[figure_id] = caption_id
        used_figures.add(figure_id)
        used_captions.add(caption_id)

    return figure_to_caption


def build_images_field(
    document_dir: Path,
    page: int,
    block_ids: list[int],
    crop_bbox: dict[str, float],
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Args:
        crop_bbox: the article crop's own page-pixel bbox
            (`crop.json["bbox"]`), used as the coordinate origin.
        blocks: pre-loaded page_json["blocks"] for this page, if the
            caller already has them (avoids re-reading/re-parsing
            page_json once per article on a page with many articles).
            Falls back to loading page_json itself when omitted.
    """
    empty = {"has_images": False, "image_count": 0, "items": []}

    if not block_ids:
        return empty

    if blocks is None:
        page_data = _load_page_json(document_dir, page)
        if not page_data:
            return empty
        blocks = page_data.get("blocks") or []

    block_id_set = set(block_ids)

    article_blocks = [
        block for block in blocks
        if block.get("id") in block_id_set and isinstance(block.get("bbox"), dict)
    ]

    figures = [block for block in article_blocks if block.get("class") == "figure"]
    captions = [block for block in article_blocks if block.get("class") == "figure_caption"]

    if not figures:
        return empty

    crop_x1 = crop_bbox.get("x1", 0.0)
    crop_y1 = crop_bbox.get("y1", 0.0)
    crop_width = max(1.0, crop_bbox.get("x2", 1.0) - crop_x1)
    crop_height = max(1.0, crop_bbox.get("y2", 1.0) - crop_y1)

    # Stable top-to-bottom numbering regardless of the raw block order
    # in page_json.
    figures = sorted(figures, key=lambda b: b["bbox"].get("y1", 0.0))

    figure_to_caption = _pair_figures_with_captions(figures, captions)
    captions_by_id = {caption["id"]: caption for caption in captions}

    items = []

    for index, figure in enumerate(figures, start=1):

        caption_text = None
        caption_id = figure_to_caption.get(figure["id"])

        if caption_id is not None:
            caption_block = captions_by_id.get(caption_id)
            if caption_block is not None:
                caption_text = str(caption_block.get("text", "") or "").strip() or None

        fig_bbox = figure["bbox"]

        rel_x1 = fig_bbox["x1"] - crop_x1
        rel_y1 = fig_bbox["y1"] - crop_y1
        rel_x2 = fig_bbox["x2"] - crop_x1
        rel_y2 = fig_bbox["y2"] - crop_y1

        image_bbox = {
            "x1": round(max(0.0, rel_x1) / crop_width * _COORDINATE_MAX, 2),
            "y1": round(max(0.0, rel_y1) / crop_height * _COORDINATE_MAX, 2),
            "x2": round(min(crop_width, rel_x2) / crop_width * _COORDINATE_MAX, 2),
            "y2": round(min(crop_height, rel_y2) / crop_height * _COORDINATE_MAX, 2),
        }

        items.append({
            "image_id": f"image_{index:03d}",
            "image_description": (
                caption_text
                if caption_text
                else "Photograph associated with this article"
            ),
            "image_bbox": image_bbox,
            "caption": caption_text,
            "confidence": _GEOMETRY_CONFIDENCE,
        })

    return {
        "has_images": len(items) > 0,
        "image_count": len(items),
        "items": items,
    }
