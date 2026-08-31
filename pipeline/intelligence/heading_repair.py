"""
Repairs a specific, narrow failure mode in the per-page boundary/crop
output: a heading that Stream B (document_order_extractor.py) saw on
a page, but that never made it into ANY final article crop's
`block_ids` on that page -- i.e. a real heading whose article region
was cropped starting below it (see local_grouper.py's own
ROOT_WIDTH_FACTOR note for the geometric cause of this in local mode;
the LLM-grouped path can produce the same shape of mistake too).

Scope, deliberately narrow:

- This ONLY adds a missing heading block to an EXISTING article crop's
  `block_ids` and re-crops that ONE article. It never creates a new
  article, never merges two existing articles, never touches any
  other article's crop, and never changes grouping/boundary logic
  itself -- Stream A's own boundaries/decisions are the only source
  of which articles exist at all.
- A heading is repaired ONLY when BOTH of these are true:
    1. It matches an actual "title"-class block in that page's
       page_json (the heading isn't just floating text with nothing
       real behind it).
    2. That title block is not already covered by any existing
       article's block_ids, AND there is a real body-text block
       belonging to exactly one existing article that the heading
       would plausibly govern (same column, positioned below it) --
       i.e. it has real "matching body text", not just a name that
       happens to resemble a heading.
- Anything short of that (no matching block, already covered, no
  plausible owning article) is left completely alone -- Stream A
  stays authoritative, per the same policy reconcile.py already uses
  for text-level reconciliation.

Reuses the SAME column-overlap/bbox primitives as local_grouper.py
(so "same column" means the same thing everywhere in this codebase)
and the SAME heading/vocabulary matching primitives as reconcile.py,
rather than redefining either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.article.local_grouper import _bbox, _overlap_ratio, COLUMN_OVERLAP
from pipeline.intelligence.reconcile import (
    _normalize_heading,
    _significant_tokens,
)

# A page_json "title" block's OCR text must overlap the Stream B
# heading text by at least this much (normalized substring
# containment, or shared significant tokens) to count as a match.
# Generous on purpose -- OCR noise on either side (the page-level
# RapidOCR pass, or Stream B's own vision read) is expected; a false
# NEGATIVE here just means a real fix is skipped (safe), a false
# POSITIVE would attach the wrong block (unsafe), so containment/
# token-overlap rather than a loose fuzzy-distance match is used.
HEADING_TOKEN_OVERLAP_MIN = 0.5

# The layout classes that carry a genuine article's body text --
# matches OCR_CLASSES in rapidocr_engine.py minus "title" itself,
# since a title cannot be its own body evidence.
BODY_CLASSES = {"plain text", "figure_caption"}


def _load_page_blocks(document_dir: Path, page: int) -> dict[int, dict]:

    page_json_path = (
        document_dir / "page_json" / f"page_{page:03d}.json"
    )

    if not page_json_path.exists():
        return {}

    try:
        data = json.loads(page_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    blocks = data.get("blocks", []) if isinstance(data, dict) else data

    return {b["id"]: b for b in blocks if "id" in b}


def _load_page_crops(document_dir: Path, page: int) -> list[dict[str, Any]]:

    page_dir = (
        document_dir / "final_articles_crops" / f"page_{page:03d}"
    )

    if not page_dir.exists():
        return []

    crops = []

    for crop_json_path in sorted(page_dir.glob("*/crop.json")):

        try:
            crop = json.loads(crop_json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        crop["_crop_json_path"] = crop_json_path
        crops.append(crop)

    return crops


def _heading_matches_block(heading: str, block: dict) -> bool:

    if (block.get("class") or "").strip().lower() != "title":
        return False

    block_text = _normalize_heading(block.get("text") or "")
    heading_norm = _normalize_heading(heading)

    if not block_text or not heading_norm:
        return False

    if heading_norm in block_text or block_text in heading_norm:
        return True

    heading_tokens = _significant_tokens(heading_norm)
    block_tokens = _significant_tokens(block_text)

    if not heading_tokens or not block_tokens:
        return False

    shared = heading_tokens & block_tokens

    overlap = len(shared) / min(len(heading_tokens), len(block_tokens))

    return overlap >= HEADING_TOKEN_OVERLAP_MIN


def _find_heading_block(heading: str, blocks_by_id: dict[int, dict]):

    best_block = None
    best_score = -1.0

    for block in blocks_by_id.values():

        if not _heading_matches_block(heading, block):
            continue

        # Prefer the closest text-length match when more than one
        # title on the page resembles the heading (rare, but a
        # newspaper can repeat a section name as a running head).
        score = -abs(
            len(block.get("text") or "") - len(heading)
        )

        if score > best_score:
            best_score = score
            best_block = block

    return best_block


def _block_already_covered(block_id: int, crops: list[dict[str, Any]]) -> bool:

    for crop in crops:

        if block_id in (crop.get("block_ids") or []):
            return True

    return False


def _find_owning_crop(
    heading_block: dict,
    blocks_by_id: dict[int, dict],
    crops: list[dict[str, Any]],
):
    """
    Among the existing crops on this page, find the one whose OWN
    body text this heading most plausibly governs: the crop that
    contains the nearest "plain text"/"figure_caption" block that
    sits below the heading, in the same column -- the same
    "governs what reads BELOW it" rule local_grouper.py's own root-
    attachment logic uses, just evaluated against already-built crops
    instead of already-chosen roots.
    """

    heading_box = _bbox(heading_block)

    if heading_box is None:
        return None

    best_crop = None
    best_distance = None

    for crop in crops:

        block_ids = crop.get("block_ids") or []

        for block_id in block_ids:

            block = blocks_by_id.get(block_id)

            if block is None:
                continue

            if (block.get("class") or "").strip().lower() not in BODY_CLASSES:
                continue

            box = _bbox(block)

            if box is None:
                continue

            if box["y1"] <= heading_box["y2"]:
                continue

            if _overlap_ratio(heading_box, box) < COLUMN_OVERLAP:
                continue

            distance = float(box["y1"] - heading_box["y2"])

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_crop = crop

    return best_crop


def _recrop(
    document_dir: Path,
    page: int,
    crop: dict[str, Any],
    heading_block: dict,
) -> None:

    heading_box = _bbox(heading_block)

    current_bbox = crop.get("bbox") or {}

    new_x1 = min(current_bbox.get("x1", heading_box["x1"]), heading_box["x1"])
    new_y1 = min(current_bbox.get("y1", heading_box["y1"]), heading_box["y1"])
    new_x2 = max(current_bbox.get("x2", heading_box["x2"]), heading_box["x2"])
    new_y2 = max(current_bbox.get("y2", heading_box["y2"]), heading_box["y2"])

    page_image_path = document_dir / "pages" / f"page_{page:03d}.png"

    if not page_image_path.exists():
        return

    with Image.open(page_image_path) as source:

        source = source.convert("RGB")

        new_x1 = max(0, min(int(round(new_x1)), source.width))
        new_y1 = max(0, min(int(round(new_y1)), source.height))
        new_x2 = max(0, min(int(round(new_x2)), source.width))
        new_y2 = max(0, min(int(round(new_y2)), source.height))

        if new_x2 <= new_x1 or new_y2 <= new_y1:
            return

        new_crop_image = source.crop((new_x1, new_y1, new_x2, new_y2))

        crop_json_path = crop["_crop_json_path"]

        output_path = crop_json_path.parent / f"page_{page:03d}.png"

        new_crop_image.save(output_path, format="PNG")

    block_ids = list(crop.get("block_ids") or [])

    if heading_block["id"] not in block_ids:
        block_ids.append(heading_block["id"])

    crop["block_ids"] = block_ids
    crop["bbox"] = {"x1": new_x1, "y1": new_y1, "x2": new_x2, "y2": new_y2}
    crop["width"] = new_x2 - new_x1
    crop["height"] = new_y2 - new_y1

    crop_to_save = {
        key: value for key, value in crop.items() if key != "_crop_json_path"
    }

    with open(crop_json_path, "w", encoding="utf-8") as f:
        json.dump(crop_to_save, f, indent=4, ensure_ascii=False)


def repair_missing_headings(
    document_dir: Path,
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Given a batch of Stream B sections ({"page", "heading", "text"}),
    repair any heading that has a real matching title block AND a
    real owning article crop, but is missing from that crop's
    block_ids. Never raises -- a failure on one section must not
    block the rest of the pipeline.

    Returns a list of repair records (for logging/visibility), one
    per heading actually repaired.
    """

    document_dir = Path(document_dir)

    repairs: list[dict[str, Any]] = []

    blocks_cache: dict[int, dict[int, dict]] = {}
    crops_cache: dict[int, list[dict[str, Any]]] = {}

    for section in sections:

        try:

            heading = (section.get("heading") or "").strip()

            if not heading:
                continue

            page = section.get("page")

            if not isinstance(page, int):
                continue

            if page not in blocks_cache:
                blocks_cache[page] = _load_page_blocks(document_dir, page)

            if page not in crops_cache:
                crops_cache[page] = _load_page_crops(document_dir, page)

            blocks_by_id = blocks_cache[page]
            crops = crops_cache[page]

            if not blocks_by_id or not crops:
                continue

            heading_block = _find_heading_block(heading, blocks_by_id)

            if heading_block is None:
                # No clear match -- Stream A stays authoritative.
                continue

            if _block_already_covered(heading_block["id"], crops):
                # Already included somewhere -- nothing to repair.
                continue

            owning_crop = _find_owning_crop(
                heading_block, blocks_by_id, crops,
            )

            if owning_crop is None:
                # No plausible existing article to attach it to --
                # this repair only ever extends an EXISTING article,
                # it never fabricates a new one.
                continue

            _recrop(document_dir, page, owning_crop, heading_block)

            # Keep the in-memory crops cache consistent in case a
            # later section in this same batch targets the same page.
            crops_cache[page] = _load_page_crops(document_dir, page)

            repairs.append(
                {
                    "page": page,
                    "heading": heading,
                    "heading_block_id": heading_block["id"],
                    "article_id": owning_crop.get("article_id"),
                }
            )

        except Exception as exc:

            print(
                "⚠ Heading repair failed for section "
                f"{section.get('page')}/{section.get('heading')!r}: "
                f"{exc}"
            )

            continue

    if repairs:

        print(
            f"Heading repair: {len(repairs)} missing heading(s) "
            "folded back into their article's boundary"
        )

        for repair in repairs:

            print(
                f"  page {repair['page']} -> {repair['article_id']}: "
                f"{repair['heading']!r}"
            )

    return repairs
