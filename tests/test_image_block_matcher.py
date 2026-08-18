import json
import shutil
import tempfile
from pathlib import Path

from pipeline.intelligence.local.image_block_matcher import build_images_field


def _make_page_json(document_dir: Path, page: int, blocks: list[dict]):

    page_json_dir = document_dir / "page_json"
    page_json_dir.mkdir(parents=True, exist_ok=True)

    with open(page_json_dir / f"page_{page:03d}.json", "w", encoding="utf-8") as f:
        json.dump({"page": page, "blocks": blocks}, f)


def test_build_images_field_pairs_figure_and_caption():

    document_dir = Path(tempfile.mkdtemp())

    try:
        blocks = [
            {"id": 1, "class": "title", "bbox": {"x1": 500, "y1": 800, "x2": 900, "y2": 850}},
            {"id": 2, "class": "figure", "bbox": {"x1": 500, "y1": 900, "x2": 900, "y2": 1300}},
            {
                "id": 3,
                "class": "figure_caption",
                "bbox": {"x1": 500, "y1": 1310, "x2": 900, "y2": 1350},
                "text": "Workers repair the damaged road after heavy rain.",
            },
            {"id": 4, "class": "plain text", "bbox": {"x1": 500, "y1": 1360, "x2": 900, "y2": 2000}},
        ]

        _make_page_json(document_dir, page=1, blocks=blocks)

        crop_bbox = {"x1": 500, "y1": 800, "x2": 900, "y2": 2000}

        images = build_images_field(
            document_dir=document_dir,
            page=1,
            block_ids=[1, 2, 3, 4],
            crop_bbox=crop_bbox,
        )

        assert images["has_images"] is True
        assert images["image_count"] == 1

        item = images["items"][0]
        assert item["caption"] == "Workers repair the damaged road after heavy rain."

        # crop is 400x1200 px; figure bbox relative to crop origin is
        # (0, 100) -> (400, 500), i.e. x spans the full width (0-1000)
        # and y spans roughly the top 8%-42% of the crop height.
        bbox = item["image_bbox"]
        assert bbox["x1"] == 0.0
        assert bbox["x2"] == 1000.0
        assert 0 < bbox["y1"] < bbox["y2"] < 1000

    finally:
        shutil.rmtree(document_dir, ignore_errors=True)


def test_build_images_field_does_not_pair_distant_caption():
    """
    Regression test for a real bug: an article with two figures but
    only one figure_caption (the other photo's caption wasn't detected
    as a separate block). The first-processed figure must NOT steal
    the only caption just because it's "nearest among what's left" --
    it should only pair with a caption that's actually close to it.
    Mirrors a real doc_000002 article: two photos ~900px apart, one
    caption directly below the second photo only.
    """
    document_dir = Path(tempfile.mkdtemp())

    try:
        blocks = [
            {"id": 0, "class": "figure", "bbox": {"x1": 837, "y1": 833, "x2": 1753, "y2": 1372}},
            {"id": 4, "class": "figure", "bbox": {"x1": 1146, "y1": 1862, "x2": 1752, "y2": 2254}},
            {
                "id": 120,
                "class": "figure_caption",
                "bbox": {"x1": 1147, "y1": 2266, "x2": 1750, "y2": 2424},
                "text": "Defence minister Rajnath Singh at the inauguration.",
            },
        ]

        _make_page_json(document_dir, page=1, blocks=blocks)

        crop_bbox = {"x1": 831, "y1": 833, "x2": 1755, "y2": 2774}

        images = build_images_field(
            document_dir=document_dir,
            page=1,
            block_ids=[0, 4, 120],
            crop_bbox=crop_bbox,
        )

        assert images["image_count"] == 2

        # Sorted top-to-bottom: image_001 is block 0 (top, no nearby
        # caption), image_002 is block 4 (bottom, paired correctly).
        assert images["items"][0]["caption"] is None
        assert images["items"][1]["caption"] == "Defence minister Rajnath Singh at the inauguration."

    finally:
        shutil.rmtree(document_dir, ignore_errors=True)


def test_build_images_field_empty_without_figure_blocks():

    document_dir = Path(tempfile.mkdtemp())

    try:
        blocks = [
            {"id": 1, "class": "plain text", "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}},
        ]

        _make_page_json(document_dir, page=1, blocks=blocks)

        images = build_images_field(
            document_dir=document_dir,
            page=1,
            block_ids=[1],
            crop_bbox={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
        )

        assert images == {"has_images": False, "image_count": 0, "items": []}

    finally:
        shutil.rmtree(document_dir, ignore_errors=True)


def test_build_images_field_empty_without_block_ids():

    images = build_images_field(
        document_dir=Path("does-not-matter"),
        page=1,
        block_ids=[],
        crop_bbox={},
    )

    assert images == {"has_images": False, "image_count": 0, "items": []}


if __name__ == "__main__":

    test_build_images_field_pairs_figure_and_caption()
    test_build_images_field_does_not_pair_distant_caption()
    test_build_images_field_empty_without_figure_blocks()
    test_build_images_field_empty_without_block_ids()

    print("=" * 60)
    print("ALL IMAGE BLOCK MATCHER TESTS PASSED")
    print("=" * 60)
