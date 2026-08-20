from __future__ import annotations

import json
import sys

import cv2

from pipeline.ocr.rapidocr_engine import RapidOCREngine

CANDIDATE_CROP_FRACTIONS = (0.06, 0.09, 0.12, 0.16, 0.20)


def _ocr_crop(reader, image, top_fraction):
    height, width = image.shape[:2]
    crop = image[0 : int(height * top_fraction), 0:width]

    result = reader(crop)

    if result is None:
        return []

    txts = getattr(result, "txts", None)
    if txts is None:
        txts = []

    scores = getattr(result, "scores", None)
    if scores is None:
        scores = []

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        boxes = []

    lines = []

    for index, text in enumerate(txts):
        text = str(text).strip()
        if not text:
            continue

        try:
            confidence = float(scores[index])
        except (IndexError, TypeError, ValueError):
            confidence = 0.0

        if index < len(boxes):
            xs = [float(point[0]) for point in boxes[index]]
            ys = [float(point[1]) for point in boxes[index]]
            bbox_fraction = (
                round(min(xs) / width, 4),
                round(min(ys) / height, 4),
                round(max(xs) / width, 4),
                round(max(ys) / height, 4),
            )
        else:
            bbox_fraction = None

        lines.append((text, round(confidence, 3), bbox_fraction))

    return lines


def main(image_path: str) -> None:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    print(f"Loading RapidOCR to inspect {image_path} ...")
    reader = RapidOCREngine().reader

    for top_fraction in CANDIDATE_CROP_FRACTIONS:
        print()
        print("=" * 60)
        print(f"TOP {top_fraction * 100:.0f}% OF PAGE")
        print("=" * 60)

        lines = _ocr_crop(reader, image, top_fraction)

        if not lines:
            print("(no text detected)")
            continue

        for text, confidence, bbox_fraction in lines:
            print(f"  conf={confidence:.2f}  bbox={bbox_fraction}  {text!r}")

    print()
    print("=" * 60)
    print("DRAFT REGISTRY ENTRY")
    print("=" * 60)

    draft = {
        "template_id": "REPLACE_ME",
        "newspaper_name": "REPLACE_ME",
        "edition": "REPLACE_ME",
        "language": "REPLACE_ME",
        "name_ocr_variants": ["REPLACE_ME"],
        "masthead_region": [0.0, 0.0, 1.0, 0.0],
        "date_region": [0.0, 0.0, 1.0, 0.0],
        "edition_region": [0.0, 0.0, 1.0, 0.0],
        "edition_patterns": ["REPLACE_ME"],
        "edition_primary_token": "REPLACE_ME",
        "date_format_hint": "REPLACE_ME",
        "sample_source": image_path,
        "verified": False,
        "notes": "",
    }

    print(json.dumps(draft, indent=4))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m pipeline.intelligence.local.masthead."
            "onboarding_tool <page_001.png>"
        )
        sys.exit(1)

    main(sys.argv[1])
