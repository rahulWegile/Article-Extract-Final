from pathlib import Path

from pipeline.block_parser import parse_results
from pipeline.sort_blocks import sort_blocks

from pipeline.preprocess.page_cleaner import PageCleaner

from pipeline.export.gemini_page_exporter import (
    GeminiPageExporter,
)

from pipeline.layout_detector import LayoutDetector
from pipeline.ocr.easyocr_engine import EasyOCREngine


DOCUMENT_DIR = Path(
    "output/documents/doc_000018"
)

PAGE_NUMBER = 3

PAGE_PATH = (
    DOCUMENT_DIR
    / "pages"
    / "page_003.png"
)


def regenerate_page_json():

    print()
    print("=" * 60)
    print("REGENERATE PAGE JSON")
    print("=" * 60)

    print(
        f"Document : {DOCUMENT_DIR}"
    )

    print(
        f"Page     : {PAGE_NUMBER}"
    )

    print(
        f"Image    : {PAGE_PATH}"
    )

    if not PAGE_PATH.exists():

        raise FileNotFoundError(
            f"Page image not found: {PAGE_PATH}"
        )

    # -------------------------------------------------
    # Layout Detector
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING LAYOUT DETECTOR")
    print("=" * 60)

    detector = LayoutDetector()

    print()
    print("Detecting layout...")

    results = detector.detect(
        str(PAGE_PATH)
    )

    blocks = parse_results(
        results
    )

    print(
        f"Detected blocks : {len(blocks)}"
    )

    blocks = sort_blocks(
        blocks
    )

    # -------------------------------------------------
    # OCR
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING EASYOCR")
    print("=" * 60)

    ocr_engine = EasyOCREngine()

    print()
    print("Running EasyOCR...")

    ocr_results = ocr_engine.process_blocks(
        str(PAGE_PATH),
        blocks,
    )

    for block, ocr in zip(
        blocks,
        ocr_results,
    ):

        block.text = ocr.text

        block.ocr_confidence = (
            ocr.confidence
        )

        block.ocr_lines = (
            ocr.lines
        )

    print(
        "OCR completed"
    )

    # -------------------------------------------------
    # Page Dimensions
    # -------------------------------------------------

    page_height, page_width = (
        results[0].orig_shape
    )

    print()
    print(
        f"Page width  : {page_width}"
    )

    print(
        f"Page height : {page_height}"
    )

    # -------------------------------------------------
    # Page Cleaner
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING NEW PAGE CLEANER")
    print("=" * 60)

    clean_blocks = PageCleaner().clean(
        blocks=blocks,
        page_width=page_width,
        page_height=page_height,
    )

    print()
    print(
        f"Clean blocks : {len(clean_blocks)}"
    )

    # -------------------------------------------------
    # Export Page JSON
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("EXPORTING PAGE JSON")
    print("=" * 60)

    json_path = GeminiPageExporter().export(
        page_number=PAGE_NUMBER,
        page_width=page_width,
        page_height=page_height,
        blocks=clean_blocks,
        output_dir=(
            DOCUMENT_DIR
            / "page_json"
        ),
    )

    print()
    print("=" * 60)
    print("PAGE JSON REGENERATED")
    print("=" * 60)

    print(
        f"Saved -> {json_path}"
    )

    # -------------------------------------------------
    # IMPORTANT VALIDATION
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("CHECKING BLOCK 3")
    print("=" * 60)

    import json

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:

        page_json = json.load(f)

    block_3 = None

    for block in page_json.get(
        "blocks",
        [],
    ):

        if block.get("id") == 3:

            block_3 = block

            break

    if block_3 is None:

        print(
            "WARNING: Block 3 was not found."
        )

    else:

        print(
            f"ID         : {block_3.get('id')}"
        )

        print(
            f"TYPE       : {block_3.get('type')}"
        )

        print(
            f"IS_GLOBAL  : {block_3.get('is_global')}"
        )

        print(
            f"BBOX       : {block_3.get('bbox')}"
        )

        print(
            f"TEXT       : "
            f"{block_3.get('text', '')[:300]}"
        )

    # -------------------------------------------------
    # Print all global blocks
    # -------------------------------------------------

    print()
    print("=" * 60)
    print("GLOBAL BLOCKS")
    print("=" * 60)

    global_blocks = [

        block

        for block in page_json.get(
            "blocks",
            []
        )

        if block.get(
            "is_global",
            False,
        )

    ]

    print(
        f"Global blocks : {len(global_blocks)}"
    )

    for block in global_blocks:

        print()

        print(
            f"ID        : {block.get('id')}"
        )

        print(
            f"TYPE      : {block.get('type')}"
        )

        print(
            f"BBOX      : {block.get('bbox')}"
        )

        print(
            f"TEXT      : "
            f"{block.get('text', '')[:200]}"
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":

    regenerate_page_json()