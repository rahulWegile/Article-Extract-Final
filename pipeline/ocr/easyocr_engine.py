import cv2
import easyocr

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from pipeline.ocr.block_cropper import BlockCropper
from pipeline.ocr.ocr_models import OCRResult


# ============================================================
# CONFIGURATION
# ============================================================

OCR_WORKERS = 2

OCR_CLASSES = {
    "plain text",
    "title",
    "figure_caption",
}


# ============================================================
# WORKER-SIDE OCR READER
# ============================================================

_worker_reader = None
_worker_cropper = None


def _init_ocr_worker():
    """
    Initialize EasyOCR once inside each worker process.

    We intentionally create only one Reader per worker.
    """

    global _worker_reader
    global _worker_cropper

    print(
        "[OCR WORKER] Loading EasyOCR..."
    )

    _worker_reader = easyocr.Reader(
        ["en"],
        gpu=False,
    )

    _worker_cropper = BlockCropper()

    print(
        "[OCR WORKER] EasyOCR loaded"
    )


# ============================================================
# OCR ONE BLOCK
# ============================================================

def _process_one_block(
    image_path,
    block_data,
):
    """
    OCR one block inside a worker.

    block_data is a simple dictionary so it can safely
    be transferred between processes.
    """

    global _worker_reader
    global _worker_cropper

    block_id = block_data["id"]

    cls = block_data["cls"]

    x1 = block_data["x1"]
    y1 = block_data["y1"]
    x2 = block_data["x2"]
    y2 = block_data["y2"]

    # ========================================================
    # Skip non-text blocks
    # ========================================================

    if cls not in OCR_CLASSES:

        return {
            "index": block_data["index"],

            "result": OCRResult(
                text="",
                confidence=0.0,

                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,

                lines=[],
            ),
        }

    # ========================================================
    # Read page image
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"Unable to read image: "
            f"{image_path}"
        )

    # ========================================================
    # Create a lightweight block object
    # ========================================================

    class Block:

        pass

    block = Block()

    block.id = block_id

    block.cls = cls

    block.x1 = x1
    block.y1 = y1
    block.x2 = x2
    block.y2 = y2

    # ========================================================
    # Crop
    # ========================================================

    crop = _worker_cropper.crop(
        image,
        block,
    )

    if crop.size == 0:

        return {
            "index": block_data["index"],

            "result": OCRResult(
                text="",
                confidence=0.0,

                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,

                lines=[],
            ),
        }

    # ========================================================
    # EasyOCR
    # ========================================================

    prediction = _worker_reader.readtext(
        crop,
        detail=1,
        paragraph=False,
    )

    text_parts = []

    confidences = []

    lines = []

    # ========================================================
    # Process OCR lines
    # ========================================================

    for item in prediction:

        if len(item) < 2:
            continue

        line_text = str(
            item[1]
        ).strip()

        if not line_text:
            continue

        line_confidence = 0.0

        if len(item) >= 3:

            try:

                line_confidence = float(
                    item[2]
                )

            except Exception:

                line_confidence = 0.0

        # ====================================================
        # Polygon coordinates
        # ====================================================

        polygon = item[0]

        if (
            polygon
            and len(polygon) >= 4
        ):

            xs = [
                point[0]
                for point in polygon
            ]

            ys = [
                point[1]
                for point in polygon
            ]

            relative_x1 = min(xs)
            relative_y1 = min(ys)

            relative_x2 = max(xs)
            relative_y2 = max(ys)

            page_x1 = (
                x1
                + relative_x1
            )

            page_y1 = (
                y1
                + relative_y1
            )

            page_x2 = (
                x1
                + relative_x2
            )

            page_y2 = (
                y1
                + relative_y2
            )

        else:

            page_x1 = x1
            page_y1 = y1

            page_x2 = x2
            page_y2 = y2

        lines.append(
            {
                "text": line_text,

                "confidence": (
                    line_confidence
                ),

                "bbox": {
                    "x1": round(
                        page_x1,
                        2,
                    ),

                    "y1": round(
                        page_y1,
                        2,
                    ),

                    "x2": round(
                        page_x2,
                        2,
                    ),

                    "y2": round(
                        page_y2,
                        2,
                    ),
                },
            }
        )

        text_parts.append(
            line_text
        )

        confidences.append(
            line_confidence
        )

    # ========================================================
    # Combine text
    # ========================================================

    text = " ".join(
        text_parts
    )

    confidence = 0.0

    if confidences:

        confidence = (
            sum(confidences)
            / len(confidences)
        )

    # ========================================================
    # Return result
    # ========================================================

    return {
        "index": block_data["index"],

        "result": OCRResult(
            text=text,

            confidence=confidence,

            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,

            lines=lines,
        ),
    }


# ============================================================
# MAIN OCR ENGINE
# ============================================================

class EasyOCREngine:

    def __init__(self):

        print(
            "EasyOCR engine configured "
            f"for {OCR_WORKERS} workers"
        )

        self.cropper = BlockCropper()

    # ========================================================
    # Process Layout Blocks
    # ========================================================

    def process_blocks(
        self,
        image_path,
        blocks,
    ):

        image_path = str(
            image_path
        )

        # ====================================================
        # Build worker-safe block data
        # ====================================================

        block_data = []

        skipped_count = 0

        for index, block in enumerate(
            blocks
        ):

            if block.cls not in OCR_CLASSES:

                skipped_count += 1

            block_data.append(
                {
                    "index": index,

                    "id": block.id,

                    "cls": block.cls,

                    "x1": block.x1,
                    "y1": block.y1,
                    "x2": block.x2,
                    "y2": block.y2,
                }
            )

        processed_count = (
            len(blocks)
            - skipped_count
        )

        print()
        print("=" * 60)
        print("PARALLEL OCR")
        print("=" * 60)

        print(
            f"Total blocks  : "
            f"{len(blocks)}"
        )

        print(
            f"OCR processed : "
            f"{processed_count}"
        )

        print(
            f"OCR skipped   : "
            f"{skipped_count}"
        )

        print(
            f"OCR workers   : "
            f"{OCR_WORKERS}"
        )

        print("=" * 60)

        # ====================================================
        # Run workers
        # ====================================================

        with ProcessPoolExecutor(
            max_workers=OCR_WORKERS,
            initializer=_init_ocr_worker,
        ) as executor:

            worker_function = partial(
                _process_one_block,
                image_path,
            )

            worker_results = list(
                executor.map(
                    worker_function,
                    block_data,
                )
            )

        # ====================================================
        # Restore original block order
        # ====================================================

        worker_results.sort(
            key=lambda item: item["index"]
        )

        results = [
            item["result"]
            for item in worker_results
        ]

        # ====================================================
        # Safety check
        # ====================================================

        if len(results) != len(blocks):

            raise RuntimeError(
                "OCR result count mismatch: "
                f"{len(results)} results "
                f"for {len(blocks)} blocks"
            )

        # ====================================================
        # Summary
        # ====================================================

        non_empty = sum(
            1
            for result in results
            if result.text.strip()
        )

        print()
        print("=" * 60)
        print("PARALLEL OCR COMPLETE")
        print("=" * 60)

        print(
            f"Total blocks      : "
            f"{len(blocks)}"
        )

        print(
            f"OCR processed     : "
            f"{processed_count}"
        )

        print(
            f"OCR skipped       : "
            f"{skipped_count}"
        )

        print(
            f"OCR results       : "
            f"{len(results)}"
        )

        print(
            f"Non-empty results : "
            f"{non_empty}"
        )

        print("=" * 60)

        return results

    # ========================================================
    # Process Final Article Crop
    # ========================================================

    def process_article_crop(
        self,
        image_path,
    ):
        """
        Run EasyOCR directly on one final
        verified article crop.

        This method remains sequential because it
        operates on one final article crop.
        """

        # Create a temporary reader for this method.

        reader = easyocr.Reader(
            ["en"],
            gpu=False,
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise FileNotFoundError(
                f"Unable to read article crop: "
                f"{image_path}"
            )

        prediction = reader.readtext(
            image,
            detail=1,
            paragraph=False,
        )

        text_parts = []

        confidences = []

        lines = []

        # ====================================================
        # Process OCR Lines
        # ====================================================

        for item in prediction:

            if len(item) < 2:
                continue

            line_text = str(
                item[1]
            ).strip()

            if not line_text:
                continue

            line_confidence = 0.0

            if len(item) >= 3:

                try:

                    line_confidence = float(
                        item[2]
                    )

                except Exception:

                    line_confidence = 0.0

            polygon = item[0]

            if (
                polygon
                and len(polygon) >= 4
            ):

                xs = [
                    float(point[0])
                    for point in polygon
                ]

                ys = [
                    float(point[1])
                    for point in polygon
                ]

                x1 = min(xs)
                y1 = min(ys)

                x2 = max(xs)
                y2 = max(ys)

            else:

                x1 = 0.0
                y1 = 0.0

                x2 = float(
                    image.shape[1]
                )

                y2 = float(
                    image.shape[0]
                )

            lines.append(
                {
                    "text": line_text,

                    "confidence": (
                        line_confidence
                    ),

                    "bbox": {
                        "x1": round(
                            x1,
                            2,
                        ),

                        "y1": round(
                            y1,
                            2,
                        ),

                        "x2": round(
                            x2,
                            2,
                        ),

                        "y2": round(
                            y2,
                            2,
                        ),
                    },
                }
            )

            text_parts.append(
                line_text
            )

            confidences.append(
                line_confidence
            )

        # ====================================================
        # Combined Article Text
        # ====================================================

        text = "\n".join(
            text_parts
        )

        confidence = 0.0

        if confidences:

            confidence = (
                sum(confidences)
                / len(confidences)
            )

        return {
            "text": text,

            "confidence": confidence,

            "lines": lines,

            "width": int(
                image.shape[1]
            ),

            "height": int(
                image.shape[0]
            ),

            "line_count": len(
                lines
            ),
        }