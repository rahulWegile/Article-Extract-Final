import cv2
import time

from rapidocr import RapidOCR

from pipeline.ocr.ocr_models import OCRResult


# ============================================================
# OCR CLASSES
# ============================================================

OCR_CLASSES = {
    "plain text",
    "title",
    "figure_caption",
}


# ============================================================
# RAPIDOCR ENGINE
# ============================================================

class RapidOCREngine:

    def __init__(self):

        print("Loading RapidOCR...")

        self.reader = RapidOCR()

        print("✓ RapidOCR loaded")

    # ========================================================
    # EMPTY OCR RESULT
    # ========================================================

    @staticmethod
    def _empty_result(block):

        return OCRResult(
            text="",
            confidence=0.0,

            x1=block.x1,
            y1=block.y1,
            x2=block.x2,
            y2=block.y2,

            lines=[],
        )

    # ========================================================
    # BOX -> XYXY
    # ========================================================

    @staticmethod
    def _box_to_xyxy(box):

        try:

            xs = [
                float(point[0])
                for point in box
            ]

            ys = [
                float(point[1])
                for point in box
            ]

            return (
                min(xs),
                min(ys),
                max(xs),
                max(ys),
            )

        except Exception:

            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

    # ========================================================
    # INTERSECTION AREA
    # ========================================================

    @staticmethod
    def _intersection_area(
        ax1,
        ay1,
        ax2,
        ay2,
        bx1,
        by1,
        bx2,
        by2,
    ):

        x1 = max(
            ax1,
            bx1,
        )

        y1 = max(
            ay1,
            by1,
        )

        x2 = min(
            ax2,
            bx2,
        )

        y2 = min(
            ay2,
            by2,
        )

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        return width * height

    # ========================================================
    # FIND BEST DOC LAYOUT BLOCK
    # ========================================================

    @staticmethod
    def _find_best_block(
        line_bbox,
        blocks,
    ):

        lx1, ly1, lx2, ly2 = (
            line_bbox
        )

        line_width = max(
            0.0,
            lx2 - lx1,
        )

        line_height = max(
            0.0,
            ly2 - ly1,
        )

        line_area = (
            line_width
            * line_height
        )

        if line_area <= 0:

            return None

        center_x = (
            lx1 + lx2
        ) / 2.0

        center_y = (
            ly1 + ly2
        ) / 2.0

        best_block = None
        best_score = 0.0

        for block in blocks:

            # ------------------------------------------------
            # Only text-related blocks
            # ------------------------------------------------

            if block.cls not in OCR_CLASSES:
                continue

            # ------------------------------------------------
            # Check center
            # ------------------------------------------------

            center_inside = (
                block.x1
                <= center_x
                <= block.x2
                and
                block.y1
                <= center_y
                <= block.y2
            )

            # ------------------------------------------------
            # Calculate intersection
            # ------------------------------------------------

            intersection = (
                RapidOCREngine._intersection_area(
                    lx1,
                    ly1,
                    lx2,
                    ly2,
                    block.x1,
                    block.y1,
                    block.x2,
                    block.y2,
                )
            )

            overlap_ratio = (
                intersection
                / line_area
            )

            # ------------------------------------------------
            # Score
            # ------------------------------------------------

            if center_inside:

                score = (
                    1.0
                    + overlap_ratio
                )

            elif overlap_ratio >= 0.20:

                score = overlap_ratio

            else:

                continue

            if score > best_score:

                best_score = score
                best_block = block

        return best_block

    # ========================================================
    # PROCESS LAYOUT BLOCKS
    # ========================================================

    def process_blocks(
        self,
        image_path,
        blocks,
    ):

        # ====================================================
        # READ PAGE ONCE
        # ====================================================

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise FileNotFoundError(
                f"Unable to read image: "
                f"{image_path}"
            )

        # ====================================================
        # RESULT STORAGE
        # ====================================================

        results = {}

        processed_blocks = []

        skipped_count = 0

        # ====================================================
        # INITIALIZE RESULTS
        # ====================================================

        for block in blocks:

            results[
                block.id
            ] = self._empty_result(
                block
            )

            # ------------------------------------------------
            # Skip non-text blocks
            # ------------------------------------------------

            if block.cls not in OCR_CLASSES:

                skipped_count += 1

                continue

            processed_blocks.append(
                block
            )

        # ====================================================
        # NO OCR BLOCKS
        # ====================================================

        if not processed_blocks:

            ordered_results = [
                results[
                    block.id
                ]
                for block in blocks
            ]

            print()
            print("=" * 60)
            print("RAPIDOCR BLOCK SUMMARY")
            print("=" * 60)

            print(
                f"Total blocks      : "
                f"{len(blocks)}"
            )

            print(
                "OCR processed     : 0"
            )

            print(
                f"OCR skipped       : "
                f"{skipped_count}"
            )

            print(
                f"OCR results       : "
                f"{len(ordered_results)}"
            )

            print(
                "Non-empty results : 0"
            )

            print(
                "Full-page calls   : 0"
            )

            print(
                "Fallback calls    : 0"
            )

            print("=" * 60)

            return ordered_results

        # ====================================================
        # ONE FULL-PAGE RAPIDOCR CALL
        # ====================================================

        print()
        print(
            "Running RapidOCR "
            "once on the full page..."
        )

        # ----------------------------------------------------
        # FULL-PAGE OCR TIMER
        # ----------------------------------------------------

        full_page_start = (
            time.perf_counter()
        )

        result = self.reader(
            image
        )

        full_page_elapsed = (
            time.perf_counter()
            - full_page_start
        )

        print(
            f"[RAPIDOCR TIMING] "
            f"Full-page OCR : "
            f"{full_page_elapsed:8.2f} sec"
        )

        # ====================================================
        # RAPIDOCR RETURNED NOTHING
        # ====================================================

        if result is None:

            print(
                "Warning: RapidOCR returned "
                "no full-page result."
            )

            ordered_results = [
                results[
                    block.id
                ]
                for block in blocks
            ]

            print()
            print("=" * 60)
            print("RAPIDOCR BLOCK SUMMARY")
            print("=" * 60)

            print(
                f"Total blocks      : "
                f"{len(blocks)}"
            )

            print(
                f"OCR processed     : "
                f"{len(processed_blocks)}"
            )

            print(
                f"OCR skipped       : "
                f"{skipped_count}"
            )

            print(
                f"OCR results       : "
                f"{len(ordered_results)}"
            )

            print(
                "Non-empty results : 0"
            )

            print(
                "Full-page calls   : 1"
            )

            print(
                "Fallback calls    : 0"
            )

            print("=" * 60)

            return ordered_results

        # ====================================================
        # EXTRACT FULL-PAGE OCR OUTPUT
        # ====================================================

        txts = getattr(
            result,
            "txts",
            None,
        )

        scores = getattr(
            result,
            "scores",
            None,
        )

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if txts is None:
            txts = []

        if scores is None:
            scores = []

        if boxes is None:
            boxes = []

        # ====================================================
        # STORE OCR LINES PER BLOCK
        # ====================================================

        block_lines = {
            block.id: []
            for block in blocks
            if block.cls in OCR_CLASSES
        }

        # ====================================================
        # MAP FULL-PAGE OCR LINES
        # TO DOCLAYOUT BLOCKS
        # ====================================================

        for index, line_text in enumerate(
            txts
        ):

            line_text = str(
                line_text
            ).strip()

            if not line_text:
                continue

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            line_confidence = 0.0

            if index < len(
                scores
            ):

                try:

                    line_confidence = float(
                        scores[index]
                    )

                except Exception:

                    line_confidence = 0.0

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            if index >= len(
                boxes
            ):

                continue

            polygon = boxes[
                index
            ]

            (
                page_x1,
                page_y1,
                page_x2,
                page_y2,
            ) = self._box_to_xyxy(
                polygon
            )

            # ------------------------------------------------
            # Find owning block
            # ------------------------------------------------

            best_block = (
                self._find_best_block(
                    (
                        page_x1,
                        page_y1,
                        page_x2,
                        page_y2,
                    ),
                    processed_blocks,
                )
            )

            if best_block is None:

                continue

            # ------------------------------------------------
            # Save line
            # ------------------------------------------------

            block_lines[
                best_block.id
            ].append(
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

        # ====================================================
        # BUILD OCR RESULT FOR EACH BLOCK
        # ====================================================

        for block in processed_blocks:

            lines = block_lines.get(
                block.id,
                [],
            )

            # ------------------------------------------------
            # Sort lines
            # ------------------------------------------------

            lines.sort(
                key=lambda item: (
                    item["bbox"]["y1"],
                    item["bbox"]["x1"],
                )
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            text_parts = [
                line["text"]
                for line in lines
            ]

            text = " ".join(
                text_parts
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidences = [
                line["confidence"]
                for line in lines
            ]

            confidence = 0.0

            if confidences:

                confidence = (
                    sum(confidences)
                    / len(confidences)
                )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            results[
                block.id
            ] = OCRResult(
                text=text,

                confidence=confidence,

                x1=block.x1,
                y1=block.y1,
                x2=block.x2,
                y2=block.y2,

                lines=lines,
            )

        # ====================================================
        # TARGETED FALLBACK OCR
        #
        # Only empty OCR blocks are processed again.
        #
        # This is deliberately NOT a return to the old
        # 61-block OCR architecture.
        #
        # Normal case:
        #
        #     1 full-page OCR
        #
        # Fallback:
        #
        #     OCR only blocks with no mapped text
        # ====================================================

        fallback_start = (
            time.perf_counter()
        )

        fallback_count = 0

        for block in processed_blocks:

            current_result = results.get(
                block.id
            )

            if current_result is None:
                continue

            current_text = (
                current_result.text
                or ""
            ).strip()

            # ------------------------------------------------
            # Already has text
            # ------------------------------------------------

            if current_text:

                continue

            # ------------------------------------------------
            # Crop coordinates
            # ------------------------------------------------

            height, width = (
                image.shape[:2]
            )

            x1 = max(
                0,
                int(block.x1),
            )

            y1 = max(
                0,
                int(block.y1),
            )

            x2 = min(
                width,
                int(block.x2),
            )

            y2 = min(
                height,
                int(block.y2),
            )

            # ------------------------------------------------
            # Invalid crop
            # ------------------------------------------------

            if x2 <= x1:
                continue

            if y2 <= y1:
                continue

            crop = image[
                y1:y2,
                x1:x2,
            ]

            if crop.size == 0:
                continue

            # ------------------------------------------------
            # Fallback OCR
            # ------------------------------------------------

            fallback_result = self.reader(
                crop
            )

            fallback_count += 1

            if fallback_result is None:

                continue

            fallback_txts = getattr(
                fallback_result,
                "txts",
                None,
            )

            fallback_scores = getattr(
                fallback_result,
                "scores",
                None,
            )

            fallback_boxes = getattr(
                fallback_result,
                "boxes",
                None,
            )

            if fallback_txts is None:
                fallback_txts = []

            if fallback_scores is None:
                fallback_scores = []

            if fallback_boxes is None:
                fallback_boxes = []

            fallback_lines = []

            fallback_text_parts = []

            fallback_confidences = []

            # ------------------------------------------------
            # Process fallback OCR lines
            # ------------------------------------------------

            for index, line_text in enumerate(
                fallback_txts
            ):

                line_text = str(
                    line_text
                ).strip()

                if not line_text:
                    continue

                # --------------------------------------------
                # Confidence
                # --------------------------------------------

                confidence = 0.0

                if index < len(
                    fallback_scores
                ):

                    try:

                        confidence = float(
                            fallback_scores[index]
                        )

                    except Exception:

                        confidence = 0.0

                # --------------------------------------------
                # Convert crop coordinates
                # to page coordinates
                # --------------------------------------------

                if index < len(
                    fallback_boxes
                ):

                    polygon = (
                        fallback_boxes[
                            index
                        ]
                    )

                    (
                        crop_x1,
                        crop_y1,
                        crop_x2,
                        crop_y2,
                    ) = self._box_to_xyxy(
                        polygon
                    )

                    page_x1 = (
                        crop_x1
                        + x1
                    )

                    page_y1 = (
                        crop_y1
                        + y1
                    )

                    page_x2 = (
                        crop_x2
                        + x1
                    )

                    page_y2 = (
                        crop_y2
                        + y1
                    )

                else:

                    page_x1 = float(
                        x1
                    )

                    page_y1 = float(
                        y1
                    )

                    page_x2 = float(
                        x2
                    )

                    page_y2 = float(
                        y2
                    )

                # --------------------------------------------
                # Store line
                # --------------------------------------------

                fallback_lines.append(
                    {
                        "text": line_text,

                        "confidence": (
                            confidence
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

                fallback_text_parts.append(
                    line_text
                )

                fallback_confidences.append(
                    confidence
                )

            # ------------------------------------------------
            # Save fallback result
            # ------------------------------------------------

            if fallback_text_parts:

                fallback_text = " ".join(
                    fallback_text_parts
                )

                if fallback_confidences:

                    fallback_confidence = (
                        sum(
                            fallback_confidences
                        )
                        / len(
                            fallback_confidences
                        )
                    )

                else:

                    fallback_confidence = 0.0

                results[
                    block.id
                ] = OCRResult(
                    text=fallback_text,

                    confidence=(
                        fallback_confidence
                    ),

                    x1=block.x1,
                    y1=block.y1,
                    x2=block.x2,
                    y2=block.y2,

                    lines=fallback_lines,
                )

        # ====================================================
        # FALLBACK TIMING
        # ====================================================

        fallback_elapsed = (
            time.perf_counter()
            - fallback_start
        )

        print(
            f"[RAPIDOCR TIMING] "
            f"Fallback OCR   : "
            f"{fallback_elapsed:8.2f} sec"
        )

        print(
            f"[RAPIDOCR TIMING] "
            f"Fallback calls  : "
            f"{fallback_count}"
        )

        # ====================================================
        # TOTAL RAPIDOCR TIMING
        # ====================================================

        total_ocr_elapsed = (
            full_page_elapsed
            + fallback_elapsed
        )

        print(
            f"[RAPIDOCR TIMING] "
            f"Measured OCR    : "
            f"{total_ocr_elapsed:8.2f} sec"
        )

        # ====================================================
        # ORIGINAL BLOCK ORDER
        # ====================================================

        ordered_results = [
            results[
                block.id
            ]
            for block in blocks
        ]

        # ====================================================
        # SAFETY CHECK
        # ====================================================

        if len(
            ordered_results
        ) != len(blocks):

            raise RuntimeError(
                "OCR result count mismatch: "
                f"{len(ordered_results)} results "
                f"for {len(blocks)} blocks"
            )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        non_empty = sum(
            1
            for result in ordered_results
            if (
                result.text
                and result.text.strip()
            )
        )

        print()
        print("=" * 60)
        print(
            "RAPIDOCR BLOCK SUMMARY"
        )
        print("=" * 60)

        print(
            f"Total blocks      : "
            f"{len(blocks)}"
        )

        print(
            f"OCR processed     : "
            f"{len(processed_blocks)}"
        )

        print(
            f"OCR skipped       : "
            f"{skipped_count}"
        )

        print(
            f"OCR results       : "
            f"{len(ordered_results)}"
        )

        print(
            f"Non-empty results : "
            f"{non_empty}"
        )

        print(
            "Full-page calls   : 1"
        )

        print(
            f"Fallback calls    : "
            f"{fallback_count}"
        )

        print("=" * 60)

        return ordered_results

    # ========================================================
    # PROCESS FINAL ARTICLE CROP
    # ========================================================

    def process_article_crop(
        self,
        image_path,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise FileNotFoundError(
                f"Unable to read article crop: "
                f"{image_path}"
            )

        result = self.reader(
            image
        )

        if result is None:

            return {
                "text": "",
                "confidence": 0.0,
                "lines": [],
                "width": int(
                    image.shape[1]
                ),
                "height": int(
                    image.shape[0]
                ),
                "line_count": 0,
            }

        txts = getattr(
            result,
            "txts",
            None,
        )

        scores = getattr(
            result,
            "scores",
            None,
        )

        boxes = getattr(
            result,
            "boxes",
            None,
        )

        if txts is None:
            txts = []

        if scores is None:
            scores = []

        if boxes is None:
            boxes = []

        text_parts = []

        confidences = []

        lines = []

        # ====================================================
        # PROCESS OCR LINES
        # ====================================================

        for index, line_text in enumerate(
            txts
        ):

            line_text = str(
                line_text
            ).strip()

            if not line_text:
                continue

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            line_confidence = 0.0

            if index < len(
                scores
            ):

                try:

                    line_confidence = float(
                        scores[index]
                    )

                except Exception:

                    line_confidence = 0.0

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            if index < len(
                boxes
            ):

                polygon = boxes[
                    index
                ]

                try:

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

                except Exception:

                    x1 = 0.0
                    y1 = 0.0

                    x2 = float(
                        image.shape[1]
                    )

                    y2 = float(
                        image.shape[0]
                    )

            else:

                x1 = 0.0
                y1 = 0.0

                x2 = float(
                    image.shape[1]
                )

                y2 = float(
                    image.shape[0]
                )

            # ------------------------------------------------
            # Save line
            # ------------------------------------------------

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
        # COMBINED TEXT
        # ====================================================

        text = "\n".join(
            text_parts
        )

        # ====================================================
        # AVERAGE CONFIDENCE
        # ====================================================

        confidence = 0.0

        if confidences:

            confidence = (
                sum(confidences)
                / len(confidences)
            )

        # ====================================================
        # RETURN
        # ====================================================

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