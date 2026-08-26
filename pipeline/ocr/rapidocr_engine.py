import cv2
import os
import threading
import time

from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

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

    # A Devanagari block whose average recognition confidence falls
    # below this is retried with contrast enhancement + upscaling
    # rather than accepted as-is (see _enhance_for_devanagari).
    # English is left alone -- 0.95 average confidence was already
    # measured there, and the enhancement is tuned for the specific
    # failure mode of small, low-contrast matras/conjuncts, not for
    # Latin script.
    LOW_CONFIDENCE_THRESHOLD = 0.55

    # Upscale factor applied only inside _enhance_for_devanagari.
    ENHANCE_SCALE = 1.6

    # process_blocks() stops issuing further per-block retries once
    # this much total time has passed since it was called (see
    # process_blocks's TIME BUDGET note). Left with real margin
    # under a 30s target -- the caller may want headroom for
    # whatever runs after OCR in the same page-processing step.
    DEFAULT_TIME_BUDGET_SECONDS = 25.0

    # Native PaddleOCR runs the SAME Devanagari weights RapidOCR's
    # ONNX export uses (confirmed: neither library ships a larger
    # Devanagari model), but scored measurably higher on real text
    # (0.9076 vs 0.9566 avg confidence, 26-line benchmark) -- a
    # difference in the two runtimes' pre/post-processing, not model
    # quality. It is recognition-only (no detector -- native
    # PaddleOCR's own detector crashes on this machine), so it is
    # used ONLY as a per-LINE retry for already-low-confidence lines,
    # never as the primary pass: unconditional whole-page use both
    # risks the 25s time budget (extrapolated ~33-42s on a dense
    # page) and pays its ~660MB load cost (vs RapidOCR's 103MB for
    # all three of its own models) on every document instead of only
    # documents that actually need it.
    #
    # Ships disabled by default -- flip via this env var only after
    # the manual accuracy spot-check in the project plan has been
    # done. Confidence-improvement alone does not prove
    # correctness-improvement.
    NATIVE_DEVANAGARI_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"

    def __init__(self, lang="en"):

        print(f"Loading RapidOCR (lang={lang})...")

        self.lang = lang

        if lang in ("hi", "hindi"):

            # PP-OCRv6 (the package default) has no Devanagari
            # recognition model -- only PP-OCRv4/v5 ship one.
            # Detection/orientation stay on their normal defaults;
            # only the recognition model is swapped, and only for
            # documents already identified as Hindi.
            self.reader = RapidOCR(
                params={
                    "Rec.lang_type": LangRec.DEVANAGARI,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )

        else:

            self.reader = RapidOCR()

        # ----------------------------------------------------
        # Native-paddle line-level retry (Hindi only, lazy, opt-in)
        # ----------------------------------------------------

        self._native_devanagari_recognizer = None

        self._native_devanagari_unavailable = False

        # Guards the lazy-load in _get_native_devanagari_recognizer
        # so the background preload thread (started from
        # process_blocks, see _start_native_devanagari_preload) and
        # the retry loop calling the same method later can never
        # both start constructing the model at once.
        self._native_load_lock = threading.Lock()

        self._native_load_thread = None

        self._native_retry_enabled = (
            lang in ("hi", "hindi")
            and os.getenv(
                "OCR_NATIVE_PADDLE_RETRY",
                "0",
            )
            == "1"
        )

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
    # WHOLE-PAGE CONTRAST NORMALIZATION (every Hindi page)
    # ========================================================

    @staticmethod
    def _normalize_contrast(image):
        """
        CLAHE contrast normalization applied to the whole page once,
        before the single full-page OCR call, for Hindi documents.

        This is deliberately different from _enhance_for_devanagari
        below: that one is a targeted, upscaled retry paid for only
        on a block that already scored poorly. This one runs
        unconditionally on every Hindi page because it is cheap
        (measured near-zero added time on a 4072x6368 page) and can
        only help or be a no-op -- it corrects the specific kind of
        degradation a real scan or phone photo introduces (uneven
        lighting across the page), which a clean digitally-rendered
        PDF simply does not have much of, so its benefit will not
        show up as strongly on a rendered PDF as it will on an
        actual scanned/photographed newspaper page.
        """

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(16, 16),
        )

        l_channel = clahe.apply(l_channel)

        merged = cv2.merge(
            [l_channel, a_channel, b_channel]
        )

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # ========================================================
    # DEVANAGARI ENHANCEMENT (low-confidence retry only)
    # ========================================================

    @classmethod
    def _enhance_for_devanagari(cls, crop):
        """
        Contrast-normalize and upscale a crop before a second OCR
        attempt on a block that scored below LOW_CONFIDENCE_THRESHOLD.

        Devanagari relies on small strokes above/below the headline
        line -- matras and stacked conjuncts -- that are the first
        detail lost to uneven scan lighting or a soft/small source
        image. CLAHE evens out local contrast (a flat global
        brightness/contrast adjustment does not help a scan with
        uneven lighting across the block), and the upscale gives the
        recognizer more pixels per stroke to work with.

        This is intentionally NOT applied to every block: it costs a
        second OCR pass, and unlike the initial full-page call
        (proven at ~0.95 average confidence in normal light), it is
        only worth paying for where the first pass already struggled.
        """

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        contrasted = clahe.apply(gray)

        upscaled = cv2.resize(
            contrasted,
            None,
            fx=cls.ENHANCE_SCALE,
            fy=cls.ENHANCE_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        blurred = cv2.GaussianBlur(upscaled, (0, 0), 3)

        sharpened = cv2.addWeighted(
            upscaled, 1.4,
            blurred, -0.4,
            0,
        )

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    # ========================================================
    # NATIVE-PADDLE LINE RETRY (Hindi, low-confidence lines only)
    # ========================================================

    def _get_native_devanagari_recognizer(self, wait_timeout=None):
        """
        Lazily build and cache the native-paddle Devanagari
        recognizer for this engine instance.

        Deliberately NOT imported at module level: a deployment
        without paddlepaddle/paddlex installed must still be able to
        import this whole file and run RapidOCR normally -- only
        this one lazy path is affected if those packages are
        missing.

        `self._native_devanagari_unavailable` is sticky once set, so
        a failed load is not retried on every subsequent block --
        every call after the first failure short-circuits to None
        immediately instead of repeating a known-doomed import/
        construction attempt.

        Guarded by `self._native_load_lock`: measured directly, this
        construction call takes 17-27s (dominated by first-inference
        JIT/graph warmup, not just building the model object -- a
        second call on an already-loaded model measured 0.03-0.045s).
        `_start_native_devanagari_preload` kicks this off in a
        background thread as early as possible (see process_blocks)
        so that cost overlaps with the full-page OCR pass instead of
        landing inside the retry loop's own time budget.

        `wait_timeout`: when the retry loop calls this (as opposed to
        the background preload thread, which always waits until
        done), it must NOT block past its own remaining time budget
        if the preload happens to still be running -- e.g. on an
        unusually fast full-page pass. Pass the remaining budget (or
        some bounded slice of it) here; if the lock can't be acquired
        within that window, this returns None WITHOUT marking the
        recognizer unavailable (it may simply still be loading), so a
        later call on the same or a later block can still succeed
        once the background load finishes.
        """

        if self._native_devanagari_unavailable:
            return None

        if self._native_devanagari_recognizer is not None:
            return self._native_devanagari_recognizer

        if wait_timeout is not None:

            acquired = self._native_load_lock.acquire(
                timeout=max(0.0, wait_timeout)
            )

            if not acquired:
                # Still loading and we're out of time to wait for
                # it this call -- not a failure, just not ready yet.
                return None

        else:

            self._native_load_lock.acquire()

        try:

            # Re-check inside the lock: another thread (the
            # background preload, most likely) may have already
            # finished the load while this call was waiting.

            if self._native_devanagari_unavailable:
                return None

            if self._native_devanagari_recognizer is not None:
                return self._native_devanagari_recognizer

            try:

                import paddlex

                self._native_devanagari_recognizer = (
                    paddlex.create_model(
                        self.NATIVE_DEVANAGARI_MODEL_NAME,
                        device="cpu",
                    )
                )

            except Exception as exc:

                self._native_devanagari_unavailable = True

                print(
                    "⚠ Native-paddle Devanagari recognizer "
                    f"unavailable, staying on RapidOCR's own "
                    f"retry for the rest of this document: {exc}"
                )

                return None

        finally:

            self._native_load_lock.release()

        return self._native_devanagari_recognizer

    def _start_native_devanagari_preload(self):
        """
        Kick off the native-paddle recognizer's cold load in a
        background thread, called as early as possible in
        process_blocks (before the full-page RapidOCR pass) so its
        measured 17-27s cold-start cost overlaps with that pass's own
        time instead of landing synchronously inside the budget-
        constrained retry loop -- confirmed by direct testing to
        otherwise consume the entire remaining budget on just the
        first weak line encountered.

        Safe to call every time process_blocks runs: a no-op once
        already loaded, already known unavailable, or already
        loading from an earlier call on this same engine instance.
        """

        if not self._native_retry_enabled:
            return

        if self._native_devanagari_recognizer is not None:
            return

        if self._native_devanagari_unavailable:
            return

        if (
            self._native_load_thread is not None
            and self._native_load_thread.is_alive()
        ):
            return

        self._native_load_thread = threading.Thread(
            target=self._get_native_devanagari_recognizer,
            daemon=True,
        )

        self._native_load_thread.start()

    def _native_paddle_recognize_line(self, line_crop):
        """
        Recognize ONE single-line crop via native paddle.

        Returns (text, confidence) or None on any failure -- this is
        the single choke point that guarantees a native-paddle crash
        (this machine's paddle install has a confirmed detector-side
        stability bug; recognition-only was confirmed working in
        testing, but this must still never be trusted to not throw)
        can never propagate out of the OCR pipeline.
        """

        model = self._get_native_devanagari_recognizer()

        if model is None:
            return None

        try:

            outputs = list(
                model.predict(input=line_crop)
            )

            if not outputs:
                return None

            result = outputs[0]

            text = str(
                result.get("rec_text", "") or ""
            ).strip()

            confidence = float(
                result.get("rec_score", 0.0) or 0.0
            )

            return text, confidence

        except Exception as exc:

            print(
                "  WARNING: native-paddle line recognition "
                f"failed, keeping existing reading: {exc}"
            )

            return None

    def _retry_block_lines_with_native_paddle(
        self,
        block,
        current_result,
        results,
        image,
        overall_start,
        time_budget_seconds,
    ):
        """
        Retry only the individually-weak lines of one block through
        native paddle, at LINE granularity.

        This is line-level, not block-level, because native paddle's
        recognizer has no detector of its own -- handed a multi-line
        block crop it would read the whole thing as one line and
        produce garbage. Each line already carries its own bbox (page
        coordinates) and confidence from the initial full-page
        RapidOCR pass, via current_result.lines.

        Mutates `results[block.id]` in place when at least one line
        improves; otherwise leaves it untouched.
        """

        height, width = image.shape[:2]

        new_lines = []

        any_changed = False

        for line in current_result.lines:

            line_confidence = (
                line.get("confidence", 0.0) or 0.0
            )

            # ------------------------------------------------
            # Time budget -- same expression/constant the outer
            # per-block loop already uses, checked per LINE here
            # so one block with many weak lines can't itself
            # blow past budget between block-level checks.
            # ------------------------------------------------

            if (
                time.perf_counter() - overall_start
                >= time_budget_seconds
            ):

                new_lines.append(line)

                continue

            # ------------------------------------------------
            # Only lines that are themselves weak -- a block can
            # average below threshold while most of its lines
            # are already fine.
            # ------------------------------------------------

            if (
                line_confidence
                >= self.LOW_CONFIDENCE_THRESHOLD
            ):

                new_lines.append(line)

                continue

            bbox = line.get("bbox") or {}

            lx1 = max(0, int(bbox.get("x1", 0)))
            ly1 = max(0, int(bbox.get("y1", 0)))
            lx2 = min(width, int(bbox.get("x2", 0)))
            ly2 = min(height, int(bbox.get("y2", 0)))

            if lx2 <= lx1 or ly2 <= ly1:

                new_lines.append(line)

                continue

            line_crop = image[ly1:ly2, lx1:lx2]

            if line_crop.size == 0:

                new_lines.append(line)

                continue

            # Native paddle's own preprocessing is resize +
            # mean/std normalization only (no contrast handling),
            # confirmed by reading its source -- this enhancement
            # step is not redundant with anything paddle does
            # internally.
            enhanced_crop = self._enhance_for_devanagari(
                line_crop
            )

            native_result = self._native_paddle_recognize_line(
                enhanced_crop
            )

            self._native_retry_attempts += 1

            if native_result is None:

                self._native_retry_failures += 1

                new_lines.append(line)

                continue

            native_text, native_confidence = native_result

            if (
                native_text
                and native_confidence > line_confidence
            ):

                any_changed = True

                new_lines.append(
                    {
                        "text": native_text,
                        "confidence": native_confidence,
                        # Recognition-only: no new geometry:
                        # the line's own bbox is reused as-is.
                        "bbox": bbox,
                    }
                )

            else:

                new_lines.append(line)

        if not any_changed:
            return

        new_lines.sort(
            key=lambda item: (
                item["bbox"]["y1"],
                item["bbox"]["x1"],
            )
        )

        text = " ".join(
            item["text"] for item in new_lines
        )

        confidences = [
            item["confidence"] for item in new_lines
        ]

        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        results[block.id] = OCRResult(
            text=text,

            confidence=confidence,

            x1=block.x1,
            y1=block.y1,
            x2=block.x2,
            y2=block.y2,

            lines=new_lines,
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
        time_budget_seconds=None,
    ):

        # ====================================================
        # TIME BUDGET
        #
        # The full-page pass and any retryable low-confidence
        # blocks together are not guaranteed to fit any particular
        # ceiling on their own -- measured 27.3s on a 103-block page
        # with 14 weak blocks, and each retry costs roughly another
        # second, so a page with more weak blocks than that WOULD
        # exceed a 30s target without an enforced stop. This is
        # tracked from the very start of the call (not just the
        # fallback loop) so the full-page pass itself counts against
        # it too.
        # ====================================================

        overall_start = time.perf_counter()

        if time_budget_seconds is None:
            time_budget_seconds = self.DEFAULT_TIME_BUDGET_SECONDS

        # Start the native-paddle recognizer's cold load now (if
        # enabled) so its 17-27s one-time cost overlaps with the
        # full-page OCR pass below instead of landing inside the
        # retry loop's own budget later.
        self._start_native_devanagari_preload()

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

        # Whole-page contrast normalization, Devanagari documents
        # only. Cheap (measured near-zero added time on a 4072x6368
        # page) and only ever helps or is a no-op, so it is applied
        # unconditionally for Hindi rather than gated behind a
        # confidence check the way the targeted per-block retry is.
        # Kept as a SEPARATE image from `image` itself -- the crop-
        # based fallback further below deliberately keeps reading
        # from the untouched original, since a block that needs its
        # own retry already gets its own purpose-built enhancement
        # there (_enhance_for_devanagari), and stacking two contrast
        # passes on top of each other has no benefit.
        ocr_input = image

        if self.lang in ("hi", "hindi"):

            ocr_input = self._normalize_contrast(
                image
            )

        result = self.reader(
            ocr_input
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

        budget_exhausted = False

        skipped_for_budget = 0

        self._native_retry_attempts = 0

        self._native_retry_failures = 0

        for block in processed_blocks:

            # ------------------------------------------------
            # Time budget check
            #
            # Checked once per block rather than once for the
            # whole loop so a page that starts well within budget
            # but has an unusually large number of weak blocks
            # still gets stopped partway through instead of
            # running every retry regardless of how long it takes.
            # ------------------------------------------------

            if (
                not budget_exhausted
                and time.perf_counter() - overall_start
                >= time_budget_seconds
            ):

                budget_exhausted = True

                print(
                    "⚠ OCR time budget "
                    f"({time_budget_seconds:.0f}s) reached -- "
                    "remaining blocks keep their current reading "
                    "instead of being retried"
                )

            if budget_exhausted:

                current_result = results.get(block.id)

                current_has_text = bool(
                    current_result
                    and (current_result.text or "").strip()
                )

                if not current_has_text:
                    skipped_for_budget += 1

                continue

            current_result = results.get(
                block.id
            )

            if current_result is None:
                continue

            current_text = (
                current_result.text
                or ""
            ).strip()

            current_confidence = (
                current_result.confidence
                or 0.0
            )

            # ------------------------------------------------
            # Decide whether this block needs a second pass.
            #
            # Empty result: always worth retrying (original
            # behaviour, any language).
            #
            # Low-confidence Devanagari: the mobile Devanagari
            # model is more sensitive to scan quality than the
            # default model, and unlike an empty result there IS
            # already a candidate reading here, so the enhanced
            # retry's output only replaces it further below if it
            # actually scores higher -- never blindly.
            # ------------------------------------------------

            needs_enhancement = (
                bool(current_text)
                and self.lang in ("hi", "hindi")
                and current_confidence
                < self.LOW_CONFIDENCE_THRESHOLD
            )

            if current_text and not needs_enhancement:

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
            # Native-paddle per-line retry (Hindi, opt-in)
            #
            # Tried first, ahead of the whole-block RapidOCR
            # retry below: recognition-only native paddle scored
            # measurably higher on real text in testing, but only
            # when applied per LINE (see
            # _retry_block_lines_with_native_paddle's docstring
            # for why block-level would corrupt multi-line
            # blocks). Falls through unchanged to the existing
            # whole-block retry when disabled, not installed, or
            # unavailable this run -- zero regression either way.
            # ------------------------------------------------

            if needs_enhancement and self._native_retry_enabled:

                # Bounded wait, not an indefinite block: on an
                # unusually fast full-page pass the background
                # preload (started at the top of process_blocks)
                # may not be done yet. Cap the wait well under
                # whatever budget remains rather than risking most
                # of it on just waiting for the load.
                remaining_budget = (
                    time_budget_seconds
                    - (time.perf_counter() - overall_start)
                )

                wait_timeout = max(
                    0.0,
                    min(remaining_budget, 5.0),
                )

                recognizer = (
                    self._get_native_devanagari_recognizer(
                        wait_timeout=wait_timeout,
                    )
                )

                if recognizer is not None:

                    self._retry_block_lines_with_native_paddle(
                        block,
                        current_result,
                        results,
                        image,
                        overall_start,
                        time_budget_seconds,
                    )

                    continue

            # ------------------------------------------------
            # Fallback OCR
            # ------------------------------------------------

            coordinate_scale = 1.0

            if needs_enhancement:

                crop_for_ocr = (
                    self._enhance_for_devanagari(crop)
                )

                coordinate_scale = self.ENHANCE_SCALE

            else:

                crop_for_ocr = crop

            fallback_result = self.reader(
                crop_for_ocr
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

                    # coordinate_scale > 1.0 when this crop was
                    # enhanced/upscaled before OCR (see
                    # _enhance_for_devanagari) -- without dividing
                    # it back out here, an enhanced block's line
                    # boxes would land at the wrong place on the
                    # page, off by the upscale factor.

                    page_x1 = (
                        crop_x1 / coordinate_scale
                        + x1
                    )

                    page_y1 = (
                        crop_y1 / coordinate_scale
                        + y1
                    )

                    page_x2 = (
                        crop_x2 / coordinate_scale
                        + x1
                    )

                    page_y2 = (
                        crop_y2 / coordinate_scale
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

                # The empty-result path (needs_enhancement False)
                # always has nothing to lose, so it always keeps
                # the fallback. The low-confidence Devanagari retry
                # DOES already have a candidate reading, so its
                # enhanced result only replaces it when it actually
                # scored higher -- an enhanced retry is not
                # guaranteed to beat the original on every block.

                keep_fallback = (
                    not needs_enhancement
                    or fallback_confidence
                    > current_confidence
                )

                if keep_fallback:

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

        if skipped_for_budget:

            print(
                f"Skipped (budget)  : "
                f"{skipped_for_budget}"
            )

        if self._native_retry_attempts:

            print(
                f"Native retries    : "
                f"{self._native_retry_attempts} "
                f"({self._native_retry_failures} failed)"
            )

        print(
            f"Total time        : "
            f"{time.perf_counter() - overall_start:6.2f} sec "
            f"(budget {time_budget_seconds:.0f}s)"
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