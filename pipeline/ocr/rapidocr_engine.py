import cv2
import os
import re
import threading
import time

from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

from pipeline.ocr.ocr_models import OCRResult
from pipeline.ocr.layout_gap_recovery import (
    page_confidence_reference,
    page_line_heights,
    recover_missed_blocks,
    recover_misclassified_text_blocks,
)


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

    # Hindi-routed pages get a larger default budget: starting the
    # native-Devanagari preload before the full-page pass (see
    # process_blocks) measurably slows that pass down on this
    # machine (CPU contention between paddlepaddle's cold load and
    # RapidOCR's own inference -- observed 14.9s -> 39.9s on the
    # same real page across two runs), which can otherwise exhaust
    # DEFAULT_TIME_BUDGET_SECONDS before a single retry gets to run
    # at all. English-routed pages are NOT affected -- they never
    # start the preload this early, so they keep the original
    # DEFAULT_TIME_BUDGET_SECONDS untouched (see "preserve current
    # English performance").
    NATIVE_DEVANAGARI_TIME_BUDGET_SECONDS = 75.0

    # Per-retry-call cap on how long to wait for the background
    # preload to finish (see _get_native_devanagari_recognizer's
    # wait_timeout). Measured cold-load time is 17-27s; a cap well
    # below that (the original 5.0s) meant the FIRST candidate
    # block/line to reach this check would almost always give up
    # before the model was ready and fall through to the same-engine
    # fallback (useless for real Devanagari content) even when the
    # overall time budget had plenty of room left -- observed on a
    # real page: 55s of a 75s budget went unused while every retry
    # timed out waiting only 5s each. This cap is chosen to cover the
    # measured cold-load range; the actual wait is still bounded by
    # whatever's left of the real budget (see remaining_budget).
    NATIVE_LOAD_WAIT_CAP_SECONDS = 30.0

    # Native PaddleOCR runs the SAME Devanagari weights RapidOCR's
    # ONNX export uses (confirmed: neither library ships a larger
    # Devanagari model), but scored measurably higher on real text
    # (0.9076 vs 0.9566 avg confidence, 26-line benchmark) -- a
    # difference in the two runtimes' pre/post-processing, not model
    # quality. It is recognition-only (no detector -- native
    # PaddleOCR's own detector crashes on this machine), so it is
    # used ONLY as a per-LINE retry for lines flagged as Devanagari
    # candidates (see _is_devanagari_candidate), never as the primary
    # pass: unconditional whole-page use both risks the 25s time
    # budget (extrapolated ~33-42s on a dense page) and pays its
    # ~660MB load cost (vs RapidOCR's 103MB for all three of its own
    # models) on every document instead of only documents/pages that
    # actually turn out to need it -- see _start_native_devanagari_preload
    # for how that cost stays lazy even now that this runs for
    # English-routed documents too (not just Hindi-routed ones).
    #
    # Manual accuracy spot-check done -- confirmed a real Devanagari
    # headline crop that gpt-5.6-luna's vision read fabricated text
    # for reads correctly through this path. On by default; set
    # OCR_NATIVE_PADDLE_RETRY=0 to fall back to RapidOCR's own retry
    # only.
    NATIVE_DEVANAGARI_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"

    # ========================================================
    # PER-BLOCK SCRIPT DETECTION
    #
    # Deciding whether a block/line is worth a second opinion from
    # the native Devanagari model, and -- when it is -- deciding
    # which of the two candidate readings to keep. Both are text-only
    # (no extra OCR cost to compute), so they gate the expensive part
    # (an actual native-paddle recognition call) rather than
    # replacing it.
    # ========================================================

    # Devanagari Unicode block.
    _DEVANAGARI_RANGE = re.compile("[ऀ-ॿ]")

    # Characters a genuinely correct reading -- English OR Devanagari
    # -- is expected to consist almost entirely of. The default
    # (English/Latin) recognizer's dictionary has no Devanagari
    # characters in it at all, so when it is pointed at real
    # Devanagari glyphs (matras, conjuncts) it cannot echo them back
    # -- it instead emits stray symbols or mismatched Latin
    # fragments, which show up here as characters outside this set.
    _CLEAN_CHAR_PATTERN = re.compile("[A-Za-z0-9ऀ-ॿ .,'\"-]")

    # A reading whose share of "garbage" (non-clean) characters meets
    # or exceeds this is treated as a script-detection candidate
    # regardless of its reported confidence -- a confidently wrong
    # reading is exactly the failure mode this exists to catch (see
    # the project's CONFIDENCE / RESULT SELECTION requirement: engine
    # confidence values are not comparable across engines/scripts).
    GARBAGE_RATIO_THRESHOLD = 0.30

    # Above this confidence, a normal (non-title) reading is trusted
    # without a second opinion -- close to the ~0.95 average already
    # measured for clean English blocks, so genuinely good English
    # text is left untouched.
    CANDIDATE_CONFIDENCE_THRESHOLD = 0.75

    # Headlines are the highest-priority case: large, clear glyphs
    # can still fool the English recognizer into a confidently wrong
    # Latin reading of real Devanagari text. Title blocks therefore
    # get a stricter (higher) confidence bar before being considered
    # "clean enough, skip the second opinion".
    TITLE_CANDIDATE_CONFIDENCE_THRESHOLD = 0.90

    def __init__(
        self,
        lang="en",
        enable_misclassified_figure_recovery=True,
        enable_low_confidence_retry=True,
    ):

        print(f"Loading RapidOCR (lang={lang})...")

        self.lang = lang

        # Both default to True, i.e. the behaviour this engine has had
        # since these passes were added. English's language pipeline
        # (pipeline/languages/english/__init__.py) turns them off to
        # match the reference "english Pproper" pipeline, whose OCR
        # engine has neither pass. Every other language, and the
        # default engine used for masthead detection, keeps them.
        self._enable_misclassified_figure_recovery = bool(
            enable_misclassified_figure_recovery
        )

        self._enable_low_confidence_retry = bool(
            enable_low_confidence_retry
        )

        # Devanagari-script languages share the same recognition
        # model -- Marathi is written in the same script as Hindi,
        # so it reuses every Hindi-tuned accuracy path in this file
        # (see self._is_devanagari_lang below) instead of getting a
        # separate, unvalidated code path.
        self._is_devanagari_lang = lang in (
            "hi", "hindi", "mr", "marathi",
        )

        # Whole-page contrast normalization (see _normalize_contrast)
        # is a generic scan/photo preprocessing step, not specific to
        # Devanagari, so it is also applied to every other RapidOCR-
        # routed non-Latin language.
        self._apply_page_contrast_normalization = lang in (
            "hi", "hindi", "mr", "marathi", "ta", "tamil",
        )

        if self._is_devanagari_lang:

            # PP-OCRv6 (the package default) has no Devanagari
            # recognition model -- only PP-OCRv4/v5 ship one.
            # Detection/orientation stay on their normal defaults;
            # only the recognition model is swapped, and only for
            # documents already identified as Hindi or Marathi.
            self.reader = RapidOCR(
                params={
                    "Rec.lang_type": LangRec.DEVANAGARI,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )

        elif lang in ("ta", "tamil"):

            # Same PP-OCRv5/mobile family as the Devanagari branch
            # above, swapped to the Tamil recognition model.
            self.reader = RapidOCR(
                params={
                    "Rec.lang_type": LangRec.TA,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )

        # NOTE: RapidOCR does ship real recognition models for Telugu
        # (LangRec.TE, PP-OCRv5/mobile) and Kannada (LangRec.KA,
        # PP-OCRv4/mobile -- its only tier for that script), and both
        # were originally routed here on that basis. A direct side-by-
        # side test against TesseractOCREngine found RapidOCR's shared
        # text detector silently returns zero detected boxes for some
        # real Telugu conjunct clusters (unfixed by lower confidence
        # thresholds, padding, or upscaling) and badly garbles Kannada
        # recognition, while Tesseract read the identical crops
        # correctly. Both languages were moved to TesseractOCREngine
        # (see pipeline/languages/telugu.py and
        # pipeline/languages/kannada.py for the measured comparison)
        # -- this class intentionally has no "te"/"ka" branch any
        # more.

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

        # Language-agnostic: an English-routed document can still
        # contain individual Devanagari blocks (a Hindi quote, a
        # vernacular sidebar) that the whole-document language
        # routing in pipeline_service.py never sees, since that
        # routing picks ONE engine for the entire document. The
        # per-block candidate check (_is_devanagari_candidate) is
        # what keeps this cheap for genuinely English-only documents
        # -- see _start_native_devanagari_preload.
        self._native_retry_enabled = (
            os.getenv(
                "OCR_NATIVE_PADDLE_RETRY",
                "1",
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
    # SCRIPT DETECTION HELPERS
    # ========================================================

    @classmethod
    def _garbage_ratio(cls, text):
        """
        Fraction of a reading's non-space characters that fall
        outside the "clean" set (see _CLEAN_CHAR_PATTERN). High for
        the kind of stray-symbol/mismatched-Latin noise the English
        recognizer produces when pointed at real Devanagari glyphs.
        """

        text = (text or "").strip()

        if not text:
            return 0.0

        stripped = text.replace(" ", "")

        if not stripped:
            return 0.0

        clean = len(
            cls._CLEAN_CHAR_PATTERN.findall(stripped)
        )

        return 1.0 - (clean / len(stripped))

    @classmethod
    def _is_devanagari_candidate(
        cls,
        text,
        confidence,
        is_title=False,
        doc_is_devanagari_script=False,
    ):
        """
        Cheap, text-only pre-filter deciding whether a block/line is
        worth a second opinion from the native Devanagari recognizer.

        Devanagari-routed documents ONLY (Hindi/Marathi --
        `doc_is_devanagari_script` must be true) -- this used to also
        run on English-routed documents, as a "catch a Devanagari
        sidebar hiding in an English page" safety net, but that
        reached further than intended: it returned True
        unconditionally for any empty-text block regardless of
        language, which meant the native model's preload was very
        likely firing on English-only documents too, and its short-
        reading/garbage-ratio checks could also flag genuine English
        bylines or punctuation-heavy fragments, sending them through
        an unnecessary and unvalidated second opinion. Restricted back
        to Devanagari-routed documents, where Devanagari characters in
        the reading are the normal, expected case and this filter's
        confidence-threshold/garbage-ratio signals are what they were
        actually validated against.
        """

        if not doc_is_devanagari_script:
            return False

        text = (text or "").strip()

        if not text:
            return True

        if (
            cls._garbage_ratio(text)
            >= cls.GARBAGE_RATIO_THRESHOLD
        ):
            return True

        threshold = (
            cls.TITLE_CANDIDATE_CONFIDENCE_THRESHOLD
            if is_title
            else cls.CANDIDATE_CONFIDENCE_THRESHOLD
        )

        return (confidence or 0.0) < threshold

    @classmethod
    def _score_reading(cls, text, confidence):
        """
        Multi-factor score used to pick between two candidate
        readings of the same line (the current reading vs. a native-
        Devanagari retry).

        Deliberately NOT a straight confidence comparison: OCR engine
        confidence values are not directly comparable across engines/
        recognizers, and a confidently WRONG Latin misreading of real
        Devanagari text must still lose to a plausible Devanagari
        reading. Genuine Devanagari content and a low garbage ratio
        both add to the score independently of confidence, so a
        readable Devanagari result can outrank a higher-confidence
        but corrupted one.
        """

        text = (text or "").strip()

        if not text:
            return -1.0

        score = float(confidence or 0.0)

        if cls._DEVANAGARI_RANGE.search(text):
            score += 0.5

        score -= cls._garbage_ratio(text)

        # A handful of characters is rarely a genuine full reading of
        # a real text line -- mildly penalize very short outputs so
        # an empty-ish native reading can't win purely on the bonuses
        # above.
        if len(text) <= 2:
            score -= 0.3

        return score

    # ========================================================
    # NATIVE-PADDLE LINE RETRY (Devanagari candidates only)
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
            # Only lines that themselves look like Devanagari
            # candidates -- a block can average below threshold
            # while most of its lines are already fine, and this
            # is the SAME per-line script-detection check used to
            # decide whether the block needed a retry at all (see
            # _is_devanagari_candidate).
            # ------------------------------------------------

            if not self._is_devanagari_candidate(
                line.get("text", ""),
                line_confidence,
                is_title=(
                    getattr(block, "cls", None) == "title"
                ),
                doc_is_devanagari_script=(
                    self._is_devanagari_lang
                ),
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

            # Multi-factor comparison, not a blind confidence
            # comparison -- see _score_reading. A readable Devanagari
            # result should strongly outrank a high-confidence but
            # corrupted Latin/English result.
            if (
                native_text
                and self._score_reading(
                    native_text, native_confidence,
                )
                > self._score_reading(
                    line.get("text", ""), line_confidence,
                )
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
        page_number=None,
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

            time_budget_seconds = (
                self.NATIVE_DEVANAGARI_TIME_BUDGET_SECONDS
                if (
                    self._is_devanagari_lang
                    and self._native_retry_enabled
                )
                else self.DEFAULT_TIME_BUDGET_SECONDS
            )

        # A Hindi-routed document already accepted the native
        # recognizer's cold-load cost before this per-block detection
        # existed, so it keeps the ORIGINAL full overlap window: start
        # the load now, before the full-page pass, so its 17-27s cost
        # overlaps with that pass instead of eating into the fallback
        # loop's own (much smaller) remaining time budget below --
        # confirmed necessary by measurement: on a real 14.9s full-
        # page pass, only ~10s of budget was left afterwards, nowhere
        # near enough for a cold load started only at that point.
        #
        # An English-routed document does NOT get this early start --
        # see the "NATIVE-PADDLE PRELOAD" section after the full-page
        # pass below, which starts it only once a block on THIS page
        # actually looks like a Devanagari candidate. Starting it here
        # unconditionally for every document (not just Hindi-routed
        # ones) would load a ~660MB model on English-only documents
        # that will never use it, violating the "don't initialize
        # PaddleOCR for English-only documents" requirement.
        if self._is_devanagari_lang:
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

        if self._apply_page_contrast_normalization:

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
        # LAYOUT GAP RECOVERY
        #
        # Purely layout-driven -- finds gaps between EXISTING blocks
        # (never from OCR output) that pass a strict absolute-size
        # cap and local-density check, then reads only that small
        # region in its own isolated OCR call (not the busy full
        # page, which can lose a large bold headline exactly the way
        # small print gets lost -- confirmed on a real page). See
        # layout_gap_recovery.py for why this shape specifically.
        #
        # `blocks` (the caller's list) and `processed_blocks` are
        # both mutated in place so every downstream consumer of
        # either -- the rest of this method's own candidate-
        # detection/retry loop below, and the caller's own use of
        # `blocks` after process_blocks returns -- sees the
        # recovered block exactly like any other.
        #
        # Hindi-only: the reference English pipeline has no
        # equivalent recovery step, so English's blocks are left
        # exactly as the layout model produced them.
        # ====================================================

        # What "confident" means for THIS page's script, measured
        # from the blocks the full-page pass already read
        # successfully, so the recovery passes below gate on a floor
        # the engine can actually reach on this script -- see
        # layout_gap_recovery's SCRIPT-RELATIVE CONFIDENCE FLOORS.
        confidence_reference = page_confidence_reference(
            result.confidence
            for result in results.values()
            if result.text.strip()
        )

        # How tall this page's body type and display type actually
        # are, per LINE -- see
        # layout_gap_recovery._classify_recovered_block.
        line_heights = page_line_heights(
            (block.cls, results[block.id].lines)
            for block in blocks
            if block.id in results
        )

        if self._is_devanagari_lang:

            next_id = (
                max((b.id for b in blocks), default=0) + 1
            )

            recovered = recover_missed_blocks(
                processed_blocks,
                image,
                self.reader,
                next_id,
                confidence_reference=confidence_reference,
                line_heights=line_heights,
            )

            for (
                new_block,
                recovered_text,
                recovered_confidence,
                recovered_lines,
            ) in recovered:

                print(
                    "Layout gap recovery: promoted a "
                    f"missed {new_block.cls} block (id="
                    f"{new_block.id}) from a verified layout gap "
                    f"-- {recovered_text[:60]!r}"
                )

                blocks.append(new_block)
                processed_blocks.append(new_block)

                results[new_block.id] = OCRResult(
                    text=recovered_text,
                    confidence=recovered_confidence,
                    x1=new_block.x1,
                    y1=new_block.y1,
                    x2=new_block.x2,
                    y2=new_block.y2,
                    lines=recovered_lines,
                )

        # ====================================================
        # MISCLASSIFIED FIGURE RECOVERY
        #
        # A different failure from the layout-gap recovery above:
        # here the detector DID draw a box, but classified it
        # "figure" (image) when the region is actually text -- see
        # layout_gap_recovery.py's MISCLASSIFIED FIGURE RECOVERY
        # section for the confirmed real-page case and the safety
        # checks that keep a genuine photograph from ever being
        # relabeled. NOT limited to Hindi/Marathi like the gap
        # recovery above -- a layout model calling bold headline
        # text an image is a visual misclassification, not a
        # script-specific OCR problem, so this runs for every
        # document.
        # ====================================================

        reclassified = (
            recover_misclassified_text_blocks(
                blocks,
                image,
                self.reader,
                confidence_reference=confidence_reference,
                page_number=page_number,
                line_heights=line_heights,
            )
            if self._enable_misclassified_figure_recovery
            else []
        )

        for (
            reclassified_block,
            recovered_text,
            recovered_confidence,
            recovered_lines,
        ) in reclassified:

            print(
                "Figure misclassification recovery: relabeled "
                f"block (id={reclassified_block.id}) from "
                f"'figure' to {reclassified_block.cls!r} -- "
                f"{recovered_text[:60]!r}"
            )

            skipped_count -= 1

            processed_blocks.append(reclassified_block)

            results[reclassified_block.id] = OCRResult(
                text=recovered_text,
                confidence=recovered_confidence,
                x1=reclassified_block.x1,
                y1=reclassified_block.y1,
                x2=reclassified_block.x2,
                y2=reclassified_block.y2,
                lines=recovered_lines,
            )

        # ====================================================
        # NATIVE-PADDLE PRELOAD (only if this page needs it)
        #
        # Scanning every block's already-computed text/confidence is
        # free (no OCR call) and tells us whether the fallback loop
        # below is actually going to want the native recognizer. A
        # Hindi-routed document is always assumed to need it (same
        # cost this document type already accepted before this
        # change); an English-routed one only pays the ~660MB/17-27s
        # load cost when a block genuinely looks like a Devanagari
        # candidate. Starting it here (rather than at the very top of
        # process_blocks) means it overlaps with the rest of this
        # fallback loop's own work instead of the full-page pass --
        # a smaller overlap window than before, but the alternative
        # (starting it unconditionally up front) would load the
        # model on every English-only document too.
        # ====================================================

        if self._native_retry_enabled:

            needs_native_preload = (
                self._is_devanagari_lang
                or any(
                    self._is_devanagari_candidate(
                        results[block.id].text,
                        results[block.id].confidence,
                        is_title=(block.cls == "title"),
                        doc_is_devanagari_script=(
                            self._is_devanagari_lang
                        ),
                    )
                    for block in processed_blocks
                )
            )

            if needs_native_preload:

                self._start_native_devanagari_preload()

                # An English-routed page that turns out to have a
                # real Devanagari candidate needs the SAME headroom a
                # Hindi-routed page gets by default -- the native
                # model's 17-27s cold load has no chance to finish (or
                # even meaningfully start) in whatever's left of a
                # 25s budget after the full-page pass already used
                # some of it. This only widens the budget for THIS
                # page, and only once a candidate has actually been
                # found on it -- an all-English page never reaches
                # this branch, so it keeps the original 25s untouched.
                if (
                    time_budget_seconds
                    < self.NATIVE_DEVANAGARI_TIME_BUDGET_SECONDS
                ):

                    print(
                        "Devanagari candidate found -- extending "
                        "OCR time budget from "
                        f"{time_budget_seconds:.0f}s to "
                        f"{self.NATIVE_DEVANAGARI_TIME_BUDGET_SECONDS:.0f}s "
                        "for this page"
                    )

                    time_budget_seconds = (
                        self.NATIVE_DEVANAGARI_TIME_BUDGET_SECONDS
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
            # Devanagari candidate: script-detected via
            # _is_devanagari_candidate -- not just "low confidence on
            # a Hindi-routed document" any more, since an English-
            # routed document can still contain individual Devanagari
            # blocks that whole-document language routing never
            # catches. Unlike an empty result there IS already a
            # candidate reading here, so the retry's output only
            # replaces it further below if it actually scores higher
            # (see _score_reading) -- never blindly.
            #
            # Low confidence (any script/language): a SEPARATE,
            # broader trigger from the Devanagari check above --
            # confirmed on a real Tamil page: a title block ("சென்னை",
            # a dateline) printed only 3px above the next block's
            # first line, with overlapping columns, made RapidOCR's
            # own full-page text DETECTOR draw a corrupted region
            # spanning across both -- 0.552 confidence, recognizable
            # Latin-letter noise mixed into otherwise-perfect Tamil.
            # The Tamil recognition model itself is not at fault (the
            # same reader gets 0.92-0.98 on this exact text once
            # re-cropped in isolation, away from the neighbouring
            # block) -- this is a detection-boundary collision that
            # can happen in ANY script whenever two blocks sit nearly
            # flush against each other, so unlike the Devanagari
            # check this is deliberately NOT gated by language.
            # Deliberately kept OUT of `needs_enhancement` itself: that
            # flag also controls the Devanagari-only contrast
            # enhancement and native-paddle retry paths below, neither
            # of which should ever fire for a non-Devanagari block just
            # because its confidence happens to be low.
            # ------------------------------------------------

            needs_enhancement = (
                bool(current_text)
                and self._is_devanagari_candidate(
                    current_text,
                    current_confidence,
                    is_title=(block.cls == "title"),
                    doc_is_devanagari_script=(
                        self._is_devanagari_lang
                    ),
                )
            )

            is_low_confidence = (
                self._enable_low_confidence_retry
                and bool(current_text)
                and current_confidence < self.LOW_CONFIDENCE_THRESHOLD
            )

            needs_retry = needs_enhancement or is_low_confidence

            if current_text and not needs_retry:

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

                # Bounded wait, not an indefinite block: the
                # background preload may still be mid-cold-load when
                # the first candidate block/line reaches this check.
                # Wait long enough to actually catch it (see
                # NATIVE_LOAD_WAIT_CAP_SECONDS), but leave a small
                # safety margin off the real remaining budget so
                # waiting for the load can't itself consume the
                # entire budget with nothing left to act on the
                # result.
                remaining_budget = (
                    time_budget_seconds
                    - (time.perf_counter() - overall_start)
                )

                wait_timeout = max(
                    0.0,
                    min(
                        remaining_budget - 3.0,
                        self.NATIVE_LOAD_WAIT_CAP_SECONDS,
                    ),
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

                # The empty-result path (current_text falsy) always
                # has nothing to lose, so it always keeps the
                # fallback. Both retry-with-an-existing-reading cases
                # -- the Devanagari-candidate retry AND the low-
                # confidence retry (see needs_retry above) -- DO
                # already have a candidate reading, so their result
                # only replaces it when it actually scores higher
                # (see _score_reading, same multi-factor comparison
                # used for the native-paddle retry) -- a retry is not
                # guaranteed to beat the original on every block.
                #
                # Keyed on `current_text` itself, not `needs_enhancement`:
                # a low-confidence-only retry (needs_enhancement False,
                # needs_retry True) still has a real existing reading to
                # lose, so it must go through the same score comparison,
                # not the "nothing to lose" branch.

                keep_fallback = (
                    not current_text
                    or self._score_reading(
                        fallback_text, fallback_confidence,
                    )
                    > self._score_reading(
                        current_text, current_confidence,
                    )
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