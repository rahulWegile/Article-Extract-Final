import os
import shutil
from pathlib import Path

import cv2
import pytesseract

from pipeline.ocr.block_cropper import BlockCropper
from pipeline.ocr.layout_gap_recovery import (
    page_confidence_reference,
    page_line_heights,
    recover_missed_blocks,
    recover_misclassified_text_blocks,
)
from pipeline.ocr.ocr_models import OCRResult


# ============================================================
# OCR CLASSES
#
# Kept identical to RapidOCREngine/EasyOCREngine's OCR_CLASSES so a
# block is skipped/processed the same way regardless of which engine
# a document's language routes it to.
# ============================================================

OCR_CLASSES = {
    "plain text",
    "title",
    "figure_caption",
}


# ============================================================
# TESSDATA LOCATION
#
# Project-local, not the system Tesseract install's own tessdata
# folder: the *_best (highest-accuracy LSTM) language files are
# downloaded once into models/ (already gitignored -- same convention
# as the YOLO layout weights in this same directory) and referenced
# here via --tessdata-dir. This is what makes the SAME language file
# work identically on a Windows dev machine and the Linux container
# image -- only the tesseract engine binary itself differs per
# platform (installed via winget on Windows, apt on Linux); the
# language data does not.
# ============================================================

TESSDATA_DIR = str(
    Path(__file__).resolve().parents[2] / "models" / "tessdata"
)


class _CropOriginBlock:
    """
    Stand-in block for a crop that is already in its own coordinate
    space. _run_tesseract offsets the line boxes it reports by the
    block's own origin; a caller that only wants the recognized TEXT
    of a bare crop (see TesseractOCREngine._read_line) has no such
    origin to give, and discards those boxes anyway.
    """

    x1 = 0
    y1 = 0
    cls = "plain text"


def _resolve_tesseract_cmd():
    """
    Locate the tesseract binary.

    TESSERACT_CMD wins if set (non-standard installs). Otherwise PATH
    is tried first -- the normal case once `apt-get install
    tesseract-ocr` has run in the container -- and only then the
    default Windows winget/UB-Mannheim install path, since a fresh
    Windows install does not add tesseract to PATH automatically.
    """

    env_path = os.getenv("TESSERACT_CMD")

    if env_path:
        return env_path

    on_path = (
        shutil.which("tesseract")
        or shutil.which("tesseract.exe")
    )

    if on_path:
        return on_path

    default_windows_path = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if os.path.exists(default_windows_path):
        return default_windows_path

    # Fall through to the bare command name -- lets a clear
    # "tesseract is not installed" error surface at call time instead
    # of construction time, matching how a missing PATH entry would
    # normally fail.
    return "tesseract"


pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd()


# ============================================================
# TESSERACT OCR ENGINE
# ============================================================

class TesseractOCREngine:
    """
    Tesseract-backed OCR engine.

    Used ONLY for languages that have no recognition model in either
    RapidOCR or EasyOCR -- currently Gujarati, Malayalam, Urdu,
    Punjabi, Bengali, Assamese, and Odia (see each language's module
    under pipeline/languages/ for the per-language routing and why
    each one lands here instead of RapidOCR). Loads
    the tessdata_best (highest-accuracy) LSTM model from TESSDATA_DIR
    for whichever `lang` is requested, not the faster/lower-accuracy
    default data bundled with the Tesseract installer itself.
    """

    def __init__(self, lang="guj"):

        self.lang = lang

        self.cropper = BlockCropper()

        traineddata_path = (
            Path(TESSDATA_DIR) / f"{lang}.traineddata"
        )

        if not traineddata_path.exists():

            raise RuntimeError(
                f"Tesseract language data not found: "
                f"{traineddata_path}. Download it from "
                f"https://github.com/tesseract-ocr/tessdata_best "
                f"into {TESSDATA_DIR}."
            )

        # Fail fast, at construction time, if the tesseract binary
        # itself is not actually runnable -- otherwise the first
        # failure would surface deep inside process_blocks on
        # whatever page happens to contain the first OCR-eligible
        # block, with a less obvious pytesseract stack trace instead
        # of a clear, immediate cause.
        try:

            pytesseract.get_tesseract_version()

        except Exception as exc:

            raise RuntimeError(
                "Tesseract executable not found or not runnable "
                f"(tesseract_cmd={pytesseract.pytesseract.tesseract_cmd!r}). "
                "Install Tesseract OCR (winget install "
                "tesseract-ocr.tesseract on Windows, `apt-get install "
                "tesseract-ocr` on Linux) or set TESSERACT_CMD to its "
                f"path. Original error: {exc}"
            ) from exc

        print(
            f"Loading Tesseract OCR (lang={lang}, "
            f"tessdata={TESSDATA_DIR})..."
        )

        print("Tesseract OCR loaded")

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
    # TESSERACT CONFIG STRING
    # ========================================================

    @staticmethod
    def _build_config(psm, oem=None):

        parts = [
            f'--tessdata-dir "{TESSDATA_DIR}"',
            f"--psm {psm}",
        ]

        if oem is not None:
            parts.append(f"--oem {oem}")

        return " ".join(parts)

    # ========================================================
    # OCR ONE BLOCK
    #
    # Multi-line "plain text" blocks use PSM 6 ("assume a single
    # uniform block of text"), Tesseract's own recommended mode for
    # an already-segmented block/paragraph crop, with a PSM 13 rescue
    # on a genuinely empty result -- see _rescue_psm below.
    #
    # TITLE/CAPTION CROPS -- why these race two PSMs
    # ------------------------------------------------
    #
    # These used to be handed PSM 7 ("single LINE") outright, on the
    # grounds that PSM 6 mis-splits a genuinely single-line RTL (Urdu)
    # crop into garbled fragments while PSM 7 reads it perfectly.
    # That holds only for crops that really are one line. A newspaper
    # display headline is routinely two or three, and confirmed
    # directly on real Gujarati (દિવ્ય ભાસ્કર) pages: PSM 7 returns
    # ZERO-to-ONE characters for a multi-line headline the layout
    # detector found perfectly well, where PSM 6 reads the same crop
    # in full --
    #
    #     psm 7: 'વ'
    #     psm 6: 'જજ બનીને ફોન ક્યો, સુકેશને 8 વર્ષની જેલ'
    #
    # A 0-2 character reading is then discarded by
    # PageCleaner.remove_noise, so the headline BLOCK disappeared off
    # the page entirely -- taking the article's boundary with it,
    # since the grouper never saw a title to start the article at.
    # Measured over 109 title/caption crops on one such document,
    # PSM 7 produced <=2 characters on 40 of them.
    #
    # Deciding single- vs multi-line from the crop's own geometry was
    # tried first and rejected: a row-wise ink projection
    # (layout_gap_recovery._split_into_line_bands) reports ONE band
    # for most of these headlines, because tightly-leaded bold
    # Gujarati lines leave no fully blank row between them. The
    # ink projection cannot tell them apart, so the reading itself is
    # what gets compared instead.
    #
    # So both are run and scored against each other (_reading_score).
    # PSM 7 stays the trusted default and is only displaced when
    # PSM 6 beats it by PSM_SWITCH_MARGIN -- deliberately a wide
    # margin, not a simple "higher score wins". On the same 109 crops
    # the two outcomes separate cleanly: every genuine PSM 6 win came
    # in at 7.9x-73.6x, while the ONE case where PSM 6 was worse (it
    # segmented a single line twice and repeated it) sat at 2.0x.
    # Net effect measured on that document: 33 blocks recovered from
    # deletion, 3 partial reads completed, 0 regressions.
    # ========================================================

    # Below this many characters, a reading is not a real block of
    # text -- it is what Tesseract returns when its layout analysis
    # rejected the crop outright. Deliberately the same threshold
    # PageCleaner.remove_noise discards a block at: a reading this
    # short is going to be thrown away downstream regardless, so
    # another pass over the crop costs nothing that was not already
    # lost.
    MIN_PLAUSIBLE_CHARS = 2

    # How much better PSM 6 must score before it displaces PSM 7 on a
    # title/caption crop. See the derivation above.
    PSM_SWITCH_MARGIN = 3.0

    @staticmethod
    def _reading_score(reading):
        """
        Rank two readings of the SAME crop against each other.

        Confidence alone is not enough -- Tesseract reports high
        confidence on the single character it salvages from a crop it
        otherwise failed to segment. Length alone is not enough
        either -- a garbled over-segmentation is long. Their product
        separates both failures from a real reading, and is only ever
        compared against another reading of the same crop, never used
        as an absolute quality measure.
        """

        text, confidence, _lines = reading

        return confidence * len(" ".join(text.split()))

    def _is_implausible(self, reading):

        text, _confidence, _lines = reading

        return (
            len(" ".join(text.split()))
            <= self.MIN_PLAUSIBLE_CHARS
        )

    # --------------------------------------------------------
    # RESCUE PASS
    #
    # Verified directly that Tesseract's legacy line-finder can
    # reject an entire Gurmukhi (Punjabi) crop as non-text regardless
    # of PSM 3/4/6/7 or OEM -- the script's thick, continuous
    # headline appears to be misread as a ruling line during layout
    # analysis, discarding the block before recognition ever runs.
    # PSM 13 ("raw line", LSTM engine only) bypasses that layout
    # analysis entirely and reads the exact same crop correctly.
    #
    # It only ever runs when every pass before it came back
    # implausibly short, so it can only turn "nothing" into
    # "something" -- it never overrides a pass that already found
    # text, and cannot regress the languages those already read
    # correctly.
    # --------------------------------------------------------

    def _rescue_psm(self, crop, block):

        return self._run_tesseract(
            crop, block, psm="13", oem="1",
        )

    def _ocr_block(self, crop, block):

        if block.cls not in ("title", "figure_caption"):

            reading = self._run_tesseract(
                crop, block, psm="6",
            )

            if self._is_implausible(reading):
                reading = self._rescue_psm(crop, block)

            return reading

        single_line = self._run_tesseract(
            crop, block, psm="7",
        )

        uniform_block = self._run_tesseract(
            crop, block, psm="6",
        )

        if (
            self._reading_score(uniform_block)
            > self.PSM_SWITCH_MARGIN
            * self._reading_score(single_line)
        ):
            best = uniform_block
        else:
            best = single_line

        if self._is_implausible(best):

            rescued = self._rescue_psm(crop, block)

            if (
                self._reading_score(rescued)
                > self._reading_score(best)
            ):
                best = rescued

        return best

    def _run_tesseract(self, crop, block, psm, oem=None):

        data = pytesseract.image_to_data(
            crop,
            lang=self.lang,
            config=self._build_config(psm, oem=oem),
            output_type=pytesseract.Output.DICT,
        )

        # ----------------------------------------------------
        # Group words back into lines using Tesseract's own
        # block/paragraph/line numbering, the same granularity
        # RapidOCR/EasyOCR report lines at.
        # ----------------------------------------------------

        lines_by_key = {}

        word_count = len(data.get("text", []))

        for i in range(word_count):

            word_text = (data["text"][i] or "").strip()

            if not word_text:
                continue

            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0

            # Tesseract reports -1 confidence for non-text layout
            # elements it still enumerates (e.g. block-level rows).
            if conf < 0:
                continue

            line_key = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            entry = lines_by_key.setdefault(
                line_key,
                {
                    "words": [],
                    "confidences": [],
                    "x1": [],
                    "y1": [],
                    "x2": [],
                    "y2": [],
                },
            )

            entry["words"].append(word_text)
            entry["confidences"].append(conf / 100.0)

            left = data["left"][i]
            top = data["top"][i]

            entry["x1"].append(left)
            entry["y1"].append(top)
            entry["x2"].append(left + data["width"][i])
            entry["y2"].append(top + data["height"][i])

        text_parts = []
        confidences = []
        lines = []

        for entry in lines_by_key.values():

            line_text = " ".join(entry["words"])

            if not line_text:
                continue

            line_confidence = (
                sum(entry["confidences"])
                / len(entry["confidences"])
            )

            text_parts.append(line_text)
            confidences.append(line_confidence)

            lines.append(
                {
                    "text": line_text,
                    "confidence": line_confidence,
                    "bbox": {
                        "x1": round(block.x1 + min(entry["x1"]), 2),
                        "y1": round(block.y1 + min(entry["y1"]), 2),
                        "x2": round(block.x1 + max(entry["x2"]), 2),
                        "y2": round(block.y1 + max(entry["y2"]), 2),
                    },
                }
            )

        text = "\n".join(text_parts)

        confidence = (
            sum(confidences) / len(confidences)
            if confidences
            else 0.0
        )

        return text, confidence, lines

    # ========================================================
    # PROCESS LAYOUT BLOCKS
    # ========================================================

    # ========================================================
    # RAPIDOCR-SHAPED LINE READER
    #
    # layout_gap_recovery calls its `reader` the way RapidOCR is
    # called -- `reader(crop, use_det=False, use_cls=False,
    # use_rec=True)` returning an object with `.txts` and `.scores`
    # -- because it deliberately BYPASSES text detection: it has
    # already isolated one printed line by ink projection and only
    # wants that line recognized.
    #
    # This adapter lets the same recovery run on Tesseract, so the
    # nine Tesseract-routed languages get it too. PSM 7 ("single
    # line") is the right STARTING point here and carries none of
    # the risk it does in _ocr_block above: the caller has already
    # split the crop into individual line-bands, so the input
    # genuinely IS one line.
    #
    # WHY THIS FALLS BACK THROUGH PSM 6 AND 13
    # -----------------------------------------
    #
    # PSM 7 alone is not enough, because "the input is one line"
    # does not guarantee PSM 7 can read it -- the same
    # zero-to-one-character failure documented for title crops in
    # _ocr_block happens here too, and here it is worse: a band that
    # reads back empty is DROPPED, so the recovery pass sees fewer
    # lines than the ink projection found and rejects the whole
    # block on its min-lines check, never reaching the OCR-quality
    # checks at all.
    #
    # Confirmed on a real Urdu (THE INQUILAB, Nastaliq) page: the
    # banner headlines detected as "figure" at 1037x121 (aspect 8.6)
    # and 765x82 (aspect 9.3) each projected to exactly one ink band,
    # PSM 7 returned NOTHING for that band, and both were rejected
    # as "0 lines recovered" -- leaving the stories beneath them
    # grouped with no headline. Two more (769x141, and 496x224 on
    # the facing page) found two bands and lost one of them the same
    # way.
    #
    # So the same escalation _ocr_block already uses is applied per
    # line: PSM 7, then PSM 6, then the PSM 13 raw-line rescue. It
    # can only ever turn a line that read back as NOTHING into one
    # that read back as something -- a plausible PSM 7 reading is
    # returned immediately and never re-raced, so no language whose
    # recovery already works can regress.
    # ========================================================

    class _LineResult:

        __slots__ = ("txts", "scores")

        def __init__(self, txts, scores):
            self.txts = txts
            self.scores = scores

    def _read_line(self, line_crop, use_det=True, use_cls=True, use_rec=True):

        best = self._run_tesseract(
            line_crop,
            _CropOriginBlock(),
            psm="7",
        )

        if self._is_implausible(best):

            for psm, oem in (("6", None), ("13", "1")):

                candidate = self._run_tesseract(
                    line_crop,
                    _CropOriginBlock(),
                    psm=psm,
                    oem=oem,
                )

                if (
                    self._reading_score(candidate)
                    > self._reading_score(best)
                ):
                    best = candidate

                if not self._is_implausible(best):
                    break

        text, confidence, _lines = best

        text = " ".join(text.split())

        if not text:
            return None

        return self._LineResult([text], [confidence])

    def process_blocks(self, image_path, blocks, page_number=None):

        image = cv2.imread(str(image_path))

        if image is None:

            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        results = []

        processed_count = 0
        skipped_count = 0

        for block in blocks:

            if block.cls not in OCR_CLASSES:

                skipped_count += 1

                results.append(self._empty_result(block))

                continue

            crop = self.cropper.crop(image, block)

            if crop.size == 0:

                results.append(self._empty_result(block))

                continue

            processed_count += 1

            text, confidence, lines = self._ocr_block(
                crop, block
            )

            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,

                    x1=block.x1,
                    y1=block.y1,
                    x2=block.x2,
                    y2=block.y2,

                    lines=lines,
                )
            )

        # What "confident" means for THIS page's script, measured
        # from the blocks the pass above already read successfully.
        # Tesseract reports 0.3-0.6 on a perfectly correct Nastaliq
        # (Urdu) reading where it reports 0.85+ on Gujarati, so a
        # single absolute floor inside either recovery pass below is
        # unreachable for one and generous for the other -- see
        # layout_gap_recovery's SCRIPT-RELATIVE CONFIDENCE FLOORS.
        confidence_reference = page_confidence_reference(
            result.confidence
            for result in results
            if result.text.strip()
        )

        # How tall this page's body type and display type actually
        # are, per LINE -- what a recovered region's own line height
        # gets compared against to decide title vs body. Measured
        # from the line boxes the pass above already reported; see
        # layout_gap_recovery._classify_recovered_block for why a
        # per-block height cannot substitute for this.
        line_heights = page_line_heights(
            (block.cls, result.lines)
            for block, result in zip(blocks, results)
        )

        # ====================================================
        # LAYOUT GAP RECOVERY
        #
        # A different failure from the one below: the layout
        # detector drew NO box at all here -- not even a
        # misclassified "figure" -- so there is nothing in `blocks`
        # to relabel. Finds gaps between EXISTING blocks that pass a
        # strict absolute-size cap and local-density check (see
        # layout_gap_recovery.py's own docstring for why it is
        # shaped this way), then reads only that small region in its
        # own isolated OCR call.
        #
        # This is the same pass rapidocr_engine.py already runs, but
        # ONLY for Devanagari there -- it was entirely unreachable
        # from every Tesseract-routed language. Confirmed on a real
        # Urdu (THE INQUILAB) page: a 166px gap between two existing
        # blocks, in a column with a page-median gap of ~6px, sat
        # directly over a real bold headline ("دردھلواروں کاٹرک
        # مرمت...") the layout detector never boxed at all -- the
        # 900px, 12-block story beneath it was then grouped with no
        # headline whatsoever.
        # ====================================================

        next_id = (
            max((b.id for b in blocks), default=0) + 1
        )

        for (
            new_block,
            recovered_text,
            recovered_confidence,
            recovered_lines,
        ) in recover_missed_blocks(
            blocks,
            image,
            self._read_line,
            next_id,
            confidence_reference=confidence_reference,
            line_heights=line_heights,
        ):

            print(
                "Layout gap recovery: promoted a missed "
                f"{new_block.cls} block (id={new_block.id}) from "
                f"a verified layout gap -- {recovered_text[:60]!r}"
            )

            blocks.append(new_block)

            processed_count += 1

            results.append(
                OCRResult(
                    text=recovered_text,
                    confidence=recovered_confidence,
                    x1=new_block.x1,
                    y1=new_block.y1,
                    x2=new_block.x2,
                    y2=new_block.y2,
                    lines=recovered_lines,
                )
            )

        # ====================================================
        # MISCLASSIFIED FIGURE RECOVERY
        #
        # The layout detector DID draw a box, but called it "figure"
        # (an image) when the region is really text -- so it is not
        # in OCR_CLASSES, is never read, never becomes a title, and
        # the story beneath it ends up with no headline at all.
        #
        # Confirmed on real Bengali (উত্তরবঙ্গ সংবাদ) pages: the
        # banner headline "ছুটির খবর দিতে ২৮ কিলোমিটার পাড়ি
        # শিক্ষকের" and the two-line "ইস্টবেঙ্গলে সই করলেন রেনিয়ার"
        # were both detected as figures and silently dropped.
        #
        # This is the same pass rapidocr_engine already runs; it was
        # simply unreachable from here, leaving all nine
        # Tesseract-routed languages without it. Every safety check
        # (size floor, line-band pre-filter, confidence and character
        # floors, top-of-page masthead guard) lives in
        # layout_gap_recovery and is shared verbatim -- only the
        # line recognizer differs, via _read_line above.
        # ====================================================

        index_by_block_id = {
            block.id: index
            for index, block in enumerate(blocks)
        }

        for (
            reclassified_block,
            recovered_text,
            recovered_confidence,
            recovered_lines,
        ) in recover_misclassified_text_blocks(
            blocks,
            image,
            self._read_line,
            confidence_reference=confidence_reference,
            page_number=page_number,
            line_heights=line_heights,
        ):

            index = index_by_block_id.get(reclassified_block.id)

            if index is None:
                continue

            print(
                "Figure misclassification recovery: relabeled "
                f"block (id={reclassified_block.id}) from "
                f"'figure' to {reclassified_block.cls!r} -- "
                f"{recovered_text[:60]!r}"
            )

            skipped_count -= 1
            processed_count += 1

            results[index] = OCRResult(
                text=recovered_text,
                confidence=recovered_confidence,
                x1=reclassified_block.x1,
                y1=reclassified_block.y1,
                x2=reclassified_block.x2,
                y2=reclassified_block.y2,
                lines=recovered_lines,
            )

        non_empty = sum(
            1 for result in results if result.text.strip()
        )

        print()
        print("=" * 60)
        print("TESSERACT OCR SUMMARY")
        print("=" * 60)

        print(f"Total blocks      : {len(blocks)}")
        print(f"OCR processed     : {processed_count}")
        print(f"OCR skipped       : {skipped_count}")
        print(f"Non-empty results : {non_empty}")

        print("=" * 60)

        return results
