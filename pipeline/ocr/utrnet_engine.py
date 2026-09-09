import math
from pathlib import Path

import cv2
import torch
from PIL import Image

from pipeline.device import resolve_device
from pipeline.ocr.block_cropper import BlockCropper
from pipeline.ocr.layout_gap_recovery import (
    page_confidence_reference,
    page_line_heights,
    recover_missed_blocks,
    recover_misclassified_text_blocks,
)
from pipeline.ocr.ocr_models import OCRResult
from pipeline.ocr.urdu_line_detector import UrduLineDetector
from pipeline.ocr.utrnet.model import Model
from pipeline.ocr.utrnet.utils import CTCLabelConverter, NormalizePAD


# ============================================================
# OCR CLASSES
#
# Kept identical to every other engine's OCR_CLASSES so a block is
# skipped/processed the same way regardless of which engine a
# document's language routes it to.
# ============================================================

OCR_CLASSES = {
    "plain text",
    "title",
    "figure_caption",
}


# ============================================================
# WEIGHTS / VOCAB LOCATIONS
#
# Project-local and gitignored, same convention TESSDATA_DIR uses in
# tesseract_engine.py -- downloaded/copied once, never committed
# (both files are 50-190MB and CC BY-NC-4.0 licensed). UrduGlyphs.txt
# is small and license-compatible with this vendored copy, so it
# ships in the repo instead.
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "utrnet"
    / "UTRNet-Large.pth"
)

VOCAB_PATH = Path(__file__).resolve().parent / "utrnet" / "UrduGlyphs.txt"

BATCH_SIZE = 20

IMG_H = 32
IMG_W = 400


# ============================================================
# UTRNET RECOGNIZER
#
# Recognition-only -- no detector of its own (unlike RapidOCR/
# Tesseract, which both segment lines internally). UTRNetOCREngine
# below is what supplies the line-level crops, via UrduLineDetector
# for whole blocks and layout_gap_recovery's own ink-projection line
# bands for the two recovery passes.
# ============================================================

class UTRNetRecognizer:

    def __init__(self, model_path, vocabulary_path, device=None):

        self.device = torch.device(resolve_device(device))

        with open(vocabulary_path, "r", encoding="utf-8") as file:
            content = file.readlines()

        # Matches the published UTRNet-Large checkpoint's own
        # vocabulary construction exactly (trailing space appended
        # as its own class) -- a mismatched vocabulary here would
        # silently shift every class index against the trained
        # weights.
        characters = "".join(
            line.strip("\n") for line in content
        ) + " "

        self.converter = CTCLabelConverter(characters)

        num_class = len(self.converter.character)

        class Opt:
            pass

        opt = Opt()

        opt.input_channel = 1
        opt.output_channel = 32
        opt.hidden_size = 256
        opt.num_class = num_class
        opt.device = self.device

        print(f"UTRNet device: {self.device}, num_class: {num_class}")

        self.model = Model(opt)

        checkpoint = torch.load(
            model_path, map_location=self.device,
        )

        self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.transform = NormalizePAD((1, IMG_H, IMG_W))

    def _prepare(self, crop_bgr):

        image = Image.fromarray(
            cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        ).convert("L")

        # UTRNet-Large was trained on horizontally-flipped Urdu line
        # images (an RTL-specific convention baked into the
        # published weights) -- this must be reproduced exactly, not
        # just "some" preprocessing choice.
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        width, height = image.size

        ratio = width / float(height)

        resized_w = min(
            IMG_W, max(1, math.ceil(IMG_H * ratio)),
        )

        image = image.resize(
            (resized_w, IMG_H), Image.Resampling.BICUBIC,
        )

        return self.transform(image)

    @torch.no_grad()
    def recognize_batch(self, crops):
        """
        crops: list of BGR numpy arrays, each already isolated to
        ONE printed line.

        Returns a list of (text, confidence) pairs, same length/
        order as `crops`. `confidence` is the mean softmax
        probability of the timesteps CTCLabelConverter.decode keeps
        (non-blank, non-repeated) -- not a value the upstream model
        reports on its own; the raw Linear/CTC head has no
        confidence output. Computed with the SAME
        blank/repeat-collapse rule decode() itself uses, so it
        reflects the confidence of the characters actually returned,
        not the whole raw timestep sequence.
        """

        if not crops:
            return []

        tensors = [self._prepare(crop) for crop in crops]

        batch = torch.stack(tensors, dim=0).to(self.device)

        preds = self.model(batch)

        probs = torch.softmax(preds, dim=2)

        preds_size = torch.IntTensor(
            [preds.size(1)] * len(crops)
        )

        confidence_values, preds_index = probs.max(2)

        texts = self.converter.decode(preds_index, preds_size)

        results = []

        for i in range(len(crops)):

            length = int(preds_size[i])

            indices = preds_index[i]
            confidences = confidence_values[i]

            kept = [
                float(confidences[j])
                for j in range(length)
                if indices[j] != 0
                and not (j > 0 and indices[j - 1] == indices[j])
            ]

            confidence = sum(kept) / len(kept) if kept else 0.0

            results.append((texts[i], confidence))

        return results


# ============================================================
# UTRNET OCR ENGINE
# ============================================================

class UTRNetOCREngine:
    """
    UTRNet-backed OCR engine for Urdu (Nastaliq/Perso-Arabic).

    Structured like TesseractOCREngine (crop-then-recognize, no
    built-in full-page pass), not RapidOCREngine -- UTRNet has no
    detector of its own, so UrduLineDetector supplies the per-block
    line segmentation Tesseract/RapidOCR get from their own engines.
    """

    class _LineResult:

        __slots__ = ("txts", "scores")

        def __init__(self, txts, scores):
            self.txts = txts
            self.scores = scores

    def __init__(
        self,
        lang="ur",
        model_path=None,
        vocabulary_path=None,
        device=None,
    ):

        self.lang = lang

        self.cropper = BlockCropper()

        model_path = Path(model_path or MODEL_PATH)
        vocabulary_path = Path(vocabulary_path or VOCAB_PATH)

        if not model_path.exists():

            raise RuntimeError(
                f"UTRNet weights not found: {model_path}. Copy "
                f"UTRNet-Large's best_norm_ED.pth "
                f"(abdur75648/UTRNet-High-Resolution-Urdu-Text-"
                f"Recognition, CC BY-NC-4.0) into "
                f"{model_path.parent}."
            )

        if not vocabulary_path.exists():

            raise RuntimeError(
                f"UTRNet vocabulary not found: {vocabulary_path}."
            )

        print(f"Loading UTRNet OCR (lang={lang})...")

        self.recognizer = UTRNetRecognizer(
            model_path, vocabulary_path, device=device,
        )

        self.line_detector = UrduLineDetector(device=device)

        print("UTRNet OCR loaded")

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
    # LINE RECOGNITION
    # ========================================================

    def _recognize_lines(self, crop, line_boxes, origin_x, origin_y):
        """
        line_boxes: (x1, y1, x2, y2) in `crop`'s own coordinate
        space. Returns line dicts (shared OCRResult.lines shape)
        with bbox translated into PAGE coordinates via origin_x/
        origin_y, sorted top-to-bottom.
        """

        crops = []
        kept_boxes = []

        for (x1, y1, x2, y2) in line_boxes:

            line_crop = crop[y1:y2, x1:x2]

            if line_crop.size == 0:
                continue

            crops.append(line_crop)
            kept_boxes.append((x1, y1, x2, y2))

        lines = []

        for batch_start in range(0, len(crops), BATCH_SIZE):

            batch_crops = crops[batch_start:batch_start + BATCH_SIZE]
            batch_boxes = kept_boxes[batch_start:batch_start + BATCH_SIZE]

            for (text, confidence), (x1, y1, x2, y2) in zip(
                self.recognizer.recognize_batch(batch_crops),
                batch_boxes,
            ):

                text = text.strip()

                if not text:
                    continue

                lines.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": {
                            "x1": round(origin_x + x1, 2),
                            "y1": round(origin_y + y1, 2),
                            "x2": round(origin_x + x2, 2),
                            "y2": round(origin_y + y2, 2),
                        },
                    }
                )

        lines.sort(
            key=lambda item: (item["bbox"]["y1"], item["bbox"]["x1"])
        )

        return lines

    def _ocr_block(self, image, block):

        crop = self.cropper.crop(image, block)

        if crop.size == 0:
            return "", 0.0, []

        line_boxes = self.line_detector.detect_lines(crop)

        # No line detected at all -- fall back to the whole crop as
        # one line rather than silently dropping the block, matching
        # the "a missed detection must not delete real content"
        # principle every other engine's recovery passes follow.
        if not line_boxes:
            height, width = crop.shape[:2]
            line_boxes = [(0, 0, width, height)]

        lines = self._recognize_lines(
            crop, line_boxes, block.x1, block.y1,
        )

        text = "\n".join(line["text"] for line in lines)

        confidence = (
            sum(line["confidence"] for line in lines) / len(lines)
            if lines
            else 0.0
        )

        return text, confidence, lines

    # ========================================================
    # RAPIDOCR-SHAPED LINE READER
    #
    # Lets layout_gap_recovery's two recovery passes (a missed block
    # the layout detector never boxed at all, and a real text block
    # misclassified as "figure") run against UTRNet -- see
    # TesseractOCREngine._read_line for why this exact
    # reader(crop, use_det=..., use_cls=..., use_rec=...) shape.
    # Unlike _ocr_block above, the caller here has ALREADY isolated
    # one printed line via ink-projection band-splitting, so no line
    # detection step is needed -- straight to recognition.
    # ========================================================

    def _read_line(self, line_crop, use_det=True, use_cls=True, use_rec=True):

        recognized = self.recognizer.recognize_batch([line_crop])

        if not recognized:
            return None

        text, confidence = recognized[0]

        text = text.strip()

        if not text:
            return None

        return self._LineResult([text], [confidence])

    # ========================================================
    # PROCESS LAYOUT BLOCKS
    # ========================================================

    def process_blocks(self, image_path, blocks, page_number=None, time_budget_seconds=None):

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

            processed_count += 1

            text, confidence, lines = self._ocr_block(image, block)

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

        # What "confident" means for THIS page's script -- UTRNet's
        # CTC-softmax confidence proxy is not on the same scale as
        # Tesseract's or RapidOCR's own reported confidence, so the
        # recovery passes below gate on a floor measured from this
        # page's own successfully-read blocks rather than an
        # absolute number tuned for a different engine. See
        # layout_gap_recovery's SCRIPT-RELATIVE CONFIDENCE FLOORS.
        confidence_reference = page_confidence_reference(
            result.confidence
            for result in results
            if result.text.strip()
        )

        line_heights = page_line_heights(
            (block.cls, result.lines)
            for block, result in zip(blocks, results)
        )

        # ====================================================
        # LAYOUT GAP RECOVERY -- a gap between existing blocks the
        # layout detector never boxed at all.
        # ====================================================

        next_id = max((b.id for b in blocks), default=0) + 1

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
        # MISCLASSIFIED FIGURE RECOVERY -- the detector drew a box
        # but called it "figure" when the region is really text.
        # ====================================================

        index_by_block_id = {
            block.id: index for index, block in enumerate(blocks)
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
        print("UTRNET OCR SUMMARY")
        print("=" * 60)

        print(f"Total blocks      : {len(blocks)}")
        print(f"OCR processed     : {processed_count}")
        print(f"OCR skipped       : {skipped_count}")
        print(f"Non-empty results : {non_empty}")

        print("=" * 60)

        return results

    # ========================================================
    # PROCESS FINAL ARTICLE CROP
    #
    # Same contract as RapidOCREngine/TesseractOCREngine's
    # process_article_crop -- LocalArticleExtractor's fallback OCR
    # path for a final article crop.
    # ========================================================

    def process_article_crop(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:

            raise FileNotFoundError(
                f"Unable to read article crop: {image_path}"
            )

        line_boxes = self.line_detector.detect_lines(image)

        if not line_boxes:
            height, width = image.shape[:2]
            line_boxes = [(0, 0, width, height)]

        lines = self._recognize_lines(image, line_boxes, 0, 0)

        text = "\n".join(line["text"] for line in lines)

        confidence = (
            sum(line["confidence"] for line in lines) / len(lines)
            if lines
            else 0.0
        )

        return {
            "text": text,
            "confidence": confidence,
            "lines": lines,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "line_count": len(lines),
        }
