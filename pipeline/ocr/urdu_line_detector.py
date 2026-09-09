"""
Per-crop Urdu text-LINE detector.

Vendors the detection half of abdur75648/urdu-text-detection (a
YOLOv8m fine-tuned on the UrduDoc dataset, CC BY-NC-4.0) for a
different job than that repo's own CLI script does: there, it runs
once over an entire page at a fixed imgsz=1280 to find every text
region on the page from scratch. Here, DocLayout-YOLO (see
pipeline/layout_detector.py) has already found and classified each
page's blocks -- this only needs to split ONE already-isolated
"plain text"/"title"/"figure_caption" block crop into its individual
printed LINES, since UTRNet (pipeline/ocr/utrnet_engine.py) is a
recognition-only model that expects one line per input image.

imgsz is derived from the CROP's own size, not fixed at the original
script's page-level 1280 -- a small block crop run at 1280 upscales
it far past its native resolution for no benefit, and a large block
crop run at a too-small fixed size can lose thin line gaps. Rounded
to a multiple of 32 (this model's stride) since ultralytics silently
does that anyway; rounding explicitly here avoids its warning on
every single call.
"""

from pathlib import Path

from ultralytics import YOLO

from pipeline.device import resolve_device


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "urdu_line_detector"
    / "yolov8m_UrduDoc.pt"
)

DEFAULT_CONFIDENCE = 0.15

STRIDE = 32
MIN_IMGSZ = 320
MAX_IMGSZ = 1280


def _round_imgsz(size):

    size = max(MIN_IMGSZ, min(MAX_IMGSZ, size))

    return int(round(size / STRIDE) * STRIDE)


class UrduLineDetector:

    def __init__(
        self,
        model_path=None,
        confidence=DEFAULT_CONFIDENCE,
        device=None,
    ):

        self.model_path = Path(model_path or MODEL_PATH)

        if not self.model_path.exists():

            raise RuntimeError(
                f"Urdu line detector weights not found: "
                f"{self.model_path}. Copy yolov8m_UrduDoc.pt "
                f"(abdur75648/urdu-text-detection, CC BY-NC-4.0) "
                f"into {self.model_path.parent}."
            )

        self.confidence = confidence

        self.device = resolve_device(device)

        print(
            f"Loading Urdu line detector "
            f"(device={self.device})..."
        )

        self.model = YOLO(str(self.model_path))

        print("Urdu line detector loaded")

    def detect_lines(self, crop):
        """
        crop: a BGR numpy array already isolated to one layout
        block (from BlockCropper).

        Returns a list of (x1, y1, x2, y2) line boxes in the crop's
        OWN coordinate space, sorted top-to-bottom -- the caller is
        responsible for translating them back to page coordinates
        (see UTRNetOCREngine, which owns the block's own x1/y1
        offset). Empty list if the crop is degenerate or the model
        finds nothing.
        """

        height, width = crop.shape[:2]

        if height <= 0 or width <= 0:
            return []

        imgsz = _round_imgsz(max(height, width))

        results = self.model.predict(
            source=crop,
            conf=self.confidence,
            imgsz=imgsz,
            # Ultralytics defaults max_det to 300 -- fine for a
            # normal paragraph/headline block, but a layout
            # misclassification that merges a whole dense column
            # into one "plain text" block could plausibly exceed
            # that and silently drop lines past the 300th. Raised
            # to match the same fix applied upstream (colleague's
            # urdu-text-detection, which runs this model per whole
            # PAGE rather than per block and hits this far more
            # easily).
            max_det=1000,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()

        line_boxes = [
            (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(width, int(round(x2))),
                min(height, int(round(y2))),
            )
            for x1, y1, x2, y2 in boxes
        ]

        line_boxes = [
            box for box in line_boxes
            if box[2] > box[0] and box[3] > box[1]
        ]

        line_boxes.sort(key=lambda box: (box[1], box[0]))

        return line_boxes
