import threading

from doclayout_yolo import YOLOv10

from pipeline.device import resolve_device


# Default confidence floor for DocLayout-YOLO's own predict() call.
# Confirmed too high for dense Nastaliq (Urdu) newspaper pages on a
# real document (THE INQUILAB, doc_000172 page 4): only 20 blocks
# detected for a full dense page at 0.20, dropping dozens of real
# articles (including an entire column) entirely invisible to the
# rest of the pipeline; re-running the SAME page at 0.08 immediately
# recovered 50+ blocks. Kept as the default for every other script,
# which this same page-4 test did not regress -- see
# pipeline/languages/base.py's LanguagePipeline.layout_confidence for
# the per-language override.
DEFAULT_CONFIDENCE = 0.20


class LayoutDetector:
    def __init__(self, model_path="models/doclayout_yolo.pt", device=None):
        self.device = resolve_device(device)
        print(f"Loading DocLayout-YOLO (device={self.device})...")
        self.model = YOLOv10(model_path)
        print("✓ Model loaded!")

        # backend/services/pipeline_service.py holds ONE LayoutDetector
        # per document and hands it to every page's prepare_page() call,
        # which now run concurrently (prepare_executor, up to
        # OCR_PAGE_CONCURRENCY threads at once -- see pipeline_service.py).
        # Ultralytics YOLO is not safe to call predict() on from more than
        # one thread at a time against the same model instance: the
        # predictor keeps mutable state (its input batch, CUDA stream)
        # on `self.model`, and two threads entering predict() at once can
        # corrupt it for both. Confirmed on a real 2-page document
        # (malyalam_removed.pdf, doc_000175): both pages' concurrent
        # detect() calls came back with 0 layout blocks each, which then
        # cascaded into an empty page_json, an empty LLM response, and
        # "RuntimeError: No final article crops were created" -- while
        # the exact same PDF detected 68 blocks per page when run
        # sequentially. A plain lock around the predict() call serializes
        # GPU/CPU inference across threads without giving up the
        # concurrency benefit prepare_executor exists for (OCR and page
        # cleanup for one page can still overlap with layout detection
        # for another; only the predict() call itself is ever
        # single-threaded at a time).
        self._lock = threading.Lock()

    def detect(self, image_path, conf=DEFAULT_CONFIDENCE):
        with self._lock:
            results = self.model.predict(
                source=image_path,
                imgsz=1024,
                conf=conf,
                device=self.device,
                save=False,
                verbose=False,
                # The model's own NMS only suppresses overlapping boxes
                # within the SAME predicted class. Confirmed on a real
                # Telugu (Namasthe Telangana) page: the same headline
                # region was boxed once as "title" and once as "plain
                # text"/"figure" at IoU 1.00, surviving as two separate
                # blocks each with their own OCR read -- this is what
                # showed up as duplicate/nested boxes in the boundary
                # editor. agnostic_nms=True runs NMS across all classes
                # together, so a near-identical box loses to whichever
                # one the model was more confident in regardless of its
                # predicted class.
                agnostic_nms=True,
            )

        return results