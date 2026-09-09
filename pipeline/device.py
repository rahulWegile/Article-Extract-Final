"""
Shared torch/ultralytics device selection.

Covers the three places in this pipeline that run a torch model and
therefore have a CPU/GPU choice to make: the shared DocLayout-YOLO
layout detector (pipeline/layout_detector.py, every language), and
the two Urdu-only models -- UTRNet itself and its YOLOv8 line
detector (pipeline/ocr/utrnet_engine.py, pipeline/ocr/
urdu_line_detector.py).

Defaults to "cpu" whichever way OCR_DEVICE is unset or blank,
matching what every one of those three already did (the layout
detector had "cpu" hardcoded; both Urdu models have only ever been
run CPU-only so far). Set OCR_DEVICE=cuda (or cuda:0, cuda:1, ... for
a specific GPU) once a CUDA-enabled torch build and an NVIDIA GPU are
actually available to move all three over -- no code changes needed
at that point.
"""

import os


def resolve_device(explicit=None):
    """
    Precedence: an explicit constructor arg (unchanged behavior for
    any caller that already passes device=...) beats OCR_DEVICE,
    which beats the "cpu" fallback.
    """

    if explicit:
        return explicit

    configured = os.environ.get("OCR_DEVICE", "").strip()

    return configured or "cpu"
