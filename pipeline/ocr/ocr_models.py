from dataclasses import dataclass, field


@dataclass
class OCRResult:

    # --------------------------------------------------
    # Combined OCR text for the layout block
    # --------------------------------------------------

    text: str

    # --------------------------------------------------
    # Average OCR confidence
    # --------------------------------------------------

    confidence: float

    # --------------------------------------------------
    # Layout block bounding box
    # --------------------------------------------------

    x1: int
    y1: int
    x2: int
    y2: int

    # --------------------------------------------------
    # Individual OCR lines
    #
    # Each line contains:
    #
    # {
    #     "text": "...",
    #     "confidence": 0.95,
    #     "bbox": {
    #         "x1": ...,
    #         "y1": ...,
    #         "x2": ...,
    #         "y2": ...
    #     }
    # }
    # --------------------------------------------------

    lines: list[dict] = field(
        default_factory=list
    )