from dataclasses import dataclass


@dataclass
class LayoutBlock:

    # Unique block id
    id: int

    # Detection class
    cls: str

    # Bounding box
    x1: int
    y1: int
    x2: int
    y2: int

    confidence: float

    # Filled after OCR
    text: str = ""

    ocr_confidence: float = 0.0

    # Filled later by Knowledge Layer
    knowledge = None

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2


def parse_results(results):

    result = results[0]

    names = result.names

    blocks = []

    for index, box in enumerate(result.boxes):

        cls = names[int(box.cls.item())]

        conf = float(box.conf.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        blocks.append(
            LayoutBlock(
                id=index,
                cls=cls,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=conf,
            )
        )
    return blocks