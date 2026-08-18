from pipeline.boundary.boundary_validator import (
    BoundaryValidator,
)

from pipeline.boundary_refiner import (
    ArticleBoundary,
)


class Block:

    def __init__(self, idx, x1, y1, x2, y2):

        self.id = idx

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2


class Knowledge:

    def __init__(self, heading, language):

        self.heading_probability = heading
        self.language = language


blocks = [

    Block(0, 0, 0, 100, 100),

    Block(1, 100, 0, 200, 100),

]

knowledge_map = {

    0: Knowledge(
        heading=0.95,
        language="en",
    ),

    1: Knowledge(
        heading=0.10,
        language="en",
    ),

}

boundary = ArticleBoundary(

    article_id=1,

    x1=0,
    y1=0,

    x2=100,
    y2=100,

    width=100,

    height=100,

    area=10000,

    blocks=[0],

)

validator = BoundaryValidator()

candidate = {

    "index": 1,

}

print("=" * 60)

print(

    validator.validate(

        boundary,

        candidate,

        blocks,

        knowledge_map,

    )

)