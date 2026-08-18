from pipeline.boundary.boundary_merger import (
    BoundaryMerger,
)

from pipeline.boundary_refiner import (
    ArticleBoundary,
)


class Block:

    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
    ):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


blocks = [

    Block(0, 0, 100, 100),

    Block(100, 0, 200, 100),

    Block(200, 0, 300, 100),

]

boundary = ArticleBoundary(

    article_id=1,

    x1=0,
    y1=0,

    x2=200,
    y2=100,

    width=200,

    height=100,

    area=20000,

    blocks=[0, 1],

)

accepted = [

    {

        "index": 2,

    }

]

merger = BoundaryMerger()

merged = merger.merge(

    boundary,

    accepted,

    blocks,

)

print("=" * 60)

print("Merged Blocks :", merged.blocks)

print(
    merged.x1,
    merged.y1,
    merged.x2,
    merged.y2,
)

print(
    merged.width,
    merged.height,
)

print(
    merged.area,
)