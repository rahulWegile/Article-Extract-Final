from pipeline.boundary.boundary_content_repair import (
    BoundaryContentRepair,
)

from pipeline.boundary_refiner import (
    ArticleBoundary,
)


class Block:

    def __init__(
        self,
        idx,
        x1,
        y1,
        x2,
        y2,
    ):

        self.id = idx

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2


class Knowledge:

    def __init__(
        self,
        caption,
        continuation,
    ):

        self.caption_probability = caption

        self.continuation_probability = continuation


blocks = [

    Block(0, 0, 0, 100, 100),

    Block(1, 110, 0, 200, 100),

    Block(2, 50, 110, 160, 140),

]

knowledge_map = {

    0: Knowledge(0.0, 0.0),

    1: Knowledge(0.0, 0.0),

    2: Knowledge(0.95, 0.0),

}

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

repair = BoundaryContentRepair()

new_boundary = repair.repair(

    boundary,

    blocks,

    knowledge_map,

)

print("=" * 60)

print(new_boundary.blocks)