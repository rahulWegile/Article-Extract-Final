from pipeline.boundary.boundary_geometry_repair import (
    BoundaryGeometryRepair,
)

from pipeline.boundary_refiner import (
    ArticleBoundary,
)


class Block:

    def __init__(self, x1, y1, x2, y2):

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2


blocks = [

    Block(10, 10, 110, 110),

    Block(110, 10, 220, 120),

]

boundary = ArticleBoundary(

    article_id=1,

    x1=15,
    y1=15,

    x2=210,
    y2=115,

    width=195,

    height=100,

    area=19500,

    blocks=[0, 1],

)

repair = BoundaryGeometryRepair()

new_boundary = repair.repair(

    boundary,

    blocks,

    page_width=1000,

    page_height=2000,

)

print("=" * 60)

print(new_boundary.x1)

print(new_boundary.y1)

print(new_boundary.x2)

print(new_boundary.y2)

print(new_boundary.area)