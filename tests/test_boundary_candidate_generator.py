from pipeline.boundary.boundary_candidate_generator import (
    BoundaryCandidateGenerator,
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

    Block(0, 0, 100, 100),

    Block(110, 0, 200, 100),

    Block(500, 500, 600, 600),

]

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

generator = BoundaryCandidateGenerator()

candidates = generator.generate(

    boundary,

    blocks,

)

print("=" * 60)

print(len(candidates))

for c in candidates:

    print(c["index"])