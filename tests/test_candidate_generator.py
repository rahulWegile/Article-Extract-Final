from pipeline.graph.candidate_generator import (
    CandidateGenerator,
)


class Block:

    def __init__(
        self,
        cls,
        x1,
        y1,
        x2,
        y2,
    ):

        self.cls = cls

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class Neighbor:

    def __init__(
        self,
        index,
        distance,
    ):

        self.index = index
        self.distance = distance


class MockSpatialIndex:

    def nearby(
        self,
        index,
        radius,
    ):

        return [

            Neighbor(1, 40),

            Neighbor(2, 55),

            Neighbor(3, 70),

        ]


blocks = [

    Block(
        "title",
        100,
        100,
        400,
        150,
    ),

    Block(
        "figure",
        100,
        160,
        300,
        300,
    ),

    Block(
        "figure_caption",
        100,
        305,
        300,
        340,
    ),

    Block(
        "plain text",
        100,
        350,
        400,
        700,
    ),

]

generator = CandidateGenerator()

edges = generator.generate(
    blocks,
    MockSpatialIndex(),
)

print()

print("=" * 60)

for edge in edges:

    print(

        edge.relation,

        edge.source,

        "->",

        edge.target,

    )