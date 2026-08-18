from pipeline.graph.connectors.title_body_connector import (
    TitleBodyConnector,
)

from pipeline.graph.edge import Edge


class Knowledge:

    def __init__(self, heading, language):

        self.heading_probability = heading
        self.language = language


class Block:

    def __init__(
        self,
        block_id,
        x1,
        y1,
        x2,
        y2,
    ):

        self.id = block_id

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


blocks = [

    Block(0, 100, 100, 400, 150),

    Block(1, 100, 160, 400, 500),

]

knowledge = {

    0: Knowledge(0.95, "en"),

    1: Knowledge(0.10, "en"),

}

edges = []

edges = TitleBodyConnector().connect(
    edges,
    blocks,
    knowledge,
)

print()

print("=" * 60)

print("Generated Edges :", len(edges))

for edge in edges:

    print(
        edge.source,
        "->",
        edge.target,
    )