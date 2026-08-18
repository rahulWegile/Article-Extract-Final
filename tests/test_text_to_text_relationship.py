from pipeline.graph.scoring.relationship_score import (
    RelationshipScore,
)

from pipeline.graph.candidate_generator import CandidateEdge


class Knowledge:

    def __init__(self):
        self.language = "en"
        self.heading_probability = 0.0


class Block:

    def __init__(
        self,
        x1,
        y1,
        x2,
        y2,
        text,
    ):

        self.cls = "plain text"

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.text = text

        self.knowledge = Knowledge()

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2


source = Block(
    100,
    100,
    500,
    200,
    "The government announced",
)

target = Block(
    102,
    205,
    505,
    320,
    "new measures today.",
)

edge = CandidateEdge(
    source=0,
    target=1,
    relation="text_to_text",
    direction="below",
)

score = RelationshipScore().score(
    edge,
    source,
    target,
    [source, target],
)

print()
print("=" * 60)
print("Text-to-Text Relationship Score :", score)
print("=" * 60)