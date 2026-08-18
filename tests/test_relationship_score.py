print("TEST STARTED")

from pipeline.graph.scoring.relationship_score import RelationshipScore

print("Imported RelationshipScore")


class Knowledge:
    def __init__(self):
        self.heading_probability = 0.98
        self.language = "en"


class Block:

    def __init__(self):

        self.cls = "title"

        self.x1 = 100
        self.y1 = 100
        self.x2 = 500
        self.y2 = 150

        self.knowledge = Knowledge()

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


title = Block()

body = Block()

body.cls = "plain text"
body.y1 = 160
body.y2 = 500
body.knowledge.heading_probability = 0.1


class Edge:

    relation = "title_to_text"


score = RelationshipScore().score(
    Edge(),
    title,
    body,
    [title, body],
)

print("Score =", score)

print("TEST FINISHED")