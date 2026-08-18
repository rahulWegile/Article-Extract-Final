from pipeline.article_merge.article_knowledge_builder import (
    ArticleKnowledgeBuilder,
)


class Knowledge:

    def __init__(
        self,
        language,
        heading,
        embedding,
        entities,
    ):

        self.language = language
        self.heading_probability = heading
        self.embedding = embedding
        self.entities = entities


class Block:

    def __init__(
        self,
        block_id,
        text,
        x1,
        y1,
        x2,
        y2,
    ):

        self.id = block_id

        self.text = text

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):

        return self.x2 - self.x1


# ----------------------------------------------------
# Dummy Blocks
# ----------------------------------------------------

blocks = [

    Block(
        0,
        "First paragraph.",
        100,
        100,
        500,
        180,
    ),

    Block(
        1,
        "Second paragraph.",
        100,
        190,
        510,
        280,
    ),

]

# ----------------------------------------------------
# Knowledge Map
# ----------------------------------------------------

knowledge_map = {

    0: Knowledge(

        language="en",

        heading=0.95,

        embedding=[1.0, 2.0, 3.0],

        entities={

            "PERSON": [

                "John",

            ],

        },

    ),

    1: Knowledge(

        language="en",

        heading=0.10,

        embedding=[2.0, 3.0, 4.0],

        entities={

            "ORG": [

                "OpenAI",

            ],

        },

    ),

}

# ----------------------------------------------------
# Dummy Article
# ----------------------------------------------------

article = {

    "id": 1,

    "nodes": [

        0,

        1,

    ],

}

# ----------------------------------------------------
# Build Article Knowledge
# ----------------------------------------------------

builder = ArticleKnowledgeBuilder()

result = builder.build(

    article,

    blocks,

    knowledge_map,

)

# ----------------------------------------------------
# Output
# ----------------------------------------------------

print()

print("=" * 60)

print(result)

print("=" * 60)