from pipeline.article_merge.article_merge_pipeline import (
    ArticleMergePipeline,
)


# ==========================================================
# Dummy Knowledge
# ==========================================================

class Knowledge:

    def __init__(
        self,
        heading_probability,
        language,
        entities,
        embedding,
    ):

        self.heading_probability = heading_probability
        self.language = language
        self.entities = entities
        self.embedding = embedding


# ==========================================================
# Dummy Block
# ==========================================================

class Block:

    def __init__(
        self,
        block_id,
        text,
        heading_probability,
        language,
        entities,
        embedding,
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

        self.knowledge = Knowledge(
            heading_probability,
            language,
            entities,
            embedding,
        )

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


# ==========================================================
# Blocks
# ==========================================================

blocks = [

    Block(

        0,

        "Paragraph one",

        0.95,

        "en",

        {

            "PERSON": ["John"]

        },

        [

            1.0,
            2.0,
            3.0,

        ],

        100,
        100,
        500,
        150,

    ),

    Block(

        1,

        "continues here.",

        0.10,

        "en",

        {

            "ORG": ["OpenAI"]

        },

        [

            3.0,
            4.0,
            5.0,

        ],

        100,
        170,
        500,
        420,

    ),

    Block(

        2,

        "Completely different article.",

        0.98,

        "en",

        {

            "PERSON": ["Alice"]

        },

        [

            8.0,
            9.0,
            10.0,

        ],

        900,
        100,
        1300,
        260,

    ),

]

# ==========================================================
# Articles
# ==========================================================

articles = [

    {

        "id": 1,

        "nodes": [

            0,

        ],

    },

    {

        "id": 2,

        "nodes": [

            1,

        ],

    },

    {

        "id": 3,

        "nodes": [

            2,

        ],

    },

]



# ==========================================================
# Knowledge Map
# ==========================================================

knowledge_map = {

    block.id: block.knowledge

    for block in blocks

}


# ==========================================================
# Run Pipeline
# ==========================================================

pipeline = ArticleMergePipeline()

merged_articles = pipeline.run(

    articles,

    blocks,

    knowledge_map,

)


# ==========================================================
# Results
# ==========================================================

print()

print("=" * 70)

print("FINAL MERGED ARTICLES")

print("=" * 70)

print(f"Count : {len(merged_articles)}")

print()

for article in merged_articles:

    print(

        f"Article {article['id']}"

    )

    print(

        f"Nodes : {article['nodes']}"

    )

    print()