from pipeline.article_growth.article_growth_pipeline import (
    ArticleGrowthPipeline,
)


class Knowledge:

    def __init__(
        self,
        heading_probability,
        language,
        embedding,
        entities,
    ):

        self.heading_probability = heading_probability
        self.language = language
        self.embedding = embedding
        self.entities = entities


class Block:

    def __init__(
        self,
        block_id,
        cls,
        x1,
        y1,
        x2,
        y2,
        column,
        text,
    ):

        self.id = block_id

        self.cls = cls

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.column = column

        self.text = text


blocks = [

    Block(
        0,
        "title",
        100,
        100,
        500,
        150,
        0,
        "Article Title",
    ),

    Block(
        1,
        "plain text",
        100,
        170,
        500,
        260,
        0,
        "Paragraph One",
    ),

    Block(
        2,
        "plain text",
        100,
        280,
        500,
        380,
        0,
        "Paragraph Two",
    ),

    Block(
        3,
        "title",
        900,
        100,
        1300,
        150,
        1,
        "Second Article",
    ),

    Block(
        4,
        "plain text",
        900,
        170,
        1300,
        260,
        1,
        "Another Paragraph",
    ),

]

knowledge_map = {

    0: Knowledge(
        0.95,
        "en",
        [1,2,3],
        {},
    ),

    1: Knowledge(
        0.20,
        "en",
        [2,3,4],
        {},
    ),

    2: Knowledge(
        0.20,
        "en",
        [3,4,5],
        {},
    ),

    3: Knowledge(
        0.97,
        "en",
        [1,2,3],
        {},
    ),

    4: Knowledge(
        0.20,
        "en",
        [2,3,4],
        {},
    ),

}

pipeline = ArticleGrowthPipeline()

articles = pipeline.run(

    blocks,

    knowledge_map,

)

print()

print("=" * 60)

print("FINAL ARTICLES")

print("=" * 60)

for article in articles:

    print(article)

print()

print("=" * 60)