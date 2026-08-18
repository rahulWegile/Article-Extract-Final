from pipeline.article_seed.seed import (
    ArticleSeed,
)

from pipeline.article_growth.article_grower import (
    ArticleGrower,
)

from pipeline.article_growth.growth_context import (
    GrowthContext,
)

from pipeline.article_growth.growth_engine import (
    GrowthEngine,
)

from pipeline.article_growth.growth_candidate_generator import (
    GrowthCandidateGenerator,
)

from pipeline.article_growth.growth_scorer import (
    GrowthScorer,
)

from pipeline.article_growth.growth_validator import (
    GrowthValidator,
)


# ----------------------------------------------------------
# Dummy Knowledge
# ----------------------------------------------------------

class Knowledge:

    def __init__(
        self,
        language,
        embedding,
        entities,
    ):

        self.language = language
        self.embedding = embedding
        self.entities = entities


# ----------------------------------------------------------
# Dummy Block
# ----------------------------------------------------------

class Block:

    def __init__(
        self,
        block_id,
        column,
        y1,
        y2,
    ):

        self.id = block_id
        self.column = column
        self.y1 = y1
        self.y2 = y2


# ----------------------------------------------------------
# Blocks
# ----------------------------------------------------------

title = Block(
    0,
    0,
    100,
    150,
)

body1 = Block(
    1,
    0,
    170,
    250,
)

body2 = Block(
    2,
    0,
    270,
    360,
)

blocks = [
    title,
    body1,
    body2,
]


# ----------------------------------------------------------
# Knowledge
# ----------------------------------------------------------

knowledge_map = {

    0: Knowledge(
        "en",
        [1, 2, 3],
        {
            "PERSON": ["John"],
        },
    ),

    1: Knowledge(
        "en",
        [2, 3, 4],
        {
            "ORG": ["OpenAI"],
        },
    ),

    2: Knowledge(
        "en",
        [3, 4, 5],
        {
            "ORG": ["Google"],
        },
    ),
}


# ----------------------------------------------------------
# Seed
# ----------------------------------------------------------

seed = ArticleSeed(

    block_id=0,

    column=0,

    heading_probability=0.95,

    language="en",

)


# ----------------------------------------------------------
# Context
# ----------------------------------------------------------

context = GrowthContext(

    blocks=blocks,

    block_lookup={
        b.id: b
        for b in blocks
    },

    knowledge_map=knowledge_map,

    candidate_generator=GrowthCandidateGenerator(),

    scorer=GrowthScorer(),

    validator=GrowthValidator(),

)


# ----------------------------------------------------------
# Grow
# ----------------------------------------------------------

engine = GrowthEngine()

grower = ArticleGrower()

article = grower.grow(

    seed,

    context,

    engine,

)


# ----------------------------------------------------------
# Results
# ----------------------------------------------------------

print()

print("=" * 70)
print("ARTICLE GROWER TEST")
print("=" * 70)

print()

print("Article ID :", article.article_id)
print("Nodes      :", article.nodes)
print("Visited    :", sorted(article.visited))
print("Bottom     :", article.bottom)
print("Column     :", article.column)
print("Last Block :", article.last_block.id)

print()

print(article)

print()

print("=" * 70)