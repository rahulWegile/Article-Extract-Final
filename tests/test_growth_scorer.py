from pipeline.article_growth.current_article import (
    CurrentArticle,
)

from pipeline.article_growth.growth_candidate import (
    GrowthCandidate,
)

from pipeline.article_growth.growth_scorer import (
    GrowthScorer,
)

from pipeline.article_seed.seed import (
    ArticleSeed,
)


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


class Block:

    def __init__(
        self,
        block_id,
        y2,
    ):

        self.id = block_id

        self.y2 = y2


seed_block = Block(

    0,

    150,

)

body = Block(

    1,

    250,

)

article = CurrentArticle(

    article_id=1,

    nodes=[0],

    visited={0},

    last_block=seed_block,

    bottom=150,

    column=0,

)

candidate = GrowthCandidate(

    seed=None,

    block=body,

    vertical_gap=20,

    same_column=True,

)

knowledge_map = {

    0: Knowledge(

        "en",

        [1,2,3],

        {

            "PERSON":[

                "John"

            ]

        },

    ),

    1: Knowledge(

        "en",

        [2,3,4],

        {

            "ORG":[

                "OpenAI"

            ]

        },

    ),

}

score = GrowthScorer().score(

    article,

    candidate,

    knowledge_map,

    {

        0: seed_block,

        1: body,

    },

)

print()

print("=" * 60)

print(score)

print("=" * 60)