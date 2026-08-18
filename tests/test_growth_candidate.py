from pipeline.article_growth.growth_candidate import (
    GrowthCandidate,
)

from pipeline.article_seed.seed import (
    ArticleSeed,
)


class Block:

    def __init__(
        self,
        block_id,
        text,
    ):

        self.id = block_id

        self.text = text


seed = ArticleSeed(

    block_id=5,

    column=1,

    heading_probability=0.96,

    language="en",

)

block = Block(

    7,

    "First paragraph of article.",

)

candidate = GrowthCandidate(

    seed=seed,

    block=block,

    vertical_gap=18,

    same_column=True,

)

print()

print("=" * 60)

print(candidate)

print()

print("=" * 60)