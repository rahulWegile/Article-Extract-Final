from pipeline.article_growth.growth_validator import (
    GrowthValidator,
)

from pipeline.article_growth.growth_candidate import (
    GrowthCandidate,
)

from pipeline.article_growth.growth_score import (
    GrowthScore,
)

from pipeline.article_seed.seed import (
    ArticleSeed,
)


class Block:

    def __init__(
        self,
        block_id,
    ):

        self.id = block_id


seed = ArticleSeed(

    block_id=0,

    column=0,

    heading_probability=0.95,

    language="en",

)

candidate = GrowthCandidate(

    seed=seed,

    block=Block(1),

    vertical_gap=25,

    same_column=True,

)

score = GrowthScore(

    geometry=0.50,

    language=0.20,

    embedding=0.40,

    entities=0.10,

    alignment=0.30,

    reading_order=0.20,

    total=1.70,

)

validator = GrowthValidator()

result = validator.validate(

    candidate,

    score,

)

print()

print("=" * 60)

print(f"Accepted : {result}")

print("=" * 60)