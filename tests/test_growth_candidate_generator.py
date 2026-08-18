from pipeline.article_growth.current_article import (
    CurrentArticle,
)

from pipeline.article_growth.growth_candidate_generator import (
    GrowthCandidateGenerator,
)


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

    260,

)

body2 = Block(

    2,

    0,

    280,

    360,

)

other_column = Block(

    3,

    1,

    170,

    250,

)

blocks = [

    title,

    body1,

    body2,

    other_column,

]

article = CurrentArticle(

    article_id=1,

    nodes=[0],

    visited={0},

    last_block=title,

    bottom=150,

    column=0,

)

generator = GrowthCandidateGenerator()

candidates = generator.generate(

    article,

    blocks,

)

print()

print("=" * 60)

for candidate in candidates:

    print(candidate)

print("=" * 60)