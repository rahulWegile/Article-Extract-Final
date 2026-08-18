from pipeline.article_seed.seed_generator import (
    SeedGenerator,
)


class Knowledge:

    def __init__(
        self,
        heading_probability,
        language,
    ):

        self.heading_probability = heading_probability
        self.language = language


class Block:

    def __init__(
        self,
        block_id,
        cls,
        column=0,
    ):

        self.id = block_id
        self.cls = cls
        self.column = column


blocks = [

    Block(
        0,
        "title",
        0,
    ),

    Block(
        1,
        "plain text",
        0,
    ),

    Block(
        2,
        "title",
        1,
    ),

    Block(
        3,
        "image",
        1,
    ),

]

knowledge_map = {

    0: Knowledge(
        0.95,
        "en",
    ),

    1: Knowledge(
        0.10,
        "en",
    ),

    2: Knowledge(
        0.91,
        "en",
    ),

    3: Knowledge(
        0.00,
        "en",
    ),

}

generator = SeedGenerator()

seeds = generator.generate(

    blocks,

    knowledge_map,

)

print()

for seed in seeds:

    print(seed)

print()