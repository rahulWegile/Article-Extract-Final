from pipeline.article_seed.seed import (
    ArticleSeed,
)

print()

print("=" * 60)

seed = ArticleSeed(

    block_id=5,

    column=1,

    heading_probability=0.96,

    language="en",

)

print(seed)

print()

print("=" * 60)