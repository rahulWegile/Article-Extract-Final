from pipeline.article_merge.article_merge_validator import (
    ArticleMergeValidator,
)

from pipeline.article_merge.article_knowledge import (
    ArticleKnowledge,
)


article_a = ArticleKnowledge(

    article_id=1,

    blocks=[],

    language="en",

    text="",

    entities={},

    embedding=[],

    heading_probability=0.90,

    left_margin=100,

    average_width=400,

    top=100,

    bottom=300,

)

article_b = ArticleKnowledge(

    article_id=2,

    blocks=[],

    language="en",

    text="",

    entities={},

    embedding=[],

    heading_probability=0.20,

    left_margin=110,

    average_width=400,

    top=320,

    bottom=520,

)

validator = ArticleMergeValidator()

result = validator.validate(

    article_a,

    article_b,

)

print()

print("=" * 60)

print("Can Merge :", result)

print("=" * 60)