from pipeline.article_merge.article_merge_scorer import (
    ArticleMergeScorer,
)

from pipeline.article_merge.article_knowledge import (
    ArticleKnowledge,
)


article_a = ArticleKnowledge(

    article_id=1,

    blocks=[],

    language="en",

    text="The government announced",

    entities={
        "PERSON": ["John"]
    },

    embedding=[1.0, 2.0, 3.0],

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

    text="new reforms today.",

    entities={
        "PERSON": ["John"]
    },

    embedding=[1.1, 2.1, 3.1],

    heading_probability=0.10,

    left_margin=105,

    average_width=395,

    top=320,

    bottom=500,

)

score = ArticleMergeScorer().score(

    article_a,

    article_b,

)

print()

print("=" * 60)

print("Merge Score :", score)

print("=" * 60)