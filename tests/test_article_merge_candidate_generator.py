from pipeline.article_merge.article_knowledge import (
    ArticleKnowledge,
)

from pipeline.article_merge.article_merge_candidate_generator import (
    ArticleMergeCandidateGenerator,
)


articles = [

    ArticleKnowledge(

        article_id=1,

        blocks=[],

        language="en",

        text="",

        entities={},

        embedding=[],

        heading_probability=0,

        left_margin=100,

        average_width=400,

        top=100,

        bottom=300,

    ),

    ArticleKnowledge(

        article_id=2,

        blocks=[],

        language="en",

        text="",

        entities={},

        embedding=[],

        heading_probability=0,

        left_margin=110,

        average_width=405,

        top=340,

        bottom=520,

    ),

    ArticleKnowledge(

        article_id=3,

        blocks=[],

        language="en",

        text="",

        entities={},

        embedding=[],

        heading_probability=0,

        left_margin=500,

        average_width=400,

        top=340,

        bottom=520,

    ),

]

generator = ArticleMergeCandidateGenerator()

candidates = generator.generate(
    articles
)

print()

print("=" * 60)

for candidate in candidates:

    print(
        candidate.source,
        "->",
        candidate.target,
    )

print("=" * 60)