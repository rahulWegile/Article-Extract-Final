from pipeline.article_growth.growth_context import (
    GrowthContext,
)


context = GrowthContext(

    blocks=[],

    block_lookup={},

    knowledge_map={},

    candidate_generator=None,

    scorer=None,

    validator=None,

)

print()

print("=" * 60)

print(context)

print()

print("=" * 60)