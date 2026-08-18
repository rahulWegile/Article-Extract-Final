from pipeline.article_growth.growth_score import (
    GrowthScore,
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

print()

print("=" * 60)

print(score)

print()

print("=" * 60)