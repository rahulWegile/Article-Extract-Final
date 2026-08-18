from pipeline.article_merge.article_merger import (
    ArticleMerger,
)

from pipeline.article_merge.article_merge_candidate import (
    ArticleMergeCandidate,
)

#
# Dummy Articles
#

articles = [

    {
        "id": 1,
        "nodes": [1, 2],
    },

    {
        "id": 2,
        "nodes": [3, 4],
    },

]

#
# Approved merge
#

candidate = ArticleMergeCandidate(

    source=1,

    target=2,

)

#
# Merge
#

merged = ArticleMerger().merge(

    articles,

    [candidate],

)

print()

print("=" * 70)

print("ARTICLE MERGER TEST")

print("=" * 70)

print(f"Merged Articles : {len(merged)}")

print()

for article in merged:

    print(article)

print()

print("=" * 70)