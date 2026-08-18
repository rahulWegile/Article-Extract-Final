from pipeline.boundary.boundary_knowledge_scorer import (
    BoundaryKnowledgeScorer,
)


class DummyKnowledge:

    def __init__(
        self,
        language,
        heading,
        caption,
        continuation,
        embedding,
        entities,
    ):
        self.language = language
        self.heading_probability = heading
        self.caption_probability = caption
        self.continuation_probability = continuation
        self.embedding = embedding
        self.entities = entities


class DummyBlock:

    def __init__(self, block_id):
        self.id = block_id


# ----------------------------------------
# Boundary blocks
# ----------------------------------------

boundary_blocks = [

    DummyBlock(1),

    DummyBlock(2),

]

candidate = DummyBlock(3)

# ----------------------------------------
# Fake Knowledge
# ----------------------------------------

knowledge_map = {

    1: DummyKnowledge(

        language="en",

        heading=0.95,

        caption=0.05,

        continuation=0.05,

        embedding=[1.0, 0.0, 0.0],

        entities={
            "people": ["Narendra Modi"],
            "organizations": [],
            "places": [],
            "dates": [],
        },

    ),

    2: DummyKnowledge(

        language="en",

        heading=0.05,

        caption=0.05,

        continuation=0.05,

        embedding=[0.9, 0.1, 0.0],

        entities={
            "people": ["Narendra Modi"],
            "organizations": [],
            "places": [],
            "dates": [],
        },

    ),

    3: DummyKnowledge(

        language="en",

        heading=0.05,

        caption=0.05,

        continuation=0.05,

        embedding=[0.95, 0.05, 0.0],

        entities={
            "people": ["Narendra Modi"],
            "organizations": [],
            "places": [],
            "dates": [],
        },

    ),

}

# ----------------------------------------

scorer = BoundaryKnowledgeScorer()

score = scorer.score(

    boundary=None,

    candidate_block=candidate,

    boundary_blocks=boundary_blocks,

    knowledge_map=knowledge_map,

)

print("=" * 60)
print("Boundary Knowledge Score")
print("=" * 60)
print(score)