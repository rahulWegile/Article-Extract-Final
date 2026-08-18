from pipeline.knowledge.block_knowledge_builder import (
    BlockKnowledgeBuilder,
)


class DummyBlock:

    def __init__(self):

        self.id = 1

        self.cls = "title"

        self.text = "Prime Minister Visits Delhi"

        self.ocr_confidence = 0.98


builder = BlockKnowledgeBuilder()

knowledge = builder.build(
    DummyBlock(),
    page_number=1,
)

print("=" * 60)
print("Block Knowledge")
print("=" * 60)

print("Block ID        :", knowledge.block_id)
print("Text            :", knowledge.text)
print("Confidence      :", knowledge.confidence)
print("Language        :", knowledge.language)
print("First Sentence  :", knowledge.first_sentence)
print("Last Sentence   :", knowledge.last_sentence)
print("Entities        :", knowledge.entities)
print("Heading Prob    :", knowledge.heading_probability)
print("Caption Prob    :", knowledge.caption_probability)
print("Continuation    :", knowledge.continuation_probability)
print("Embedding Size  :", len(knowledge.embedding))
print("Metadata        :", knowledge.metadata)