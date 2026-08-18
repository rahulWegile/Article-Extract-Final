from pipeline.knowledge.embedding_generator import (
    EmbeddingGenerator,
)

generator = EmbeddingGenerator()

texts = [

    "Prime Minister visits Delhi",

    "Government announces new policy",

    "India wins cricket match",

    "Prime Minister visits Delhi",

]

for text in texts:

    embedding = generator.generate(text)

    print("-" * 60)

    print(text)

    print(len(embedding))

    print(embedding[:5])