"""
Embedding-similarity category classifier.

`import_final_articles.py::get_category` stores whatever string is
present under `article["category"]` as free text (no DB enum), so the
taxonomy below is our own choice, not a fixed external contract. Kept
intentionally small and generic since this is explicitly a best-effort
field per the migration plan.
"""

from __future__ import annotations

import numpy as np

from pipeline.knowledge.embedding_generator import EmbeddingGenerator

# Each category is represented by a few canonical example sentences,
# embedded once and averaged into a centroid. Mixes clean full-sentence
# prose with terse newspaper-headline phrasing -- headlines are a
# different register (abbreviation-heavy, verb-dropping) that scores
# poorly against prose-only centroids even when topically correct.
_TAXONOMY: dict[str, list[str]] = {
    "politics": [
        "The government announced a new policy after debate in parliament.",
        "The minister addressed reporters about the upcoming election.",
        "Party wins assembly seat, ending decades-long rival's hold",
        "Court acquits former minister in high-profile case",
        "Parliamentary panel recommends changes to companies law",
    ],
    "sports": [
        "The team won the match after a thrilling final over.",
        "The player scored the winning goal in the championship.",
        "India wins series decider by five wickets",
        "Star batsman scores century on Test debut",
    ],
    "business": [
        "The company reported quarterly profits and rising stock prices.",
        "The market saw major investment in the new economic policy.",
        "Govt to sell stake in state-run insurer via share sale",
        "Companies Act amendment eases rules for managing directors",
        "Panel recommends cutting minimum age limit for company directors",
    ],
    "entertainment": [
        "The actor's new film released in theatres this week.",
        "The singer performed at a sold-out concert.",
        "Bollywood star's next film to release this festive season",
    ],
    "crime": [
        "Police arrested a suspect in connection with the robbery.",
        "The court sentenced the accused after the investigation.",
        "Man arrested after stabbing attack at private school",
        "Accused's house to be demolished over illegal construction",
    ],
    "national": [
        "The country marked the national holiday with celebrations.",
        "Officials reviewed the state of infrastructure across the nation.",
        "Supreme Court rules on citizens' fundamental rights",
        "Census phase to begin next month across northern states",
        "Farmers announce fresh round of protests over new laws",
    ],
    "international": [
        "World leaders met to discuss the international crisis.",
        "The two countries signed a bilateral trade agreement.",
        "UN officials call for ceasefire amid rising tensions",
    ],
    "opinion": [
        "This editorial argues that reform is overdue.",
        "In my view, the policy fails to address the real problem.",
    ],
    "technology": [
        "The tech company announced a new social media content policy.",
        "Regulators questioned the platform's handling of user content.",
        "Social media giant faces backlash over content moderation call",
        "Tech platform rejects apology for taking down political video",
    ],
    "health": [
        "Hospitals reported a record number of organ transplants this year.",
        "Health officials announced new guidelines for disease prevention.",
        "India records highest-ever number of organ donations",
    ],
    "local": [
        "The city administration announced new civic infrastructure plans.",
        "Local police increased patrols following a spate of incidents.",
        "Teacher attacked inside school premises by stalker",
        "High court directs state to clear pending employee dues",
    ],
}

_SIMILARITY_FLOOR = 0.20


class CategoryClassifier:

    def __init__(self, embedding_generator: EmbeddingGenerator | None = None):

        self._embedder = embedding_generator or EmbeddingGenerator()
        self._centroids: dict[str, np.ndarray] = {}

        for category, examples in _TAXONOMY.items():

            vectors = [
                np.asarray(self._embedder.generate(example), dtype=np.float32)
                for example in examples
            ]

            vectors = [v for v in vectors if v.size]

            if not vectors:
                continue

            centroid = np.mean(vectors, axis=0)
            norm = np.linalg.norm(centroid)

            if norm > 0:
                centroid = centroid / norm

            self._centroids[category] = centroid

    def classify(self, text: str) -> str:

        text = (text or "").strip()

        if not text or not self._centroids:
            return "other"

        vector = np.asarray(self._embedder.generate(text), dtype=np.float32)

        if vector.size == 0:
            return "other"

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        best_category = "other"
        best_similarity = _SIMILARITY_FLOOR

        for category, centroid in self._centroids.items():

            similarity = float(np.dot(vector, centroid))

            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        return best_category
