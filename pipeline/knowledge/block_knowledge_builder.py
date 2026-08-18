from pipeline.models.block_knowledge import BlockKnowledge

from pipeline.knowledge.sentence_features import SentenceFeatures
from pipeline.knowledge.language_detector import LanguageDetector
from pipeline.knowledge.entity_extractor import EntityExtractor
from pipeline.knowledge.heading_classifier import HeadingClassifier
from pipeline.knowledge.caption_classifier import CaptionClassifier
from pipeline.knowledge.continuation_classifier import (
    ContinuationClassifier,
)
from pipeline.knowledge.embedding_generator import (
    EmbeddingGenerator,
)


class BlockKnowledgeBuilder:
    """
    Build a complete BlockKnowledge object from
    an OCR block.

    This is the single entry point for the
    Knowledge Layer.
    """

    def __init__(self):

        self.sentence = SentenceFeatures()

        self.language = LanguageDetector()

        self.entities = EntityExtractor()

        self.heading = HeadingClassifier()

        self.caption = CaptionClassifier()

        self.continuation = ContinuationClassifier()

        self.embedding = EmbeddingGenerator()

    # -----------------------------------------------------

    def build(
        self,
        block,
        page_number=None,
    ):

        knowledge = BlockKnowledge(

            block_id=block.id,

        )

        # ------------------------------------------
        # OCR
        # ------------------------------------------

        knowledge.text = block.text

        knowledge.confidence = getattr(
            block,
            "ocr_confidence",
            0.0,
        )

        # ------------------------------------------
        # Sentence Features
        # ------------------------------------------

        features = self.sentence.extract(
            knowledge.text,
        )

        knowledge.first_sentence = features[
            "first_sentence"
        ]

        knowledge.last_sentence = features[
            "last_sentence"
        ]

        knowledge.first_word = features[
            "first_word"
        ]

        knowledge.last_word = features[
            "last_word"
        ]

        knowledge.starts_with_lowercase = features[
            "starts_with_lowercase"
        ]

        knowledge.ends_with_punctuation = features[
            "ends_with_punctuation"
        ]

        # ------------------------------------------
        # Language
        # ------------------------------------------

        knowledge.language = self.language.detect(
            knowledge.text,
        )

        # ------------------------------------------
        # Entities
        # ------------------------------------------

        knowledge.entities = self.entities.extract(
            knowledge.text,
        )

        # ------------------------------------------
        # Heading
        # ------------------------------------------

        knowledge.heading_probability = (
            self.heading.predict(
                knowledge.text,
            )
        )

        # ------------------------------------------
        # Caption
        # ------------------------------------------

        knowledge.caption_probability = (
            self.caption.predict(
                knowledge.text,
            )
        )

        # ------------------------------------------
        # Continuation
        # ------------------------------------------

        knowledge.continuation_probability = (
            self.continuation.predict(
                knowledge.text,
            )
        )

        # ------------------------------------------
        # Embedding
        # ------------------------------------------

        knowledge.embedding = (
            self.embedding.generate(
                knowledge.text,
            )
        )

        # ------------------------------------------
        # Metadata
        # ------------------------------------------

        knowledge.metadata = {

            "page": page_number,

            "block_type": getattr(
                block,
                "cls",
                "",
            ),

        }

        return knowledge