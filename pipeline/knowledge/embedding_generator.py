from pathlib import Path
import hashlib
import pickle

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generate semantic embeddings for OCR text.

    Features:
    - Loads model only once
    - Automatic disk cache
    - Reuses embeddings for repeated text
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    CACHE_DIR = Path("cache/embeddings")

    def __init__(self):

        self.CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

    # --------------------------------------------------

    def generate(self, text: str):

        if not text:

            return []

        cache_file = self._cache_file(text)

        # ----------------------------
        # Cache Hit
        # ----------------------------

        if cache_file.exists():

            with open(cache_file, "rb") as f:

                return pickle.load(f)

        # ----------------------------
        # Generate Embedding
        # ----------------------------

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        # ----------------------------
        # Save Cache
        # ----------------------------

        with open(cache_file, "wb") as f:

            pickle.dump(
                embedding,
                f,
            )

        return embedding.tolist()

    # --------------------------------------------------

    def _cache_file(
        self,
        text: str,
    ):

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        return self.CACHE_DIR / f"{digest}.pkl"