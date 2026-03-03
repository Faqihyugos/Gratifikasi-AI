"""Sentence embedding using sentence-transformers."""
import os
from typing import List
from sentence_transformers import SentenceTransformer

from libs.common.logging import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingModel:
    """Wrapper around sentence-transformers for embedding generation."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info("Embedding model loaded")

    def encode(self, text: str) -> List[float]:
        """Encode a single text into a vector."""
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts."""
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
