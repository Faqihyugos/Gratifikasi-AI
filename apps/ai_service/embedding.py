"""Sentence embedding using fastembed (ONNX-based, no PyTorch required)."""
import os
from typing import List, Optional
from fastembed import TextEmbedding

from libs.common.logging import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingModel:
    """Wrapper around fastembed for ONNX-based embedding generation (no PyTorch)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model (fastembed): %s", model_name)
        self.model = TextEmbedding(model_name_or_path=model_name)
        self.model_name = model_name
        self._vector_size: Optional[int] = None
        logger.info("Embedding model loaded")

    def encode(self, text: str) -> List[float]:
        """Encode a single text into a vector."""
        vectors = list(self.model.embed([text]))
        return vectors[0].tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts."""
        return [v.tolist() for v in self.model.embed(texts)]

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            sample = list(self.model.embed(["test"]))
            self._vector_size = len(sample[0])
        return self._vector_size
