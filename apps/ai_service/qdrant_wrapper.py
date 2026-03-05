"""Qdrant vector store wrapper."""
import os
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from libs.common.logging import get_logger

logger = get_logger(__name__)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "gratifikasi_cases")
VECTOR_SIZE = 384  # default for MiniLM


class QdrantWrapper:
    """Async Qdrant client wrapper."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.client = AsyncQdrantClient(host=host, port=port)
        self.collection_name = collection_name

    async def init_collection(self, vector_size: int = VECTOR_SIZE) -> None:
        """Create collection if it does not exist."""
        try:
            await self.client.get_collection(self.collection_name)
            logger.info("Qdrant collection '%s' already exists", self.collection_name)
        except Exception:
            logger.info("Creating Qdrant collection '%s'", self.collection_name)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self.collection_name)

    async def search(
        self, vector: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload or {},
            }
            for r in response.points
        ]

    async def upsert(
        self,
        record_id: int,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """Upsert a single vector with payload."""
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=record_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.info("Upserted record %s into Qdrant", record_id)
