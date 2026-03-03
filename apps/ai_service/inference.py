"""Inference pipeline combining similarity search and classifier."""
from typing import Dict, Any, List, Optional
from .embedding import EmbeddingModel
from .qdrant_wrapper import QdrantWrapper
from .mlflow_loader import ModelLoader
from libs.common.logging import get_logger

logger = get_logger(__name__)


async def run_inference(
    text: str,
    top_k: int,
    similarity_threshold: float,
    embedding: EmbeddingModel,
    qdrant: QdrantWrapper,
    model_loader: ModelLoader,
) -> Dict[str, Any]:
    """
    Run the full inference pipeline:
    1. Embed text
    2. Similarity search in Qdrant
    3. If best score >= threshold -> use similarity result
    4. Else -> run classifier
    """
    vector = embedding.encode(text)
    similar_results = await qdrant.search(vector=vector, top_k=top_k)

    similar_case_ids: List[str] = []
    best_score = 0.0
    best_payload: Optional[Dict[str, Any]] = None

    if similar_results:
        best = similar_results[0]
        best_score = best["score"]
        best_payload = best["payload"]
        similar_case_ids = [r["id"] for r in similar_results]

    if best_score >= similarity_threshold and best_payload:
        label = best_payload.get("final_label") or best_payload.get("label") or "UNKNOWN"
        logger.info("Similarity match: score=%.4f label=%s", best_score, label)
        return {
            "label": label,
            "confidence": float(best_score),
            "source": "similarity",
            "similar_case_ids": similar_case_ids,
            "model_version": model_loader.model_info.get("stage", "n/a"),
        }

    result = model_loader.predict(text)
    logger.info(
        "Classifier prediction: label=%s confidence=%.4f",
        result["label"],
        result["confidence"],
    )
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "source": "classifier",
        "similar_case_ids": similar_case_ids,
        "model_version": model_loader.model_info.get("stage", "n/a"),
    }
