"""Inference pipeline combining similarity search and classifier."""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .embedding import EmbeddingModel
from .qdrant_wrapper import QdrantWrapper
from .mlflow_loader import ModelLoader
from libs.common.logging import get_logger

logger = get_logger(__name__)


def _build_similar_cases(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw Qdrant results to SimilarCase dicts expected by the frontend."""
    cases = []
    for r in results:
        payload = r.get("payload", {})
        cases.append({
            "id": int(r["id"]) if str(r["id"]).isdigit() else r["id"],
            "similarity_score": round(float(r["score"]), 4),
            "final_label": payload.get("final_label") or "Unknown",
            "preview": (payload.get("preview") or "")[:200],
        })
    return cases


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
    timestamp = datetime.now(timezone.utc).isoformat()
    model_info = model_loader.model_info
    model_version = model_info.get("model_name") or model_info.get("stage", "n/a")
    model_run_id = model_info.get("run_id", "")

    vector = embedding.encode(text)
    similar_results = await qdrant.search(vector=vector, top_k=top_k)
    similar_cases = _build_similar_cases(similar_results)

    best_score = similar_cases[0]["similarity_score"] if similar_cases else 0.0
    best_label = similar_cases[0]["final_label"] if similar_cases else None

    if best_score >= similarity_threshold and best_label:
        label = best_label
        logger.info("Similarity match: score=%.4f label=%s", best_score, label)
        return {
            "label": label,
            "confidence": float(best_score),
            "source": "similarity",
            "similar_cases": similar_cases,
            "model_version": model_version,
            "model_run_id": model_run_id,
            "timestamp": timestamp,
            "probabilities": {label: float(best_score)},
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
        "similar_cases": similar_cases,
        "model_version": model_version,
        "model_run_id": model_run_id,
        "timestamp": timestamp,
        "probabilities": result.get("probabilities", {}),
    }
