"""Prediction endpoint."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from ..inference import run_inference

router = APIRouter()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)


class SimilarCaseOut(BaseModel):
    id: Any
    similarity_score: float
    final_label: str
    preview: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    source: str
    similar_cases: List[SimilarCaseOut]
    model_version: str
    model_run_id: str
    timestamp: str
    probabilities: Dict[str, float]


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest):
    result = await run_inference(
        text=body.text,
        top_k=body.top_k,
        similarity_threshold=body.similarity_threshold,
        embedding=request.app.state.embedding,
        qdrant=request.app.state.qdrant,
        model_loader=request.app.state.model_loader,
    )
    return PredictResponse(**result)
