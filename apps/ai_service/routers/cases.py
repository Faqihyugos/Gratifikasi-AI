"""Cases upsert endpoint."""
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/cases")


class UpsertRequest(BaseModel):
    record_id: int
    text: str
    final_label: str
    value_estimation: Optional[float] = None
    created_at: Optional[str] = None


@router.post("/upsert")
async def upsert_case(request: Request, body: UpsertRequest):
    """Embed text and upsert into Qdrant."""
    embedding = request.app.state.embedding
    qdrant = request.app.state.qdrant

    vector = embedding.encode(body.text)

    # Ensure vector size matches collection
    await qdrant.init_collection(vector_size=len(vector))

    payload = {
        "record_id": body.record_id,
        "final_label": body.final_label,
        "preview": body.text[:200],
        "value_estimation": body.value_estimation,
        "created_at": body.created_at,
    }

    await qdrant.upsert(
        record_id=body.record_id,
        vector=vector,
        payload=payload,
    )

    return {"status": "ok", "record_id": body.record_id}
