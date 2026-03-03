"""Model info endpoint."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/model")
async def model_info(request: Request):
    return request.app.state.model_loader.model_info
