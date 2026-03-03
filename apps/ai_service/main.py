"""FastAPI AI Inference Service main application."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from libs.common.logging import configure_logging, get_logger
from .mlflow_loader import ModelLoader
from .qdrant_wrapper import QdrantWrapper
from .embedding import EmbeddingModel
from .routers import predict, model_info, cases, health

configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and connections on startup."""
    logger.info("Starting AI service...")

    app.state.embedding = EmbeddingModel()
    logger.info("Embedding model loaded")

    app.state.qdrant = QdrantWrapper()
    await app.state.qdrant.init_collection()
    logger.info("Qdrant collection initialized")

    app.state.model_loader = ModelLoader()
    await app.state.model_loader.load()
    logger.info("Classifier model loaded: %s", app.state.model_loader.model_info)

    yield

    logger.info("Shutting down AI service")


app = FastAPI(
    title="Gratifikasi AI Service",
    description="AI-based gratification classification service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(model_info.router)
app.include_router(cases.router)
