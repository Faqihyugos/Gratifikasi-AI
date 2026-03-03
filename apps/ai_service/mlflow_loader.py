"""MLflow model loader with fallback."""
import os
from typing import Optional, Dict, Any
import mlflow
import mlflow.transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from libs.common.logging import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.environ.get("MODEL_URI", "models:/gratifikasi_classifier/Production")
FALLBACK_MODEL = os.environ.get(
    "FALLBACK_BASE_MODEL", "indobenchmark/indobert-base-p1"
)
LABEL2ID = {"Milik Negara": 0, "Bukan Milik Negara": 1}
ID2LABEL = {0: "Milik Negara", 1: "Bukan Milik Negara"}


class ModelLoader:
    """Loads the classifier model from MLflow registry with fallback."""

    def __init__(self) -> None:
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.model_info: Dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load(self) -> None:
        """Load model from MLflow or fallback to base model."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        loaded = False

        try:
            logger.info("Loading model from MLflow: %s", MODEL_URI)
            components = mlflow.transformers.load_model(MODEL_URI, return_type="components")
            self.tokenizer = components["tokenizer"]
            self.model = components["model"].to(self.device)
            self.model.eval()

            model_name = MODEL_URI.split("/")[-2] if "models:/" in MODEL_URI else "unknown"
            stage = MODEL_URI.split("/")[-1] if "models:/" in MODEL_URI else "unknown"

            self.model_info = {
                "uri": MODEL_URI,
                "model_name": model_name,
                "stage": stage,
                "source": "mlflow_registry",
            }
            loaded = True
            logger.info("Loaded model from MLflow registry: %s", MODEL_URI)

        except Exception as exc:
            logger.warning(
                "Could not load model from MLflow (%s). Falling back to base model.",
                exc,
            )

        if not loaded:
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Load base model as fallback."""
        logger.warning("Loading fallback model: %s", FALLBACK_MODEL)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                FALLBACK_MODEL,
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            ).to(self.device)
            self.model.eval()
            self.model_info = {
                "uri": FALLBACK_MODEL,
                "model_name": FALLBACK_MODEL,
                "stage": "fallback",
                "source": "local_fallback",
            }
            logger.warning("Fallback model loaded. Predictions may be random until trained.")
        except Exception as exc:
            logger.error("Failed to load fallback model: %s", exc)
            self.model_info = {"error": str(exc), "source": "none"}

    def predict(self, text: str) -> Dict[str, Any]:
        """Run inference and return label + confidence."""
        if self.model is None or self.tokenizer is None:
            return {"label": "UNKNOWN", "confidence": 0.0}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        label = ID2LABEL.get(pred_idx, "UNKNOWN")
        confidence = probs[pred_idx].item()

        return {"label": label, "confidence": float(confidence)}
