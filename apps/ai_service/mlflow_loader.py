"""MLflow ONNX model loader — uses onnxruntime, no PyTorch required."""
import os
from typing import Optional, Dict, Any, List
import numpy as np
import mlflow
import onnxruntime as ort
from transformers import AutoTokenizer

from libs.common.logging import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.environ.get("MODEL_URI", "models:/gratifikasi_classifier/Production")
LABEL2ID = {"Milik Negara": 0, "Bukan Milik Negara": 1}
ID2LABEL = {0: "Milik Negara", 1: "Bukan Milik Negara"}
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))


class ModelLoader:
    """Loads the ONNX classifier from MLflow artifact store and runs inference via onnxruntime."""

    def __init__(self) -> None:
        self.tokenizer: Optional[Any] = None
        self.session: Optional[ort.InferenceSession] = None
        self.model_info: Dict[str, Any] = {}
        self._input_names: List[str] = []
        self.device = "cpu"  # kept for interface compatibility

    async def load(self) -> None:
        """Download the onnx_model artifact from MLflow and initialise onnxruntime session."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        loaded = False

        try:
            model_name = MODEL_URI.split("/")[-2] if "models:/" in MODEL_URI else "unknown"
            stage = MODEL_URI.split("/")[-1] if "models:/" in MODEL_URI else "unknown"

            logger.info("Fetching ONNX model from MLflow: %s", MODEL_URI)
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            matching = [v for v in versions if v.current_stage == stage]
            if not matching:
                raise ValueError(f"No model version at stage '{stage}' for '{model_name}'")

            run_id = sorted(matching, key=lambda v: int(v.version), reverse=True)[0].run_id
            local_dir = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="onnx_model"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.session = ort.InferenceSession(
                f"{local_dir}/model.onnx",
                providers=["CPUExecutionProvider"],
            )
            self._input_names = [inp.name for inp in self.session.get_inputs()]

            # Enrich model_info with run metrics for the model-info page
            run = client.get_run(run_id)
            m = run.data.metrics
            p = run.data.params
            training_date = (
                run.info.start_time / 1000
                if run.info.start_time else None
            )
            import datetime as _dt
            training_date_str = (
                _dt.datetime.utcfromtimestamp(training_date).isoformat()
                if training_date else None
            )
            self.model_info = {
                "uri": MODEL_URI,
                "model_name": model_name,
                "stage": stage,
                "source": "mlflow_onnx",
                "run_id": run_id,
                "eval_f1": m.get("eval_f1", 0.0),
                "eval_accuracy": m.get("eval_accuracy", 0.0),
                "dataset_size": int(p.get("train_size", 0)) + int(p.get("eval_size", 0)),
                "training_date": training_date_str,
            }
            loaded = True
            logger.info("ONNX model loaded from MLflow: %s", MODEL_URI)

        except Exception as exc:
            logger.warning("Could not load ONNX model from MLflow (%s). Using fallback.", exc)

        if not loaded:
            self._load_fallback()

    def _load_fallback(self) -> None:
        """Mark service as degraded — no ONNX model available yet (pre-training state)."""
        logger.warning(
            "No ONNX model available. Classifier will return UNKNOWN until first training run."
        )
        self.session = None
        self.tokenizer = None
        self.model_info = {"stage": "none", "source": "unavailable"}

    def predict(self, text: str) -> Dict[str, Any]:
        """Run ONNX inference and return label + confidence."""
        if self.session is None or self.tokenizer is None:
            return {"label": "UNKNOWN", "confidence": 0.0}

        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        ort_inputs = {
            k: v.astype(np.int64)
            for k, v in inputs.items()
            if k in self._input_names
        }

        logits = self.session.run(None, ort_inputs)[0][0]  # shape (num_labels,)
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / exp_l.sum()
        pred_id = int(np.argmax(probs))
        probabilities = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "label": ID2LABEL.get(pred_id, "UNKNOWN"),
            "confidence": float(probs[pred_id]),
            "probabilities": probabilities,
        }
