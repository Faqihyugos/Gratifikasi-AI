"""MLflow utilities for training pipeline."""
import os
import logging
import mlflow
import mlflow.transformers

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_REGISTRY_NAME", "gratifikasi_classifier")
PROMOTE_F1_THRESHOLD = float(os.environ.get("PROMOTE_F1_THRESHOLD", "0.85"))


def setup_mlflow(tracking_uri: str, experiment_name: str = "gratifikasi_training") -> None:
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured: %s / %s", tracking_uri, experiment_name)


def log_training_run(
    params: dict,
    metrics: dict,
    output_dir: str,
    tokenizer,
    model,
    input_example: str,
) -> tuple:
    """
    Log training run to MLflow, register model, and optionally promote to Staging.

    Returns (run_id, model_version).
    """
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )

        logger.info("Model logged and registered as: %s", MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    version = latest.version

    f1 = metrics.get("eval_f1", 0.0)
    if f1 >= PROMOTE_F1_THRESHOLD:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info(
            "Auto-promoted model version %s to Staging (F1=%.4f >= %.4f)",
            version, f1, PROMOTE_F1_THRESHOLD,
        )
    else:
        logger.info(
            "Model version %s NOT promoted to Staging (F1=%.4f < %.4f)",
            version, f1, PROMOTE_F1_THRESHOLD,
        )

    return run_id, version
