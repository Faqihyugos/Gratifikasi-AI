#!/usr/bin/env python
"""
CLI script to promote a registered model version to Production.

Usage:
    python scripts/promote_to_production.py \
        --model-name gratifikasi_classifier \
        --version 3
"""
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a model version to Production stage in MLflow Model Registry"
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_REGISTRY_NAME", "gratifikasi_classifier"),
        help="Name of the registered model",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Model version number to promote",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow Tracking URI",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        help="Archive existing Production versions",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    print(f"Promoting model '{args.model_name}' version {args.version} to Production...")

    client.transition_model_version_stage(
        name=args.model_name,
        version=args.version,
        stage="Production",
        archive_existing_versions=args.archive_existing,
    )

    print(f"Done. Model '{args.model_name}' v{args.version} is now in Production.")

    mv = client.get_model_version(args.model_name, args.version)
    print(f"Current stage: {mv.current_stage}")
    print(f"Run ID: {mv.run_id}")


if __name__ == "__main__":
    main()
