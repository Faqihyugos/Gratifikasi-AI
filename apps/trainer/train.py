"""
Fine-tuning pipeline for Gratifikasi classifier.

Usage:
    docker compose run --rm trainer
or:
    uv run python apps/trainer/train.py
"""
import os
import sys
import logging
import random
import numpy as np
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apps.trainer.data_loader import load_training_data  # noqa: E402
from apps.trainer.metrics import compute_metrics  # noqa: E402
from apps.trainer.mlflow_utils import setup_mlflow, log_training_run  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "indobenchmark/indobert-base-p1")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/gratifikasi_model")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "256"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
SEED = int(os.environ.get("SEED", "42"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "3"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.15"))

LABEL2ID = {"Milik Negara": 0, "Bukan Milik Negara": 1}
ID2LABEL = {0: "Milik Negara", 1: "Bukan Milik Negara"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)
    logger.info("Starting training pipeline")
    logger.info("Base model: %s", BASE_MODEL)

    samples = load_training_data(
        database_url=DATABASE_URL,
        train_data_path=TRAIN_DATA_PATH if TRAIN_DATA_PATH else None,
    )

    if len(samples) < 10:
        logger.error(
            "Not enough training samples (%d). Need at least 10.", len(samples)
        )
        sys.exit(1)

    texts = [s["text"] for s in samples]
    labels = [LABEL2ID[s["label"]] for s in samples]

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    split = dataset.train_test_split(
        test_size=VAL_SPLIT, seed=SEED, stratify_by_column="label"
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info("Dataset: %d train, %d eval", len(train_dataset), len(eval_dataset))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=SEED,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    eval_results = trainer.evaluate()
    logger.info("Evaluation results: %s", eval_results)

    setup_mlflow(MLFLOW_TRACKING_URI)

    params = {
        "base_model": BASE_MODEL,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "max_length": MAX_LENGTH,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "seed": SEED,
    }

    run_id, model_version = log_training_run(
        params=params,
        metrics=eval_results,
        output_dir=OUTPUT_DIR,
        tokenizer=tokenizer,
        model=trainer.model,
        input_example="Penerimaan hadiah berupa uang tunai sebesar Rp 500.000",
    )

    logger.info("Training complete!")
    logger.info("MLflow run_id: %s", run_id)
    logger.info("Registered model version: %s", model_version)
    logger.info("Artifacts in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
