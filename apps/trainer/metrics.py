"""Evaluation metrics for the fine-tuning pipeline."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, f1 (macro) for Trainer using sklearn."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }
