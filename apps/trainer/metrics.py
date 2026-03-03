"""Evaluation metrics for the fine-tuning pipeline."""
import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, f1 (macro) for Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    prec = precision_metric.compute(
        predictions=predictions, references=labels, average="macro", zero_division=0
    )
    rec = recall_metric.compute(
        predictions=predictions, references=labels, average="macro", zero_division=0
    )
    f1 = f1_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1["f1"],
    }
