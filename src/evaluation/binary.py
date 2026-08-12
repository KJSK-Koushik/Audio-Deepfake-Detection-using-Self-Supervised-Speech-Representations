"""Binary classification metrics used by reproducible experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float | None]:
    """Return EER and its score threshold for positive-class scores."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.nanargmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = (false_positive_rate[index] + false_negative_rate[index]) / 2.0
    threshold = float(thresholds[index])
    return float(eer), threshold if np.isfinite(threshold) else None


def binary_classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    predictions = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    eer, eer_threshold = equal_error_rate(labels, scores)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": matrix.tolist(),
    }
