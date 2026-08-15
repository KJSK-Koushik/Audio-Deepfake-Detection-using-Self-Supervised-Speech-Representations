import numpy as np

from src.evaluation.binary import binary_classification_metrics, equal_error_rate


def test_perfect_binary_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = binary_classification_metrics(labels, scores)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_equal_error_rate_for_uninformative_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])

    eer, threshold = equal_error_rate(labels, scores)

    assert eer == 0.5
    assert threshold is None
