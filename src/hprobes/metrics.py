"""Threshold calibration and binary classification metrics."""

from __future__ import annotations

import math
from typing import Any, Sequence

def calibrate_threshold(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, dict[str, Any]]:
    if not scores or len(scores) != len(labels):
        raise ValueError("Threshold calibration requires equal nonempty scores and labels")
    unique = sorted(set(float(score) for score in scores))
    candidates = [math.nextafter(unique[0], -math.inf)]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(math.nextafter(unique[-1], math.inf))
    positives = sum(labels)
    negatives = len(labels) - positives
    choices = []
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
        tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
        balanced = 0.5 * (tp / positives + tn / negatives) if positives and negatives else (tp + tn) / len(labels)
        accuracy = sum(y == p for y, p in zip(labels, predictions)) / len(labels)
        gap = abs(sum(predictions) / len(predictions) - positives / len(labels))
        choices.append((balanced, accuracy, -gap, -threshold, threshold))
    selected = max(choices)
    return float(selected[-1]), {
        "objective": "maximum source-validation balanced accuracy",
        "balanced_accuracy": selected[0],
        "accuracy": selected[1],
        "n": len(labels),
        "n_correct": int(sum(labels)),
        "has_both_classes": len(set(labels)) == 2,
    }


def binary_metrics(labels: Sequence[int], predictions: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    if not labels:
        return {"n": 0}
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
    balanced = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None
    return {
        "n": len(labels),
        "n_correct": int(sum(labels)),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auc,
        "confusion_matrix": {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn},
    }
