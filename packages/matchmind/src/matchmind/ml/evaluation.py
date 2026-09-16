"""Evaluation helpers for binary VAEP probability models."""

from __future__ import annotations

from typing import Any


def binary_probability_report(
    y_true: Any,
    probabilities: Any,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
    inference_seconds: float | None = None,
) -> dict[str, Any]:
    """Evaluate a binary probability model without choosing on the test set."""

    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    truth = np.asarray(y_true, dtype=np.int8)
    predicted_probability = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or predicted_probability.ndim != 1:
        raise ValueError("binary evaluation inputs must be one-dimensional")
    if len(truth) != len(predicted_probability) or not len(truth):
        raise ValueError("binary evaluation inputs must have equal non-zero lengths")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("binary evaluation requires both target classes")
    if not np.isfinite(predicted_probability).all() or (
        (predicted_probability < 0) | (predicted_probability > 1)
    ).any():
        raise ValueError("predicted probabilities must be finite and in [0, 1]")

    hard_prediction = predicted_probability >= threshold
    observed, predicted = calibration_curve(
        truth,
        predicted_probability,
        n_bins=calibration_bins,
        strategy="quantile",
    )
    latency = None
    if inference_seconds is not None:
        latency = {
            "total_seconds": inference_seconds,
            "microseconds_per_row": inference_seconds / len(truth) * 1_000_000,
        }
    return {
        "row_count": int(len(truth)),
        "positive_count": int(truth.sum()),
        "positive_frequency": float(truth.mean()),
        "pr_auc": float(average_precision_score(truth, predicted_probability)),
        "roc_auc": float(roc_auc_score(truth, predicted_probability)),
        "log_loss": float(log_loss(truth, predicted_probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(truth, predicted_probability)),
        "threshold": float(threshold),
        "precision": float(precision_score(truth, hard_prediction, zero_division=0)),
        "recall": float(recall_score(truth, hard_prediction, zero_division=0)),
        "calibration_curve": [
            {
                "mean_predicted_probability": float(x),
                "observed_positive_frequency": float(y),
            }
            for x, y in zip(predicted, observed, strict=True)
        ],
        "inference_latency": latency,
    }


__all__ = ["binary_probability_report"]
