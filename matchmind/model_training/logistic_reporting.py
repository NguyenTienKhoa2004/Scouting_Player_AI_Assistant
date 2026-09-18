"""Build the validation report for logistic baseline models."""

from __future__ import annotations

from typing import Any

from .logistic_contracts import (
    LOGISTIC_BASELINE_VERSION,
    PREPROCESSING_VERSION,
    TARGETS,
    LogisticTrainingResult,
    PreparedLogisticData,
)


def build_logistic_report(
    data: PreparedLogisticData,
    result: LogisticTrainingResult,
    *,
    epochs: int,
    batch_size: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "baseline_version": LOGISTIC_BASELINE_VERSION,
        "algorithm": "logistic regression with SGD",
        "random_seed": random_seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "feature_count": len(data.features),
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "fit_split": "train",
            "algorithm": "StandardScaler",
        },
        "class_weights": {
            target: {
                "negative_count": int(result.class_counts[target][0]),
                "positive_count": int(result.class_counts[target][1]),
                **result.class_weights[target],
            }
            for target in TARGETS
        },
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "metrics": result.metrics,
    }


__all__ = ["build_logistic_report"]
