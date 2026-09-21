"""Evaluate logistic baselines on the final validation phase."""

from __future__ import annotations

from typing import Any

from .evaluation import binary_probability_report
from .training_files import find_training_file
from .xgboost_contracts import PreparedXGBoostData, TARGETS


def evaluate_logistic_baseline(
    data: PreparedXGBoostData,
    *,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    import joblib
    import numpy as np

    evaluation_indices = data.phase_indices["calibration_evaluation"]
    matrix = data.validation.matrix[evaluation_indices]
    preprocessor = joblib.load(
        find_training_file(
            data.root,
            data.manifest,
            "logistic_preprocessor.joblib",
        )
    )
    models = {
        "scores": joblib.load(
            find_training_file(
                data.root,
                data.manifest,
                "score_logistic_baseline.joblib",
            )
        ),
        "concedes": joblib.load(
            find_training_file(
                data.root,
                data.manifest,
                "concede_logistic_baseline.joblib",
            )
        ),
    }

    probability_chunks: dict[str, list[Any]] = {
        target: [] for target in TARGETS
    }
    for start in range(0, matrix.shape[0], batch_size):
        dense = matrix[start : start + batch_size].toarray()
        scaled = preprocessor.transform(dense)
        for target in TARGETS:
            probability_chunks[target].append(
                models[target].predict_proba(scaled)[:, 1]
            )

    return {
        target: binary_probability_report(
            data.validation.targets[target][evaluation_indices],
            np.concatenate(probability_chunks[target]),
        )
        for target in TARGETS
    }


__all__ = ["evaluate_logistic_baseline"]
