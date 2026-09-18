"""Fit and evaluate the two logistic baseline models."""

from __future__ import annotations

import time
from typing import Any

from .evaluation import binary_probability_report
from .logistic_contracts import LogisticTrainingResult, PreparedLogisticData, TARGETS
from .logistic_data import iter_split_batches


def train_logistic_models(
    data: PreparedLogisticData,
    *,
    epochs: int,
    batch_size: int,
    random_seed: int,
    progress: Any | None,
) -> LogisticTrainingResult:
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(copy=False)
    class_counts = {target: np.zeros(2, dtype=np.int64) for target in TARGETS}
    train_rows = 0
    for batch_number, (matrix, targets) in enumerate(
        iter_split_batches(
            data.dataset_path,
            data.features,
            "train",
            batch_size=batch_size,
        ),
        start=1,
    ):
        scaler.partial_fit(matrix)
        train_rows += len(matrix)
        for target in TARGETS:
            class_counts[target] += np.bincount(targets[target], minlength=2)
        if progress is not None and batch_number % 10 == 0:
            progress(f"prepared training batch {batch_number}: {train_rows} rows")
    if train_rows == 0:
        raise ValueError("Training split contains no rows")

    class_weights = {
        target: balanced_weights(counts) for target, counts in class_counts.items()
    }
    classifiers = {
        target: SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            learning_rate="optimal",
            average=True,
            random_state=random_seed,
        )
        for target in TARGETS
    }

    for epoch in range(epochs):
        rng = np.random.default_rng(random_seed + epoch)
        for matrix, targets in iter_split_batches(
            data.dataset_path,
            data.features,
            "train",
            batch_size=batch_size,
        ):
            scaler.transform(matrix, copy=False)
            order = rng.permutation(len(matrix))
            for target in TARGETS:
                truth = targets[target][order]
                weights = class_weights[target]
                sample_weight = np.where(
                    truth == 1,
                    weights["positive"],
                    weights["negative"],
                )
                classifiers[target].partial_fit(
                    matrix[order],
                    truth,
                    classes=np.array([0, 1], dtype=np.int8),
                    sample_weight=sample_weight,
                )
        if progress is not None:
            progress(f"completed logistic epoch {epoch + 1}/{epochs}")

    metrics = evaluate_logistic_models(
        data,
        scaler=scaler,
        classifiers=classifiers,
        batch_size=batch_size,
    )
    return LogisticTrainingResult(
        scaler=scaler,
        classifiers=classifiers,
        class_counts=class_counts,
        class_weights=class_weights,
        metrics=metrics,
    )


def evaluate_logistic_models(
    data: PreparedLogisticData,
    *,
    scaler: Any,
    classifiers: dict[str, Any],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    truths: dict[str, list[Any]] = {target: [] for target in TARGETS}
    probabilities: dict[str, list[Any]] = {target: [] for target in TARGETS}
    inference_seconds = {target: 0.0 for target in TARGETS}

    for matrix, targets in iter_split_batches(
        data.dataset_path,
        data.features,
        "validation",
        batch_size=batch_size,
    ):
        scaler.transform(matrix, copy=False)
        for target in TARGETS:
            started = time.perf_counter()
            probability = classifiers[target].predict_proba(matrix)[:, 1]
            inference_seconds[target] += time.perf_counter() - started
            truths[target].append(targets[target])
            probabilities[target].append(probability)

    if not truths[TARGETS[0]]:
        raise ValueError("Validation split contains no rows")
    return {
        target: binary_probability_report(
            np.concatenate(truths[target]),
            np.concatenate(probabilities[target]),
            inference_seconds=inference_seconds[target],
        )
        for target in TARGETS
    }


def balanced_weights(counts: Any) -> dict[str, float]:
    if len(counts) != 2 or (counts <= 0).any():
        raise ValueError("Training split must contain both target classes")
    total = float(counts.sum())
    return {
        "negative": total / (2.0 * float(counts[0])),
        "positive": total / (2.0 * float(counts[1])),
    }


__all__ = ["balanced_weights", "evaluate_logistic_models", "train_logistic_models"]
