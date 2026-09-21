"""High-level training flow for the two logistic baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .logistic_contracts import (
    LOGISTIC_BASELINE_REPORT_FILENAME,
    LOGISTIC_BASELINE_VERSION,
    PREPROCESSING_VERSION,
    LogisticBaselinePaths,
)
from .logistic_data import prepare_logistic_data
from .logistic_reporting import build_logistic_report
from .logistic_storage import save_logistic_results
from .logistic_training import train_logistic_models


class LogisticBaselineTrainer:
    """Train simple reference models for comparison with XGBoost."""

    def write(
        self,
        artifact_directory: Path,
        *,
        epochs: int = 3,
        batch_size: int = 16_384,
        random_seed: int = 20260914,
        progress: Any | None = None,
    ) -> LogisticBaselinePaths:
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")

        data = prepare_logistic_data(artifact_directory)
        result = train_logistic_models(
            data,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            progress=progress,
        )
        report = build_logistic_report(
            data,
            result,
            epochs=epochs,
            batch_size=batch_size,
            random_seed=random_seed,
        )
        paths = save_logistic_results(data, result, report)

        if progress is not None:
            progress(f"completed logistic baseline training: {paths.directory}")
        return paths


__all__ = [
    "LOGISTIC_BASELINE_REPORT_FILENAME",
    "LOGISTIC_BASELINE_VERSION",
    "PREPROCESSING_VERSION",
    "LogisticBaselinePaths",
    "LogisticBaselineTrainer",
]
