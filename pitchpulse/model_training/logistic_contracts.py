"""Data passed between logistic-baseline training steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGISTIC_BASELINE_VERSION = "vaep-logistic-sgd-baseline-v1"
PREPROCESSING_VERSION = "standard-scaler-train-only-v1"
LOGISTIC_BASELINE_REPORT_FILENAME = "logistic_baseline_report.json"
TARGETS = ("scores", "concedes")


@dataclass(frozen=True, slots=True)
class LogisticBaselinePaths:
    directory: Path
    preprocessor: Path
    score_model: Path
    concede_model: Path
    report: Path
    training_manifest: Path


@dataclass(slots=True)
class PreparedLogisticData:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    features: tuple[str, ...]
    dataset_path: Path


@dataclass(slots=True)
class LogisticTrainingResult:
    scaler: Any
    classifiers: dict[str, Any]
    class_counts: dict[str, Any]
    class_weights: dict[str, dict[str, float]]
    metrics: dict[str, dict[str, Any]]


__all__ = [
    "LOGISTIC_BASELINE_REPORT_FILENAME",
    "LOGISTIC_BASELINE_VERSION",
    "PREPROCESSING_VERSION",
    "TARGETS",
    "LogisticBaselinePaths",
    "LogisticTrainingResult",
    "PreparedLogisticData",
]
