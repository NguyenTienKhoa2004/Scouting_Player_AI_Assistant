"""Shared contracts and configuration for VAEP XGBoost training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


XGBOOST_BUNDLE_VERSION = "vaep-xgboost-calibrated-v1"
XGBOOST_REPORT_FILENAME = "xgboost_validation_report.json"
VALIDATION_PARTITION_VERSION = "match-level-validation-50-25-25-v1"
TARGETS = ("scores", "concedes")

DEFAULT_XGBOOST_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 250,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 250,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 350,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
    },
)


class XGBoostTrainingError(ValueError):
    """Raised when XGBoost training cannot complete without leakage."""


@dataclass(frozen=True, slots=True)
class XGBoostBundlePaths:
    directory: Path
    score_model: Path
    concede_model: Path
    score_calibration: Path
    concede_calibration: Path
    report: Path
    training_manifest: Path


@dataclass(slots=True)
class LoadedSplit:
    matrix: Any
    targets: dict[str, Any]
    match_ids: Any


@dataclass(slots=True)
class PreparedXGBoostData:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    features: tuple[str, ...]
    train: LoadedSplit
    validation: LoadedSplit
    phase_matches: dict[str, set[int]]
    phase_indices: dict[str, Any]


@dataclass(slots=True)
class TargetResult:
    model: Any
    calibrator: Any
    report: dict[str, Any]


__all__ = [
    "DEFAULT_XGBOOST_CANDIDATES",
    "LoadedSplit",
    "PreparedXGBoostData",
    "TARGETS",
    "TargetResult",
    "VALIDATION_PARTITION_VERSION",
    "XGBOOST_BUNDLE_VERSION",
    "XGBOOST_REPORT_FILENAME",
    "XGBoostBundlePaths",
    "XGBoostTrainingError",
]
