"""Validation report construction for the calibrated XGBoost bundle."""

from __future__ import annotations

from typing import Any, Mapping

from .calibration import CALIBRATION_VERSION
from .xgboost_contracts import (
    TARGETS,
    VALIDATION_PARTITION_VERSION,
    XGBOOST_BUNDLE_VERSION,
    LoadedSplit,
    TargetResult,
)
from .xgboost_data import split_summary


def build_validation_report(
    *,
    features: tuple[str, ...],
    feature_manifest: Mapping[str, Any],
    random_seed: int,
    train: LoadedSplit,
    phase_matches: Mapping[str, set[int]],
    phase_indices: Mapping[str, Any],
    target_results: Mapping[str, TargetResult],
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "bundle_version": XGBOOST_BUNDLE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "validation_partition_version": VALIDATION_PARTITION_VERSION,
        "feature_count": len(features),
        "feature_allowlist_sha256": feature_manifest["feature_allowlist_sha256"],
        "input_matrix": {
            "format": "scipy_csr_float32",
            "xgboost_matrix": "DMatrix",
        },
        "random_seed": random_seed,
        "test_split_accessed": False,
        "data": {
            "train": split_summary(train),
            "validation_partitions": {
                phase: {
                    "match_count": len(phase_matches[phase]),
                    "row_count": int(len(indices)),
                    "purpose": {
                        "tuning": "hyperparameter selection and early stopping",
                        "calibration_fit": "fit probability calibrators",
                        "calibration_evaluation": (
                            "select calibration method and compare baseline"
                        ),
                    }[phase],
                }
                for phase, indices in phase_indices.items()
            },
        },
        "targets": {target: target_results[target].report for target in TARGETS},
    }
    report["ready_for_test_evaluation"] = all(
        target_results[target].report["beats_logistic_baseline"]
        for target in TARGETS
    )
    return report


__all__ = ["build_validation_report"]
