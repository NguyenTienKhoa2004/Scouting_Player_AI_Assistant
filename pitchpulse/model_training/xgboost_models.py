"""High-level training flow for the two VAEP XGBoost models."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .xgboost_baseline import evaluate_logistic_baseline
from .xgboost_contracts import (
    DEFAULT_XGBOOST_CANDIDATES,
    TARGETS,
    VALIDATION_PARTITION_VERSION,
    XGBOOST_BUNDLE_VERSION,
    XGBOOST_REPORT_FILENAME,
    XGBoostBundlePaths,
    XGBoostTrainingError,
)
from .xgboost_data import prepare_training_data
from .xgboost_reporting import build_validation_report
from .xgboost_storage import save_training_results
from .xgboost_training import fit_all_targets


class XGBoostVaepTrainer:
    """Run the XGBoost training pipeline from prepared artifacts."""

    def write(
        self,
        artifact_directory: Path,
        *,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        batch_size: int = 16_384,
        random_seed: int = 20260917,
        early_stopping_rounds: int = 40,
        n_jobs: int = -1,
        progress: Any | None = None,
    ) -> XGBoostBundlePaths:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "Install the project requirements before training XGBoost"
            ) from exc

        if min(batch_size, early_stopping_rounds) <= 0:
            raise ValueError("Training sizes must be positive")

        requested_candidates = (
            DEFAULT_XGBOOST_CANDIDATES if candidates is None else candidates
        )
        if not requested_candidates:
            raise ValueError("At least one XGBoost candidate is required")
        model_candidates = [
            {**DEFAULT_XGBOOST_CANDIDATES[0], **dict(candidate)}
            for candidate in requested_candidates
        ]
        if any(int(item["n_estimators"]) <= 0 for item in model_candidates):
            raise ValueError("n_estimators must be positive")
        data = prepare_training_data(
            artifact_directory,
            batch_size=batch_size,
            progress=progress,
        )
        baseline_metrics = evaluate_logistic_baseline(
            data,
            batch_size=batch_size,
        )

        target_results = fit_all_targets(
            xgb=xgb,
            train=data.train,
            validation=data.validation,
            phase_indices=data.phase_indices,
            candidates=model_candidates,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            baseline_metrics=baseline_metrics,
            progress=progress,
        )
        report = build_validation_report(
            features=data.features,
            feature_manifest=data.feature_manifest,
            random_seed=random_seed,
            train=data.train,
            phase_matches=data.phase_matches,
            phase_indices=data.phase_indices,
            target_results=target_results,
        )
        paths = save_training_results(
            root=data.root,
            manifest_path=data.manifest_path,
            manifest=data.manifest,
            target_results=target_results,
            report=report,
        )

        if progress is not None:
            progress(f"completed XGBoost training: {paths.directory}")
        return paths


__all__ = [
    "DEFAULT_XGBOOST_CANDIDATES",
    "TARGETS",
    "VALIDATION_PARTITION_VERSION",
    "XGBOOST_BUNDLE_VERSION",
    "XGBOOST_REPORT_FILENAME",
    "XGBoostBundlePaths",
    "XGBoostTrainingError",
    "XGBoostVaepTrainer",
]
