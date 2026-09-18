"""Load input files and save XGBoost training results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .xgboost_contracts import (
    TARGETS,
    XGBOOST_BUNDLE_VERSION,
    XGBOOST_REPORT_FILENAME,
    TargetResult,
    XGBoostBundlePaths,
)
from .training_files import write_json


def save_training_results(
    *,
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    target_results: Mapping[str, TargetResult],
    report: Mapping[str, Any],
) -> XGBoostBundlePaths:
    import joblib

    output = root / XGBOOST_BUNDLE_VERSION
    output.mkdir(parents=True, exist_ok=True)
    paths = XGBoostBundlePaths(
        directory=output,
        score_model=output / "score_xgboost.json",
        concede_model=output / "concede_xgboost.json",
        score_calibration=output / "score_calibration.joblib",
        concede_calibration=output / "concede_calibration.joblib",
        report=output / XGBOOST_REPORT_FILENAME,
        training_manifest=manifest_path,
    )

    target_results["scores"].model.save_model(paths.score_model)
    target_results["concedes"].model.save_model(paths.concede_model)
    joblib.dump(target_results["scores"].calibrator, paths.score_calibration)
    joblib.dump(target_results["concedes"].calibrator, paths.concede_calibration)
    write_json(paths.report, report)

    generated_files = {
        "score_xgboost.json": paths.score_model,
        "concede_xgboost.json": paths.concede_model,
        "score_calibration.joblib": paths.score_calibration,
        "concede_calibration.joblib": paths.concede_calibration,
        XGBOOST_REPORT_FILENAME: paths.report,
    }
    updated = dict(manifest)
    updated["stage"] = "xgboost_training"
    updated["status"] = "xgboost_validation_complete"
    updated["artifacts"] = {
        **dict(manifest.get("artifacts") or {}),
        **{
            name: {"path": str(path)}
            for name, path in generated_files.items()
        },
    }
    updated["models"] = {
        **dict(manifest.get("models") or {}),
        "status": "xgboost_validation_complete",
        "xgboost_bundle_version": XGBOOST_BUNDLE_VERSION,
        "calibration": {
            target: target_results[target].calibrator.method for target in TARGETS
        },
        "xgboost_validation_metrics": {
            target: target_results[target].report["calibrated_evaluation_metrics"]
            for target in TARGETS
        },
    }
    updated["ready_for_test_evaluation"] = bool(
        report["ready_for_test_evaluation"]
    )
    write_json(manifest_path, updated)
    return paths


__all__ = ["save_training_results"]
