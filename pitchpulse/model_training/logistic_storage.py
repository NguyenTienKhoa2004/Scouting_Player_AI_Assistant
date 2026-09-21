"""Save logistic baseline models and their validation report."""

from __future__ import annotations

from typing import Any, Mapping

from .logistic_contracts import (
    LOGISTIC_BASELINE_REPORT_FILENAME,
    LOGISTIC_BASELINE_VERSION,
    TARGETS,
    LogisticBaselinePaths,
    LogisticTrainingResult,
    PreparedLogisticData,
)
from .training_files import write_json


def save_logistic_results(
    data: PreparedLogisticData,
    result: LogisticTrainingResult,
    report: Mapping[str, Any],
) -> LogisticBaselinePaths:
    import joblib

    output = data.root / LOGISTIC_BASELINE_VERSION
    output.mkdir(parents=True, exist_ok=True)
    paths = LogisticBaselinePaths(
        directory=output,
        preprocessor=output / "logistic_preprocessor.joblib",
        score_model=output / "score_logistic_baseline.joblib",
        concede_model=output / "concede_logistic_baseline.joblib",
        report=output / LOGISTIC_BASELINE_REPORT_FILENAME,
        training_manifest=data.manifest_path,
    )
    joblib.dump(result.scaler, paths.preprocessor)
    joblib.dump(result.classifiers["scores"], paths.score_model)
    joblib.dump(result.classifiers["concedes"], paths.concede_model)
    write_json(paths.report, report)

    generated_files = {
        paths.preprocessor.name: paths.preprocessor,
        paths.score_model.name: paths.score_model,
        paths.concede_model.name: paths.concede_model,
        paths.report.name: paths.report,
    }
    updated = dict(data.manifest)
    updated["stage"] = "logistic_baseline_training"
    updated["status"] = "logistic_baselines_evaluated"
    updated["artifacts"] = {
        **dict(data.manifest.get("artifacts") or {}),
        **{
            name: {"path": str(path)}
            for name, path in generated_files.items()
        },
    }
    updated["models"] = {
        **dict(data.manifest.get("models") or {}),
        "status": "logistic_baselines_evaluated",
        "baseline_version": LOGISTIC_BASELINE_VERSION,
        "baseline_validation_metrics": {
            target: result.metrics[target] for target in TARGETS
        },
    }
    updated["ready_for_xgboost"] = True
    write_json(data.manifest_path, updated)
    return paths


__all__ = ["save_logistic_results"]
