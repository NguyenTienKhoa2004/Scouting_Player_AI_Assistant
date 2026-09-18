"""Fit XGBoost candidates, then calibrate and evaluate the winner."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from .calibration import fit_probability_calibrator
from .evaluation import binary_probability_report
from .xgboost_contracts import LoadedSplit, TARGETS, TargetResult, XGBoostTrainingError


def fit_all_targets(
    *,
    xgb: Any,
    train: LoadedSplit,
    validation: LoadedSplit,
    phase_indices: dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    random_seed: int,
    early_stopping_rounds: int,
    n_jobs: int,
    baseline_metrics: Mapping[str, dict[str, Any]],
    progress: Any | None,
) -> dict[str, TargetResult]:
    tuning_rows = phase_indices["tuning"]
    calibration_rows = phase_indices["calibration_fit"]
    evaluation_rows = phase_indices["calibration_evaluation"]

    train_matrix = xgb.DMatrix(train.matrix, nthread=n_jobs)
    tuning_matrix = xgb.DMatrix(validation.matrix[tuning_rows], nthread=n_jobs)
    calibration_matrix = xgb.DMatrix(
        validation.matrix[calibration_rows], nthread=n_jobs
    )
    evaluation_matrix = xgb.DMatrix(
        validation.matrix[evaluation_rows], nthread=n_jobs
    )

    results: dict[str, TargetResult] = {}
    for target_index, target in enumerate(TARGETS):
        if progress is not None:
            progress(
                f"training {len(candidates)} XGBoost candidates for {target}"
            )
        results[target] = fit_target(
            xgb=xgb,
            target=target,
            train=train,
            validation=validation,
            phase_indices=phase_indices,
            train_matrix=train_matrix,
            tuning_matrix=tuning_matrix,
            calibration_matrix=calibration_matrix,
            evaluation_matrix=evaluation_matrix,
            candidates=candidates,
            random_seed=random_seed + target_index,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            baseline_metrics=baseline_metrics[target],
        )
    return results


def fit_target(
    *,
    xgb: Any,
    target: str,
    train: LoadedSplit,
    validation: LoadedSplit,
    phase_indices: dict[str, Any],
    train_matrix: Any,
    tuning_matrix: Any,
    calibration_matrix: Any,
    evaluation_matrix: Any,
    candidates: Sequence[Mapping[str, Any]],
    random_seed: int,
    early_stopping_rounds: int,
    n_jobs: int,
    baseline_metrics: dict[str, Any],
) -> TargetResult:
    train_truth = train.targets[target]
    negative_count = int((train_truth == 0).sum())
    positive_count = int((train_truth == 1).sum())
    if not negative_count or not positive_count:
        raise XGBoostTrainingError(
            f"Training split must contain both classes for {target}"
        )

    tuning_rows = phase_indices["tuning"]
    calibration_rows = phase_indices["calibration_fit"]
    evaluation_rows = phase_indices["calibration_evaluation"]
    tuning_truth = validation.targets[target][tuning_rows]
    train_matrix.set_label(train_truth)
    tuning_matrix.set_label(tuning_truth)

    trained_candidates: list[tuple[Any, dict[str, Any]]] = []
    for index, parameters in enumerate(candidates):
        model = train_candidate(
            xgb=xgb,
            train_matrix=train_matrix,
            tuning_matrix=tuning_matrix,
            parameters=parameters,
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=n_jobs,
            scale_pos_weight=negative_count / positive_count,
        )
        candidate_report = {
            "candidate": index,
            "parameters": dict(parameters),
            "best_iteration": best_iteration(model),
            "tuning_metrics": binary_probability_report(
                tuning_truth,
                predict(model, tuning_matrix),
            ),
        }
        trained_candidates.append((model, candidate_report))

    model, selected = max(
        trained_candidates,
        key=lambda item: candidate_rank(item[1]["tuning_metrics"]),
    )
    calibration_probabilities = predict(model, calibration_matrix)
    calibrator = fit_probability_calibrator(
        calibration_probabilities,
        validation.targets[target][calibration_rows],
        random_seed=random_seed,
    )

    started = time.perf_counter()
    raw_probabilities = predict(model, evaluation_matrix)
    calibrated_probabilities = calibrator.predict(raw_probabilities)
    inference_seconds = time.perf_counter() - started
    raw_metrics = binary_probability_report(
        validation.targets[target][evaluation_rows],
        raw_probabilities,
    )
    calibrated_metrics = binary_probability_report(
        validation.targets[target][evaluation_rows],
        calibrated_probabilities,
        inference_seconds=inference_seconds,
    )
    return TargetResult(
        model=model,
        calibrator=calibrator,
        report={
            "selected_candidate": selected["candidate"],
            "parameters": selected["parameters"],
            "candidate_results": [item[1] for item in trained_candidates],
            "best_iteration": selected["best_iteration"],
            "negative_count": negative_count,
            "positive_count": positive_count,
            "calibration": calibrator.method,
            "raw_evaluation_metrics": raw_metrics,
            "calibrated_evaluation_metrics": calibrated_metrics,
            "logistic_baseline_metrics": baseline_metrics,
            "beats_logistic_baseline": beats_baseline(
                calibrated_metrics, baseline_metrics
            ),
        },
    )


def train_candidate(
    *,
    xgb: Any,
    train_matrix: Any,
    tuning_matrix: Any,
    parameters: Mapping[str, Any],
    random_seed: int,
    early_stopping_rounds: int,
    n_jobs: int,
    scale_pos_weight: float,
) -> Any:
    model_parameters = dict(parameters)
    num_boost_round = int(model_parameters.pop("n_estimators"))
    return xgb.train(
        {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "eval_metric": "aucpr",
            "seed": random_seed,
            "nthread": n_jobs,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": 0,
            **model_parameters,
        },
        train_matrix,
        num_boost_round=num_boost_round,
        evals=[(tuning_matrix, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )


def candidate_rank(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Higher PR-AUC wins; log-loss and Brier score break ties."""

    return (
        float(metrics["pr_auc"]),
        -float(metrics["log_loss"]),
        -float(metrics["brier_score"]),
    )


def predict(model: Any, matrix: Any) -> Any:
    selected_iteration = best_iteration(model)
    iteration_range = (
        (0, selected_iteration + 1) if selected_iteration is not None else (0, 0)
    )
    return model.predict(matrix, iteration_range=iteration_range)


def beats_baseline(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    probability_quality_not_worse = (
        candidate["brier_score"] <= baseline["brier_score"]
        or candidate["log_loss"] <= baseline["log_loss"]
    )
    return bool(
        candidate["pr_auc"] > baseline["pr_auc"]
        and probability_quality_not_worse
    )


def best_iteration(model: Any) -> int | None:
    try:
        return int(model.best_iteration)
    except (AttributeError, TypeError):
        return None


__all__ = [
    "beats_baseline",
    "best_iteration",
    "candidate_rank",
    "fit_all_targets",
    "fit_target",
    "predict",
    "train_candidate",
]
