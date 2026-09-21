"""One-time evaluation of frozen VAEP models on the untouched test split."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pitchpulse.shared.file_io import file_sha256
from pitchpulse.shared.paths import PROJECT_ROOT
from pitchpulse.vaep_features.feature_dataset_loader import baseline_feature_allowlist

from .evaluation import binary_probability_report
from .training_files import read_json, verify_training_file, write_json
from .xgboost_contracts import TARGETS, XGBOOST_BUNDLE_VERSION
from .xgboost_training import predict


FROZEN_EVALUATION_VERSION = "vaep-frozen-test-evaluation-v1"
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT / "configs" / "models" / "vaep-test-evaluation-v1.json"
)
OUTPUT_FILENAMES = {
    "freeze": "frozen_model_manifest.json",
    "metrics": "test_metrics.json",
    "calibration": "calibration_report.json",
    "sanity": "sanity_checks.json",
    "errors": "error_analysis.json",
}


class FrozenEvaluationError(ValueError):
    """Raised when the frozen test evaluation contract is not satisfied."""


@dataclass(frozen=True, slots=True)
class FrozenEvaluationPaths:
    directory: Path
    freeze_manifest: Path
    test_metrics: Path
    calibration_report: Path
    sanity_checks: Path
    error_analysis: Path
    training_manifest: Path


@dataclass(slots=True)
class TestEvaluationData:
    matrix: Any
    targets: dict[str, Any]
    match_ids: Any
    action_ids: Any
    competition_ids: Any
    season_ids: Any
    action_types: Any
    game_states: Any
    successful_progressive: Any
    dangerous_turnover: Any
    shot_action: Any
    goal_action: Any


class FrozenModelEvaluator:
    """Freeze the selected bundle and evaluate its test split once."""

    def write(
        self,
        artifact_directory: Path,
        *,
        policy_path: Path = DEFAULT_POLICY_PATH,
        batch_size: int = 16_384,
        n_jobs: int = -1,
        progress: Any | None = None,
    ) -> FrozenEvaluationPaths:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        root = Path(artifact_directory).resolve()
        manifest_path = root / "training_manifest.json"
        manifest = read_json(manifest_path)
        paths = _output_paths(root, manifest_path)
        if manifest.get("status") == "test_evaluation_complete":
            _verify_completed_outputs(root, manifest, paths)
            if progress is not None:
                progress("test evaluation already complete; verified saved reports")
            return paths
        _require_ready_for_test(manifest)
        _refuse_partial_test_outputs(paths)

        policy = _read_policy(policy_path)
        required = {
            name: verify_training_file(root, manifest, name)
            for name in (
                "model_dataset.parquet",
                "split_assignments.parquet",
                "feature_allowlist.json",
                "score_xgboost.json",
                "concede_xgboost.json",
                "score_calibration.joblib",
                "concede_calibration.joblib",
                "xgboost_validation_report.json",
            )
        }
        features = _validated_features(required["feature_allowlist.json"])
        validation_report = read_json(required["xgboost_validation_report.json"])
        validation_passes = _validation_gate_results(validation_report)
        if policy["quality_gates"][
            "require_both_models_beat_logistic_validation"
        ] and not all(validation_passes.values()):
            failed = [
                target
                for target, passed in validation_passes.items()
                if not passed
            ]
            raise FrozenEvaluationError(
                "Frozen models are not eligible for test evaluation; validation "
                f"baseline gate failed for {failed}"
            )

        if progress is not None:
            progress("loading and validating frozen XGBoost models and calibrators")
        models, calibrators = _load_frozen_bundle(
            required,
            feature_count=len(features),
        )
        freeze_manifest = _build_freeze_manifest(
            manifest=manifest,
            policy=policy,
            policy_path=policy_path,
            required=required,
            validation_passes=validation_passes,
        )

        if progress is not None:
            progress("opening the untouched test split")
        data = load_test_evaluation_data(
            required["model_dataset.parquet"],
            required["split_assignments.parquet"],
            features,
            batch_size=batch_size,
        )
        _reconcile_test_data(data, manifest)

        if progress is not None:
            progress("running frozen score and concede inference")
        metrics, probabilities = _evaluate_targets(
            data=data,
            models=models,
            calibrators=calibrators,
            policy=policy,
            n_jobs=n_jobs,
        )
        quality_gate = _quality_gate(
            metrics,
            validation_passes=validation_passes,
            policy=policy,
        )
        test_report = _build_test_report(
            manifest=manifest,
            policy=policy,
            data=data,
            metrics=metrics,
            quality_gate=quality_gate,
        )
        calibration_report = _build_calibration_report(
            metrics,
            calibrators=calibrators,
            policy=policy,
        )
        sanity_report = _build_sanity_report(data, probabilities)
        error_report = _build_error_report(
            data,
            probabilities,
            policy=policy,
        )

        paths.directory.mkdir(parents=True, exist_ok=True)
        write_json(paths.freeze_manifest, freeze_manifest)
        write_json(paths.test_metrics, test_report)
        write_json(paths.calibration_report, calibration_report)
        write_json(paths.sanity_checks, sanity_report)
        write_json(paths.error_analysis, error_report)
        _update_training_manifest(
            manifest_path,
            manifest,
            paths,
            policy=policy,
            quality_gate=quality_gate,
            metrics=metrics,
        )
        if progress is not None:
            result = "passed" if quality_gate["passed"] else "blocked"
            progress(f"completed frozen test evaluation: promotion {result}")
        return paths


def load_test_evaluation_data(
    dataset_path: Path,
    split_path: Path,
    features: tuple[str, ...],
    *,
    batch_size: int,
) -> TestEvaluationData:
    """Load only the test split, retaining metadata needed for diagnostics."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from scipy import sparse

    assignments = pq.read_table(
        split_path,
        columns=["match_id", "competition_id", "season_id", "split"],
    )
    test_assignments = assignments.filter(pc.equal(assignments["split"], "test"))
    assignment_rows = test_assignments.to_pylist()
    match_metadata = {
        int(row["match_id"]): (
            int(row["competition_id"]),
            int(row["season_id"]),
        )
        for row in assignment_rows
    }
    if not match_metadata or len(match_metadata) != len(assignment_rows):
        raise FrozenEvaluationError(
            "Test split assignments must contain unique complete matches"
        )

    required_features = {
        "actiontype_pass_a0",
        "actiontype_bad_touch_a0",
        "result_fail_a0",
        "result_success_a0",
        "dx_a0",
        "goalscore_diff",
    }
    missing = sorted(required_features.difference(features))
    if missing:
        raise FrozenEvaluationError(
            f"Evaluation diagnostics require missing features: {missing}"
        )
    action_features = tuple(
        name
        for name in features
        if name.startswith("actiontype_")
        and name.endswith("_a0")
        and "_result_" not in name
    )
    feature_index = {name: index for index, name in enumerate(features)}
    columns = ["match_id", "action_id", *features, *TARGETS, "split"]
    scanner = ds.dataset(dataset_path, format="parquet").scanner(
        columns=columns,
        filter=ds.field("split") == "test",
        batch_size=batch_size,
        use_threads=False,
    )

    matrices: list[Any] = []
    targets: dict[str, list[Any]] = {target: [] for target in TARGETS}
    values: dict[str, list[Any]] = {
        "match_ids": [],
        "action_ids": [],
        "action_types": [],
        "game_states": [],
        "successful_progressive": [],
        "dangerous_turnover": [],
        "shot_action": [],
        "goal_action": [],
    }
    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])
        if table.num_rows == 0:
            continue
        dense = np.column_stack(
            [table[name].to_numpy(zero_copy_only=False) for name in features]
        ).astype(np.float32, copy=False)
        if not np.isfinite(dense).all():
            raise FrozenEvaluationError(
                "Test model features contain null or non-finite values"
            )
        matrix = sparse.csr_matrix(dense)
        matrix.eliminate_zeros()
        matrices.append(matrix)

        match_ids = table["match_id"].to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        action_ids = table["action_id"].to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        values["match_ids"].append(match_ids)
        values["action_ids"].append(action_ids)
        for target in TARGETS:
            targets[target].append(
                table[target]
                .to_numpy(zero_copy_only=False)
                .astype(np.int8, copy=False)
            )

        action_matrix = dense[
            :, [feature_index[name] for name in action_features]
        ]
        selected = action_matrix.argmax(axis=1)
        has_action = action_matrix.max(axis=1) > 0.5
        action_names = np.asarray(
            [
                name.removeprefix("actiontype_").removesuffix("_a0")
                for name in action_features
            ],
            dtype=object,
        )
        values["action_types"].append(
            np.where(has_action, action_names[selected], "unknown")
        )
        goal_diff = dense[:, feature_index["goalscore_diff"]]
        values["game_states"].append(
            np.where(
                goal_diff > 0,
                "leading",
                np.where(goal_diff < 0, "trailing", "drawing"),
            )
        )
        success = dense[:, feature_index["result_success_a0"]] > 0.5
        progressive = success & (dense[:, feature_index["dx_a0"]] > 0)
        bad_touch = dense[:, feature_index["actiontype_bad_touch_a0"]] > 0.5
        failed_pass = (
            (dense[:, feature_index["actiontype_pass_a0"]] > 0.5)
            & (dense[:, feature_index["result_fail_a0"]] > 0.5)
        )
        shot_indices = [
            feature_index[name]
            for name in (
                "actiontype_shot_a0",
                "actiontype_shot_penalty_a0",
                "actiontype_shot_freekick_a0",
            )
            if name in feature_index
        ]
        shot = dense[:, shot_indices].max(axis=1) > 0.5
        values["successful_progressive"].append(progressive)
        values["dangerous_turnover"].append(bad_touch | failed_pass)
        values["shot_action"].append(shot)
        values["goal_action"].append(shot & success)

    if not matrices:
        raise FrozenEvaluationError("Test split contains no eligible rows")
    match_ids = np.concatenate(values["match_ids"])
    observed_matches = {int(value) for value in np.unique(match_ids)}
    if observed_matches != set(match_metadata):
        raise FrozenEvaluationError(
            "Test dataset match IDs do not reconcile with split assignments"
        )
    competition_ids = np.asarray(
        [match_metadata[int(match_id)][0] for match_id in match_ids],
        dtype=np.int64,
    )
    season_ids = np.asarray(
        [match_metadata[int(match_id)][1] for match_id in match_ids],
        dtype=np.int64,
    )
    return TestEvaluationData(
        matrix=sparse.vstack(matrices, format="csr", dtype=np.float32),
        targets={target: np.concatenate(parts) for target, parts in targets.items()},
        match_ids=match_ids,
        action_ids=np.concatenate(values["action_ids"]),
        competition_ids=competition_ids,
        season_ids=season_ids,
        action_types=np.concatenate(values["action_types"]),
        game_states=np.concatenate(values["game_states"]),
        successful_progressive=np.concatenate(values["successful_progressive"]),
        dangerous_turnover=np.concatenate(values["dangerous_turnover"]),
        shot_action=np.concatenate(values["shot_action"]),
        goal_action=np.concatenate(values["goal_action"]),
    )


def _output_paths(root: Path, manifest_path: Path) -> FrozenEvaluationPaths:
    output = root / "reports"
    return FrozenEvaluationPaths(
        directory=output,
        freeze_manifest=output / OUTPUT_FILENAMES["freeze"],
        test_metrics=output / OUTPUT_FILENAMES["metrics"],
        calibration_report=output / OUTPUT_FILENAMES["calibration"],
        sanity_checks=output / OUTPUT_FILENAMES["sanity"],
        error_analysis=output / OUTPUT_FILENAMES["errors"],
        training_manifest=manifest_path,
    )


def _require_ready_for_test(manifest: Mapping[str, Any]) -> None:
    if manifest.get("status") != "xgboost_validation_complete":
        raise FrozenEvaluationError(
            "Test evaluation requires status xgboost_validation_complete"
        )
    model_status = (manifest.get("models") or {}).get("status")
    if model_status != "xgboost_validation_complete":
        raise FrozenEvaluationError(
            "Test evaluation requires frozen XGBoost validation artifacts"
        )


def _refuse_partial_test_outputs(paths: FrozenEvaluationPaths) -> None:
    existing = [
        path.name
        for path in (
            paths.freeze_manifest,
            paths.test_metrics,
            paths.calibration_report,
            paths.sanity_checks,
            paths.error_analysis,
        )
        if path.exists()
    ]
    if existing:
        raise FrozenEvaluationError(
            "Refusing to reopen the test split with partial prior outputs: "
            f"{existing}"
        )


def _read_policy(path: Path) -> dict[str, Any]:
    policy = read_json(Path(path))
    if policy.get("schema_version") != 1 or not isinstance(
        policy.get("policy_version"), str
    ):
        raise FrozenEvaluationError("Unsupported test evaluation policy")
    thresholds = policy.get("thresholds") or {}
    for target in TARGETS:
        threshold = thresholds.get(target)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise FrozenEvaluationError(f"Invalid threshold for {target}")
    gates = policy.get("quality_gates") or {}
    for name in (
        "maximum_brier_to_positive_frequency_ratio",
        "minimum_pr_auc_to_positive_frequency_ratio",
    ):
        if not isinstance(gates.get(name), (int, float)) or gates[name] <= 0:
            raise FrozenEvaluationError(f"Invalid quality gate {name}")
    return policy


def _validated_features(path: Path) -> tuple[str, ...]:
    manifest = read_json(path)
    features = tuple(manifest.get("feature_columns") or ())
    if features != baseline_feature_allowlist():
        raise FrozenEvaluationError(
            "Frozen evaluation feature allowlist does not match the baseline contract"
        )
    return features


def _validation_gate_results(report: Mapping[str, Any]) -> dict[str, bool]:
    target_reports = report.get("targets") or {}
    return {
        target: bool((target_reports.get(target) or {}).get("beats_logistic_baseline"))
        for target in TARGETS
    }


def _load_frozen_bundle(
    required: Mapping[str, Path],
    *,
    feature_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import joblib
    import numpy as np
    import xgboost as xgb

    models: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    for target, stem in (("scores", "score"), ("concedes", "concede")):
        model = xgb.Booster()
        model.load_model(required[f"{stem}_xgboost.json"])
        if model.num_features() != feature_count:
            raise FrozenEvaluationError(
                f"{target} model expects {model.num_features()} features; "
                f"contract declares {feature_count}"
            )
        calibrator = joblib.load(required[f"{stem}_calibration.joblib"])
        method = getattr(calibrator, "method", None)
        if method not in {"sigmoid", "isotonic"}:
            raise FrozenEvaluationError(
                f"Unsupported {target} calibration method: {method!r}"
            )
        probe = np.asarray(calibrator.predict([0.25, 0.75]), dtype=np.float64)
        if probe.shape != (2,) or not np.isfinite(probe).all() or (
            (probe < 0) | (probe > 1)
        ).any():
            raise FrozenEvaluationError(f"Invalid {target} calibrator output")
        models[target] = model
        calibrators[target] = calibrator
    return models, calibrators


def _build_freeze_manifest(
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    policy_path: Path,
    required: Mapping[str, Path],
    validation_passes: Mapping[str, bool],
) -> dict[str, Any]:
    frozen_names = (
        "score_xgboost.json",
        "concede_xgboost.json",
        "score_calibration.joblib",
        "concede_calibration.joblib",
    )
    return {
        "schema_version": 1,
        "freeze_version": FROZEN_EVALUATION_VERSION,
        "bundle_version": (manifest.get("models") or {}).get(
            "xgboost_bundle_version", XGBOOST_BUNDLE_VERSION
        ),
        "dataset_fingerprint": (manifest.get("source") or {}).get(
            "dataset_fingerprint"
        ),
        "feature_allowlist_sha256": (manifest.get("features") or {}).get(
            "feature_allowlist_sha256"
        ),
        "evaluation_policy": {
            "path": str(Path(policy_path).resolve()),
            "sha256": file_sha256(Path(policy_path)),
            "version": policy["policy_version"],
        },
        "thresholds": dict(policy["thresholds"]),
        "validation_baseline_gate": dict(validation_passes),
        "artifacts": {
            name: {
                "path": str(required[name]),
                "sha256": file_sha256(required[name]),
            }
            for name in frozen_names
        },
        "status": "frozen_for_untouched_test",
    }


def _reconcile_test_data(
    data: TestEvaluationData,
    manifest: Mapping[str, Any],
) -> None:
    import numpy as np

    row_count = int(data.matrix.shape[0])
    declared = (((manifest.get("split") or {}).get("splits") or {}).get("test") or {})
    expected_rows = declared.get("eligible_label_count")
    if isinstance(expected_rows, int) and row_count != expected_rows:
        raise FrozenEvaluationError(
            f"Test row count mismatch: expected {expected_rows}, got {row_count}"
        )
    expected_matches = declared.get("match_count")
    observed_matches = len(np.unique(data.match_ids))
    if isinstance(expected_matches, int) and observed_matches != expected_matches:
        raise FrozenEvaluationError(
            "Test match count does not reconcile with the training manifest"
        )
    for target in TARGETS:
        if set(np.unique(data.targets[target])) != {0, 1}:
            raise FrozenEvaluationError(
                f"Test split requires both target classes for {target}"
            )


def _evaluate_targets(
    *,
    data: TestEvaluationData,
    models: Mapping[str, Any],
    calibrators: Mapping[str, Any],
    policy: Mapping[str, Any],
    n_jobs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import xgboost as xgb

    matrix = xgb.DMatrix(data.matrix, nthread=n_jobs)
    metrics: dict[str, Any] = {}
    probabilities: dict[str, Any] = {}
    for target in TARGETS:
        started = time.perf_counter()
        raw = np.asarray(predict(models[target], matrix), dtype=np.float64)
        calibrated = np.asarray(
            calibrators[target].predict(raw), dtype=np.float64
        )
        inference_seconds = time.perf_counter() - started
        threshold = float(policy["thresholds"][target])
        bins = int(policy["calibration_bins"])
        metrics[target] = {
            "raw": binary_probability_report(
                data.targets[target],
                raw,
                threshold=threshold,
                calibration_bins=bins,
            ),
            "calibrated": binary_probability_report(
                data.targets[target],
                calibrated,
                threshold=threshold,
                calibration_bins=bins,
                inference_seconds=inference_seconds,
            ),
        }
        probabilities[target] = calibrated
    return metrics, probabilities


def _quality_gate(
    metrics: Mapping[str, Any],
    *,
    validation_passes: Mapping[str, bool],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    gates = policy["quality_gates"]
    blockers: list[str] = []
    checks: dict[str, Any] = {}
    for target in TARGETS:
        report = metrics[target]["calibrated"]
        frequency = float(report["positive_frequency"])
        pr_ratio = float(report["pr_auc"]) / frequency
        brier_ratio = float(report["brier_score"]) / frequency
        target_checks = {
            "beats_logistic_baseline_on_validation": bool(
                validation_passes[target]
            ),
            "pr_auc_to_positive_frequency_ratio": pr_ratio,
            "pr_auc_gate_passed": pr_ratio
            >= gates["minimum_pr_auc_to_positive_frequency_ratio"],
            "brier_to_positive_frequency_ratio": brier_ratio,
            "brier_gate_passed": brier_ratio
            <= gates["maximum_brier_to_positive_frequency_ratio"],
        }
        for name in (
            "beats_logistic_baseline_on_validation",
            "pr_auc_gate_passed",
            "brier_gate_passed",
        ):
            if not target_checks[name]:
                blockers.append(f"models:{target}:{name}")
        checks[target] = target_checks
    return {"passed": not blockers, "blockers": blockers, "checks": checks}


def _build_test_report(
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    data: TestEvaluationData,
    metrics: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    return {
        "schema_version": 1,
        "evaluation_version": FROZEN_EVALUATION_VERSION,
        "policy_version": policy["policy_version"],
        "bundle_version": (manifest.get("models") or {}).get(
            "xgboost_bundle_version", XGBOOST_BUNDLE_VERSION
        ),
        "split": "test",
        "test_split_accessed": True,
        "data": {
            "row_count": int(data.matrix.shape[0]),
            "match_count": int(len(np.unique(data.match_ids))),
            "feature_count": int(data.matrix.shape[1]),
            "competition_count": int(len(np.unique(data.competition_ids))),
            "targets": {
                target: {
                    "positive_count": int(data.targets[target].sum()),
                    "positive_frequency": float(data.targets[target].mean()),
                }
                for target in TARGETS
            },
        },
        "thresholds": dict(policy["thresholds"]),
        "targets": dict(metrics),
        "quality_gate": dict(quality_gate),
    }


def _build_calibration_report(
    metrics: Mapping[str, Any],
    *,
    calibrators: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target in TARGETS:
        raw = metrics[target]["raw"]
        calibrated = metrics[target]["calibrated"]
        targets[target] = {
            "method": calibrators[target].method,
            "raw_curve": raw["calibration_curve"],
            "calibrated_curve": calibrated["calibration_curve"],
            "raw_brier_score": raw["brier_score"],
            "calibrated_brier_score": calibrated["brier_score"],
            "brier_improvement": raw["brier_score"]
            - calibrated["brier_score"],
            "raw_log_loss": raw["log_loss"],
            "calibrated_log_loss": calibrated["log_loss"],
            "log_loss_improvement": raw["log_loss"]
            - calibrated["log_loss"],
        }
    return {
        "schema_version": 1,
        "evaluation_version": FROZEN_EVALUATION_VERSION,
        "policy_version": policy["policy_version"],
        "targets": targets,
    }


def _build_sanity_report(
    data: TestEvaluationData,
    probabilities: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    score = probabilities["scores"]
    concede = probabilities["concedes"]
    structural = {
        target: {
            "finite": bool(np.isfinite(values).all()),
            "within_unit_interval": bool(
                ((values >= 0) & (values <= 1)).all()
            ),
        }
        for target, values in probabilities.items()
    }
    observations = {
        "shots_score_probability": _cohort_observation(
            score, data.shot_action, expected="higher"
        ),
        "goals_score_probability": _cohort_observation(
            score, data.goal_action, expected="higher"
        ),
        "successful_progressive_score_probability": _cohort_observation(
            score, data.successful_progressive, expected="higher"
        ),
        "dangerous_turnover_concede_probability": _cohort_observation(
            concede, data.dangerous_turnover, expected="higher"
        ),
    }
    return {
        "schema_version": 1,
        "evaluation_version": FROZEN_EVALUATION_VERSION,
        "blocking_checks_passed": all(
            item["finite"] and item["within_unit_interval"]
            for item in structural.values()
        ),
        "structural_checks": structural,
        "football_observations": observations,
        "football_observations_are_blocking": False,
        "scope_note": (
            "These are post-action cohort checks. Perspective-safe before/after "
            "VAEP change checks belong to Task 11."
        ),
    }


def _cohort_observation(values: Any, mask: Any, *, expected: str) -> dict[str, Any]:
    import numpy as np

    count = int(np.asarray(mask).sum())
    overall = float(np.asarray(values).mean())
    cohort = float(np.asarray(values)[mask].mean()) if count else None
    direction_passed = bool(cohort is not None and cohort > overall)
    return {
        "row_count": count,
        "overall_mean_probability": overall,
        "cohort_mean_probability": cohort,
        "difference": None if cohort is None else cohort - overall,
        "expected_direction": expected,
        "direction_observed": direction_passed,
    }


def _build_error_report(
    data: TestEvaluationData,
    probabilities: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target in TARGETS:
        truth = data.targets[target]
        predicted = probabilities[target]
        threshold = float(policy["thresholds"][target])
        targets[target] = {
            "by_competition": _group_reports(
                truth,
                predicted,
                data.competition_ids,
                threshold=threshold,
                minimum_rows=int(policy["minimum_group_rows"]),
            ),
            "by_action_type": _group_reports(
                truth,
                predicted,
                data.action_types,
                threshold=threshold,
                minimum_rows=int(policy["minimum_group_rows"]),
            ),
            "by_game_state": _group_reports(
                truth,
                predicted,
                data.game_states,
                threshold=threshold,
                minimum_rows=int(policy["minimum_group_rows"]),
            ),
            "largest_errors": _largest_errors(
                data,
                truth,
                predicted,
                threshold=threshold,
                limit=int(policy["error_examples_per_target"]),
            ),
        }
    return {
        "schema_version": 1,
        "evaluation_version": FROZEN_EVALUATION_VERSION,
        "split": "test",
        "targets": targets,
    }


def _group_reports(
    truth: Any,
    predicted: Any,
    groups: Any,
    *,
    threshold: float,
    minimum_rows: int,
) -> dict[str, Any]:
    import numpy as np

    reports: dict[str, Any] = {}
    for group in sorted(np.unique(groups), key=lambda value: str(value)):
        mask = groups == group
        group_truth = truth[mask]
        summary: dict[str, Any] = {
            "row_count": int(mask.sum()),
            "positive_count": int(group_truth.sum()),
            "positive_frequency": float(group_truth.mean()),
        }
        if len(group_truth) >= minimum_rows and set(np.unique(group_truth)) == {0, 1}:
            summary["metrics"] = binary_probability_report(
                group_truth,
                predicted[mask],
                threshold=threshold,
                calibration_bins=min(10, max(2, len(group_truth) // 50)),
            )
        else:
            summary["metrics"] = None
            summary["unavailable_reason"] = (
                "requires minimum rows and both target classes"
            )
        reports[str(group)] = summary
    return reports


def _largest_errors(
    data: TestEvaluationData,
    truth: Any,
    predicted: Any,
    *,
    threshold: float,
    limit: int,
) -> dict[str, Any]:
    import numpy as np

    false_positive = np.flatnonzero((truth == 0) & (predicted >= threshold))
    false_negative = np.flatnonzero((truth == 1) & (predicted < threshold))
    false_positive = false_positive[
        np.argsort(predicted[false_positive])[::-1][:limit]
    ]
    false_negative = false_negative[
        np.argsort(predicted[false_negative])[:limit]
    ]

    def rows(indices: Any) -> list[dict[str, Any]]:
        return [
            {
                "match_id": int(data.match_ids[index]),
                "action_id": int(data.action_ids[index]),
                "competition_id": int(data.competition_ids[index]),
                "season_id": int(data.season_ids[index]),
                "action_type": str(data.action_types[index]),
                "game_state": str(data.game_states[index]),
                "label": int(truth[index]),
                "probability": float(predicted[index]),
                "absolute_error": float(abs(truth[index] - predicted[index])),
            }
            for index in indices
        ]

    return {
        "false_positive_count": int(
            ((truth == 0) & (predicted >= threshold)).sum()
        ),
        "false_negative_count": int(
            ((truth == 1) & (predicted < threshold)).sum()
        ),
        "false_positive_examples": rows(false_positive),
        "false_negative_examples": rows(false_negative),
    }


def _update_training_manifest(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    paths: FrozenEvaluationPaths,
    *,
    policy: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    output_files = {
        path.name: path
        for path in (
            paths.freeze_manifest,
            paths.test_metrics,
            paths.calibration_report,
            paths.sanity_checks,
            paths.error_analysis,
        )
    }
    records = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in output_files.items()
    }
    updated = dict(manifest)
    updated["stage"] = "test_evaluation"
    updated["status"] = "test_evaluation_complete"
    updated["artifacts"] = {
        **dict(manifest.get("artifacts") or {}),
        **records,
    }
    models = dict(manifest.get("models") or {})
    models["status"] = "frozen_test_evaluated"
    models["test_metrics"] = {
        target: dict(metrics[target]["calibrated"]) for target in TARGETS
    }
    reports = dict(models.get("reports") or {})
    reports.update(records)
    models["reports"] = reports
    updated["models"] = models
    updated["test_evaluation"] = {
        "version": FROZEN_EVALUATION_VERSION,
        "policy_version": policy["policy_version"],
        "test_split_accessed": True,
        "quality_gate": dict(quality_gate),
    }
    existing_promotion = dict(manifest.get("production_promotion") or {})
    existing_promotion["allowed"] = bool(quality_gate["passed"])
    existing_promotion["blockers"] = list(quality_gate["blockers"])
    updated["production_promotion"] = existing_promotion
    write_json(manifest_path, updated)


def _verify_completed_outputs(
    root: Path,
    manifest: Mapping[str, Any],
    paths: FrozenEvaluationPaths,
) -> None:
    filenames = (
        paths.freeze_manifest.name,
        paths.test_metrics.name,
        paths.calibration_report.name,
        paths.sanity_checks.name,
        paths.error_analysis.name,
        "score_xgboost.json",
        "concede_xgboost.json",
        "score_calibration.joblib",
        "concede_calibration.joblib",
    )
    for filename in filenames:
        verify_training_file(root, manifest, filename)


__all__ = [
    "DEFAULT_POLICY_PATH",
    "FROZEN_EVALUATION_VERSION",
    "FrozenEvaluationError",
    "FrozenEvaluationPaths",
    "FrozenModelEvaluator",
    "TestEvaluationData",
    "load_test_evaluation_data",
]
