"""Perspective-safe action valuation with the frozen VAEP model bundle."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pitchpulse.shared.file_io import file_sha256
from pitchpulse.vaep_features.feature_dataset_loader import baseline_feature_allowlist

from .training_files import read_json, verify_training_file, write_json
from .xgboost_training import predict


ACTION_VALUE_VERSION = "socceraction-1.5.3-vaep-action-values-v1"
ACTION_VALUES_FILENAME = "action_values.parquet"
ACTION_VALUES_MANIFEST_FILENAME = "action_values_manifest.json"
PREDICTION_ORIGINS = {
    "train": "in_sample",
    "validation": "model_selection",
    "test": "untouched_test",
}
_MODEL_FILES = (
    "score_xgboost.json",
    "concede_xgboost.json",
    "score_calibration.joblib",
    "concede_calibration.joblib",
)
_ACTION_COLUMNS = (
    "match_id",
    "action_id",
    "player_id",
    "team_id",
    "period_id",
    "time_seconds",
    "type_name",
    "result_name",
    "mapping_version",
    "state_version",
)


class ActionValuationError(ValueError):
    """Raised when frozen-model action valuation violates its contract."""


@dataclass(frozen=True, slots=True)
class ActionValuePaths:
    directory: Path
    action_values: Path
    manifest: Path
    training_manifest: Path


def perspective_safe_values(
    actions: Any,
    p_scores_after: Any,
    p_concedes_after: Any,
) -> Any:
    """Apply socceraction's pinned VAEP formula to one ordered match.

    The returned ``p_*_before`` columns are reconstructed from the exact
    probabilities used by socceraction after its team-perspective and restart
    adjustments. This avoids maintaining a second, subtly different formula.
    """

    import numpy as np
    import pandas as pd
    from socceraction.vaep.formula import defensive_value, offensive_value

    required = {"team_id", "time_seconds", "type_name", "result_name"}
    missing = sorted(required.difference(actions.columns))
    if missing:
        raise ActionValuationError(f"Actions lack VAEP formula columns: {missing}")
    frame = actions.reset_index(drop=True)
    if "match_id" in frame and frame["match_id"].nunique(dropna=False) != 1:
        raise ActionValuationError("VAEP formula input must contain exactly one match")
    scores = np.asarray(p_scores_after, dtype=np.float64)
    concedes = np.asarray(p_concedes_after, dtype=np.float64)
    if scores.ndim != 1 or concedes.ndim != 1:
        raise ActionValuationError("VAEP probabilities must be one-dimensional")
    if len(frame) == 0 or len(scores) != len(frame) or len(concedes) != len(frame):
        raise ActionValuationError("Actions and VAEP probabilities must align")
    if not np.isfinite(scores).all() or not np.isfinite(concedes).all():
        raise ActionValuationError("VAEP probabilities must be finite")
    if ((scores < 0) | (scores > 1)).any() or (
        (concedes < 0) | (concedes > 1)
    ).any():
        raise ActionValuationError("VAEP probabilities must lie in [0, 1]")

    score_series = pd.Series(scores, index=frame.index, name="scores")
    concede_series = pd.Series(concedes, index=frame.index, name="concedes")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module=r"socceraction\..*"
        )
        offensive = np.asarray(
            offensive_value(frame, score_series, concede_series),
            dtype=np.float64,
        )
        defensive = np.asarray(
            defensive_value(frame, score_series, concede_series),
            dtype=np.float64,
        )

    # These identities expose the perspective-adjusted probabilities that the
    # pinned library actually subtracted, including goal/gap/restart overrides.
    score_before = scores - offensive
    concede_before = concedes + defensive
    vaep = offensive + defensive
    values = np.column_stack(
        (score_before, scores, concede_before, concedes, offensive, defensive, vaep)
    )
    if not np.isfinite(values).all():
        raise ActionValuationError("socceraction produced non-finite VAEP values")
    if (
        ((score_before < -1e-12) | (score_before > 1 + 1e-12)).any()
        or ((concede_before < -1e-12) | (concede_before > 1 + 1e-12)).any()
    ):
        raise ActionValuationError("Adjusted before-probabilities lie outside [0, 1]")

    return pd.DataFrame(
        {
            "p_score_before": np.clip(score_before, 0.0, 1.0),
            "p_score_after": scores,
            "p_concede_before": np.clip(concede_before, 0.0, 1.0),
            "p_concede_after": concedes,
            "offensive_value": offensive,
            "defensive_value": defensive,
            "vaep_value": vaep,
        }
    )


class ActionValueWriter:
    """Infer calibrated probabilities and write one VAEP row per eligible action."""

    def write(
        self,
        artifact_directory: Path,
        *,
        batch_size: int = 16_384,
        n_jobs: int = -1,
        progress: Any | None = None,
    ) -> ActionValuePaths:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        root = Path(artifact_directory).resolve()
        manifest_path = root / "training_manifest.json"
        manifest = read_json(manifest_path)
        paths = ActionValuePaths(
            directory=root,
            action_values=root / ACTION_VALUES_FILENAME,
            manifest=root / ACTION_VALUES_MANIFEST_FILENAME,
            training_manifest=manifest_path,
        )
        if manifest.get("status") == "action_valuation_complete":
            self._verify_completed(root, manifest, paths)
            if progress is not None:
                progress("action valuation already complete; verified saved artifact")
            return paths
        self._require_promoted_bundle(manifest)
        if paths.action_values.exists() or paths.manifest.exists():
            raise ActionValuationError(
                "Refusing to overwrite partial action-valuation outputs"
            )

        required = {
            name: verify_training_file(root, manifest, name)
            for name in (
                "model_dataset.parquet",
                "feature_allowlist.json",
                "xgboost_validation_report.json",
                *_MODEL_FILES,
            )
        }
        features = self._validated_features(required["feature_allowlist.json"])
        actions_path, plan03_manifest = self._validated_actions_source(manifest)
        models, calibrators = self._load_bundle(required, len(features))
        lineage = self._lineage(
            manifest,
            plan03_manifest,
            required["xgboost_validation_report.json"],
        )

        prediction_temp = root / ".action_probabilities.tmp.parquet"
        values_temp = root / ".action_values.tmp.parquet"
        prediction_temp.unlink(missing_ok=True)
        values_temp.unlink(missing_ok=True)
        try:
            if progress is not None:
                progress("running calibrated frozen-model inference for all splits")
            prediction_counts = self._write_predictions(
                dataset_path=required["model_dataset.parquet"],
                output_path=prediction_temp,
                features=features,
                models=models,
                calibrators=calibrators,
                batch_size=batch_size,
                n_jobs=n_jobs,
                progress=progress,
            )
            if progress is not None:
                progress("applying perspective-safe socceraction VAEP formulas")
            value_stats = self._write_action_values(
                actions_path=actions_path,
                predictions_path=prediction_temp,
                output_path=values_temp,
                lineage=lineage,
                progress=progress,
            )
            self._reconcile_counts(manifest, prediction_counts, value_stats)
            values_temp.replace(paths.action_values)
            value_manifest = self._value_manifest(
                manifest=manifest,
                actions_path=actions_path,
                dataset_path=required["model_dataset.parquet"],
                output_path=paths.action_values,
                lineage=lineage,
                stats=value_stats,
            )
            write_json(paths.manifest, value_manifest)
            self._update_training_manifest(manifest_path, manifest, paths, value_manifest)
        finally:
            prediction_temp.unlink(missing_ok=True)
            values_temp.unlink(missing_ok=True)

        if progress is not None:
            progress(f"completed action valuation: {paths.action_values}")
        return paths

    @staticmethod
    def _require_promoted_bundle(manifest: Mapping[str, Any]) -> None:
        if manifest.get("status") != "test_evaluation_complete":
            raise ActionValuationError(
                "Action valuation requires completed untouched-test evaluation"
            )
        if (manifest.get("models") or {}).get("status") != "frozen_test_evaluated":
            raise ActionValuationError("Action valuation requires frozen model artifacts")
        if not bool((manifest.get("production_promotion") or {}).get("allowed")):
            raise ActionValuationError("Action valuation requires a passed promotion gate")

    @staticmethod
    def _validated_features(path: Path) -> tuple[str, ...]:
        declared = tuple(read_json(path).get("feature_columns") or ())
        expected = baseline_feature_allowlist()
        if declared != expected:
            raise ActionValuationError(
                "Valuation feature allowlist does not match the baseline contract"
            )
        return declared

    @staticmethod
    def _validated_actions_source(
        manifest: Mapping[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        source = manifest.get("source") or {}
        directory_value = source.get("plan03_directory") or source.get(
            "feature_artifact_directory"
        )
        if not isinstance(directory_value, str) or not directory_value:
            raise ActionValuationError("Training manifest lacks its Plan 03 directory")
        directory = Path(directory_value).resolve()
        source_manifest_path = directory / "manifest.json"
        if not source_manifest_path.is_file():
            raise FileNotFoundError(f"Plan 03 manifest is missing: {source_manifest_path}")
        expected_manifest_hash = source.get("plan03_manifest_sha256")
        if (
            not isinstance(expected_manifest_hash, str)
            or file_sha256(source_manifest_path) != expected_manifest_hash
        ):
            raise ActionValuationError("Plan 03 manifest hash does not match training lineage")
        source_manifest = read_json(source_manifest_path)
        if source_manifest.get("dataset_fingerprint") != source.get(
            "dataset_fingerprint"
        ):
            raise ActionValuationError("Plan 03 dataset fingerprint does not reconcile")
        actions_path = directory / "actions.parquet"
        expected_actions_hash = (source_manifest.get("files") or {}).get(
            actions_path.name
        )
        if (
            not actions_path.is_file()
            or not isinstance(expected_actions_hash, str)
            or file_sha256(actions_path) != expected_actions_hash
        ):
            raise ActionValuationError("Plan 03 actions hash does not match its manifest")
        return actions_path, source_manifest

    @staticmethod
    def _load_bundle(
        required: Mapping[str, Path], feature_count: int
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
                raise ActionValuationError(
                    f"{target} model expects {model.num_features()} features; "
                    f"contract declares {feature_count}"
                )
            calibrator = joblib.load(required[f"{stem}_calibration.joblib"])
            probe = np.asarray(calibrator.predict([0.25, 0.75]), dtype=np.float64)
            if (
                getattr(calibrator, "method", None) not in {"sigmoid", "isotonic"}
                or probe.shape != (2,)
                or not np.isfinite(probe).all()
                or ((probe < 0) | (probe > 1)).any()
            ):
                raise ActionValuationError(f"Invalid {target} calibration artifact")
            models[target] = model
            calibrators[target] = calibrator
        return models, calibrators

    @staticmethod
    def _lineage(
        manifest: Mapping[str, Any],
        source_manifest: Mapping[str, Any],
        validation_report_path: Path,
    ) -> dict[str, Any]:
        versions = manifest.get("versions") or {}
        models = manifest.get("models") or {}
        required_versions = {
            "action_mapping_version": versions.get("action_mapping"),
            "state_contract_version": versions.get("state_contract"),
            "feature_version": versions.get("feature"),
            "target_policy_version": versions.get("target_policy"),
            "split_version": versions.get("split"),
            "score_model_version": models.get("xgboost_bundle_version"),
            "concede_model_version": models.get("xgboost_bundle_version"),
        }
        if any(not isinstance(value, str) or not value for value in required_versions.values()):
            raise ActionValuationError("Training manifest has incomplete valuation lineage")
        if source_manifest.get("mapping_version") != required_versions[
            "action_mapping_version"
        ]:
            raise ActionValuationError("Action mapping lineage does not reconcile")
        calibration_version = read_json(validation_report_path).get(
            "calibration_version"
        )
        if not isinstance(calibration_version, str) or not calibration_version:
            raise ActionValuationError("Training manifest lacks calibration lineage")
        return {
            **required_versions,
            "score_calibration_version": calibration_version,
            "concede_calibration_version": calibration_version,
            "analytics_run_id": source_manifest.get("analytics_run_id"),
            "dataset_fingerprint": source_manifest.get("dataset_fingerprint"),
        }

    @staticmethod
    def _write_predictions(
        *,
        dataset_path: Path,
        output_path: Path,
        features: tuple[str, ...],
        models: Mapping[str, Any],
        calibrators: Mapping[str, Any],
        batch_size: int,
        n_jobs: int,
        progress: Any | None,
    ) -> dict[str, int]:
        """Predict score/concede probabilities without loading the whole file.

        ``features`` are the only columns passed to XGBoost.  The three identity
        columns are copied to the output so later stages can match every
        probability back to its action and data split.

        The outer loop preserves the source Parquet row groups.  The inner loop
        reads one small batch at a time, so feature data for the entire dataset
        is never held in memory at once.
        """

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import xgboost as xgb
        from scipy import sparse

        source = pq.ParquetFile(dataset_path)
        identity_columns = ["match_id", "action_id", "split"]
        columns_to_read = [*identity_columns, *features]
        required_columns = set(columns_to_read)
        missing = sorted(required_columns.difference(source.schema_arrow.names))
        if missing:
            raise ActionValuationError(f"Model dataset lacks columns: {missing}")

        # Give the two models explicit names here.  ``models`` and
        # ``calibrators`` are dictionaries created by ``_load_bundle``.
        score_model = models["scores"]
        concede_model = models["concedes"]
        score_calibrator = calibrators["scores"]
        concede_calibrator = calibrators["concedes"]

        writer = None
        counts = {split: 0 for split in PREDICTION_ORIGINS}
        seen_rows = 0
        try:
            for row_group_index in range(source.num_row_groups):
                prediction_batches: list[Any] = []

                for record_batch in source.iter_batches(
                    batch_size=batch_size,
                    row_groups=[row_group_index],
                    columns=columns_to_read,
                    use_threads=False,
                ):
                    batch_table = pa.Table.from_batches([record_batch])
                    feature_matrix = np.column_stack(
                        [
                            batch_table[name].to_numpy(zero_copy_only=False)
                            for name in features
                        ]
                    ).astype(np.float32, copy=False)
                    if not np.isfinite(feature_matrix).all():
                        raise ActionValuationError(
                            "Model dataset contains null or non-finite features"
                        )

                    xgboost_matrix = xgb.DMatrix(
                        sparse.csr_matrix(feature_matrix),
                        nthread=n_jobs,
                    )

                    score_probabilities = ActionValueWriter._predict_probabilities(
                        target="scores",
                        model=score_model,
                        calibrator=score_calibrator,
                        matrix=xgboost_matrix,
                        expected_rows=batch_table.num_rows,
                    )
                    concede_probabilities = (
                        ActionValueWriter._predict_probabilities(
                            target="concedes",
                            model=concede_model,
                            calibrator=concede_calibrator,
                            matrix=xgboost_matrix,
                            expected_rows=batch_table.num_rows,
                        )
                    )

                    for split in batch_table["split"].to_pylist():
                        if split not in counts:
                            raise ActionValuationError(
                                f"Model dataset contains invalid split {split!r}"
                            )
                        counts[split] += 1

                    prediction_batches.append(
                        pa.table(
                            {
                                "match_id": batch_table["match_id"],
                                "action_id": batch_table["action_id"],
                                "split": batch_table["split"],
                                "p_score_after": pa.array(
                                    score_probabilities, type=pa.float64()
                                ),
                                "p_concede_after": pa.array(
                                    concede_probabilities, type=pa.float64()
                                ),
                            }
                        )
                    )
                    seen_rows += batch_table.num_rows

                if not prediction_batches:
                    raise ActionValuationError(
                        f"Model dataset row group {row_group_index} is empty"
                    )

                # Write one output row group for each input row group.  The next
                # stage relies on this one-to-one alignment with actions.parquet.
                row_group_table = pa.concat_tables(prediction_batches)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, row_group_table.schema, compression="zstd"
                    )
                writer.write_table(
                    row_group_table,
                    row_group_size=row_group_table.num_rows,
                )
                if progress is not None:
                    progress(
                        f"inferred row group {row_group_index + 1}/"
                        f"{source.num_row_groups}: {row_group_table.num_rows} actions"
                    )

            if writer is None or seen_rows == 0:
                raise ActionValuationError("Model dataset contains no eligible rows")
            writer.close()
            writer = None
        except BaseException:
            if writer is not None:
                writer.close()
            output_path.unlink(missing_ok=True)
            raise
        finally:
            source.close()
        return counts

    @staticmethod
    def _predict_probabilities(
        *,
        target: str,
        model: Any,
        calibrator: Any,
        matrix: Any,
        expected_rows: int,
    ) -> Any:
        """Run one model, calibrate its output, and validate the probabilities."""

        import numpy as np

        raw_probabilities = predict(model, matrix)
        probabilities = np.asarray(
            calibrator.predict(raw_probabilities),
            dtype=np.float64,
        )
        if (
            probabilities.shape != (expected_rows,)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ActionValuationError(
                f"Invalid calibrated {target} probabilities"
            )
        return probabilities

    @classmethod
    def _write_action_values(
        cls,
        *,
        actions_path: Path,
        predictions_path: Path,
        output_path: Path,
        lineage: Mapping[str, Any],
        progress: Any | None,
    ) -> dict[str, Any]:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        actions_file = pq.ParquetFile(actions_path)
        predictions_file = pq.ParquetFile(predictions_path)
        missing = sorted(set(_ACTION_COLUMNS).difference(actions_file.schema_arrow.names))
        if missing:
            raise ActionValuationError(f"Plan 03 actions lack columns: {missing}")
        if actions_file.num_row_groups != predictions_file.num_row_groups:
            raise ActionValuationError(
                "Plan 03 actions and model dataset row groups are not aligned"
            )

        writer = pq.ParquetWriter(
            output_path, cls._action_value_schema(pa), compression="zstd"
        )
        pending: Any | None = None
        completed_matches: set[int] = set()
        row_count = 0
        split_counts = {split: 0 for split in PREDICTION_ORIGINS}
        origin_counts = {origin: 0 for origin in PREDICTION_ORIGINS.values()}
        try:
            for row_group in range(actions_file.num_row_groups):
                action_table = actions_file.read_row_group(
                    row_group, columns=list(_ACTION_COLUMNS)
                )
                prediction_table = predictions_file.read_row_group(row_group)
                aligned = cls._align_action_rows(action_table, prediction_table, pa)
                frame = aligned.to_pandas()
                if pending is not None:
                    frame = pd.concat([pending, frame], ignore_index=True)
                last_match = int(frame["match_id"].iloc[-1])
                complete = frame.loc[frame["match_id"] != last_match]
                pending = frame.loc[frame["match_id"] == last_match].copy()
                for match_id, match in complete.groupby("match_id", sort=False):
                    match_int = int(match_id)
                    if match_int in completed_matches:
                        raise ActionValuationError(
                            f"Match {match_int} is not contiguous in action artifacts"
                        )
                    valued = cls._value_match(match, lineage, pa)
                    writer.write_table(valued, row_group_size=valued.num_rows)
                    completed_matches.add(match_int)
                    row_count += valued.num_rows
                    cls._add_split_counts(match, split_counts, origin_counts)
                if progress is not None:
                    progress(
                        f"valued aligned row group {row_group + 1}/"
                        f"{actions_file.num_row_groups}"
                    )
            if pending is None or pending.empty:
                raise ActionValuationError("No aligned action probabilities were found")
            final_match = int(pending["match_id"].iloc[0])
            if final_match in completed_matches:
                raise ActionValuationError(
                    f"Match {final_match} is not contiguous in action artifacts"
                )
            valued = cls._value_match(pending, lineage, pa)
            writer.write_table(valued, row_group_size=valued.num_rows)
            completed_matches.add(final_match)
            row_count += valued.num_rows
            cls._add_split_counts(pending, split_counts, origin_counts)
            writer.close()
            writer = None
        except BaseException:
            if writer is not None:
                writer.close()
            output_path.unlink(missing_ok=True)
            raise
        finally:
            actions_file.close()
            predictions_file.close()
        return {
            "row_count": row_count,
            "match_count": len(completed_matches),
            "split_counts": split_counts,
            "prediction_origin_counts": origin_counts,
        }

    @staticmethod
    def _align_action_rows(actions: Any, predictions: Any, pa: Any) -> Any:
        action_keys = list(
            zip(actions["match_id"].to_pylist(), actions["action_id"].to_pylist())
        )
        positions = {key: index for index, key in enumerate(action_keys)}
        if len(positions) != len(action_keys):
            raise ActionValuationError("Plan 03 action row group contains duplicate keys")
        prediction_keys = list(
            zip(
                predictions["match_id"].to_pylist(),
                predictions["action_id"].to_pylist(),
            )
        )
        if len(set(prediction_keys)) != len(prediction_keys):
            raise ActionValuationError("Probability row group contains duplicate keys")
        try:
            indices = [positions[key] for key in prediction_keys]
        except KeyError as exc:
            raise ActionValuationError(
                f"Probability has no matching Plan 03 action: {exc.args[0]}"
            ) from exc
        if indices != sorted(indices):
            raise ActionValuationError("Action probabilities are not in source action order")
        selected = actions.take(pa.array(indices, type=pa.int64()))
        return pa.table(
            {
                **{name: selected[name] for name in _ACTION_COLUMNS},
                "split": predictions["split"],
                "p_score_after": predictions["p_score_after"],
                "p_concede_after": predictions["p_concede_after"],
            }
        )

    @classmethod
    def _value_match(cls, match: Any, lineage: Mapping[str, Any], pa: Any) -> Any:
        import numpy as np
        import pandas as pd

        frame = match.reset_index(drop=True)
        action_ids = frame["action_id"].to_numpy(dtype=np.int64)
        if len(np.unique(action_ids)) != len(action_ids) or (
            len(action_ids) > 1 and (np.diff(action_ids) <= 0).any()
        ):
            raise ActionValuationError("Match actions must be unique and ordered")
        splits = set(frame["split"].tolist())
        if len(splits) != 1 or not splits.issubset(PREDICTION_ORIGINS):
            raise ActionValuationError("Every match must belong to one valid split")
        if set(frame["mapping_version"].dropna()) != {
            lineage["action_mapping_version"]
        }:
            raise ActionValuationError("Action mapping version changed within valuation")
        if set(frame["state_version"].dropna()) != {
            lineage["state_contract_version"]
        }:
            raise ActionValuationError("Action state version changed within valuation")

        values = perspective_safe_values(
            frame,
            frame["p_score_after"].to_numpy(dtype=np.float64),
            frame["p_concede_after"].to_numpy(dtype=np.float64),
        )
        split = next(iter(splits))
        origin = PREDICTION_ORIGINS[split]
        row_count = len(frame)
        player_values = [
            None if pd.isna(value) else int(value) for value in frame["player_id"]
        ]
        payload: dict[str, Any] = {
            "match_id": pa.array(frame["match_id"], type=pa.int64()),
            "action_id": pa.array(frame["action_id"], type=pa.int64()),
            "player_id": pa.array(player_values, type=pa.int64()),
            "team_id": pa.array(frame["team_id"], type=pa.int64()),
        }
        for name in (
            "p_score_before",
            "p_score_after",
            "p_concede_before",
            "p_concede_after",
            "offensive_value",
            "defensive_value",
            "vaep_value",
        ):
            payload[name] = pa.array(values[name], type=pa.float64())
        payload.update(
            {
                "split": pa.array([split] * row_count, type=pa.string()),
                "prediction_origin": pa.array([origin] * row_count, type=pa.string()),
                "analytics_run_id": pa.array(
                    [lineage.get("analytics_run_id")] * row_count,
                    type=pa.int64(),
                ),
                "action_value_version": pa.array(
                    [ACTION_VALUE_VERSION] * row_count, type=pa.string()
                ),
            }
        )
        for name in (
            "action_mapping_version",
            "state_contract_version",
            "feature_version",
            "target_policy_version",
            "split_version",
            "score_model_version",
            "concede_model_version",
            "score_calibration_version",
            "concede_calibration_version",
        ):
            payload[name] = pa.array([lineage[name]] * row_count, type=pa.string())
        return pa.table(payload, schema=cls._action_value_schema(pa))

    @staticmethod
    def _add_split_counts(
        match: Any,
        split_counts: dict[str, int],
        origin_counts: dict[str, int],
    ) -> None:
        split = str(match["split"].iloc[0])
        count = len(match)
        split_counts[split] += count
        origin_counts[PREDICTION_ORIGINS[split]] += count

    @staticmethod
    def _action_value_schema(pa: Any) -> Any:
        return pa.schema(
            [
                pa.field("match_id", pa.int64(), nullable=False),
                pa.field("action_id", pa.int64(), nullable=False),
                pa.field("player_id", pa.int64(), nullable=True),
                pa.field("team_id", pa.int64(), nullable=False),
                pa.field("p_score_before", pa.float64(), nullable=False),
                pa.field("p_score_after", pa.float64(), nullable=False),
                pa.field("p_concede_before", pa.float64(), nullable=False),
                pa.field("p_concede_after", pa.float64(), nullable=False),
                pa.field("offensive_value", pa.float64(), nullable=False),
                pa.field("defensive_value", pa.float64(), nullable=False),
                pa.field("vaep_value", pa.float64(), nullable=False),
                pa.field("split", pa.string(), nullable=False),
                pa.field("prediction_origin", pa.string(), nullable=False),
                pa.field("analytics_run_id", pa.int64(), nullable=True),
                pa.field("action_value_version", pa.string(), nullable=False),
                pa.field("action_mapping_version", pa.string(), nullable=False),
                pa.field("state_contract_version", pa.string(), nullable=False),
                pa.field("feature_version", pa.string(), nullable=False),
                pa.field("target_policy_version", pa.string(), nullable=False),
                pa.field("split_version", pa.string(), nullable=False),
                pa.field("score_model_version", pa.string(), nullable=False),
                pa.field("concede_model_version", pa.string(), nullable=False),
                pa.field("score_calibration_version", pa.string(), nullable=False),
                pa.field("concede_calibration_version", pa.string(), nullable=False),
            ]
        )

    @staticmethod
    def _reconcile_counts(
        manifest: Mapping[str, Any],
        prediction_counts: Mapping[str, int],
        value_stats: Mapping[str, Any],
    ) -> None:
        declared = (manifest.get("split") or {}).get("splits") or {}
        expected = {
            split: int((declared.get(split) or {}).get("eligible_label_count", -1))
            for split in PREDICTION_ORIGINS
        }
        if dict(prediction_counts) != expected:
            raise ActionValuationError(
                "Inference row counts do not reconcile with the training manifest"
            )
        if value_stats.get("split_counts") != expected:
            raise ActionValuationError(
                "Action-value row counts do not reconcile with inference"
            )
        expected_matches = sum(
            int((declared.get(split) or {}).get("match_count", -1))
            for split in PREDICTION_ORIGINS
        )
        if value_stats.get("match_count") != expected_matches:
            raise ActionValuationError(
                "Action-value match count does not reconcile with split assignments"
            )

    @staticmethod
    def _value_manifest(
        *,
        manifest: Mapping[str, Any],
        actions_path: Path,
        dataset_path: Path,
        output_path: Path,
        lineage: Mapping[str, Any],
        stats: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "action_value_version": ACTION_VALUE_VERSION,
            "formula_provider": "socceraction.vaep.formula",
            "formula_semantics": "socceraction==1.5.3",
            "dataset_fingerprint": lineage["dataset_fingerprint"],
            "row_count": stats["row_count"],
            "match_count": stats["match_count"],
            "split_counts": dict(stats["split_counts"]),
            "prediction_origin_counts": dict(stats["prediction_origin_counts"]),
            "lineage": dict(lineage),
            "source_artifacts": {
                "actions.parquet": {
                    "path": str(actions_path),
                    "sha256": file_sha256(actions_path),
                },
                "model_dataset.parquet": {
                    "path": str(dataset_path),
                    "sha256": file_sha256(dataset_path),
                },
            },
            "files": {output_path.name: file_sha256(output_path)},
            "training_status_at_start": manifest.get("status"),
        }

    @staticmethod
    def _update_training_manifest(
        manifest_path: Path,
        manifest: Mapping[str, Any],
        paths: ActionValuePaths,
        value_manifest: Mapping[str, Any],
    ) -> None:
        updated = dict(manifest)
        updated["stage"] = "action_valuation"
        updated["status"] = "action_valuation_complete"
        artifacts = dict(manifest.get("artifacts") or {})
        for path in (paths.action_values, paths.manifest):
            artifacts[path.name] = {
                "path": str(path),
                "sha256": file_sha256(path),
            }
        updated["artifacts"] = artifacts
        updated["action_valuation"] = {
            "version": ACTION_VALUE_VERSION,
            "formula_semantics": "socceraction==1.5.3",
            "row_count": value_manifest["row_count"],
            "match_count": value_manifest["match_count"],
            "split_counts": dict(value_manifest["split_counts"]),
            "prediction_origin_counts": dict(
                value_manifest["prediction_origin_counts"]
            ),
        }
        write_json(manifest_path, updated)

    @staticmethod
    def _verify_completed(
        root: Path,
        manifest: Mapping[str, Any],
        paths: ActionValuePaths,
    ) -> None:
        for filename in (ACTION_VALUES_FILENAME, ACTION_VALUES_MANIFEST_FILENAME):
            verify_training_file(root, manifest, filename)
        value_manifest = read_json(paths.manifest)
        expected = (value_manifest.get("files") or {}).get(ACTION_VALUES_FILENAME)
        if not isinstance(expected, str) or file_sha256(paths.action_values) != expected:
            raise ActionValuationError("Action-values hash does not match its manifest")


__all__ = [
    "ACTION_VALUES_FILENAME",
    "ACTION_VALUES_MANIFEST_FILENAME",
    "ACTION_VALUE_VERSION",
    "ActionValuationError",
    "ActionValuePaths",
    "ActionValueWriter",
    "PREDICTION_ORIGINS",
    "perspective_safe_values",
]
