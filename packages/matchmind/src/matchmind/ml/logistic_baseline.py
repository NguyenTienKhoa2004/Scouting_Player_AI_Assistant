"""Memory-bounded training for the two VAEP logistic baselines."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .feature_artifacts import baseline_feature_allowlist
from .dependencies import capture_runtime_dependencies
from .evaluation import binary_probability_report
from .training_manifest import require_corpus_adequate_for_promotion


LOGISTIC_BASELINE_VERSION = "vaep-logistic-sgd-baseline-v1"
PREPROCESSING_VERSION = "standard-scaler-train-only-v1"
LOGISTIC_BASELINE_REPORT_FILENAME = "logistic_baseline_report.json"


@dataclass(frozen=True, slots=True)
class LogisticBaselinePaths:
    directory: Path
    preprocessor: Path
    score_model: Path
    concede_model: Path
    report: Path
    training_manifest: Path


class LogisticBaselineTrainer:
    """Fit deterministic SGD logistic models from Parquet batches."""

    TARGETS = ("scores", "concedes")

    def write(
        self,
        label_directory: Path,
        *,
        epochs: int = 3,
        batch_size: int = 16_384,
        random_seed: int = 20260914,
        progress: Any | None = None,
    ) -> LogisticBaselinePaths:
        try:
            import joblib
            import numpy as np
            import pyarrow.parquet as pq
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "logistic baseline training requires pyarrow, numpy, joblib, and "
                "scikit-learn; install requirements.txt"
            ) from exc
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")

        root = Path(label_directory).resolve()
        training_manifest_path = root / "training_manifest.json"
        training_manifest = self._read_json(training_manifest_path)
        require_corpus_adequate_for_promotion(training_manifest)
        feature_manifest = self._read_json(root / "feature_allowlist.json")
        features = tuple(feature_manifest.get("feature_columns") or ())
        if features != baseline_feature_allowlist():
            raise ValueError("training feature allowlist is not the Plan 04 baseline")
        dataset_path = root / "model_dataset.parquet"
        expected_hash = (
            training_manifest.get("artifacts", {})
            .get(dataset_path.name, {})
            .get("sha256")
        )
        if progress is not None:
            progress("verifying model_dataset.parquet hash")
        if not isinstance(expected_hash, str) or self._sha256(dataset_path) != expected_hash:
            raise ValueError("model_dataset.parquet hash does not match training manifest")

        parquet_file = pq.ParquetFile(dataset_path)
        required = {*features, *self.TARGETS, "split"}
        if not required.issubset(parquet_file.schema_arrow.names):
            raise ValueError("model dataset is missing baseline training columns")
        parquet_file.close()

        scaler = StandardScaler(copy=False)
        counts = {target: np.zeros(2, dtype=np.int64) for target in self.TARGETS}
        train_rows = 0
        for batch_number, (matrix, targets) in enumerate(
            self._iter_split_batches(
                dataset_path, features, "train", batch_size=batch_size
            ),
            start=1,
        ):
            self._require_finite(matrix)
            scaler.partial_fit(matrix)
            train_rows += len(matrix)
            for target in self.TARGETS:
                counts[target] += np.bincount(targets[target], minlength=2)
            if progress is not None and batch_number % 10 == 0:
                progress(f"fitted preprocessing batch {batch_number}: {train_rows} rows")
        if train_rows == 0:
            raise ValueError("training split contains no rows")
        class_weights = {
            target: self._balanced_weights(values) for target, values in counts.items()
        }

        classifiers = {
            target: SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.0001,
                fit_intercept=True,
                learning_rate="optimal",
                average=True,
                random_state=random_seed,
            )
            for target in self.TARGETS
        }
        fitted = {target: False for target in self.TARGETS}
        for epoch in range(epochs):
            seen = 0
            rng = np.random.default_rng(random_seed + epoch)
            for batch_number, (matrix, targets) in enumerate(
                self._iter_split_batches(
                    dataset_path, features, "train", batch_size=batch_size
                ),
                start=1,
            ):
                self._require_finite(matrix)
                scaler.transform(matrix, copy=False)
                order = rng.permutation(len(matrix))
                for target in self.TARGETS:
                    truth = targets[target][order]
                    weights = class_weights[target]
                    sample_weight = np.where(
                        truth == 1, weights["positive"], weights["negative"]
                    )
                    classifiers[target].partial_fit(
                        matrix[order],
                        truth,
                        classes=np.array([0, 1], dtype=np.int8),
                        sample_weight=sample_weight,
                    )
                    fitted[target] = True
                seen += len(matrix)
                if progress is not None and batch_number % 10 == 0:
                    progress(
                        f"trained epoch {epoch + 1}/{epochs}, batch {batch_number}: "
                        f"{seen} rows"
                    )
        if not all(fitted.values()):
            raise ValueError("both logistic baselines must be fitted")

        truths: dict[str, list[Any]] = {target: [] for target in self.TARGETS}
        probabilities: dict[str, list[Any]] = {target: [] for target in self.TARGETS}
        inference_seconds = {target: 0.0 for target in self.TARGETS}
        validation_rows = 0
        for batch_number, (matrix, targets) in enumerate(
            self._iter_split_batches(
                dataset_path, features, "validation", batch_size=batch_size
            ),
            start=1,
        ):
            self._require_finite(matrix)
            scaler.transform(matrix, copy=False)
            validation_rows += len(matrix)
            for target in self.TARGETS:
                started = time.perf_counter()
                probability = classifiers[target].predict_proba(matrix)[:, 1]
                inference_seconds[target] += time.perf_counter() - started
                truths[target].append(targets[target])
                probabilities[target].append(probability)
            if progress is not None and batch_number % 10 == 0:
                progress(f"evaluated validation batch {batch_number}: {validation_rows} rows")
        if validation_rows == 0:
            raise ValueError("validation split contains no rows")

        metrics = {
            target: binary_probability_report(
                np.concatenate(truths[target]),
                np.concatenate(probabilities[target]),
                inference_seconds=inference_seconds[target],
            )
            for target in self.TARGETS
        }
        output = root / LOGISTIC_BASELINE_VERSION
        output.mkdir(parents=True, exist_ok=True)
        paths = LogisticBaselinePaths(
            directory=output,
            preprocessor=output / "logistic_preprocessor.joblib",
            score_model=output / "score_logistic_baseline.joblib",
            concede_model=output / "concede_logistic_baseline.joblib",
            report=output / LOGISTIC_BASELINE_REPORT_FILENAME,
            training_manifest=training_manifest_path,
        )
        self._joblib_dump(joblib, scaler, paths.preprocessor)
        self._joblib_dump(joblib, classifiers["scores"], paths.score_model)
        self._joblib_dump(joblib, classifiers["concedes"], paths.concede_model)
        report = {
            "schema_version": 1,
            "baseline_version": LOGISTIC_BASELINE_VERSION,
            "algorithm": "sklearn.SGDClassifier(loss='log_loss')",
            "optimization": "out_of_core_partial_fit",
            "random_seed": random_seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "feature_count": len(features),
            "feature_allowlist_sha256": feature_manifest[
                "feature_allowlist_sha256"
            ],
            "preprocessing": {
                "version": PREPROCESSING_VERSION,
                "fit_split": "train",
                "algorithm": "sklearn.StandardScaler.partial_fit",
                "missing_value_policy": "reject",
            },
            "class_weights": {
                target: {
                    "fit_split": "train",
                    "negative_count": int(counts[target][0]),
                    "positive_count": int(counts[target][1]),
                    **class_weights[target],
                }
                for target in self.TARGETS
            },
            "evaluation_split": "validation",
            "test_split_accessed": False,
            "metrics": metrics,
        }
        self._write_json(paths.report, report)
        runtime_path = output / "runtime_dependencies.json"
        self._write_json(runtime_path, capture_runtime_dependencies().as_dict())
        generated = (
            paths.preprocessor,
            paths.score_model,
            paths.concede_model,
            paths.report,
            runtime_path,
        )
        artifact_records = {
            path.name: {"path": str(path), "sha256": self._sha256(path)}
            for path in generated
        }
        updated = dict(training_manifest)
        updated["stage"] = "baseline_training"
        updated["status"] = "logistic_baselines_evaluated"
        updated["runtime_dependencies"] = capture_runtime_dependencies().as_dict()
        updated["artifacts"] = {
            **dict(training_manifest.get("artifacts") or {}),
            **artifact_records,
        }
        models = dict(training_manifest.get("models") or {})
        models.update(
            {
                "status": "logistic_baselines_evaluated",
                "preprocessing": {
                    **report["preprocessing"],
                    "artifact": artifact_records[paths.preprocessor.name],
                },
                "fitted_artifacts": {
                    "score_logistic_baseline": {
                        "version": f"score-{LOGISTIC_BASELINE_VERSION}",
                        **artifact_records[paths.score_model.name],
                    },
                    "concede_logistic_baseline": {
                        "version": f"concede-{LOGISTIC_BASELINE_VERSION}",
                        **artifact_records[paths.concede_model.name],
                    },
                },
                "reports": {
                    "logistic_baseline": artifact_records[paths.report.name]
                },
                "baseline_validation_metrics": metrics,
            }
        )
        updated["models"] = models
        updated["production_promotion"] = {
            "corpus_adequate": True,
            "allowed": False,
            "blockers": ["models:xgboost_not_trained_or_evaluated"],
        }
        self._write_json(training_manifest_path, updated)
        if progress is not None:
            progress(f"completed logistic baselines: {output}")
        return paths

    @staticmethod
    def _iter_split_batches(
        dataset_path: Path,
        features: tuple[str, ...],
        split: str,
        *,
        batch_size: int,
    ) -> Iterator[tuple[Any, dict[str, Any]]]:
        import numpy as np
        import pyarrow as pa
        import pyarrow.dataset as ds

        columns = [*features, "scores", "concedes", "split"]
        scanner = ds.dataset(dataset_path, format="parquet").scanner(
            columns=columns,
            filter=ds.field("split") == split,
            batch_size=batch_size,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            selected = pa.Table.from_batches([batch])
            if selected.num_rows == 0:
                continue
            matrix = np.column_stack(
                [
                    selected[name].to_numpy(zero_copy_only=False)
                    for name in features
                ]
            ).astype(np.float32, copy=False)
            targets = {
                target: selected[target]
                .to_numpy(zero_copy_only=False)
                .astype(np.int8, copy=False)
                for target in LogisticBaselineTrainer.TARGETS
            }
            yield matrix, targets

    @staticmethod
    def _balanced_weights(counts: Any) -> dict[str, float]:
        if len(counts) != 2 or (counts <= 0).any():
            raise ValueError("training split must contain both target classes")
        total = float(counts.sum())
        return {
            "negative": total / (2.0 * float(counts[0])),
            "positive": total / (2.0 * float(counts[1])),
        }

    @staticmethod
    def _require_finite(matrix: Any) -> None:
        import numpy as np

        if not np.isfinite(matrix).all():
            raise ValueError("model features contain null or non-finite values")

    @staticmethod
    def _joblib_dump(joblib: Any, value: Any, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        joblib.dump(value, temporary, compress=3)
        temporary.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "LOGISTIC_BASELINE_REPORT_FILENAME",
    "LOGISTIC_BASELINE_VERSION",
    "PREPROCESSING_VERSION",
    "LogisticBaselinePaths",
    "LogisticBaselineTrainer",
]
