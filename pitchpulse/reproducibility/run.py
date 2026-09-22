"""Rebuild Plan 04 independently and record its reproducibility receipt."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from pitchpulse.labeling_and_splitting.label_artifacts import (  # noqa: E402
    ChunkedTargetLabelWriter,
)
from pitchpulse.labeling_and_splitting.split_artifacts import (  # noqa: E402
    ChunkedSplitArtifactWriter,
)
from pitchpulse.model_dataset.artifacts import ChunkedModelDatasetWriter  # noqa: E402
from pitchpulse.model_dataset.finalize import ChunkedPreparationFinalizer  # noqa: E402
from pitchpulse.reproducibility.verification import (  # noqa: E402
    compare_independent_runs,
    finalize_reproducibility_report,
)
from pitchpulse.model_training.frozen_evaluation import FrozenModelEvaluator  # noqa: E402
from pitchpulse.model_training.logistic_baseline import LogisticBaselineTrainer  # noqa: E402
from pitchpulse.model_training.valuation import ActionValueWriter  # noqa: E402
from pitchpulse.model_training.xgboost_models import XGBoostVaepTrainer  # noqa: E402
from pitchpulse.pipelines.settings import (  # noqa: E402
    DATASET_MANIFEST,
    MODEL_OUTPUT,
    latest_artifact,
)
from pitchpulse.player_vaep.artifacts import PlayerAggregationWriter  # noqa: E402
from pitchpulse.shared.file_io import read_json, write_json  # noqa: E402
from pitchpulse.shared.file_io import file_sha256  # noqa: E402
from pitchpulse.shared.paths import PROJECT_ROOT  # noqa: E402


DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "artifacts" / "reproducibility" / "plan04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently rerun Plan 04 and compare every artifact"
    )
    parser.add_argument(
        "--reference-directory",
        type=Path,
        help="Promoted Plan 04 directory; defaults to the latest run",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
        help="Empty output root used for the independent rerun",
    )
    parser.add_argument(
        "--compare-existing",
        type=Path,
        help="Compare an already completed candidate instead of rebuilding",
    )
    parser.add_argument(
        "--resume-existing",
        type=Path,
        help="Resume an interrupted independent rerun from its run directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = (
        args.reference_directory.resolve()
        if args.reference_directory
        else latest_artifact(MODEL_OUTPUT, "training_manifest.json")
    )
    if args.compare_existing and args.resume_existing:
        raise ValueError("use only one of --compare-existing and --resume-existing")
    if args.compare_existing:
        candidate = args.compare_existing.resolve()
    elif args.resume_existing:
        candidate = args.resume_existing.resolve()
        _complete_candidate(candidate, read_json(reference / "training_manifest.json"))
    else:
        candidate = _independent_rerun(reference, args.candidate_root.resolve())
    print("Comparing independent artifact inventories...", flush=True)
    report = compare_independent_runs(reference, candidate)
    report_path = finalize_reproducibility_report(reference, report)
    print(f"Reproducibility passed: {report_path}", flush=True)


def _independent_rerun(reference: Path, candidate_root: Path) -> Path:
    if candidate_root.exists() and any(candidate_root.iterdir()):
        raise ValueError(
            f"candidate root must be empty for an independent rerun: {candidate_root}"
        )
    reference_manifest = read_json(reference / "training_manifest.json")
    source = reference_manifest.get("source") or {}
    feature_value = source.get("feature_artifact_directory") or source.get(
        "feature_dataset_directory"
    )
    if not isinstance(feature_value, str):
        raise ValueError("reference manifest lacks its Plan 03 artifact directory")
    feature_directory = Path(feature_value).resolve()
    progress = lambda message: print(message, flush=True)

    labels = ChunkedTargetLabelWriter().write(
        feature_directory,
        dataset_manifest_path=DATASET_MANIFEST,
        output_root=candidate_root,
        progress=progress,
    )
    ChunkedSplitArtifactWriter().write(
        labels.directory,
        dataset_manifest_path=DATASET_MANIFEST,
        progress=progress,
    )
    ChunkedModelDatasetWriter().write(
        feature_directory, labels.directory, progress=progress
    )
    ChunkedPreparationFinalizer().write(
        feature_directory,
        labels.directory,
        dataset_manifest_path=DATASET_MANIFEST,
        progress=progress,
    )
    _complete_candidate(labels.directory, reference_manifest)
    return labels.directory


def _complete_candidate(candidate: Path, reference_manifest: dict[str, object]) -> None:
    """Complete or resume deterministic stages after preparation."""

    progress = lambda message: print(message, flush=True)
    manifest_path = candidate / "training_manifest.json"
    manifest = read_json(manifest_path)
    source = dict(manifest.get("source") or {})
    reference_source = reference_manifest.get("source") or {}
    source.update(
        {
            "feature_artifact_directory": reference_source.get(
                "feature_artifact_directory"
            )
            or reference_source.get("feature_dataset_directory"),
            "plan03_directory": reference_source.get("plan03_directory")
            or reference_source.get("feature_artifact_directory")
            or reference_source.get("feature_dataset_directory"),
            "plan03_manifest_sha256": reference_source.get(
                "feature_manifest_sha256"
            )
            or source.get("feature_manifest_sha256")
            or reference_source.get("plan03_manifest_sha256"),
        }
    )
    manifest["source"] = source
    write_json(manifest_path, manifest)

    status = manifest.get("status")
    if status in {"ready_for_training", "experimental"}:
        LogisticBaselineTrainer().write(candidate, progress=progress)
        status = "logistic_baselines_evaluated"
    if status == "logistic_baselines_evaluated":
        XGBoostVaepTrainer().write(candidate, progress=progress)
        status = "xgboost_validation_complete"
    if status == "xgboost_validation_complete":
        FrozenModelEvaluator().write(candidate, progress=progress)
        status = "test_evaluation_complete"
    if status == "test_evaluation_complete":
        ActionValueWriter().write(candidate, progress=progress)
        status = "action_valuation_complete"
    if status == "action_valuation_complete":
        PlayerAggregationWriter().write(
            candidate, DATASET_MANIFEST, progress=progress
        )
        status = "player_aggregation_complete"
    _record_runtime_dependencies(candidate)
    _complete_artifact_hashes(candidate)
    if status != "player_aggregation_complete":
        raise ValueError(f"cannot resume candidate from status {status!r}")


def _record_runtime_dependencies(candidate: Path) -> None:
    package_names = (
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "socceraction",
        "xgboost",
    )
    receipt = {
        "schema_version": "pitchpulse-runtime-dependencies-v1",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
    }
    outputs = (
        candidate / "vaep-logistic-sgd-baseline-v1" / "runtime_dependencies.json",
        candidate
        / "vaep-xgboost-calibrated-v1"
        / "xgboost_runtime_dependencies.json",
    )
    manifest_path = candidate / "training_manifest.json"
    manifest = read_json(manifest_path)
    artifacts = dict(manifest.get("artifacts") or {})
    for path in outputs:
        write_json(path, receipt)
        artifacts[path.name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    manifest["artifacts"] = artifacts
    manifest["runtime_dependencies"] = receipt
    write_json(manifest_path, manifest)


def _complete_artifact_hashes(candidate: Path) -> None:
    """Make the final candidate manifest hash-complete before comparison."""

    manifest_path = candidate / "training_manifest.json"
    manifest = read_json(manifest_path)
    artifacts = dict(manifest.get("artifacts") or {})
    for name, raw_record in artifacts.items():
        record = dict(raw_record or {})
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = candidate / path
        if not path.is_file():
            raise FileNotFoundError(f"declared artifact is missing: {name}")
        record["path"] = str(path.resolve())
        record["sha256"] = file_sha256(path)
        artifacts[name] = record
    manifest["artifacts"] = artifacts
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Plan 04 reproducibility failed: {exc}") from exc
