"""Hash-complete artifact auditing and independent Plan 04 run comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pitchpulse.shared.file_io import file_sha256, read_json, write_json


REPRODUCIBILITY_VERSION = "vaep-reproducibility-v1"
REPRODUCIBILITY_REPORT_FILENAME = "reproducibility_report.json"

# PostgreSQL persistence is Task 13 and intentionally is not part of an
# independent filesystem rerun. The receipt itself is added after comparison.
REQUIRED_REPRODUCIBLE_ARTIFACTS = frozenset(
    {
        "action_labels.parquet",
        "target_policy.json",
        "target_audit.json",
        "label_manifest.json",
        "split_assignments.parquet",
        "split_manifest.json",
        "model_dataset.parquet",
        "feature_allowlist.json",
        "model_dataset_manifest.json",
        "runtime_dependencies.json",
        "logistic_preprocessor.joblib",
        "score_logistic_baseline.joblib",
        "concede_logistic_baseline.joblib",
        "logistic_baseline_report.json",
        "xgboost_runtime_dependencies.json",
        "score_xgboost.json",
        "concede_xgboost.json",
        "score_calibration.joblib",
        "concede_calibration.joblib",
        "xgboost_validation_report.json",
        "frozen_model_manifest.json",
        "test_metrics.json",
        "calibration_report.json",
        "sanity_checks.json",
        "error_analysis.json",
        "action_values.parquet",
        "action_values_manifest.json",
        "player_vaep.parquet",
        "player_vaep_manifest.json",
    }
)

_BYTE_EXACT_SUFFIXES = frozenset({".parquet", ".joblib"})
_BYTE_EXACT_JSON = frozenset({"score_xgboost.json", "concede_xgboost.json"})
_VOLATILE_JSON_KEYS = frozenset(
    {
        "artifact_directory",
        "directory",
        "inference_latency",
        "path",
        "sha256",
        "source_feature_artifact_directory",
        "source_label_manifest",
        "training_status_at_start",
    }
)


class ReproducibilityError(ValueError):
    """Raised when artifact integrity or reproducibility does not hold."""


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    name: str
    byte_sha256: str
    comparison_sha256: str
    comparison: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactInventory:
    directory: Path
    manifest: dict[str, Any]
    artifacts: dict[str, ArtifactDigest]
    fingerprint: str


def audit_training_artifacts(artifact_directory: Path) -> ArtifactInventory:
    """Verify every required artifact against its training-manifest hash."""

    root = Path(artifact_directory).resolve()
    manifest = read_json(root / "training_manifest.json")
    declared = manifest.get("artifacts")
    if not isinstance(declared, Mapping):
        raise ReproducibilityError("training manifest lacks an artifact inventory")

    missing = sorted(REQUIRED_REPRODUCIBLE_ARTIFACTS.difference(declared))
    if missing:
        raise ReproducibilityError(
            f"training manifest is not hash-complete; missing={missing}"
        )

    inventory: dict[str, ArtifactDigest] = {}
    for name in sorted(REQUIRED_REPRODUCIBLE_ARTIFACTS):
        record = declared.get(name)
        if not isinstance(record, Mapping):
            raise ReproducibilityError(f"invalid artifact record for {name}")
        path = Path(str(record.get("path", ""))).resolve()
        expected = record.get("sha256")
        if not path.is_file():
            raise ReproducibilityError(f"artifact is missing: {path}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ReproducibilityError(f"artifact has no valid SHA-256: {name}")
        actual = file_sha256(path)
        if actual != expected:
            raise ReproducibilityError(f"artifact hash mismatch: {name}")
        comparison, comparison_hash = _comparison_digest(path, name, actual)
        inventory[name] = ArtifactDigest(
            name=name,
            byte_sha256=actual,
            comparison_sha256=comparison_hash,
            comparison=comparison,
            size_bytes=path.stat().st_size,
        )

    fingerprint = _inventory_fingerprint(inventory)
    return ArtifactInventory(
        directory=root,
        manifest=manifest,
        artifacts=inventory,
        fingerprint=fingerprint,
    )


def compare_independent_runs(
    reference_directory: Path,
    candidate_directory: Path,
) -> dict[str, Any]:
    """Compare a clean independent rerun with the promoted reference run."""

    reference = audit_training_artifacts(reference_directory)
    candidate = audit_training_artifacts(candidate_directory)
    _require_same_inputs(reference.manifest, candidate.manifest)

    comparisons: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for name in sorted(REQUIRED_REPRODUCIBLE_ARTIFACTS):
        expected = reference.artifacts[name]
        actual = candidate.artifacts[name]
        matched = expected.comparison_sha256 == actual.comparison_sha256
        if not matched:
            mismatches.append(name)
        comparisons[name] = {
            "comparison": expected.comparison,
            "matched": matched,
            "reference_sha256": expected.comparison_sha256,
            "candidate_sha256": actual.comparison_sha256,
            "reference_byte_sha256": expected.byte_sha256,
            "candidate_byte_sha256": actual.byte_sha256,
            "size_bytes": expected.size_bytes,
        }

    return {
        "schema_version": 1,
        "reproducibility_version": REPRODUCIBILITY_VERSION,
        "status": "passed" if not mismatches else "failed",
        "verification_mode": "independent_full_plan04_rerun",
        "reference_directory": str(reference.directory),
        "candidate_directory": str(candidate.directory),
        "source": _input_identity(reference.manifest),
        "required_artifact_count": len(REQUIRED_REPRODUCIBLE_ARTIFACTS),
        "byte_exact_artifact_count": sum(
            item.comparison == "byte_exact" for item in reference.artifacts.values()
        ),
        "canonical_json_artifact_count": sum(
            item.comparison == "canonical_json"
            for item in reference.artifacts.values()
        ),
        "reference_fingerprint": reference.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "mismatches": mismatches,
        "artifacts": comparisons,
    }


def finalize_reproducibility_report(
    reference_directory: Path,
    report: Mapping[str, Any],
) -> Path:
    """Persist a passed receipt and add its hash to the training manifest."""

    if report.get("status") != "passed":
        raise ReproducibilityError(
            f"independent rerun differs: {report.get('mismatches', [])}"
        )
    root = Path(reference_directory).resolve()
    manifest_path = root / "training_manifest.json"
    manifest = read_json(manifest_path)
    report_path = root / REPRODUCIBILITY_REPORT_FILENAME
    write_json(report_path, report)
    report_hash = file_sha256(report_path)

    updated = dict(manifest)
    updated["artifacts"] = {
        **dict(manifest.get("artifacts") or {}),
        report_path.name: {"path": str(report_path), "sha256": report_hash},
    }
    updated["reproducibility"] = {
        "status": "passed",
        "version": REPRODUCIBILITY_VERSION,
        "verification_mode": report["verification_mode"],
        "required_artifact_count": report["required_artifact_count"],
        "reference_fingerprint": report["reference_fingerprint"],
        "candidate_fingerprint": report["candidate_fingerprint"],
        "report": {"path": str(report_path), "sha256": report_hash},
    }
    write_json(manifest_path, updated)
    return report_path


def _comparison_digest(path: Path, name: str, byte_hash: str) -> tuple[str, str]:
    if path.suffix.lower() in _BYTE_EXACT_SUFFIXES or name in _BYTE_EXACT_JSON:
        return "byte_exact", byte_hash
    if path.suffix.lower() != ".json":
        raise ReproducibilityError(f"unsupported reproducibility artifact: {name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if name in {"runtime_dependencies.json", "xgboost_runtime_dependencies.json"}:
        value = _core_runtime_dependencies(value)
    canonical = _canonical_json(value)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "canonical_json", hashlib.sha256(encoded).hexdigest()


def _core_runtime_dependencies(value: Any) -> dict[str, Any]:
    """Compare the pinned model runtime, not incidental transitive packages."""

    if not isinstance(value, Mapping):
        raise ReproducibilityError("runtime dependency artifact must be an object")
    packages = value.get("packages") or {}
    required = (
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "socceraction",
        "xgboost",
    )
    if not isinstance(packages, Mapping) or any(name not in packages for name in required):
        raise ReproducibilityError("runtime dependency artifact lacks pinned packages")
    return {
        "schema_version": value.get("schema_version"),
        "python": value.get("python"),
        "platform": value.get("platform"),
        "packages": {name: packages[name] for name in required},
    }


def _canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json(child)
            for key, child in sorted(value.items())
            if key not in _VOLATILE_JSON_KEYS
            and not str(key).endswith("_path")
            and not str(key).endswith("_directory")
        }
    if isinstance(value, list):
        return [_canonical_json(child) for child in value]
    return value


def _input_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source = manifest.get("source") or {}
    return {
        "dataset_fingerprint": source.get("dataset_fingerprint"),
        "versions": dict(manifest.get("versions") or {}),
        "split_assignment_fingerprint": (manifest.get("split") or {}).get(
            "assignment_fingerprint"
        ),
    }


def _require_same_inputs(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    if _input_identity(reference) != _input_identity(candidate):
        raise ReproducibilityError(
            "independent run does not use the same immutable inputs and policies"
        )


def _inventory_fingerprint(artifacts: Mapping[str, ArtifactDigest]) -> str:
    digest = hashlib.sha256()
    for name, artifact in sorted(artifacts.items()):
        digest.update(f"{name}|{artifact.comparison_sha256}\n".encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "ArtifactDigest",
    "ArtifactInventory",
    "REPRODUCIBILITY_REPORT_FILENAME",
    "REPRODUCIBILITY_VERSION",
    "REQUIRED_REPRODUCIBLE_ARTIFACTS",
    "ReproducibilityError",
    "audit_training_artifacts",
    "compare_independent_runs",
    "finalize_reproducibility_report",
]
