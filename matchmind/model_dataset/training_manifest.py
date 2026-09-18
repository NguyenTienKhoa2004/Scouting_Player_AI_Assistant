"""Build the small manifest consumed by model training."""

from __future__ import annotations

from typing import Any

from matchmind.vaep_features.feature_dataset_loader import FeatureDataset
TRAINING_MANIFEST_FILENAME = "training_manifest.json"
TRAINING_MANIFEST_VERSION = "vaep-training-manifest-v1"
PREPARATION_VERSION = "vaep-preparation-v2"


class TrainingCorpusNotReady(RuntimeError):
    """Raised when the prepared dataset is not large or complete enough."""


def build_preparation_training_manifest(
    *,
    source_artifacts: FeatureDataset,
    corpus_assessment: dict[str, Any],
    split_manifest: dict[str, Any],
    feature_allowlist_manifest: dict[str, Any],
    artifacts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build the manifest now; later training stages append fitted artifacts."""

    corpus_adequate = bool(
        corpus_assessment.get("adequate_for_production_evaluation")
    )
    blockers = list(corpus_assessment.get("blockers", []))
    return {
        "schema_version": 1,
        "training_manifest_version": TRAINING_MANIFEST_VERSION,
        "stage": "data_preparation",
        "status": "ready_for_training" if corpus_adequate else "experimental",
        "source": {
            "feature_dataset_directory": str(source_artifacts.directory),
            "feature_manifest_sha256": _sha256_from_split_manifest(split_manifest),
            "analytics_run_id": source_artifacts.lineage.analytics_run_id,
            "dataset_fingerprint": source_artifacts.lineage.dataset_fingerprint,
        },
        "versions": {
            "action_mapping": source_artifacts.lineage.mapping_version,
            "coordinate_system": source_artifacts.lineage.coordinate_system_version,
            "state_contract": source_artifacts.lineage.state_contract_version,
            "feature": source_artifacts.lineage.feature_version,
            "target_policy": split_manifest["target_policy_version"],
            "split": split_manifest["split_version"],
            "model_dataset": split_manifest["model_dataset_version"],
        },
        "corpus": corpus_assessment,
        "split": {
            "policy": split_manifest["policy"],
            "counts": split_manifest["counts"],
            "splits": split_manifest["splits"],
            "assignment_fingerprint": split_manifest["assignment_fingerprint"],
        },
        "features": feature_allowlist_manifest,
        "artifacts": artifacts,
        "models": {"status": "not_trained"},
        "ready_for_training": corpus_adequate,
        "training_blockers": blockers,
    }


def require_training_corpus_ready(manifest: dict[str, Any]) -> None:
    if manifest.get("ready_for_training") is not True:
        blockers = manifest.get("training_blockers", [])
        raise TrainingCorpusNotReady(
            f"Training dataset is not ready; blockers={blockers}"
        )


def _sha256_from_split_manifest(split_manifest: dict[str, Any]) -> str:
    value = split_manifest.get("source_feature_manifest_sha256")
    if not isinstance(value, str):
        raise ValueError("split manifest lacks source feature manifest SHA-256")
    return value


__all__ = [
    "PREPARATION_VERSION",
    "TRAINING_MANIFEST_FILENAME",
    "TRAINING_MANIFEST_VERSION",
    "TrainingCorpusNotReady",
    "build_preparation_training_manifest",
    "require_training_corpus_ready",
]
