"""Reproducible preparation-stage training manifest and promotion gate."""

from __future__ import annotations

from typing import Any

from matchmind.vaep_features.artifact_loader import FeatureArtifacts
from matchmind.shared.runtime_dependencies import capture_runtime_dependencies


TRAINING_MANIFEST_FILENAME = "training_manifest.json"
TRAINING_MANIFEST_VERSION = "vaep-training-manifest-v1"


class ProductionPromotionBlocked(RuntimeError):
    """Raised when a model bundle does not meet its declared promotion gates."""


def build_preparation_training_manifest(
    *,
    source_artifacts: FeatureArtifacts,
    corpus_assessment: dict[str, Any],
    split_manifest: dict[str, Any],
    feature_allowlist_manifest: dict[str, Any],
    artifacts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Build the manifest now; later training stages append fitted artifacts."""

    corpus_adequate = bool(
        corpus_assessment.get("adequate_for_production_evaluation")
    )
    corpus_blockers = [
        f"corpus:{value}" for value in corpus_assessment.get("blockers", [])
    ]
    return {
        "schema_version": 1,
        "training_manifest_version": TRAINING_MANIFEST_VERSION,
        "stage": "data_preparation",
        "status": "ready_for_training" if corpus_adequate else "experimental",
        "source": {
            "feature_artifact_directory": str(source_artifacts.directory),
            "feature_manifest_sha256": _sha256_from_split_manifest(split_manifest),
            # Legacy aliases retained for consumers of training manifest v1.
            "plan03_directory": str(source_artifacts.directory),
            "plan03_manifest_sha256": _sha256_from_split_manifest(split_manifest),
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
        "runtime_dependencies": capture_runtime_dependencies().as_dict(),
        "artifacts": artifacts,
        "models": {
            "status": "not_trained",
            "preprocessing": None,
            "calibration": None,
            "fitted_artifacts": {},
            "reports": {},
        },
        "production_promotion": {
            "corpus_adequate": corpus_adequate,
            "allowed": False,
            "blockers": corpus_blockers + ["models:not_trained_or_evaluated"],
        },
    }


def require_corpus_adequate_for_promotion(manifest: dict[str, Any]) -> None:
    """Refuse promotion when the materialized corpus failed its policy."""

    promotion = manifest.get("production_promotion")
    if not isinstance(promotion, dict) or promotion.get("corpus_adequate") is not True:
        blockers = promotion.get("blockers", []) if isinstance(promotion, dict) else []
        raise ProductionPromotionBlocked(
            "production promotion requires an adequate materialized corpus; "
            f"blockers={blockers}"
        )


def _sha256_from_split_manifest(split_manifest: dict[str, Any]) -> str:
    value = split_manifest.get("source_feature_manifest_sha256")
    if value is None:
        # Backward compatibility for preparation artifacts created before the
        # domain-based feature artifact naming was introduced.
        value = split_manifest.get("source_plan03_manifest_sha256")
    if not isinstance(value, str):
        raise ValueError("split manifest lacks source feature manifest SHA-256")
    return value


__all__ = [
    "TRAINING_MANIFEST_FILENAME",
    "TRAINING_MANIFEST_VERSION",
    "ProductionPromotionBlocked",
    "build_preparation_training_manifest",
    "require_corpus_adequate_for_promotion",
]
