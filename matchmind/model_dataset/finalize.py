"""Finish model-dataset preparation without loading everything into memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.vaep_features.action_state import STATE_CONTRACT_VERSION
from matchmind.spadl.converter import (
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
)
from matchmind.vaep_features.feature_builder import BASE_FEATURE_VERSION
from matchmind.shared.file_io import file_sha256, read_json, write_json

from matchmind.vaep_features.feature_dataset_loader import (
    FeatureDataset,
    FeatureDatasetLineage,
)
from matchmind.corpus.training_dataset_validator import (
    load_training_corpus_manifest,
    single_file_training_corpus,
)
from .feature_allowlist import (
    FEATURE_ALLOWLIST_FILENAME,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_VERSION,
)
from .artifacts import MODEL_DATASET_MANIFEST_FILENAME
from .training_manifest import (
    PREPARATION_VERSION,
    TRAINING_MANIFEST_FILENAME,
    build_preparation_training_manifest,
    require_training_corpus_ready,
)


@dataclass(frozen=True, slots=True)
class FinalizedPreparationPaths:
    directory: Path
    training_manifest: Path


class ChunkedPreparationFinalizer:
    """Run the corpus gate and write the preparation training manifest."""

    def write(
        self,
        feature_artifact_directory: Path,
        label_directory: Path,
        *,
        corpus_manifest_path: Path | None = None,
        match_metadata_path: Path | None = None,
        progress: Any | None = None,
    ) -> FinalizedPreparationPaths:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "preparation finalization requires pyarrow; install requirements.txt"
            ) from exc

        if (corpus_manifest_path is None) == (match_metadata_path is None):
            raise ValueError(
                "provide exactly one of corpus_manifest_path or match_metadata_path"
            )
        corpus = (
            load_training_corpus_manifest(corpus_manifest_path)
            if corpus_manifest_path is not None
            else single_file_training_corpus(match_metadata_path)
        )
        feature_artifact_directory = Path(feature_artifact_directory).resolve()
        directory = Path(label_directory).resolve()
        feature_manifest = read_json(feature_artifact_directory / "manifest.json")
        quality = read_json(
            feature_artifact_directory / "conversion_quality_report.json"
        )
        label_manifest = read_json(directory / "label_manifest.json")
        split_manifest = read_json(directory / "split_manifest.json")
        model_manifest = read_json(
            directory / MODEL_DATASET_MANIFEST_FILENAME
        )
        allowlist_manifest = read_json(directory / FEATURE_ALLOWLIST_FILENAME)

        fingerprint = str(feature_manifest.get("dataset_fingerprint"))
        for label, manifest in (
            ("labels", label_manifest),
            ("splits", split_manifest),
            ("model dataset", model_manifest),
            ("feature allowlist", allowlist_manifest),
        ):
            if manifest.get("source_dataset_fingerprint") != fingerprint:
                raise ValueError(f"{label} do not belong to the selected feature artifact artifact")

        expected_versions = {
            "mapping_version": ACTION_MAPPING_VERSION,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
            "feature_version": BASE_FEATURE_VERSION,
        }
        for name, expected in expected_versions.items():
            if feature_manifest.get(name) != expected or quality.get(name) != expected:
                raise ValueError(f"feature artifact {name} does not match the baseline")
        state_version = feature_manifest.get(
            "state_contract_version", quality.get("state_contract_version")
        )
        if state_version != STATE_CONTRACT_VERSION:
            raise ValueError("feature artifact state contract does not match the baseline")
        if model_manifest.get("model_dataset_version") != MODEL_DATASET_VERSION:
            raise ValueError("model dataset uses an unexpected version")

        artifact_paths = (
            directory / "action_labels.parquet",
            directory / "target_policy.json",
            directory / "target_audit.json",
            directory / "label_manifest.json",
            directory / "split_assignments.parquet",
            directory / "split_manifest.json",
            directory / MODEL_DATASET_FILENAME,
            directory / FEATURE_ALLOWLIST_FILENAME,
            directory / MODEL_DATASET_MANIFEST_FILENAME,
        )
        declared = {
            **(label_manifest.get("files") or {}),
            **(split_manifest.get("files") or {}),
            **(model_manifest.get("files") or {}),
        }
        hashes: dict[str, str] = {}
        for path in artifact_paths:
            if progress is not None:
                progress(f"verifying {path.name}")
            digest = file_sha256(path)
            expected = declared.get(path.name)
            if expected is not None and digest != expected:
                raise ValueError(f"{path.name} hash does not match its manifest")
            hashes[path.name] = digest

        assignments = pq.read_table(
            directory / "split_assignments.parquet", columns=["match_id"]
        )
        materialized_ids = set(assignments["match_id"].to_pylist())
        if progress is not None:
            progress("running corpus adequacy gate")
        enriched_split = dict(split_manifest)
        enriched_split.update(
            {
                "preparation_version": PREPARATION_VERSION,
                "model_dataset_version": MODEL_DATASET_VERSION,
                "source_feature_artifact_directory": str(feature_artifact_directory),
                "source_feature_manifest_sha256": file_sha256(
                    feature_artifact_directory / "manifest.json"
                ),
            }
        )
        corpus_assessment = corpus.assess(materialized_ids, enriched_split)
        source_artifacts = FeatureDataset(
            directory=feature_artifact_directory,
            actions=None,
            features=None,
            manifest=feature_manifest,
            quality_report=quality,
            lineage=FeatureDatasetLineage(
                analytics_run_id=feature_manifest.get("analytics_run_id"),
                dataset_fingerprint=fingerprint,
                mapping_version=ACTION_MAPPING_VERSION,
                coordinate_system_version=COORDINATE_SYSTEM_VERSION,
                state_contract_version=STATE_CONTRACT_VERSION,
                feature_version=BASE_FEATURE_VERSION,
            ),
            feature_allowlist=(),
        )
        training_manifest = build_preparation_training_manifest(
            source_artifacts=source_artifacts,
            corpus_assessment=corpus_assessment,
            split_manifest=enriched_split,
            feature_allowlist_manifest=allowlist_manifest,
            artifacts={
                path.name: {"path": str(path), "sha256": hashes[path.name]}
                for path in artifact_paths
            },
        )
        target = directory / TRAINING_MANIFEST_FILENAME
        write_json(target, training_manifest)
        require_training_corpus_ready(training_manifest)
        if progress is not None:
            progress(f"completed training manifest: {target}")
        return FinalizedPreparationPaths(directory=directory, training_manifest=target)

__all__ = ["ChunkedPreparationFinalizer", "FinalizedPreparationPaths"]
