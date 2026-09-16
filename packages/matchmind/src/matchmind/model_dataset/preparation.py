"""Materialize the versioned Plan 04 target and split artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.vaep_features.artifact_loader import FeatureArtifacts
from matchmind.corpus.training_dataset_validator import (
    TrainingCorpus,
    load_training_corpus_manifest,
    single_file_training_corpus,
)
from .builder import (
    FEATURE_ALLOWLIST_FILENAME,
    MODEL_DATASET_FILENAME,
    MODEL_DATASET_VERSION,
    ModelDatasetBuilder,
)
from matchmind.labeling_and_splitting.splits import (
    ChronologicalMatchSplitter,
    split_context,
)
from matchmind.labeling_and_splitting.targets import (
    TARGET_POLICY_VERSION,
    TargetLabelBuilder,
)
from .training_manifest import (
    TRAINING_MANIFEST_FILENAME,
    build_preparation_training_manifest,
)


PREPARATION_VERSION = "vaep-preparation-v2"


@dataclass(frozen=True, slots=True)
class PreparationArtifactPaths:
    directory: Path
    labels: Path
    target_policy: Path
    target_audit: Path
    split_assignments: Path
    split_manifest: Path
    model_dataset: Path
    feature_allowlist: Path
    training_manifest: Path


class VaepDataPreparationWriter:
    """Create deterministic labels and match splits from verified feature artifacts."""

    def write(
        self,
        source_artifacts: FeatureArtifacts,
        *,
        match_metadata_path: Path | None = None,
        corpus_manifest_path: Path | None = None,
        output_root: Path,
    ) -> PreparationArtifactPaths:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Plan 04 artifact writing requires pyarrow; install requirements.txt"
            ) from exc

        corpus = self._load_corpus(
            match_metadata_path=match_metadata_path,
            corpus_manifest_path=corpus_manifest_path,
        )
        matches = corpus.matches
        metadata_sha256 = corpus.metadata_fingerprint
        splitter = ChronologicalMatchSplitter()

        # Assignments do not depend on labels. The first pass supplies split context
        # to the target audit; the second adds label balance to the split manifest.
        initial_splits = splitter.build(
            source_artifacts.actions,
            matches,
            dataset_fingerprint=source_artifacts.lineage.dataset_fingerprint,
            target_policy_version=TARGET_POLICY_VERSION,
            match_metadata_sha256=metadata_sha256,
        )
        target_dataset = TargetLabelBuilder().build(
            source_artifacts.actions,
            analytics_run_id=source_artifacts.lineage.analytics_run_id,
            match_context=split_context(initial_splits.assignments),
        )
        split_dataset = splitter.build(
            source_artifacts.actions,
            matches,
            dataset_fingerprint=source_artifacts.lineage.dataset_fingerprint,
            target_policy_version=TARGET_POLICY_VERSION,
            labels=target_dataset.labels,
            match_metadata_sha256=metadata_sha256,
        )
        if not split_dataset.assignments.equals(initial_splits.assignments):
            raise RuntimeError("split assignments changed while adding label audit")
        model_dataset = ModelDatasetBuilder().build(
            source_artifacts,
            target_dataset.labels,
            split_dataset.assignments,
        )

        run_name = (
            f"prepare-{source_artifacts.lineage.dataset_fingerprint[:16]}-"
            f"{PREPARATION_VERSION}"
        )
        directory = Path(output_root).resolve() / run_name
        directory.mkdir(parents=True, exist_ok=True)
        paths = PreparationArtifactPaths(
            directory=directory,
            labels=directory / "action_labels.parquet",
            target_policy=directory / "target_policy.json",
            target_audit=directory / "target_audit.json",
            split_assignments=directory / "split_assignments.parquet",
            split_manifest=directory / "split_manifest.json",
            model_dataset=directory / MODEL_DATASET_FILENAME,
            feature_allowlist=directory / FEATURE_ALLOWLIST_FILENAME,
            training_manifest=directory / TRAINING_MANIFEST_FILENAME,
        )

        audit = dict(target_dataset.audit)
        audit["analytics_run_id"] = source_artifacts.lineage.analytics_run_id
        audit["source_dataset_fingerprint"] = source_artifacts.lineage.dataset_fingerprint
        audit["source_action_count"] = source_artifacts.actions.num_rows

        self._write_parquet(pq, target_dataset.labels, paths.labels)
        self._write_parquet(pq, split_dataset.assignments, paths.split_assignments)
        self._write_parquet(pq, model_dataset.table, paths.model_dataset)
        self._write_json(paths.target_policy, target_dataset.policy)
        self._write_json(paths.target_audit, audit)
        self._write_json(paths.feature_allowlist, model_dataset.allowlist_manifest)

        manifest = dict(split_dataset.manifest)
        manifest["preparation_version"] = PREPARATION_VERSION
        manifest["model_dataset_version"] = MODEL_DATASET_VERSION
        manifest["source_feature_artifact_directory"] = str(source_artifacts.directory)
        manifest["source_feature_manifest_sha256"] = self._sha256(
            source_artifacts.directory / "manifest.json"
        )
        # Legacy aliases retained for consumers of existing manifest schema v1.
        manifest["source_plan03_directory"] = manifest[
            "source_feature_artifact_directory"
        ]
        manifest["source_plan03_manifest_sha256"] = manifest[
            "source_feature_manifest_sha256"
        ]
        manifest["source_training_corpus"] = {
            "corpus_id": corpus.corpus_id,
            "manifest_path": (
                str(corpus.manifest_path) if corpus.manifest_path is not None else None
            ),
            "manifest_sha256": corpus.manifest_sha256,
            "metadata_fingerprint": corpus.metadata_fingerprint,
        }
        manifest["files"] = {
            path.name: self._sha256(path)
            for path in (
                paths.labels,
                paths.target_policy,
                paths.target_audit,
                paths.split_assignments,
                paths.model_dataset,
                paths.feature_allowlist,
            )
        }
        self._write_json(paths.split_manifest, manifest)

        corpus_assessment = corpus.assess(
            set(source_artifacts.actions["match_id"].to_pylist()),
            manifest,
        )
        preparation_files = (
            paths.labels,
            paths.target_policy,
            paths.target_audit,
            paths.split_assignments,
            paths.split_manifest,
            paths.model_dataset,
            paths.feature_allowlist,
        )
        training_manifest = build_preparation_training_manifest(
            source_artifacts=source_artifacts,
            corpus_assessment=corpus_assessment,
            split_manifest=manifest,
            feature_allowlist_manifest=model_dataset.allowlist_manifest,
            artifacts={
                path.name: {
                    "path": str(path),
                    "sha256": self._sha256(path),
                }
                for path in preparation_files
            },
        )
        self._write_json(paths.training_manifest, training_manifest)
        return paths

    @staticmethod
    def _load_corpus(
        *,
        match_metadata_path: Path | None,
        corpus_manifest_path: Path | None,
    ) -> TrainingCorpus:
        if (match_metadata_path is None) == (corpus_manifest_path is None):
            raise ValueError(
                "provide exactly one of match_metadata_path or corpus_manifest_path"
            )
        if corpus_manifest_path is not None:
            return load_training_corpus_manifest(corpus_manifest_path)
        assert match_metadata_path is not None
        return single_file_training_corpus(match_metadata_path)

    @staticmethod
    def _write_parquet(pq: Any, table: Any, target: Path) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(target)

    @staticmethod
    def _write_json(target: Path, value: dict[str, Any]) -> None:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "PREPARATION_VERSION",
    "PreparationArtifactPaths",
    "VaepDataPreparationWriter",
]
