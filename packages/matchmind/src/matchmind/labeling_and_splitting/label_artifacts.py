"""Memory-bounded VAEP label artifacts derived from feature artifact actions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matchmind.vaep_features.action_state import STATE_CONTRACT_VERSION
from matchmind.spadl.converter import (
    ACTION_MAPPING_VERSION,
    COORDINATE_SYSTEM_VERSION,
)
from matchmind.vaep_features.feature_builder import BASE_FEATURE_VERSION

from matchmind.corpus.training_dataset_validator import (
    load_training_corpus_manifest,
    single_file_training_corpus,
)
from .targets import TARGET_POLICY_VERSION, TargetLabelBuilder, baseline_target_policy


LABEL_MANIFEST_FILENAME = "label_manifest.json"


@dataclass(frozen=True, slots=True)
class LabelArtifactPaths:
    directory: Path
    labels: Path
    target_policy: Path
    target_audit: Path
    manifest: Path


class ChunkedTargetLabelWriter:
    """Generate labels one feature artifact Parquet row group at a time."""

    def write(
        self,
        feature_artifact_directory: Path,
        *,
        corpus_manifest_path: Path | None = None,
        match_metadata_path: Path | None = None,
        output_root: Path,
        progress: Callable[[str], None] | None = None,
    ) -> LabelArtifactPaths:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "VAEP label generation requires pyarrow; install requirements.txt"
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
        directory = Path(feature_artifact_directory).resolve()
        manifest = self._read_json(directory / "manifest.json")
        quality = self._read_json(directory / "conversion_quality_report.json")
        fingerprint = self._validate_lineage(manifest, quality)
        actions_path = directory / "actions.parquet"
        expected_actions_hash = (manifest.get("files") or {}).get(
            actions_path.name
        )
        if progress is not None:
            progress("verifying feature artifact actions hash")
        if not isinstance(expected_actions_hash, str) or self._sha256(
            actions_path
        ) != expected_actions_hash:
            raise ValueError("feature artifact actions.parquet hash does not match manifest")
        if progress is not None:
            progress("feature artifact actions hash verified")

        output_directory = (
            Path(output_root).resolve() / f"labels-{fingerprint[:16]}"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        paths = LabelArtifactPaths(
            directory=output_directory,
            labels=output_directory / "action_labels.parquet",
            target_policy=output_directory / "target_policy.json",
            target_audit=output_directory / "target_audit.json",
            manifest=output_directory / LABEL_MANIFEST_FILENAME,
        )
        temporary_labels = paths.labels.with_suffix(".parquet.tmp")
        temporary_labels.unlink(missing_ok=True)

        parquet_file = pq.ParquetFile(actions_path)
        digest = hashlib.sha256()
        digest.update(ACTION_MAPPING_VERSION.encode())
        digest.update(BASE_FEATURE_VERSION.encode())
        digest.update(str(quality["event_count"]).encode())
        expected_match_ids = set(corpus.match_ids)
        seen_match_ids: set[int] = set()
        audits: list[dict[str, Any]] = []
        label_writer = None
        label_count = 0
        match_context = {
            match.match_id: {
                "competition_id": match.competition_id,
                "season_id": match.season_id,
            }
            for match in corpus.matches
        }

        try:
            for row_group in range(parquet_file.num_row_groups):
                actions = parquet_file.read_row_group(row_group)
                batch_match_ids = set(actions["match_id"].to_pylist())
                overlap = seen_match_ids & batch_match_ids
                if overlap:
                    raise ValueError(
                        "a match spans multiple feature artifact row groups: "
                        f"{sorted(overlap)}"
                    )
                seen_match_ids.update(batch_match_ids)
                self._update_fingerprint(digest, actions)
                target = TargetLabelBuilder().build(
                    actions,
                    analytics_run_id=manifest.get("analytics_run_id"),
                    match_context=match_context,
                )
                if label_writer is None:
                    label_writer = pq.ParquetWriter(
                        temporary_labels,
                        target.labels.schema,
                        compression="zstd",
                    )
                label_writer.write_table(target.labels)
                label_count += target.labels.num_rows
                audits.append(target.audit)
                if progress is not None:
                    progress(
                        f"labeled batch {row_group + 1}/{parquet_file.num_row_groups}: "
                        f"{len(batch_match_ids)} matches, "
                        f"{target.labels.num_rows} actions"
                    )
            if label_writer is None:
                raise ValueError("feature artifact actions contain no row groups")
            label_writer.close()
            label_writer = None

            if digest.hexdigest() != fingerprint:
                raise ValueError("feature artifact dataset fingerprint does not reconcile")
            if seen_match_ids != expected_match_ids:
                raise ValueError(
                    "feature artifact match IDs do not exactly match the selected corpus"
                )
            if label_count != int(quality["action_count"]):
                raise ValueError(
                    "label count does not match the feature artifact action count"
                )

            temporary_labels.replace(paths.labels)
            audit = self._merge_audits(audits)
            audit["analytics_run_id"] = manifest.get("analytics_run_id")
            audit["source_dataset_fingerprint"] = fingerprint
            audit["source_action_count"] = label_count
            self._write_json(paths.target_policy, baseline_target_policy())
            self._write_json(paths.target_audit, audit)
            label_manifest = {
                "schema_version": 1,
                "source_feature_artifact_directory": str(directory),
                "source_plan03_directory": str(directory),
                "source_dataset_fingerprint": fingerprint,
                "target_policy_version": TARGET_POLICY_VERSION,
                "match_count": len(seen_match_ids),
                "action_count": label_count,
                "files": {
                    paths.labels.name: self._sha256(paths.labels),
                    paths.target_policy.name: self._sha256(paths.target_policy),
                    paths.target_audit.name: self._sha256(paths.target_audit),
                },
            }
            self._write_json(paths.manifest, label_manifest)
            if progress is not None:
                progress(f"completed label artifact: {output_directory}")
            return paths
        except BaseException:
            if label_writer is not None:
                label_writer.close()
            temporary_labels.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_lineage(
        manifest: dict[str, Any], quality: dict[str, Any]
    ) -> str:
        expected = {
            "mapping_version": ACTION_MAPPING_VERSION,
            "coordinate_system_version": COORDINATE_SYSTEM_VERSION,
            "state_contract_version": STATE_CONTRACT_VERSION,
            "feature_version": BASE_FEATURE_VERSION,
            "include_360": False,
        }
        for field, value in expected.items():
            if quality.get(field) != value:
                raise ValueError(
                    f"feature artifact quality {field} must be {value!r}"
                )
            if field != "state_contract_version" and manifest.get(field) != value:
                raise ValueError(
                    f"feature artifact manifest {field} must be {value!r}"
                )
        fingerprint = manifest.get("dataset_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("feature artifact manifest has an invalid dataset fingerprint")
        return fingerprint

    @staticmethod
    def _update_fingerprint(digest: Any, actions: Any) -> None:
        columns = (
            "match_id",
            "source",
            "source_event_id",
            "source_event_index",
            "source_action_index",
        )
        values = {column: actions[column].to_pylist() for column in columns}
        for row in zip(*(values[column] for column in columns), strict=True):
            digest.update(("|".join(str(value) for value in row) + "\n").encode())

    @classmethod
    def _merge_audits(cls, audits: list[dict[str, Any]]) -> dict[str, Any]:
        if not audits:
            raise ValueError("cannot merge an empty target audit")

        def merge_summaries(values: list[dict[str, Any]]) -> dict[str, Any]:
            fields = (
                "total_count",
                "eligible_count",
                "excluded_count",
                "scores_positive_count",
                "concedes_positive_count",
            )
            merged = {
                field: sum(int(value.get(field, 0)) for value in values)
                for field in fields
            }
            eligible = merged["eligible_count"]
            merged["scores_positive_rate"] = (
                round(merged["scores_positive_count"] / eligible, 12)
                if eligible
                else None
            )
            merged["concedes_positive_rate"] = (
                round(merged["concedes_positive_count"] / eligible, 12)
                if eligible
                else None
            )
            return merged

        breakdowns: dict[str, dict[str, Any]] = {}
        for field in ("match", "competition", "season", "split"):
            grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
            for audit in audits:
                for key, value in audit["breakdowns"][field].items():
                    grouped[str(key)].append(value)
            breakdowns[field] = {
                key: merge_summaries(values)
                for key, values in sorted(grouped.items())
            }

        checks: dict[str, Any] = {
            "one_label_row_per_source_action": all(
                bool(audit["checks"]["one_label_row_per_source_action"])
                for audit in audits
            ),
            "labels_generated_per_match": all(
                bool(audit["checks"]["labels_generated_per_match"])
                for audit in audits
            ),
            "window_action_count": audits[0]["checks"]["window_action_count"],
        }
        for field in (
            "first_actions",
            "last_eligible_actions",
            "possession_changes",
            "period_boundaries",
            "match_endings",
        ):
            checks[field] = merge_summaries(
                [audit["checks"][field] for audit in audits]
            )
        for field, mismatch_field in (
            ("goals", "current_action_scores_mismatches"),
            ("own_goals", "current_action_concedes_mismatches"),
        ):
            checks[field] = merge_summaries(
                [audit["checks"][field] for audit in audits]
            )
            checks[field][mismatch_field] = sum(
                int(audit["checks"][field][mismatch_field]) for audit in audits
            )
        checks["shootouts"] = {
            field: sum(int(audit["checks"]["shootouts"][field]) for audit in audits)
            for field in ("source_action_count", "excluded_count")
        }
        checks["shootouts"].update(
            {
                "excluded_before_label_generation": all(
                    bool(
                        audit["checks"]["shootouts"][
                            "excluded_before_label_generation"
                        ]
                    )
                    for audit in audits
                ),
                "all_labels_null": all(
                    bool(audit["checks"]["shootouts"]["all_labels_null"])
                    for audit in audits
                ),
            }
        )
        checks["malformed_sequences_excluded_count"] = sum(
            int(audit["checks"]["malformed_sequences_excluded_count"])
            for audit in audits
        )
        exclusions = Counter()
        for audit in audits:
            exclusions.update(audit["exclusions_by_reason"])
        return {
            "schema_version": 1,
            "target_policy_version": TARGET_POLICY_VERSION,
            "counts": merge_summaries([audit["counts"] for audit in audits]),
            "exclusions_by_reason": dict(sorted(exclusions.items())),
            "breakdowns": breakdowns,
            "checks": checks,
        }

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
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
    "LABEL_MANIFEST_FILENAME",
    "ChunkedTargetLabelWriter",
    "LabelArtifactPaths",
]
