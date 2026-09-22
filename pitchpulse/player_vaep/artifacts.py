"""Artifact workflow for player minutes and VAEP summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pitchpulse.shared.file_io import file_sha256

from pitchpulse.model_training.training_files import (
    read_json,
    verify_training_file,
    write_json,
)

from .calculations import (
    MINUTES_POLICY_VERSION,
    PLAYER_AGGREGATION_VERSION,
    PlayerAggregationError,
    PlayerVaepAggregator,
    calculate_player_minutes,
    match_duration_seconds,
)
from .sources import ActionTypeReader, DatasetMatchReader


PLAYER_VAEP_FILENAME = "player_vaep.parquet"
PLAYER_VAEP_MANIFEST_FILENAME = "player_vaep_manifest.json"


@dataclass(frozen=True, slots=True)
class PlayerAggregationPaths:
    directory: Path
    player_vaep: Path
    manifest: Path
    training_manifest: Path


class PlayerAggregationWriter:
    """Create reproducible player aggregates without loading all actions at once."""

    def write(
        self,
        artifact_directory: Path,
        dataset_manifest_path: Path,
        *,
        minimum_minutes: float = 450.0,
        progress: Any | None = None,
    ) -> PlayerAggregationPaths:
        import pyarrow.parquet as pq

        root = Path(artifact_directory).resolve()
        training_manifest_path = root / "training_manifest.json"
        training_manifest = read_json(training_manifest_path)
        paths = PlayerAggregationPaths(
            directory=root,
            player_vaep=root / PLAYER_VAEP_FILENAME,
            manifest=root / PLAYER_VAEP_MANIFEST_FILENAME,
            training_manifest=training_manifest_path,
        )
        if training_manifest.get("status") == "player_aggregation_complete":
            self._verify_completed(root, training_manifest, paths)
            if progress is not None:
                progress("player aggregation already complete; verified saved artifact")
            return paths
        self._require_action_values(training_manifest)
        if paths.player_vaep.exists() or paths.manifest.exists():
            raise PlayerAggregationError(
                "Refusing to overwrite partial player-aggregation outputs"
            )

        action_values_path = verify_training_file(
            root, training_manifest, "action_values.parquet"
        )
        action_values_manifest_path = verify_training_file(
            root, training_manifest, "action_values_manifest.json"
        )
        action_values_manifest = read_json(action_values_manifest_path)
        actions_path = self._verified_actions_path(action_values_manifest)
        dataset = DatasetMatchReader(dataset_manifest_path)
        action_types = ActionTypeReader(actions_path)
        aggregator = PlayerVaepAggregator(minimum_minutes=minimum_minutes)

        parquet = pq.ParquetFile(action_values_path)
        seen_matches: set[int] = set()
        for row_group_index in range(parquet.num_row_groups):
            values = parquet.read_row_group(
                row_group_index,
                columns=[
                    "match_id",
                    "action_id",
                    "player_id",
                    "team_id",
                    "offensive_value",
                    "defensive_value",
                    "vaep_value",
                ],
            ).to_pandas()
            match_ids = {int(value) for value in values["match_id"].unique()}
            if len(match_ids) != 1:
                raise PlayerAggregationError(
                    "Each action-value row group must contain exactly one match"
                )
            match_id = next(iter(match_ids))
            if match_id in seen_matches:
                raise PlayerAggregationError(
                    f"Match {match_id} spans multiple action-value row groups"
                )
            seen_matches.add(match_id)

            typed_values = self._attach_action_types(
                values, action_types.read_match(match_id)
            )
            source = dataset.source(match_id)
            events = dataset.read_array(source.events_path)
            lineups = dataset.read_array(source.lineup_path)
            duration = match_duration_seconds(events)
            minutes = calculate_player_minutes(lineups, duration)
            aggregator.add_match(typed_values, source.context, minutes)
            if progress is not None and (
                len(seen_matches) == 1 or len(seen_matches) % 100 == 0
            ):
                progress(f"aggregated {len(seen_matches)} matches")

        self._reconcile_input(training_manifest, aggregator, seen_matches)
        rows = aggregator.rows()
        temporary = root / ".player_vaep.tmp.parquet"
        temporary.unlink(missing_ok=True)
        try:
            table = self._table(rows)
            pq.write_table(table, temporary, compression="zstd", row_group_size=10_000)
            temporary.replace(paths.player_vaep)
            output_manifest = self._build_manifest(
                training_manifest=training_manifest,
                action_values_manifest=action_values_manifest,
                action_values_path=action_values_path,
                actions_path=actions_path,
                dataset_manifest_path=Path(dataset_manifest_path).resolve(),
                paths=paths,
                rows=rows,
                match_count=len(seen_matches),
                aggregator=aggregator,
            )
            write_json(paths.manifest, output_manifest)
            self._update_training_manifest(
                training_manifest, paths, output_manifest
            )
        finally:
            temporary.unlink(missing_ok=True)

        if progress is not None:
            progress(f"completed player aggregation: {paths.player_vaep}")
        return paths

    @staticmethod
    def _require_action_values(manifest: Mapping[str, Any]) -> None:
        if manifest.get("status") != "action_valuation_complete":
            raise PlayerAggregationError(
                "Player aggregation requires completed action valuation"
            )

    @staticmethod
    def _verified_actions_path(action_manifest: Mapping[str, Any]) -> Path:
        record = (action_manifest.get("source_artifacts") or {}).get(
            "actions.parquet"
        )
        if not isinstance(record, Mapping):
            raise PlayerAggregationError(
                "Action-value manifest does not declare actions.parquet"
            )
        path = Path(str(record.get("path", ""))).resolve()
        expected_hash = record.get("sha256")
        if not path.is_file():
            raise PlayerAggregationError(f"Plan 03 actions are missing: {path}")
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise PlayerAggregationError("Plan 03 action hash does not match lineage")
        return path

    @staticmethod
    def _attach_action_types(values: Any, actions: Any) -> Any:
        if actions.duplicated(["match_id", "action_id"]).any():
            raise PlayerAggregationError("Plan 03 action keys are not unique")
        merged = values.merge(
            actions,
            on=["match_id", "action_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["type_name"].isna().any():
            raise PlayerAggregationError(
                "An action value has no matching Plan 03 action type"
            )
        difference = (
            merged["offensive_value"]
            + merged["defensive_value"]
            - merged["vaep_value"]
        ).abs()
        if (difference > 1e-10).any():
            raise PlayerAggregationError(
                "Action VAEP does not reconcile with offensive and defensive values"
            )
        return merged

    @staticmethod
    def _reconcile_input(
        manifest: Mapping[str, Any],
        aggregator: PlayerVaepAggregator,
        seen_matches: set[int],
    ) -> None:
        expected = manifest.get("action_valuation") or {}
        if aggregator.input_action_count != int(expected.get("row_count", -1)):
            raise PlayerAggregationError(
                "Player aggregation row count does not match action valuation"
            )
        if len(seen_matches) != int(expected.get("match_count", -1)):
            raise PlayerAggregationError(
                "Player aggregation match count does not match action valuation"
            )
        if (
            aggregator.known_player_action_count
            + aggregator.unknown_player_action_count
            != aggregator.input_action_count
        ):
            raise PlayerAggregationError("Known and unknown player counts do not reconcile")

    @classmethod
    def _table(cls, rows: list[dict[str, Any]]) -> Any:
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("aggregation_version", pa.string(), nullable=False),
                pa.field("minutes_policy_version", pa.string(), nullable=False),
                pa.field("aggregation_level", pa.string(), nullable=False),
                pa.field("match_id", pa.int64(), nullable=True),
                pa.field("player_id", pa.int64(), nullable=False),
                pa.field("team_id", pa.int64(), nullable=False),
                pa.field("competition_id", pa.int64(), nullable=False),
                pa.field("season_id", pa.int64(), nullable=False),
                pa.field("position", pa.string(), nullable=False),
                pa.field("action_type", pa.string(), nullable=False),
                pa.field("minutes_played", pa.float64(), nullable=False),
                pa.field("match_count", pa.int64(), nullable=False),
                pa.field("action_count", pa.int64(), nullable=False),
                pa.field("player_vaep", pa.float64(), nullable=False),
                pa.field("offensive_vaep", pa.float64(), nullable=False),
                pa.field("defensive_vaep", pa.float64(), nullable=False),
                pa.field("vaep_per_90", pa.float64(), nullable=True),
                pa.field("minimum_minutes_eligible", pa.bool_(), nullable=False),
            ]
        )
        return pa.Table.from_pylist(rows, schema=schema)

    @staticmethod
    def _build_manifest(
        *,
        training_manifest: Mapping[str, Any],
        action_values_manifest: Mapping[str, Any],
        action_values_path: Path,
        actions_path: Path,
        dataset_manifest_path: Path,
        paths: PlayerAggregationPaths,
        rows: list[dict[str, Any]],
        match_count: int,
        aggregator: PlayerVaepAggregator,
    ) -> dict[str, Any]:
        total_rows = [
            row
            for row in rows
            if row["aggregation_level"] == "competition_season"
            and row["action_type"] == "all"
        ]
        if sum(row["action_count"] for row in total_rows) != (
            aggregator.known_player_action_count
        ):
            raise PlayerAggregationError("Player action totals do not reconcile")
        summed_vaep = sum(row["player_vaep"] for row in total_rows)
        summed_offensive = sum(row["offensive_vaep"] for row in total_rows)
        summed_defensive = sum(row["defensive_vaep"] for row in total_rows)
        comparisons = (
            (summed_vaep, aggregator.known_player_vaep, "VAEP"),
            (summed_offensive, aggregator.known_offensive_vaep, "offensive VAEP"),
            (summed_defensive, aggregator.known_defensive_vaep, "defensive VAEP"),
        )
        for actual, expected, label in comparisons:
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
                raise PlayerAggregationError(f"Player {label} totals do not reconcile")
        return {
            "schema_version": 1,
            "aggregation_version": PLAYER_AGGREGATION_VERSION,
            "minutes_policy": {
                "version": MINUTES_POLICY_VERSION,
                "source": "StatsBomb lineup position intervals",
                "open_interval_end": "last non-shootout event, including added time",
                "malformed_interval": "ignore rows whose end precedes their start",
                "position": "most minutes in a match; alphabetical tie-break",
                "minimum_minutes": aggregator.minimum_minutes,
            },
            "dataset_fingerprint": action_values_manifest.get("dataset_fingerprint"),
            "match_count": match_count,
            "player_count": len({row["player_id"] for row in total_rows}),
            "row_count": len(rows),
            "audit": {
                "input_action_count": aggregator.input_action_count,
                "known_player_action_count": aggregator.known_player_action_count,
                "unknown_player_action_count": aggregator.unknown_player_action_count,
                "known_player_actions_without_minutes": (
                    aggregator.known_player_actions_without_minutes
                ),
                "known_offensive_vaep": aggregator.known_offensive_vaep,
                "known_defensive_vaep": aggregator.known_defensive_vaep,
                "known_player_vaep": aggregator.known_player_vaep,
                "reconciled": True,
            },
            "source_artifacts": {
                "action_values.parquet": {
                    "path": str(action_values_path),
                    "sha256": file_sha256(action_values_path),
                },
                "actions.parquet": {
                    "path": str(actions_path),
                    "sha256": file_sha256(actions_path),
                },
                "dataset_manifest": {
                    "path": str(dataset_manifest_path),
                    "sha256": file_sha256(dataset_manifest_path),
                },
            },
            "files": {paths.player_vaep.name: file_sha256(paths.player_vaep)},
            "training_status_at_start": training_manifest.get("status"),
        }

    @staticmethod
    def _update_training_manifest(
        manifest: Mapping[str, Any],
        paths: PlayerAggregationPaths,
        output_manifest: Mapping[str, Any],
    ) -> None:
        updated = dict(manifest)
        updated["stage"] = "player_aggregation"
        updated["status"] = "player_aggregation_complete"
        artifacts = dict(manifest.get("artifacts") or {})
        for path in (paths.player_vaep, paths.manifest):
            artifacts[path.name] = {
                "path": str(path),
                "sha256": file_sha256(path),
            }
        updated["artifacts"] = artifacts
        updated["player_aggregation"] = {
            "version": PLAYER_AGGREGATION_VERSION,
            "minutes_policy_version": MINUTES_POLICY_VERSION,
            "match_count": output_manifest["match_count"],
            "player_count": output_manifest["player_count"],
            "row_count": output_manifest["row_count"],
            "minimum_minutes": output_manifest["minutes_policy"]["minimum_minutes"],
            "audit": dict(output_manifest["audit"]),
        }
        write_json(paths.training_manifest, updated)

    @staticmethod
    def _verify_completed(
        root: Path,
        manifest: Mapping[str, Any],
        paths: PlayerAggregationPaths,
    ) -> None:
        for filename in (PLAYER_VAEP_FILENAME, PLAYER_VAEP_MANIFEST_FILENAME):
            verify_training_file(root, manifest, filename)
        output_manifest = read_json(paths.manifest)
        expected = (output_manifest.get("files") or {}).get(PLAYER_VAEP_FILENAME)
        if not isinstance(expected, str) or file_sha256(paths.player_vaep) != expected:
            raise PlayerAggregationError("Player-VAEP hash does not match its manifest")


__all__ = [
    "PLAYER_VAEP_FILENAME",
    "PLAYER_VAEP_MANIFEST_FILENAME",
    "PlayerAggregationPaths",
    "PlayerAggregationWriter",
]
