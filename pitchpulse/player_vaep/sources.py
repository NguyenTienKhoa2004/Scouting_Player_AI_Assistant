"""Small readers for the immutable inputs used by player aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pitchpulse.shared.file_io import read_json

from .calculations import MatchContext, PlayerAggregationError


@dataclass(frozen=True, slots=True)
class MatchSource:
    context: MatchContext
    lineup_path: Path
    events_path: Path


class DatasetMatchReader:
    """Resolve match metadata, lineup JSON, and event JSON from the dataset pin."""

    def __init__(self, dataset_manifest_path: Path) -> None:
        self.manifest_path = Path(dataset_manifest_path).resolve()
        self.manifest = read_json(self.manifest_path)
        self.data_root = self._data_root()
        self.matches = self._index_matches()

    def source(self, match_id: int) -> MatchSource:
        try:
            context = self.matches[match_id]
        except KeyError as exc:
            raise PlayerAggregationError(
                f"Match {match_id} is absent from the pinned dataset"
            ) from exc
        return MatchSource(
            context=context,
            lineup_path=self.data_root / "lineups" / f"{match_id}.json",
            events_path=self.data_root / "events" / f"{match_id}.json",
        )

    @staticmethod
    def read_array(path: Path) -> list[dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PlayerAggregationError(f"Required source file is missing: {path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PlayerAggregationError(f"Could not read source JSON: {path}") from exc
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise PlayerAggregationError(f"Expected a JSON object array in {path}")
        return value

    def _data_root(self) -> Path:
        bronze = self.manifest.get("bronze")
        if not isinstance(bronze, Mapping):
            raise PlayerAggregationError("Dataset manifest lacks the bronze contract")
        value = bronze.get("data_root")
        if not isinstance(value, str) or not value:
            raise PlayerAggregationError("Dataset manifest lacks bronze.data_root")
        path = Path(value)
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        path = path.resolve()
        if not path.is_dir():
            raise PlayerAggregationError(f"Dataset data root does not exist: {path}")
        return path

    def _index_matches(self) -> dict[int, MatchContext]:
        selections = self.manifest.get("selections")
        if not isinstance(selections, list) or not selections:
            raise PlayerAggregationError("Dataset manifest has no selections")
        result: dict[int, MatchContext] = {}
        for selection in selections:
            if not isinstance(selection, Mapping):
                raise PlayerAggregationError("Dataset selection must be an object")
            competition_id = _positive_id(
                selection.get("competition_id"), "competition_id"
            )
            season_id = _positive_id(selection.get("season_id"), "season_id")
            relative_path = selection.get("matches_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise PlayerAggregationError("Dataset selection lacks matches_path")
            for match in self.read_array(self.data_root / relative_path):
                match_id = _positive_id(match.get("match_id"), "match_id")
                context = MatchContext(match_id, competition_id, season_id)
                if match_id in result:
                    raise PlayerAggregationError(
                        f"Duplicate match {match_id} in dataset selections"
                    )
                result[match_id] = context
        return result


class ActionTypeReader:
    """Read SPADL action types one match at a time from aligned Parquet groups."""

    def __init__(self, actions_path: Path) -> None:
        import pyarrow.parquet as pq

        self.path = Path(actions_path).resolve()
        self.parquet = pq.ParquetFile(self.path)
        required = {"match_id", "action_id", "type_name"}
        missing = sorted(required.difference(self.parquet.schema_arrow.names))
        if missing:
            raise PlayerAggregationError(f"Plan 03 actions lack columns: {missing}")
        self.row_groups = self._index_row_groups()

    def read_match(self, match_id: int) -> Any:
        import pyarrow as pa

        indices = self.row_groups.get(match_id)
        if not indices:
            raise PlayerAggregationError(
                f"No Plan 03 action rows found for match {match_id}"
            )
        tables = [
            self.parquet.read_row_group(
                index, columns=["match_id", "action_id", "type_name"]
            )
            for index in indices
        ]
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        frame = table.to_pandas()
        return frame[frame["match_id"] == match_id].reset_index(drop=True)

    def _index_row_groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for index in range(self.parquet.num_row_groups):
            match_ids = set(
                self.parquet.read_row_group(index, columns=["match_id"])[
                    "match_id"
                ].to_pylist()
            )
            for match_id in match_ids:
                result.setdefault(int(match_id), []).append(index)
        return result


def _positive_id(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlayerAggregationError(f"{name} must be a positive integer")
    return value


__all__ = ["ActionTypeReader", "DatasetMatchReader", "MatchSource"]
