"""Read StatsBomb Open Data JSON without normalizing or mutating it."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pitchpulse.shared.paths import STATSBOMB_BRONZE_DATA_ROOT

DEFAULT_DATA_ROOT = STATSBOMB_BRONZE_DATA_ROOT

JsonObject = dict[str, Any]
RecordKind = Literal["match", "lineup", "event", "three_sixty"]


class RawDataError(Exception):
    """Base error raised while reading source data."""


class RawDataFileNotFoundError(RawDataError):
    """A required StatsBomb source file does not exist."""


class RawDataFormatError(RawDataError):
    """A StatsBomb source file is not valid JSON in the expected shape."""


@dataclass(frozen=True, slots=True)
class RawRecord:
    """One untouched JSON object plus ingestion metadata.

    ``source_record_index`` is the zero-based position in the source JSON array.
    ``match_id`` comes from match metadata or the event/lineup filename; it is not
    injected into ``payload``.
    """

    kind: RecordKind
    match_id: int
    source_file: Path
    source_record_index: int
    payload: JsonObject
    source: str = "statsbomb"


@dataclass(frozen=True, slots=True)
class RawMatchBundle:
    """Raw match metadata, lineup objects, and events for one match."""

    match: RawRecord
    lineups: tuple[RawRecord, ...]
    events: tuple[RawRecord, ...]
    three_sixty: tuple[RawRecord, ...] = ()


class StatsBombRawReader:
    """Read one StatsBomb competition-season selection, one file at a time."""

    def __init__(
        self,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        competition_id: int = 43,
        season_id: int = 106,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.competition_id = self._valid_id(competition_id, "competition_id")
        self.season_id = self._valid_id(season_id, "season_id")

    @property
    def matches_path(self) -> Path:
        return (
            self.data_root
            / "matches"
            / str(self.competition_id)
            / f"{self.season_id}.json"
        )

    def events_path(self, match_id: int) -> Path:
        return self.data_root / "events" / f"{self._valid_id(match_id, 'match_id')}.json"

    def lineups_path(self, match_id: int) -> Path:
        return self.data_root / "lineups" / f"{self._valid_id(match_id, 'match_id')}.json"

    def three_sixty_path(self, match_id: int) -> Path:
        """Return the optional StatsBomb 360 path for a match."""
        return (
            self.data_root
            / "three-sixty"
            / f"{self._valid_id(match_id, 'match_id')}.json"
        )

    def iter_matches(self) -> Iterator[RawRecord]:
        """Yield raw match records in the order stored by StatsBomb."""
        for index, payload in enumerate(self._read_json_array(self.matches_path)):
            match_id = self._payload_match_id(payload, self.matches_path, index)
            yield RawRecord(
                kind="match",
                match_id=match_id,
                source_file=self.matches_path,
                source_record_index=index,
                payload=payload,
            )

    def read_match(self, match_id: int) -> RawRecord:
        """Return one match metadata record by ID."""
        wanted = self._valid_id(match_id, "match_id")
        for record in self.iter_matches():
            if record.match_id == wanted:
                return record
        raise RawDataFormatError(
            f"Match {wanted} is not listed in {self.matches_path}"
        )

    def read_events(self, match_id: int) -> tuple[RawRecord, ...]:
        """Read every raw event for one match."""
        wanted = self._valid_id(match_id, "match_id")
        path = self.events_path(wanted)
        return tuple(
            RawRecord(
                kind="event",
                match_id=wanted,
                source_file=path,
                source_record_index=index,
                payload=payload,
            )
            for index, payload in enumerate(self._read_json_array(path))
        )

    def read_lineups(self, match_id: int) -> tuple[RawRecord, ...]:
        """Read every raw team-lineup object for one match."""
        wanted = self._valid_id(match_id, "match_id")
        path = self.lineups_path(wanted)
        return tuple(
            RawRecord(
                kind="lineup",
                match_id=wanted,
                source_file=path,
                source_record_index=index,
                payload=payload,
            )
            for index, payload in enumerate(self._read_json_array(path))
        )

    def read_three_sixty(
        self, match_id: int, *, required: bool = False
    ) -> tuple[RawRecord, ...]:
        """Read optional StatsBomb 360 frames for one match.

        Competitions without 360 data remain valid. Callers may request strict
        behavior with ``required=True`` for manifests that guarantee coverage.
        """
        wanted = self._valid_id(match_id, "match_id")
        path = self.three_sixty_path(wanted)
        if not path.is_file() and not required:
            return ()
        return tuple(
            RawRecord(
                kind="three_sixty",
                match_id=wanted,
                source_file=path,
                source_record_index=index,
                payload=payload,
            )
            for index, payload in enumerate(self._read_json_array(path))
        )

    def read_match_bundle(self, match_id: int) -> RawMatchBundle:
        """Read match metadata, lineups, and events for one match."""
        match = self.read_match(match_id)
        return RawMatchBundle(
            match=match,
            lineups=self.read_lineups(match.match_id),
            events=self.read_events(match.match_id),
            three_sixty=self.read_three_sixty(match.match_id),
        )

    def iter_events(self, match_ids: Iterable[int] | None = None) -> Iterator[RawRecord]:
        """Yield events across matches while loading only one event file at a time."""
        selected_ids = (
            (record.match_id for record in self.iter_matches())
            if match_ids is None
            else (self._valid_id(value, "match_id") for value in match_ids)
        )
        for match_id in selected_ids:
            yield from self.read_events(match_id)

    def iter_match_bundles(self) -> Iterator[RawMatchBundle]:
        """Yield complete raw bundles while retaining only one match at a time."""
        for match in self.iter_matches():
            yield RawMatchBundle(
                match=match,
                lineups=self.read_lineups(match.match_id),
                events=self.read_events(match.match_id),
                three_sixty=self.read_three_sixty(match.match_id),
            )

    @staticmethod
    def _valid_id(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _payload_match_id(payload: JsonObject, path: Path, index: int) -> int:
        value = payload.get("match_id")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RawDataFormatError(
                f"Record {index} in {path} has an invalid match_id: {value!r}"
            )
        return value

    @staticmethod
    def _read_json_array(path: Path) -> list[JsonObject]:
        try:
            with path.open("r", encoding="utf-8") as source_file:
                payload = json.load(source_file)
        except FileNotFoundError as exc:
            raise RawDataFileNotFoundError(f"Source file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise RawDataFormatError(
                f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: "
                f"{exc.msg}"
            ) from exc
        except OSError as exc:
            raise RawDataError(f"Could not read {path}: {exc}") from exc

        if not isinstance(payload, list):
            raise RawDataFormatError(f"Expected a JSON array in {path}")

        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise RawDataFormatError(
                    f"Record {index} in {path} must be a JSON object"
                )
        return payload
