"""Minimal match metadata contract shared by dataset selection and splitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path


class MatchMetadataError(ValueError):
    """Raised when StatsBomb match metadata is invalid."""


@dataclass(frozen=True, slots=True)
class MatchMetadata:
    match_id: int
    competition_id: int
    season_id: int
    match_date: date
    kick_off: time

    @property
    def chronological_key(self) -> tuple[date, time, int, int, int]:
        return (
            self.match_date,
            self.kick_off,
            self.competition_id,
            self.season_id,
            self.match_id,
        )

    @property
    def match_datetime(self) -> datetime:
        return datetime.combine(self.match_date, self.kick_off)


def load_statsbomb_match_metadata(path: Path) -> tuple[MatchMetadata, ...]:
    """Read the minimal chronological contract from a StatsBomb matches file."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchMetadataError(f"cannot read match metadata {source}: {exc}") from exc
    if not isinstance(value, list):
        raise MatchMetadataError("StatsBomb matches metadata must be a JSON array")

    matches: list[MatchMetadata] = []
    for row in value:
        if not isinstance(row, dict):
            raise MatchMetadataError("each match metadata row must be an object")
        competition = row.get("competition")
        season = row.get("season")
        if not isinstance(competition, dict) or not isinstance(season, dict):
            raise MatchMetadataError("match metadata lacks competition or season")
        try:
            kick_off_text = str(row.get("kick_off") or "00:00:00")
            matches.append(
                MatchMetadata(
                    match_id=int(row["match_id"]),
                    competition_id=int(competition["competition_id"]),
                    season_id=int(season["season_id"]),
                    match_date=date.fromisoformat(str(row["match_date"])),
                    kick_off=time.fromisoformat(kick_off_text),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchMetadataError(f"invalid match metadata row: {exc}") from exc

    ids = [match.match_id for match in matches]
    if len(ids) != len(set(ids)):
        raise MatchMetadataError("match metadata contains duplicate match_id values")
    return tuple(matches)


__all__ = ["MatchMetadata", "MatchMetadataError", "load_statsbomb_match_metadata"]
