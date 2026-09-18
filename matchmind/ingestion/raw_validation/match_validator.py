"""Raw StatsBomb match-metadata validation."""

from __future__ import annotations

from ..reader import RawRecord, StatsBombRawReader
from .context import RawValidationContext


def validate_match(
    record: RawRecord,
    reader: StatsBombRawReader,
    context: RawValidationContext,
) -> frozenset[int]:
    payload = record.payload
    competition = context.object(record, payload.get("competition"), "competition")
    season = context.object(record, payload.get("season"), "season")
    home = context.object(record, payload.get("home_team"), "home_team")
    away = context.object(record, payload.get("away_team"), "away_team")

    competition_id = context.positive_int(
        record,
        competition.get("competition_id") if competition else None,
        "competition.competition_id",
    )
    season_id = context.positive_int(
        record,
        season.get("season_id") if season else None,
        "season.season_id",
    )
    context.nonempty_text(
        record,
        competition.get("competition_name") if competition else None,
        "competition.competition_name",
    )
    context.nonempty_text(
        record,
        season.get("season_name") if season else None,
        "season.season_name",
    )
    if competition_id is not None and competition_id != reader.competition_id:
        context.add(
            record,
            "competition_mismatch",
            "competition.competition_id",
            f"expected {reader.competition_id}, got {competition_id}",
        )
    if season_id is not None and season_id != reader.season_id:
        context.add(
            record,
            "season_mismatch",
            "season.season_id",
            f"expected {reader.season_id}, got {season_id}",
        )

    home_id = context.positive_int(
        record,
        home.get("home_team_id") if home else None,
        "home_team.home_team_id",
    )
    away_id = context.positive_int(
        record,
        away.get("away_team_id") if away else None,
        "away_team.away_team_id",
    )
    context.nonempty_text(
        record,
        home.get("home_team_name") if home else None,
        "home_team.home_team_name",
    )
    context.nonempty_text(
        record,
        away.get("away_team_name") if away else None,
        "away_team.away_team_name",
    )
    context.nonnegative_int(record, payload.get("home_score"), "home_score")
    context.nonnegative_int(record, payload.get("away_score"), "away_score")
    context.iso_date(record, payload.get("match_date"), "match_date")

    if home_id is not None and away_id is not None and home_id == away_id:
        context.add(
            record,
            "match_teams_not_distinct",
            "away_team.away_team_id",
            "home and away team IDs must differ",
        )
    return frozenset(value for value in (home_id, away_id) if value is not None)


__all__ = ["validate_match"]
