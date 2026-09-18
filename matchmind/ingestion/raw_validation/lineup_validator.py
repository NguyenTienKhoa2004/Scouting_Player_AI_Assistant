"""Raw StatsBomb lineup validation."""

from __future__ import annotations

from typing import Any

from ..reader import RawMatchBundle, RawRecord
from .context import RawValidationContext


def validate_lineups(
    bundle: RawMatchBundle,
    team_ids: frozenset[int],
    context: RawValidationContext,
) -> dict[int, int]:
    seen_lineup_teams: set[int] = set()
    player_teams: dict[int, int] = {}
    for record in bundle.lineups:
        payload = record.payload
        team_id = context.positive_int(record, payload.get("team_id"), "team_id")
        context.nonempty_text(record, payload.get("team_name"), "team_name")
        if team_id is not None:
            if team_id not in team_ids:
                context.add(
                    record,
                    "lineup_team_not_in_match",
                    "team_id",
                    f"team {team_id} does not participate in the match",
                )
            if team_id in seen_lineup_teams:
                context.add(
                    record,
                    "duplicate_team_lineup",
                    "team_id",
                    f"team {team_id} has multiple lineup records",
                )
            seen_lineup_teams.add(team_id)

        players = payload.get("lineup")
        if not isinstance(players, list):
            context.add(record, "lineup_not_array", "lineup", "lineup must be an array")
            continue
        for player_index, player in enumerate(players):
            prefix = f"lineup[{player_index}]"
            if not isinstance(player, dict):
                context.add(
                    record,
                    "lineup_player_not_object",
                    prefix,
                    "lineup player must be an object",
                )
                continue
            player_id = context.positive_int(
                record, player.get("player_id"), f"{prefix}.player_id"
            )
            context.nonempty_text(
                record, player.get("player_name"), f"{prefix}.player_name"
            )
            if player_id is not None and team_id is not None:
                previous_team = player_teams.get(player_id)
                if previous_team is not None:
                    context.add(
                        record,
                        "duplicate_lineup_player",
                        f"{prefix}.player_id",
                        f"player {player_id} occurs more than once in the match",
                    )
                else:
                    player_teams[player_id] = team_id
            _validate_positions(record, player.get("positions"), prefix, context)

    missing = team_ids - seen_lineup_teams
    if missing:
        context.add(
            bundle.match,
            "match_lineup_missing",
            "lineups",
            f"missing lineup records for teams {sorted(missing)}",
        )
    return player_teams


def _validate_positions(
    record: RawRecord,
    value: Any,
    player_prefix: str,
    context: RawValidationContext,
) -> None:
    field = f"{player_prefix}.positions"
    if not isinstance(value, list):
        context.add(record, "positions_not_array", field, "positions must be an array")
        return
    for index, position in enumerate(value):
        prefix = f"{field}[{index}]"
        if not isinstance(position, dict):
            context.add(
                record,
                "position_not_object",
                prefix,
                "position must be an object",
            )
            continue
        context.positive_int(
            record, position.get("position_id"), f"{prefix}.position_id"
        )
        context.nonempty_text(
            record, position.get("position"), f"{prefix}.position"
        )
        context.clock(record, position.get("from"), f"{prefix}.from", required=True)
        context.clock(record, position.get("to"), f"{prefix}.to", required=False)
        context.period(
            record,
            position.get("from_period"),
            f"{prefix}.from_period",
            required=True,
        )
        context.period(
            record,
            position.get("to_period"),
            f"{prefix}.to_period",
            required=False,
        )
        context.nonempty_text(
            record, position.get("start_reason"), f"{prefix}.start_reason"
        )
        context.nonempty_text(
            record, position.get("end_reason"), f"{prefix}.end_reason"
        )


__all__ = ["validate_lineups"]
