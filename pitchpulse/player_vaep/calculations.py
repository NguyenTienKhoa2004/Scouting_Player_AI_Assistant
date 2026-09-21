"""Calculate player minutes and summarize action VAEP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PLAYER_AGGREGATION_VERSION = "vaep-player-aggregation-v1"
MINUTES_POLICY_VERSION = "statsbomb-lineup-intervals-added-time-v1"
TOTAL_ACTION_TYPE = "all"
UNKNOWN_POSITION = "Unknown"


class PlayerAggregationError(ValueError):
    """Raised when player minutes or action values cannot be reconciled."""


@dataclass(frozen=True, slots=True)
class MatchContext:
    match_id: int
    competition_id: int
    season_id: int


@dataclass(frozen=True, slots=True)
class PlayerMatchMinutes:
    player_id: int
    team_id: int
    minutes_played: float
    primary_position: str


@dataclass(slots=True)
class _Totals:
    minutes_played: float = 0.0
    match_count: int = 0
    action_count: int = 0
    offensive_vaep: float = 0.0
    defensive_vaep: float = 0.0
    player_vaep: float = 0.0


def clock_to_seconds(value: str) -> int:
    """Convert StatsBomb's cumulative ``MM:SS`` match clock to seconds."""

    try:
        minutes_text, seconds_text = value.split(":", maxsplit=1)
        minutes = int(minutes_text)
        seconds = int(seconds_text)
    except (AttributeError, ValueError) as exc:
        raise PlayerAggregationError(f"Invalid lineup clock: {value!r}") from exc
    if minutes < 0 or seconds < 0 or seconds > 59:
        raise PlayerAggregationError(f"Invalid lineup clock: {value!r}")
    return minutes * 60 + seconds


def match_duration_seconds(events: Iterable[Mapping[str, Any]]) -> int:
    """Return elapsed match seconds, including added and extra time.

    Penalty-shootout events (period 5) are deliberately ignored because the
    baseline VAEP target and action-value artifacts exclude shootouts.
    """

    latest = 0
    for event in events:
        period = event.get("period")
        minute = event.get("minute")
        second = event.get("second")
        if not isinstance(period, int) or period < 1 or period > 4:
            continue
        if not isinstance(minute, int) or not isinstance(second, int):
            continue
        if minute < 0 or second < 0 or second > 59:
            continue
        latest = max(latest, minute * 60 + second)
    if latest <= 0:
        raise PlayerAggregationError("Could not determine match duration from events")
    return latest


def calculate_player_minutes(
    lineup_teams: Iterable[Mapping[str, Any]],
    duration_seconds: int,
) -> dict[tuple[int, int], PlayerMatchMinutes]:
    """Calculate one on-pitch duration and primary position per player."""

    if duration_seconds <= 0:
        raise PlayerAggregationError("Match duration must be positive")

    result: dict[tuple[int, int], PlayerMatchMinutes] = {}
    for team in lineup_teams:
        team_id = _positive_id(team.get("team_id"), "team_id")
        players = team.get("lineup")
        if not isinstance(players, list):
            raise PlayerAggregationError("Lineup team must contain a player list")
        for player in players:
            if not isinstance(player, Mapping):
                raise PlayerAggregationError("Lineup player must be an object")
            player_id = _positive_id(player.get("player_id"), "player_id")
            intervals: list[tuple[int, int]] = []
            position_seconds: dict[str, int] = {}
            positions = player.get("positions")
            if not isinstance(positions, list):
                raise PlayerAggregationError("Lineup player positions must be a list")
            for position in positions:
                if not isinstance(position, Mapping):
                    raise PlayerAggregationError("Lineup position must be an object")
                start = clock_to_seconds(position.get("from"))
                end_value = position.get("to")
                end = (
                    duration_seconds
                    if end_value is None
                    else clock_to_seconds(end_value)
                )
                start = min(start, duration_seconds)
                end = min(end, duration_seconds)
                # A small number of StatsBomb open-data records contain an
                # impossible cross-period tactical interval (for example,
                # period 4 back to period 2). Other valid intervals for the
                # same player remain usable, so ignore only the malformed row.
                if end < start:
                    continue
                if end == start:
                    continue
                intervals.append((start, end))
                name = position.get("position")
                if isinstance(name, str) and name:
                    position_seconds[name] = position_seconds.get(name, 0) + end - start

            if not intervals:
                continue
            seconds_played = sum(end - start for start, end in _merge_intervals(intervals))
            primary_position = (
                min(position_seconds, key=lambda name: (-position_seconds[name], name))
                if position_seconds
                else UNKNOWN_POSITION
            )
            key = (player_id, team_id)
            if key in result:
                raise PlayerAggregationError(
                    f"Duplicate lineup entry for player {player_id}, team {team_id}"
                )
            result[key] = PlayerMatchMinutes(
                player_id=player_id,
                team_id=team_id,
                minutes_played=seconds_played / 60.0,
                primary_position=primary_position,
            )
    return result


class PlayerVaepAggregator:
    """Accumulate match and competition-season player VAEP summaries."""

    def __init__(self, *, minimum_minutes: float = 450.0) -> None:
        if minimum_minutes < 0:
            raise ValueError("minimum_minutes must be non-negative")
        self.minimum_minutes = float(minimum_minutes)
        self._season_totals: dict[tuple[Any, ...], _Totals] = {}
        self._match_rows: list[dict[str, Any]] = []
        self.input_action_count = 0
        self.known_player_action_count = 0
        self.unknown_player_action_count = 0
        self.known_player_actions_without_minutes = 0
        self.known_offensive_vaep = 0.0
        self.known_defensive_vaep = 0.0
        self.known_player_vaep = 0.0

    def add_match(
        self,
        actions: Any,
        context: MatchContext,
        minutes: Mapping[tuple[int, int], PlayerMatchMinutes],
    ) -> None:
        """Add one match's action rows; ``actions`` is a pandas DataFrame."""

        required = {
            "player_id",
            "team_id",
            "type_name",
            "offensive_value",
            "defensive_value",
            "vaep_value",
        }
        missing = sorted(required.difference(actions.columns))
        if missing:
            raise PlayerAggregationError(f"Action values lack columns: {missing}")
        self.input_action_count += len(actions)
        unknown = actions[actions["player_id"].isna()]
        self.unknown_player_action_count += len(unknown)
        known = actions[actions["player_id"].notna()].copy()
        if known.empty:
            return
        known["player_id"] = known["player_id"].astype("int64")
        self.known_player_action_count += len(known)
        self.known_offensive_vaep += float(known["offensive_value"].sum())
        self.known_defensive_vaep += float(known["defensive_value"].sum())
        self.known_player_vaep += float(known["vaep_value"].sum())

        all_actions = self._group_actions(known, ["player_id", "team_id"])
        typed_actions = self._group_actions(
            known, ["player_id", "team_id", "type_name"]
        )
        for group in all_actions:
            self._add_group(group, TOTAL_ACTION_TYPE, context, minutes)
        for group in typed_actions:
            self._add_group(group, str(group["type_name"]), context, minutes)

    @staticmethod
    def _group_actions(frame: Any, columns: list[str]) -> list[dict[str, Any]]:
        grouped = frame.groupby(columns, sort=True, dropna=False).agg(
            action_count=("vaep_value", "size"),
            offensive_vaep=("offensive_value", "sum"),
            defensive_vaep=("defensive_value", "sum"),
            player_vaep=("vaep_value", "sum"),
        )
        return grouped.reset_index().to_dict("records")

    def _add_group(
        self,
        group: Mapping[str, Any],
        action_type: str,
        context: MatchContext,
        minutes: Mapping[tuple[int, int], PlayerMatchMinutes],
    ) -> None:
        player_id = int(group["player_id"])
        team_id = int(group["team_id"])
        playing_time = minutes.get((player_id, team_id))
        if playing_time is None:
            minutes_played = 0.0
            position = UNKNOWN_POSITION
            if action_type == TOTAL_ACTION_TYPE:
                self.known_player_actions_without_minutes += int(group["action_count"])
        else:
            minutes_played = playing_time.minutes_played
            position = playing_time.primary_position

        values = {
            "action_count": int(group["action_count"]),
            "offensive_vaep": float(group["offensive_vaep"]),
            "defensive_vaep": float(group["defensive_vaep"]),
            "player_vaep": float(group["player_vaep"]),
        }
        self._match_rows.append(
            self._row(
                aggregation_level="match",
                match_id=context.match_id,
                player_id=player_id,
                team_id=team_id,
                competition_id=context.competition_id,
                season_id=context.season_id,
                position=position,
                action_type=action_type,
                minutes_played=minutes_played,
                match_count=1,
                **values,
            )
        )

        key = (
            player_id,
            team_id,
            context.competition_id,
            context.season_id,
            position,
            action_type,
        )
        total = self._season_totals.setdefault(key, _Totals())
        total.minutes_played += minutes_played
        total.match_count += 1
        total.action_count += values["action_count"]
        total.offensive_vaep += values["offensive_vaep"]
        total.defensive_vaep += values["defensive_vaep"]
        total.player_vaep += values["player_vaep"]

    def rows(self) -> list[dict[str, Any]]:
        season_rows: list[dict[str, Any]] = []
        for key in sorted(self._season_totals):
            player_id, team_id, competition_id, season_id, position, action_type = key
            total = self._season_totals[key]
            season_rows.append(
                self._row(
                    aggregation_level="competition_season",
                    match_id=None,
                    player_id=player_id,
                    team_id=team_id,
                    competition_id=competition_id,
                    season_id=season_id,
                    position=position,
                    action_type=action_type,
                    minutes_played=total.minutes_played,
                    match_count=total.match_count,
                    action_count=total.action_count,
                    offensive_vaep=total.offensive_vaep,
                    defensive_vaep=total.defensive_vaep,
                    player_vaep=total.player_vaep,
                )
            )
        match_rows = sorted(
            self._match_rows,
            key=lambda row: (
                row["match_id"],
                row["player_id"],
                row["team_id"],
                row["position"],
                row["action_type"],
            ),
        )
        return season_rows + match_rows

    def _row(self, **values: Any) -> dict[str, Any]:
        minutes = float(values["minutes_played"])
        player_vaep = float(values["player_vaep"])
        return {
            "aggregation_version": PLAYER_AGGREGATION_VERSION,
            "minutes_policy_version": MINUTES_POLICY_VERSION,
            **values,
            "vaep_per_90": None if minutes <= 0 else player_vaep / minutes * 90.0,
            "minimum_minutes_eligible": minutes >= self.minimum_minutes,
        }


def _positive_id(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlayerAggregationError(f"{name} must be a positive integer")
    return value


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


__all__ = [
    "MINUTES_POLICY_VERSION",
    "PLAYER_AGGREGATION_VERSION",
    "MatchContext",
    "PlayerAggregationError",
    "PlayerMatchMinutes",
    "PlayerVaepAggregator",
    "calculate_player_minutes",
    "clock_to_seconds",
    "match_duration_seconds",
]
