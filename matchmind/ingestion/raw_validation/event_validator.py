"""Raw StatsBomb event and cross-event validation."""

from __future__ import annotations

from datetime import time
from typing import Any
from uuid import UUID

from ..normalizer import (
    ENDPOINT_DETAIL_KEYS,
    EVENT_DETAIL_KEYS,
    EVENT_TYPE_MAP,
    IMPLICIT_COMPLETE_TYPES,
    OUTCOME_DETAIL_KEYS,
)
from ..reader import RawMatchBundle, RawRecord
from ..validator import (
    ENDPOINT_REQUIRED_TYPES,
    LOCATION_OPTIONAL_TYPES,
    OUTCOME_REQUIRED_TYPES,
    PERIOD_START_MINUTES,
    PLAYER_OPTIONAL_TYPES,
)
from .context import RawValidationContext


_RAW_OUTCOME_REQUIRED_TYPES = OUTCOME_REQUIRED_TYPES - IMPLICIT_COMPLETE_TYPES
RelatedLink = tuple[RawRecord, UUID, tuple[UUID, ...]]


def validate_events(
    bundle: RawMatchBundle,
    team_ids: frozenset[int],
    player_teams: dict[int, int],
    seen_event_ids: dict[UUID, int],
    context: RawValidationContext,
) -> frozenset[UUID]:
    event_ids: set[UUID] = set()
    event_indexes: set[int] = set()
    ordered_indexes: list[int] = []
    possession_teams: dict[int, int] = {}
    related_links: list[RelatedLink] = []

    for record in bundle.events:
        payload = record.payload
        event_id = context.uuid(record, payload.get("id"), "id")
        if event_id is not None:
            previous_match = seen_event_ids.get(event_id)
            if previous_match is not None:
                context.add(
                    record,
                    "duplicate_event_id",
                    "id",
                    f"event UUID already occurs in match {previous_match}",
                )
            else:
                seen_event_ids[event_id] = record.match_id
            event_ids.add(event_id)

        event_index = context.positive_int(record, payload.get("index"), "index")
        if event_index is not None:
            if event_index in event_indexes:
                context.add(
                    record,
                    "duplicate_event_index",
                    "index",
                    f"event index {event_index} is duplicated",
                )
            event_indexes.add(event_index)
            ordered_indexes.append(event_index)

        team_id = _nested_positive_int(record, payload, "team", "id", context)
        if team_id is not None and team_id not in team_ids:
            context.add(
                record,
                "event_team_not_in_match",
                "team.id",
                f"team {team_id} does not participate in the match",
            )
        possession_team_id = _nested_positive_int(
            record, payload, "possession_team", "id", context
        )
        if possession_team_id is not None and possession_team_id not in team_ids:
            context.add(
                record,
                "possession_team_not_in_match",
                "possession_team.id",
                f"team {possession_team_id} does not participate in the match",
            )
        possession_id = context.positive_int(
            record, payload.get("possession"), "possession"
        )
        if possession_id is not None and possession_team_id is not None:
            previous_team = possession_teams.get(possession_id)
            if previous_team is not None and previous_team != possession_team_id:
                context.add(
                    record,
                    "possession_team_inconsistent",
                    "possession_team.id",
                    f"possession {possession_id} was previously assigned to "
                    f"team {previous_team}",
                )
            else:
                possession_teams[possession_id] = possession_team_id

        source_type = _nested_text(record, payload, "type", "name", context)
        event_type = EVENT_TYPE_MAP.get(source_type) if source_type else None
        if source_type is not None and event_type is None:
            context.add(
                record,
                "unknown_event_type",
                "type.name",
                f"StatsBomb event type {source_type!r} is not mapped",
            )

        player_id = _optional_nested_positive_int(
            record, payload, "player", "id", context
        )
        if event_type is not None:
            if player_id is None and event_type not in PLAYER_OPTIONAL_TYPES:
                context.add(
                    record,
                    "player_required",
                    "player.id",
                    f"player is required for {source_type}",
                )
            _validate_person_team(
                record, player_id, team_id, player_teams, "player", context
            )

        _validate_event_time(record, payload, context)
        context.optional_nonnegative_number(
            record, payload.get("duration"), "duration"
        )
        context.optional_bool(
            record, payload.get("under_pressure"), "under_pressure"
        )
        context.optional_bool(record, payload.get("counterpress"), "counterpress")
        _nested_text(record, payload, "play_pattern", "name", context)

        if event_type is not None:
            context.location(
                record,
                payload.get("location"),
                "location",
                required=event_type not in LOCATION_OPTIONAL_TYPES,
            )
            detail = _event_detail(record, payload, event_type, context)
            _validate_event_detail(
                record,
                event_type,
                source_type or event_type,
                detail,
                team_id,
                player_teams,
                context,
            )

        related = _related_ids(record, payload.get("related_events"), context)
        if event_id is not None:
            related_links.append((record, event_id, related))

    _validate_event_indexes(bundle, ordered_indexes, context)
    _validate_related_links(related_links, frozenset(event_ids), context)
    return frozenset(event_ids)


def _validate_event_indexes(
    bundle: RawMatchBundle,
    ordered_indexes: list[int],
    context: RawValidationContext,
) -> None:
    if ordered_indexes != sorted(ordered_indexes):
        context.add(
            bundle.match,
            "event_indexes_not_ordered",
            "events.index",
            "event indexes must increase in source record order",
        )
    expected_indexes = list(range(1, len(bundle.events) + 1))
    if (
        len(ordered_indexes) == len(bundle.events)
        and sorted(ordered_indexes) != expected_indexes
    ):
        context.add(
            bundle.match,
            "event_index_sequence_invalid",
            "events.index",
            "event indexes must form the complete sequence 1..N",
        )


def _validate_event_time(
    record: RawRecord,
    payload: dict[str, Any],
    context: RawValidationContext,
) -> None:
    period = context.period(record, payload.get("period"), "period", required=True)
    timestamp_value = payload.get("timestamp")
    timestamp: time | None = None
    if not isinstance(timestamp_value, str):
        context.add(
            record,
            "timestamp_invalid",
            "timestamp",
            "timestamp must be HH:MM:SS.sss text",
        )
    else:
        try:
            timestamp = time.fromisoformat(timestamp_value)
            if timestamp.tzinfo is not None:
                raise ValueError
        except ValueError:
            context.add(
                record,
                "timestamp_invalid",
                "timestamp",
                "timestamp must be a timezone-free HH:MM:SS.sss value",
            )
    minute = context.nonnegative_int(record, payload.get("minute"), "minute")
    second = context.bounded_int(record, payload.get("second"), "second", 0, 59)
    if period is None or timestamp is None or minute is None or second is None:
        return
    period_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    expected_minute, expected_second = divmod(
        PERIOD_START_MINUTES[period] * 60 + period_seconds, 60
    )
    if minute != expected_minute or second != expected_second:
        context.add(
            record,
            "time_fields_inconsistent",
            "timestamp",
            "period/timestamp does not agree with minute/second: "
            f"expected {expected_minute}:{expected_second:02d}",
        )


def _event_detail(
    record: RawRecord,
    payload: dict[str, Any],
    event_type: str,
    context: RawValidationContext,
) -> dict[str, Any] | None:
    detail_key = EVENT_DETAIL_KEYS.get(event_type)
    if detail_key is None or payload.get(detail_key) is None:
        return None
    return context.object(record, payload.get(detail_key), detail_key)


def _validate_event_detail(
    record: RawRecord,
    event_type: str,
    source_type: str,
    detail: dict[str, Any] | None,
    team_id: int | None,
    player_teams: dict[int, int],
    context: RawValidationContext,
) -> None:
    detail_key = EVENT_DETAIL_KEYS.get(event_type, event_type)
    endpoint_key = ENDPOINT_DETAIL_KEYS.get(event_type)
    endpoint_detail = detail if endpoint_key == EVENT_DETAIL_KEYS.get(event_type) else None
    endpoint = endpoint_detail.get("end_location") if endpoint_detail else None
    if endpoint_key is not None:
        context.location(
            record,
            endpoint,
            f"{endpoint_key}.end_location",
            required=event_type in ENDPOINT_REQUIRED_TYPES,
        )

    if detail is not None:
        _optional_named_object(record, detail.get("type"), "type", detail_key, context)
        _optional_named_object(
            record, detail.get("body_part"), "body_part", detail_key, context
        )

    outcome_key = OUTCOME_DETAIL_KEYS.get(event_type)
    outcome_detail = detail if outcome_key == EVENT_DETAIL_KEYS.get(event_type) else None
    outcome = outcome_detail.get("outcome") if outcome_detail else None
    if outcome is None and event_type in _RAW_OUTCOME_REQUIRED_TYPES:
        context.add(
            record,
            "outcome_required",
            f"{outcome_key}.outcome",
            f"outcome is required for {source_type}",
        )
    elif outcome is not None:
        outcome_object = context.object(record, outcome, f"{outcome_key}.outcome")
        if outcome_object is not None:
            context.nonempty_text(
                record,
                outcome_object.get("name"),
                f"{outcome_key}.outcome.name",
            )

    recipient = detail.get("recipient") if detail else None
    if recipient is not None:
        recipient_object = context.object(record, recipient, f"{detail_key}.recipient")
        recipient_id = context.positive_int(
            record,
            recipient_object.get("id") if recipient_object else None,
            f"{detail_key}.recipient.id",
        )
        if event_type != "pass":
            context.add(
                record,
                "recipient_not_allowed",
                f"{detail_key}.recipient",
                "recipient is only defined for pass events",
            )
        _validate_person_team(
            record,
            recipient_id,
            team_id,
            player_teams,
            "recipient",
            context,
        )

    if event_type == "shot":
        xg = detail.get("statsbomb_xg") if detail else None
        if xg is None:
            context.add(
                record,
                "xg_required",
                "shot.statsbomb_xg",
                "statsbomb_xg is required for shot events",
            )
        else:
            context.bounded_number(record, xg, "shot.statsbomb_xg", 0.0, 1.0)


def _validate_person_team(
    record: RawRecord,
    person_id: int | None,
    event_team_id: int | None,
    player_teams: dict[int, int],
    role: str,
    context: RawValidationContext,
) -> None:
    if person_id is None:
        return
    player_team = player_teams.get(person_id)
    if player_team is None:
        context.add(
            record,
            f"{role}_not_in_match_lineup",
            f"{role}.id",
            f"{role} {person_id} is not in a match lineup",
        )
    elif event_team_id is not None and player_team != event_team_id:
        context.add(
            record,
            f"{role}_team_mismatch",
            f"{role}.id",
            f"{role} {person_id} belongs to team {player_team}, "
            f"not {event_team_id}",
        )


def _related_ids(
    record: RawRecord,
    value: Any,
    context: RawValidationContext,
) -> tuple[UUID, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        context.add(
            record,
            "related_events_not_array",
            "related_events",
            "related_events must be an array",
        )
        return ()
    result: list[UUID] = []
    for index, item in enumerate(value):
        event_id = context.uuid(record, item, f"related_events[{index}]")
        if event_id is not None:
            result.append(event_id)
    if len(result) != len(set(result)):
        context.add(
            record,
            "duplicate_related_event",
            "related_events",
            "related_events must not contain duplicates",
        )
    return tuple(result)


def _validate_related_links(
    links: list[RelatedLink],
    event_ids: frozenset[UUID],
    context: RawValidationContext,
) -> None:
    for record, event_id, related_ids in links:
        for index, related_id in enumerate(related_ids):
            if related_id == event_id:
                context.add(
                    record,
                    "related_event_self_reference",
                    f"related_events[{index}]",
                    "an event cannot relate to itself",
                )
            elif related_id not in event_ids:
                context.add(
                    record,
                    "related_event_not_found",
                    f"related_events[{index}]",
                    f"event {related_id} is not present in the same match",
                )


def _nested_positive_int(
    record: RawRecord,
    payload: dict[str, Any],
    key: str,
    child: str,
    context: RawValidationContext,
) -> int | None:
    nested = context.object(record, payload.get(key), key)
    return context.positive_int(
        record, nested.get(child) if nested else None, f"{key}.{child}"
    )


def _optional_nested_positive_int(
    record: RawRecord,
    payload: dict[str, Any],
    key: str,
    child: str,
    context: RawValidationContext,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    nested = context.object(record, value, key)
    return context.positive_int(
        record, nested.get(child) if nested else None, f"{key}.{child}"
    )


def _nested_text(
    record: RawRecord,
    payload: dict[str, Any],
    key: str,
    child: str,
    context: RawValidationContext,
) -> str | None:
    nested = context.object(record, payload.get(key), key)
    return context.nonempty_text(
        record, nested.get(child) if nested else None, f"{key}.{child}"
    )


def _optional_named_object(
    record: RawRecord,
    value: Any,
    field: str,
    detail_key: str,
    context: RawValidationContext,
) -> None:
    if value is None:
        return
    prefix = f"{detail_key}.{field}"
    nested = context.object(record, value, prefix)
    if nested is not None:
        context.nonempty_text(record, nested.get("name"), f"{prefix}.name")


__all__ = ["validate_events"]
