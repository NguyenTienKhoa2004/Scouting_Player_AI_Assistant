"""Adapter from PitchPulse's canonical events to socceraction SPADL."""

from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any
from uuid import UUID

import pandas as pd
import socceraction.spadl as spadl
from socceraction.spadl.statsbomb import convert_to_actions

from pitchpulse.ingestion.normalizer import CanonicalEvent

from .input_validator import SpadlInput


SOCCERACTION_VERSION = version("socceraction")
ACTION_MAPPING_VERSION = f"socceraction-{SOCCERACTION_VERSION}-statsbomb-spadl"
COORDINATE_SYSTEM_VERSION = "socceraction-spadl-105x68-v1"
SPADL_FIELD_LENGTH = 105.0
SPADL_FIELD_WIDTH = 68.0


class ConversionError(ValueError):
    """A source event cannot safely be represented by the library adapter."""


@dataclass(frozen=True, slots=True)
class SpadlAction:
    """One socceraction SPADL row plus PitchPulse trace/context fields."""

    match_id: int
    action_id: int
    source: str
    source_event_id: UUID
    source_event_index: int
    source_action_index: int
    mapping_version: str
    coordinate_system_version: str
    period_id: int
    time_seconds: float
    minute: int
    team_id: int
    player_id: int
    possession_id: int
    possession_team_id: int
    type_id: int
    type_name: str
    subtype_name: str | None
    result_id: int
    result_name: str
    bodypart_id: int
    bodypart_name: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    duration: float | None
    recipient_id: int | None
    play_pattern: str
    under_pressure: bool
    counterpress: bool
    has_360: bool
    synthetic: bool

    def as_dict(self) -> dict[str, Any]:
        values = {field: getattr(self, field) for field in self.__dataclass_fields__}
        values["source_event_id"] = str(self.source_event_id)
        return values


@dataclass(frozen=True, slots=True)
class SpadlConversionReport:
    mapping_version: str
    coordinate_system_version: str
    match_count: int
    event_count: int
    mapped_event_count: int
    excluded_event_count: int
    action_count: int
    synthetic_action_count: int
    exclusions_by_reason: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapping_version": self.mapping_version,
            "coordinate_system_version": self.coordinate_system_version,
            "match_count": self.match_count,
            "event_count": self.event_count,
            "mapped_event_count": self.mapped_event_count,
            "excluded_event_count": self.excluded_event_count,
            "action_count": self.action_count,
            "synthetic_action_count": self.synthetic_action_count,
            "exclusions_by_reason": dict(self.exclusions_by_reason),
        }


@dataclass(frozen=True, slots=True)
class SpadlConversionResult:
    actions: tuple[SpadlAction, ...]
    report: SpadlConversionReport


class EventToActionConverter:
    """Delegate StatsBomb-to-SPADL semantics to socceraction 1.5.3."""

    def convert(self, inputs: SpadlInput) -> SpadlConversionResult:
        inputs.report.raise_for_errors()
        events_by_match: defaultdict[int, list[CanonicalEvent]] = defaultdict(list)
        for event in inputs.events:
            events_by_match[event.match_id].append(event)

        actions: list[SpadlAction] = []
        mapped_keys: set[tuple[str, UUID]] = set()
        exclusions: Counter[str] = Counter()
        synthetic_count = 0

        for match_id in sorted(events_by_match):
            events = sorted(
                events_by_match[match_id],
                key=lambda event: (event.source_event_index, event.source_event_id.int),
            )
            home_team_id = inputs.home_team_by_match.get(match_id)
            if home_team_id is None:
                raise ConversionError(f"match {match_id} has no home-team context")

            event_by_id = {str(event.source_event_id): event for event in events}
            event_frame = pd.DataFrame(
                [self._to_socceraction_event(event) for event in events]
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=FutureWarning, module=r"socceraction\..*"
                    )
                    converted = convert_to_actions(
                        event_frame,
                        home_team_id=home_team_id,
                        xy_fidelity_version=2,
                        shot_fidelity_version=2,
                    )
                converted = spadl.add_names(converted)
            except Exception as exc:
                raise ConversionError(
                    f"socceraction failed to convert match {match_id}: {exc}"
                ) from exc

            original_ids = list(converted["original_event_id"])
            anchor_ids = self._fill_synthetic_provenance(original_ids)
            source_indexes: Counter[str] = Counter()
            for row_position, (_, row) in enumerate(converted.iterrows()):
                original_id = original_ids[row_position]
                synthetic = self._is_missing(original_id)
                anchor_id = anchor_ids[row_position]
                source_event = event_by_id.get(anchor_id)
                if source_event is None:
                    raise ConversionError(
                        f"socceraction emitted unknown event ID {anchor_id!r} "
                        f"for match {match_id}"
                    )
                if not synthetic:
                    mapped_keys.add((source_event.source, source_event.source_event_id))

                source_action_index = source_indexes[anchor_id]
                source_indexes[anchor_id] += 1
                actions.append(
                    self._to_action(
                        row,
                        source_event,
                        action_id=row_position,
                        source_action_index=source_action_index,
                        synthetic=synthetic,
                        has_360=(
                            not synthetic
                            and (source_event.source, source_event.source_event_id)
                            in inputs.three_sixty_by_event
                        ),
                    )
                )
                synthetic_count += synthetic

            for event in events:
                if (event.source, event.source_event_id) not in mapped_keys:
                    exclusions[f"socceraction_non_action:{event.event_type}"] += 1

        report = SpadlConversionReport(
            mapping_version=ACTION_MAPPING_VERSION,
            coordinate_system_version=COORDINATE_SYSTEM_VERSION,
            match_count=len(events_by_match),
            event_count=len(inputs.events),
            mapped_event_count=len(mapped_keys),
            excluded_event_count=len(inputs.events) - len(mapped_keys),
            action_count=len(actions),
            synthetic_action_count=synthetic_count,
            exclusions_by_reason=tuple(sorted(exclusions.items())),
        )
        return SpadlConversionResult(tuple(actions), report)

    @staticmethod
    def _to_socceraction_event(event: CanonicalEvent) -> dict[str, Any]:
        payload = event.raw_details
        if not isinstance(payload, dict):
            raise ConversionError(
                f"event {event.source_event_id} raw_details must be a StatsBomb object"
            )

        type_value = payload.get("type")
        source_type = type_value.get("name") if isinstance(type_value, dict) else None
        if source_type is None:
            raise ConversionError(
                f"event {event.source_event_id} raw_details.type.name is required"
            )

        extra = {
            key: value
            for key, value in payload.items()
            if isinstance(value, dict)
            and not ("id" in value and "name" in value)
        }
        return {
            "game_id": event.match_id,
            "event_id": str(event.source_event_id),
            "period_id": event.period,
            "timestamp": payload.get("timestamp", event.timestamp.isoformat()),
            "team_id": event.team_id,
            "player_id": event.player_id,
            "type_name": source_type,
            "location": payload.get("location"),
            "extra": extra,
        }

    @classmethod
    def _fill_synthetic_provenance(cls, values: list[Any]) -> list[str]:
        real = [None if cls._is_missing(value) else str(value) for value in values]
        if not any(real):
            raise ConversionError("socceraction produced actions without source events")
        result: list[str] = []
        for index, value in enumerate(real):
            if value is not None:
                result.append(value)
                continue
            following = next((item for item in real[index + 1 :] if item is not None), None)
            preceding = next(
                (item for item in reversed(real[:index]) if item is not None), None
            )
            result.append(following or preceding or "")
        return result

    @staticmethod
    def _is_missing(value: Any) -> bool:
        missing = pd.isna(value)
        return bool(missing) if not hasattr(missing, "__len__") else False

    @staticmethod
    def _to_action(
        row: Any,
        event: CanonicalEvent,
        *,
        action_id: int,
        source_action_index: int,
        synthetic: bool,
        has_360: bool,
    ) -> SpadlAction:
        if EventToActionConverter._is_missing(row.player_id):
            raise ConversionError(
                f"action from event {event.source_event_id} has no player_id"
            )
        coordinates = [row.start_x, row.start_y, row.end_x, row.end_y]
        if any(not math.isfinite(float(value)) for value in coordinates):
            raise ConversionError(
                f"action from event {event.source_event_id} has invalid coordinates"
            )
        return SpadlAction(
            match_id=event.match_id,
            action_id=action_id,
            source=event.source,
            source_event_id=event.source_event_id,
            source_event_index=event.source_event_index,
            source_action_index=source_action_index,
            mapping_version=ACTION_MAPPING_VERSION,
            coordinate_system_version=COORDINATE_SYSTEM_VERSION,
            period_id=int(row.period_id),
            time_seconds=float(row.time_seconds),
            minute=event.minute,
            team_id=int(row.team_id),
            player_id=int(row.player_id),
            possession_id=event.possession_id,
            possession_team_id=event.possession_team_id,
            type_id=int(row.type_id),
            type_name=str(row.type_name),
            subtype_name=event.event_subtype,
            result_id=int(row.result_id),
            result_name=str(row.result_name),
            bodypart_id=int(row.bodypart_id),
            bodypart_name=str(row.bodypart_name),
            start_x=float(row.start_x),
            start_y=float(row.start_y),
            end_x=float(row.end_x),
            end_y=float(row.end_y),
            duration=event.duration,
            recipient_id=event.recipient_id,
            play_pattern=event.play_pattern,
            under_pressure=event.under_pressure,
            counterpress=event.counterpress,
            has_360=has_360,
            synthetic=synthetic,
        )


def convert_events_to_actions(inputs: SpadlInput) -> SpadlConversionResult:
    return EventToActionConverter().convert(inputs)


__all__ = [
    "ACTION_MAPPING_VERSION",
    "COORDINATE_SYSTEM_VERSION",
    "ConversionError",
    "EventToActionConverter",
    "SOCCERACTION_VERSION",
    "SPADL_FIELD_LENGTH",
    "SPADL_FIELD_WIDTH",
    "SpadlAction",
    "SpadlConversionReport",
    "SpadlConversionResult",
    "convert_events_to_actions",
]
