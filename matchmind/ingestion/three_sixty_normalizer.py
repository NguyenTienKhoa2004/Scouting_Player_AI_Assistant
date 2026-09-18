"""Normalize StatsBomb 360 records."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .normalizer import NormalizationError, StatsBombEventNormalizer
from .reader import RawRecord


@dataclass(frozen=True, slots=True)
class Canonical360Frame:
    """One StatsBomb 360 snapshot linked to an event UUID."""

    source: str
    source_event_id: UUID
    source_record_index: int
    match_id: int
    visible_area: tuple[float, ...]
    freeze_frame: tuple[dict[str, Any], ...]
    raw_details: dict[str, Any]

    def as_serializable_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_event_id": str(self.source_event_id),
            "source_record_index": self.source_record_index,
            "match_id": self.match_id,
            "visible_area": list(self.visible_area),
            "freeze_frame": list(self.freeze_frame),
        }


class StatsBomb360Normalizer:
    """Normalize one StatsBomb 360 frame."""

    def normalize(self, record: RawRecord) -> Canonical360Frame:
        if record.kind != "three_sixty":
            raise NormalizationError(
                "unexpected_record_kind",
                f"Expected a three_sixty record, got {record.kind!r}",
                record,
            )
        payload = record.payload
        event_id = StatsBombEventNormalizer._uuid(
            payload.get("event_uuid"), "event_uuid", record
        )
        visible_area_value = payload.get("visible_area")
        if not isinstance(visible_area_value, list):
            raise NormalizationError(
                "malformed_visible_area", "visible_area must be an array", record
            )
        visible_area = tuple(
            StatsBombEventNormalizer._float(
                value, f"visible_area[{index}]", record
            )
            for index, value in enumerate(visible_area_value)
        )

        freeze_frame_value = payload.get("freeze_frame")
        if not isinstance(freeze_frame_value, list):
            raise NormalizationError(
                "malformed_freeze_frame", "freeze_frame must be an array", record
            )
        freeze_frame: list[dict[str, Any]] = []
        for index, player in enumerate(freeze_frame_value):
            if not isinstance(player, dict):
                raise NormalizationError(
                    "malformed_freeze_frame_player",
                    f"freeze_frame[{index}] must be an object",
                    record,
                )
            freeze_frame.append(copy.deepcopy(player))

        return Canonical360Frame(
            source=record.source,
            source_event_id=event_id,
            source_record_index=record.source_record_index,
            match_id=record.match_id,
            visible_area=visible_area,
            freeze_frame=tuple(freeze_frame),
            raw_details=copy.deepcopy(payload),
        )

    def normalize_many(
        self, records: Iterable[RawRecord]
    ) -> Iterator[Canonical360Frame]:
        for record in records:
            yield self.normalize(record)


__all__ = ["Canonical360Frame", "StatsBomb360Normalizer"]
