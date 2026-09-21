"""Raw StatsBomb 360 validation."""

from __future__ import annotations

from uuid import UUID

from ..reader import RawMatchBundle
from .context import RawValidationContext


def validate_three_sixty(
    bundle: RawMatchBundle,
    event_ids: frozenset[UUID],
    context: RawValidationContext,
) -> None:
    seen_ids: set[UUID] = set()
    for record in bundle.three_sixty:
        payload = record.payload
        event_id = context.uuid(record, payload.get("event_uuid"), "event_uuid")
        if event_id is not None:
            if event_id not in event_ids:
                context.add(
                    record,
                    "three_sixty_event_not_found",
                    "event_uuid",
                    "360 frame does not link to an event in the same match",
                )
            if event_id in seen_ids:
                context.add(
                    record,
                    "duplicate_three_sixty_event",
                    "event_uuid",
                    "multiple 360 frames link to the same event",
                )
            seen_ids.add(event_id)

        visible = payload.get("visible_area")
        if not isinstance(visible, list) or len(visible) < 6 or len(visible) % 2:
            context.add(
                record,
                "visible_area_invalid",
                "visible_area",
                "visible_area must contain at least three x/y pairs",
            )
        else:
            for index, value in enumerate(visible):
                context.finite_number(record, value, f"visible_area[{index}]")

        freeze_frame = payload.get("freeze_frame")
        if not isinstance(freeze_frame, list):
            context.add(
                record,
                "freeze_frame_not_array",
                "freeze_frame",
                "freeze_frame must be an array",
            )
            continue
        actor_count = 0
        for index, player in enumerate(freeze_frame):
            prefix = f"freeze_frame[{index}]"
            if not isinstance(player, dict):
                context.add(
                    record,
                    "freeze_frame_player_not_object",
                    prefix,
                    "freeze-frame player must be an object",
                )
                continue
            for flag in ("teammate", "actor", "keeper"):
                if not isinstance(player.get(flag), bool):
                    context.add(
                        record,
                        "freeze_frame_flag_invalid",
                        f"{prefix}.{flag}",
                        f"{flag} must be boolean",
                    )
            actor_count += player.get("actor") is True
            location = player.get("location")
            if not isinstance(location, list) or len(location) < 2:
                context.add(
                    record,
                    "freeze_frame_location_invalid",
                    f"{prefix}.location",
                    "location must contain x and y",
                )
            else:
                context.finite_number(record, location[0], f"{prefix}.location[0]")
                context.finite_number(record, location[1], f"{prefix}.location[1]")
        if actor_count > 1:
            context.add(
                record,
                "freeze_frame_multiple_actors",
                "freeze_frame",
                "freeze_frame cannot contain more than one actor",
            )


__all__ = ["validate_three_sixty"]
