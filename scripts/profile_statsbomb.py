"""Create a repeatable profile of a StatsBomb competition-season dataset.

The default selection is the dataset pinned in
datasets/statsbomb-world-cup-2022.yaml.

Examples:
    py -3.12 scripts/profile_statsbomb.py
    py -3.12 scripts/profile_statsbomb.py --strict
    py -3.12 scripts/profile_statsbomb.py --no-write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "open-data" / "data"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "statsbomb-world-cup-2022-profile.json"
)

PINNED_SELECTION = {
    "competition_id": 43,
    "season_id": 106,
}

EXPECTED_BASELINE = {
    "match_file_sha256": (
        "e526f97e3f01894f851863fc9cc379e19f792825c57c0780949b3cd1415ab652"
    ),
    "matches": 64,
    "event_files": 64,
    "lineup_files": 64,
    "three_sixty_files": 64,
    "raw_three_sixty_frames": 203_882,
    "raw_events": 234_637,
    "unique_source_event_ids": 234_637,
    "duplicate_source_event_ids": 0,
    "event_types": 33,
    "shots": 1_494,
    "shots_with_xg": 1_494,
    "periods": [1, 2, 3, 4, 5],
    "maximum_minute": 127,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile StatsBomb matches, events, nulls, coordinates, and xG."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"StatsBomb data directory (default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument("--competition-id", type=int, default=43)
    parser.add_argument("--season-id", type=int, default=106)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary without writing the JSON report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if the pinned dataset differs from its baseline.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested_id(event: dict[str, Any], key: str) -> Any:
    value = event.get(key)
    return value.get("id") if isinstance(value, dict) else None


def nested_name(event: dict[str, Any], key: str) -> str | None:
    value = event.get(key)
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    return str(name) if name is not None else None


def inspect_coordinate(
    value: Any,
    coordinate_range: dict[str, float | None],
    problems: Counter[str],
) -> None:
    if not isinstance(value, list) or len(value) < 2:
        problems["malformed"] += 1
        return

    source_x, source_y = value[0], value[1]
    if (
        isinstance(source_x, bool)
        or isinstance(source_y, bool)
        or not isinstance(source_x, (int, float))
        or not isinstance(source_y, (int, float))
    ):
        problems["non_numeric"] += 1
        return

    coordinate_range["x_min"] = (
        source_x
        if coordinate_range["x_min"] is None
        else min(coordinate_range["x_min"], source_x)
    )
    coordinate_range["x_max"] = (
        source_x
        if coordinate_range["x_max"] is None
        else max(coordinate_range["x_max"], source_x)
    )
    coordinate_range["y_min"] = (
        source_y
        if coordinate_range["y_min"] is None
        else min(coordinate_range["y_min"], source_y)
    )
    coordinate_range["y_max"] = (
        source_y
        if coordinate_range["y_max"] is None
        else max(coordinate_range["y_max"], source_y)
    )

    if not 0 <= source_x <= 120:
        problems["x_out_of_range"] += 1
    if not 0 <= source_y <= 80:
        problems["y_out_of_range"] += 1


def compare_baseline(actual: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for field, expected in EXPECTED_BASELINE.items():
        observed = actual[field]
        if observed != expected:
            mismatches.append(
                {"field": field, "expected": expected, "actual": observed}
            )
    return {"passed": not mismatches, "mismatches": mismatches}


def build_profile(data_root: Path, competition_id: int, season_id: int) -> dict[str, Any]:
    match_file = data_root / "matches" / str(competition_id) / f"{season_id}.json"
    matches = load_json(match_file)
    if not isinstance(matches, list):
        raise SystemExit(f"Expected a JSON array in {match_file}")

    event_types: Counter[str] = Counter()
    periods: Counter[int] = Counter()
    missing_player_by_type: Counter[str] = Counter()
    missing_location_by_type: Counter[str] = Counter()
    missing_required_fields: Counter[str] = Counter()
    start_coordinate_problems: Counter[str] = Counter()
    endpoint_coordinate_problems: Counter[str] = Counter()
    start_coordinate_range: dict[str, float | None] = {
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
    }
    endpoint_coordinate_range = start_coordinate_range.copy()
    endpoint_counts: Counter[str] = Counter()
    schema_versions: set[tuple[Any, Any, Any]] = set()
    source_event_ids: set[str] = set()
    duplicate_ids: Counter[str] = Counter()

    raw_events = 0
    shots = 0
    shots_with_xg = 0
    maximum_minute: int | None = None
    event_file_count = 0
    lineup_file_count = 0
    three_sixty_file_count = 0
    raw_three_sixty_frames = 0
    linked_three_sixty_frames = 0
    duplicate_three_sixty_event_ids = 0
    three_sixty_event_ids: set[str] = set()
    missing_event_files: list[int] = []
    missing_lineup_files: list[int] = []
    missing_three_sixty_files: list[int] = []

    for match in matches:
        if not isinstance(match, dict) or match.get("match_id") is None:
            raise SystemExit(f"Malformed match record in {match_file}")

        match_id = int(match["match_id"])
        metadata = match.get("metadata") or {}
        schema_versions.add(
            (
                metadata.get("data_version"),
                metadata.get("shot_fidelity_version"),
                metadata.get("xy_fidelity_version"),
            )
        )

        event_file = data_root / "events" / f"{match_id}.json"
        lineup_file = data_root / "lineups" / f"{match_id}.json"
        three_sixty_file = data_root / "three-sixty" / f"{match_id}.json"

        if lineup_file.is_file():
            lineup_file_count += 1
        else:
            missing_lineup_files.append(match_id)
        if three_sixty_file.is_file():
            three_sixty_file_count += 1
        else:
            missing_three_sixty_files.append(match_id)

        if not event_file.is_file():
            missing_event_files.append(match_id)
            continue
        event_file_count += 1

        events = load_json(event_file)
        if not isinstance(events, list):
            raise SystemExit(f"Expected a JSON array in {event_file}")

        match_event_ids: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                missing_required_fields["event_object"] += 1
                continue

            raw_events += 1
            event_type = nested_name(event, "type")
            if event_type is None:
                missing_required_fields["event_type"] += 1
                event_type = "<missing>"
            event_types[event_type] += 1

            event_id = event.get("id")
            if not isinstance(event_id, str) or not event_id:
                missing_required_fields["source_event_id"] += 1
            elif event_id in source_event_ids:
                duplicate_ids[event_id] += 1
            else:
                source_event_ids.add(event_id)
            if isinstance(event_id, str) and event_id:
                match_event_ids.add(event_id)

            if nested_id(event, "team") is None:
                missing_required_fields["team_id"] += 1
            if nested_id(event, "player") is None:
                missing_player_by_type[event_type] += 1
            if event.get("timestamp") is None:
                missing_required_fields["timestamp"] += 1

            period = event.get("period")
            if isinstance(period, int) and not isinstance(period, bool):
                periods[period] += 1
            else:
                missing_required_fields["period"] += 1

            minute = event.get("minute")
            if isinstance(minute, int) and not isinstance(minute, bool):
                maximum_minute = (
                    minute if maximum_minute is None else max(maximum_minute, minute)
                )
            else:
                missing_required_fields["minute"] += 1
            if event.get("second") is None:
                missing_required_fields["second"] += 1

            location = event.get("location")
            if location is None:
                missing_location_by_type[event_type] += 1
            else:
                inspect_coordinate(
                    location, start_coordinate_range, start_coordinate_problems
                )

            for detail_name in ("pass", "carry", "shot", "goalkeeper"):
                detail = event.get(detail_name)
                if not isinstance(detail, dict) or "end_location" not in detail:
                    continue
                endpoint_counts[detail_name] += 1
                inspect_coordinate(
                    detail["end_location"],
                    endpoint_coordinate_range,
                    endpoint_coordinate_problems,
                )

            if event_type == "Shot":
                shots += 1
                shot = event.get("shot")
                if isinstance(shot, dict) and shot.get("statsbomb_xg") is not None:
                    shots_with_xg += 1

        if three_sixty_file.is_file():
            frames = load_json(three_sixty_file)
            if not isinstance(frames, list):
                raise SystemExit(f"Expected a JSON array in {three_sixty_file}")
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                raw_three_sixty_frames += 1
                event_uuid = frame.get("event_uuid")
                if isinstance(event_uuid, str) and event_uuid in match_event_ids:
                    linked_three_sixty_frames += 1
                if isinstance(event_uuid, str):
                    if event_uuid in three_sixty_event_ids:
                        duplicate_three_sixty_event_ids += 1
                    three_sixty_event_ids.add(event_uuid)

    duplicate_occurrences = sum(duplicate_ids.values())
    actual_baseline = {
        "match_file_sha256": sha256(match_file),
        "matches": len(matches),
        "event_files": event_file_count,
        "lineup_files": lineup_file_count,
        "three_sixty_files": three_sixty_file_count,
        "raw_three_sixty_frames": raw_three_sixty_frames,
        "raw_events": raw_events,
        "unique_source_event_ids": len(source_event_ids),
        "duplicate_source_event_ids": duplicate_occurrences,
        "event_types": len(event_types),
        "shots": shots,
        "shots_with_xg": shots_with_xg,
        "periods": sorted(periods),
        "maximum_minute": maximum_minute,
    }

    pinned = {
        "competition_id": competition_id,
        "season_id": season_id,
    } == PINNED_SELECTION

    profile: dict[str, Any] = {
        "selection": {
            "competition_id": competition_id,
            "season_id": season_id,
            "match_file": str(match_file.resolve()),
            "is_pinned_selection": pinned,
        },
        "source_versions": [
            {
                "data_version": values[0],
                "shot_fidelity_version": values[1],
                "xy_fidelity_version": values[2],
            }
            for values in sorted(schema_versions, key=lambda item: str(item))
        ],
        "files": {
            "matches": 1,
            "events": event_file_count,
            "lineups": lineup_file_count,
            "three_sixty": three_sixty_file_count,
            "missing_event_match_ids": sorted(missing_event_files),
            "missing_lineup_match_ids": sorted(missing_lineup_files),
            "missing_three_sixty_match_ids": sorted(missing_three_sixty_files),
        },
        "counts": {
            "matches": len(matches),
            "raw_events": raw_events,
            "unique_source_event_ids": len(source_event_ids),
            "duplicate_source_event_occurrences": duplicate_occurrences,
            "event_types": len(event_types),
            "shots": shots,
            "shots_with_xg": shots_with_xg,
            "three_sixty_frames": raw_three_sixty_frames,
            "linked_three_sixty_frames": linked_three_sixty_frames,
            "duplicate_three_sixty_event_ids": duplicate_three_sixty_event_ids,
        },
        "reconciliation": {
            "equation": "raw_events = unique_source_event_ids + duplicate_occurrences",
            "passed": raw_events == len(source_event_ids) + duplicate_occurrences,
        },
        "event_type_counts": dict(sorted(event_types.items())),
        "period_counts": {str(key): periods[key] for key in sorted(periods)},
        "maximum_minute": maximum_minute,
        "missing_player_by_event_type": dict(sorted(missing_player_by_type.items())),
        "missing_location_by_event_type": dict(
            sorted(missing_location_by_type.items())
        ),
        "missing_required_fields": dict(sorted(missing_required_fields.items())),
        "coordinates": {
            "source_pitch": {"x": [0, 120], "y": [0, 80]},
            "start_location_observed": start_coordinate_range,
            "start_location_problems": dict(sorted(start_coordinate_problems.items())),
            "endpoint_observed": endpoint_coordinate_range,
            "endpoint_counts": dict(sorted(endpoint_counts.items())),
            "endpoint_problems": dict(sorted(endpoint_coordinate_problems.items())),
        },
        "duplicate_source_event_ids": dict(sorted(duplicate_ids.items())),
    }

    profile["baseline_check"] = (
        compare_baseline(actual_baseline)
        if pinned
        else {"passed": None, "mismatches": [], "reason": "Not the pinned selection"}
    )
    return profile


def print_summary(profile: dict[str, Any]) -> None:
    selection = profile["selection"]
    counts = profile["counts"]
    files = profile["files"]
    check = profile["baseline_check"]

    print(
        "StatsBomb profile "
        f"competition={selection['competition_id']} season={selection['season_id']}"
    )
    print(
        f"files: matches=1 events={files['events']} lineups={files['lineups']} "
        f"three_sixty={files['three_sixty']}"
    )
    print(
        f"records: matches={counts['matches']} raw_events={counts['raw_events']} "
        f"unique_ids={counts['unique_source_event_ids']} "
        f"duplicates={counts['duplicate_source_event_occurrences']}"
    )
    print(
        f"360: frames={counts['three_sixty_frames']} "
        f"linked={counts['linked_three_sixty_frames']} "
        f"duplicates={counts['duplicate_three_sixty_event_ids']}"
    )
    print(
        f"classes: event_types={counts['event_types']} shots={counts['shots']} "
        f"shots_with_xg={counts['shots_with_xg']}"
    )
    print(
        f"time: periods={list(profile['period_counts'])} "
        f"maximum_minute={profile['maximum_minute']}"
    )
    if check["passed"] is True:
        print("baseline: PASS")
    elif check["passed"] is False:
        print("baseline: FAIL")
        for mismatch in check["mismatches"]:
            print(
                f"  {mismatch['field']}: expected={mismatch['expected']!r} "
                f"actual={mismatch['actual']!r}"
            )
    else:
        print("baseline: SKIPPED (selection is not the pinned dataset)")


def main() -> None:
    args = parse_args()
    profile = build_profile(
        args.data_root.resolve(), args.competition_id, args.season_id
    )
    print_summary(profile)

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"report: {args.output.resolve()}")

    if args.strict and profile["baseline_check"]["passed"] is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
