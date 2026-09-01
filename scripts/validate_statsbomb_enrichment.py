"""Validate lineup intervals and StatsBomb 360 without writing PostgreSQL."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchmind.data import (  # noqa: E402
    Canonical360Validator,
    CanonicalLineupIntervalValidator,
    NormalizationError,
    RawDataError,
    StatsBomb360Normalizer,
    StatsBombLineupNormalizer,
    StatsBombRawReader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate VAEP-ready lineup intervals and StatsBomb 360."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--match-id", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = StatsBombRawReader()
    lineup_normalizer = StatsBombLineupNormalizer()
    lineup_validator = CanonicalLineupIntervalValidator()
    frame_normalizer = StatsBomb360Normalizer()
    frame_validator = Canonical360Validator()
    issue_counts: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    raw_intervals = valid_intervals = invalid_intervals = 0
    raw_frames = valid_frames = invalid_frames = 0

    match_ids = (
        [record.match_id for record in reader.iter_matches()]
        if args.all
        else [args.match_id]
    )
    try:
        for match_id in match_ids:
            bundle = reader.read_match_bundle(match_id)
            event_ids = {
                UUID(str(record.payload["id"])) for record in bundle.events
            }

            for record in bundle.lineups:
                players = record.payload.get("lineup")
                positions_count = (
                    sum(
                        len(player.get("positions", []))
                        for player in players
                        if isinstance(player, dict)
                    )
                    if isinstance(players, list)
                    else 1
                )
                raw_intervals += positions_count
                try:
                    intervals = lineup_normalizer.normalize(record)
                except NormalizationError as exc:
                    invalid_intervals += positions_count
                    issue_counts[f"lineup.normalization.{exc.code}"] += positions_count
                    continue
                for interval in intervals:
                    if (
                        interval.to_seconds is not None
                        and interval.to_seconds < interval.from_seconds
                    ):
                        quality_flags["lineup.to_before_from"] += 1
                    if (
                        interval.to_period is not None
                        and interval.to_period < interval.from_period
                    ):
                        quality_flags["lineup.to_period_before_from_period"] += 1
                    result = lineup_validator.validate(interval)
                    if result.is_valid:
                        valid_intervals += 1
                    else:
                        invalid_intervals += 1
                        for issue in result.issues:
                            issue_counts[f"lineup.validation.{issue.code}"] += 1

            for record in bundle.three_sixty:
                raw_frames += 1
                try:
                    frame = frame_normalizer.normalize(record)
                except NormalizationError as exc:
                    invalid_frames += 1
                    issue_counts[f"360.normalization.{exc.code}"] += 1
                    continue
                result = frame_validator.validate(
                    frame, known_event_ids=event_ids
                )
                if result.is_valid:
                    valid_frames += 1
                else:
                    invalid_frames += 1
                    for issue in result.issues:
                        issue_counts[f"360.validation.{issue.code}"] += 1
    except (RawDataError, KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    reconciled = (
        raw_intervals == valid_intervals + invalid_intervals
        and raw_frames == valid_frames + invalid_frames
    )
    print(f"matches: {len(match_ids)}")
    print(f"raw_lineup_intervals: {raw_intervals}")
    print(f"valid_lineup_intervals: {valid_intervals}")
    print(f"invalid_lineup_intervals: {invalid_intervals}")
    print(f"raw_360: {raw_frames}")
    print(f"valid_360: {valid_frames}")
    print(f"invalid_360: {invalid_frames}")
    print(f"reconciled: {reconciled}")
    for code, count in sorted(issue_counts.items()):
        print(f"  {code}: {count}")
    for flag, count in sorted(quality_flags.items()):
        print(f"  quality.{flag}: {count}")

    if args.strict and (not reconciled or invalid_intervals or invalid_frames):
        sys.exit(1)


if __name__ == "__main__":
    main()
