"""Normalize and validate StatsBomb events without writing to PostgreSQL."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchmind.data import (  # noqa: E402
    CanonicalEventValidator,
    EventValidationContext,
    NormalizationError,
    RawDataError,
    StatsBombEventNormalizer,
    StatsBombRawReader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and validate one match or the pinned dataset."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--match-id", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any event is invalid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = StatsBombRawReader()
    normalizer = StatsBombEventNormalizer()

    try:
        context = EventValidationContext.from_reader(reader)
        raw_records = (
            reader.iter_events()
            if args.all
            else iter(reader.read_events(args.match_id))
        )
        validator = CanonicalEventValidator(context)

        raw_count = 0
        normalized_count = 0
        valid_count = 0
        invalid_count = 0
        issue_counts: Counter[str] = Counter()

        for record in raw_records:
            raw_count += 1
            try:
                event = normalizer.normalize(record)
            except NormalizationError as exc:
                invalid_count += 1
                issue_counts[f"normalization.{exc.code}"] += 1
                if invalid_count <= 10:
                    print(f"INVALID {exc}", file=sys.stderr)
                continue

            normalized_count += 1
            result = validator.validate(event)
            if result.is_valid:
                valid_count += 1
                continue

            invalid_count += 1
            for issue in result.issues:
                issue_counts[f"validation.{issue.code}"] += 1
            if invalid_count <= 10:
                details = "; ".join(
                    f"{issue.code}: {issue.message}" for issue in result.issues
                )
                print(
                    f"INVALID {record.source_file}, "
                    f"record {record.source_record_index}: {details}",
                    file=sys.stderr,
                )
    except (RawDataError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"raw_events: {raw_count}")
    print(f"normalized_events: {normalized_count}")
    print(f"valid_events: {valid_count}")
    print(f"invalid_events: {invalid_count}")
    print(f"reconciled: {raw_count == valid_count + invalid_count}")
    for code, count in sorted(issue_counts.items()):
        print(f"  {code}: {count}")

    if args.strict and invalid_count:
        sys.exit(1)


if __name__ == "__main__":
    main()

