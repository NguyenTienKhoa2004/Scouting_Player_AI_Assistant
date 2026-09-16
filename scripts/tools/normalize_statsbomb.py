"""Run StatsBomb-to-canonical normalization without writing to PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "matchmind" / "src"))

from matchmind.ingestion import (  # noqa: E402
    NormalizationError,
    RawDataError,
    StatsBombEventNormalizer,
    StatsBombRawReader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize one match or the full pinned StatsBomb dataset."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--match-id", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Print the first N normalized events as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any event cannot be normalized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show < 0:
        raise SystemExit("--show must be zero or greater")

    reader = StatsBombRawReader()
    normalizer = StatsBombEventNormalizer()
    raw_records = (
        reader.iter_events()
        if args.all
        else iter(reader.read_events(args.match_id))
    )

    raw_count = 0
    normalized_count = 0
    errors: Counter[str] = Counter()
    shown = 0

    try:
        for record in raw_records:
            raw_count += 1
            try:
                event = normalizer.normalize(record)
            except NormalizationError as exc:
                errors[exc.code] += 1
                if sum(errors.values()) <= 10:
                    print(f"ERROR {exc}", file=sys.stderr)
                continue

            normalized_count += 1
            if shown < args.show:
                print(
                    json.dumps(
                        event.as_serializable_dict(), ensure_ascii=False, indent=2
                    )
                )
                shown += 1
    except (RawDataError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"raw_events: {raw_count}")
    print(f"normalized_events: {normalized_count}")
    print(f"normalization_errors: {sum(errors.values())}")
    for code, count in sorted(errors.items()):
        print(f"  {code}: {count}")

    if args.strict and errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
