"""Smoke-test and inspect the reusable StatsBomb raw reader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchmind.ingestion import RawDataError, StatsBombRawReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read one raw StatsBomb match bundle.")
    parser.add_argument("match_id", type=int, help="StatsBomb match ID")
    parser.add_argument(
        "--show-events",
        type=int,
        default=0,
        help="Print the first N event payloads as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.show_events < 0:
        raise SystemExit("--show-events must be zero or greater")

    try:
        bundle = StatsBombRawReader().read_match_bundle(args.match_id)
    except (RawDataError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    match = bundle.match.payload
    home_team = match.get("home_team", {}).get("home_team_name", "unknown")
    away_team = match.get("away_team", {}).get("away_team_name", "unknown")
    print(f"match_id: {bundle.match.match_id}")
    print(f"match: {home_team} vs {away_team}")
    print(f"lineup objects: {len(bundle.lineups)}")
    print(f"events: {len(bundle.events)}")

    for record in bundle.events[: args.show_events]:
        print(
            json.dumps(
                {
                    "source_file": str(record.source_file),
                    "source_record_index": record.source_record_index,
                    "match_id": record.match_id,
                    "payload": record.payload,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
