"""Validate PostgreSQL inputs consumed by SPADL conversion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matchmind.feature_engineering import SpadlInputContractError  # noqa: E402
from matchmind.storage.postgres.feature_reader import (  # noqa: E402
    PostgresSpadlInputReader,
)
from matchmind.storage.postgres.ingestion_writer import connect  # noqa: E402


DEFAULT_DATABASE_URL = "postgresql://matchmind:1234567@localhost:5433/matchmind"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast validation of inputs required by SPADL conversion."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--match-id",
        type=int,
        action="append",
        dest="match_ids",
        help="Validate selected matches; repeat for multiple matches.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable validation report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with connect(args.database_url) as connection:
            inputs = PostgresSpadlInputReader(connection).load(args.match_ids)
        report = inputs.report
    except SpadlInputContractError as exc:
        report = exc.report
        _print_report(report.as_dict(), as_json=args.json)
        raise SystemExit(1) from exc

    _print_report(report.as_dict(), as_json=args.json)


def _print_report(report: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    status = "PASS" if report["valid"] else "FAIL"
    print(f"spadl_input: {status}")
    for field in (
        "match_count",
        "event_count",
        "lineup_interval_count",
        "three_sixty_count",
        "events_with_subtype",
    ):
        print(f"{field}: {report[field]}")
    for issue in report["issues"]:  # type: ignore[union-attr]
        code = issue["code"]  # type: ignore[index]
        entity = issue["entity"]  # type: ignore[index]
        message = issue["message"]  # type: ignore[index]
        print(f"  {code} [{entity}]: {message}")
    for warning in report["warnings"]:  # type: ignore[union-attr]
        code = warning["code"]  # type: ignore[index]
        entity = warning["entity"]  # type: ignore[index]
        message = warning["message"]  # type: ignore[index]
        print(f"  WARNING {code} [{entity}]: {message}")


if __name__ == "__main__":
    main()
