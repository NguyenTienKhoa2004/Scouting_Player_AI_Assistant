"""Browse StatsBomb event JSON files in manageable pages.

Examples:
    python scripts/tools/view_events.py 3857276
    python scripts/tools/view_events.py 3857276 --page-size 100 --type Shot
    python scripts/tools/view_events.py 3857276 --raw --once
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "statsbomb-open-data"
    / "data"
    / "events"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View a StatsBomb event file 100 events at a time."
    )
    parser.add_argument(
        "match_id",
        help="StatsBomb match ID, for example 3857276.",
    )
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=DEFAULT_EVENTS_DIR,
        help=f"Directory containing event JSON files (default: {DEFAULT_EVENTS_DIR}).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Number of events per page (default: 100).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Start at this 1-based event number after filtering (default: 1).",
    )
    parser.add_argument(
        "--type",
        dest="event_type",
        help="Only show one event type, for example Shot or Pass.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print complete JSON objects for each page instead of summaries.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one page and exit without opening the interactive prompt.",
    )
    args = parser.parse_args()

    if args.page_size < 1:
        parser.error("--page-size must be at least 1")
    if args.start < 1:
        parser.error("--start must be at least 1")

    return args


def load_events(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            events = json.load(file)
    except FileNotFoundError as exc:
        raise SystemExit(f"Event file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(events, list):
        raise SystemExit(f"Expected a JSON array in {path}")
    return events


def nested_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name", "-"))
    return "-"


def format_location(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 2:
        return "-"
    return f"({value[0]}, {value[1]})"


def event_outcome(event: dict[str, Any]) -> str:
    """Find the source outcome without flattening away its event-specific meaning."""
    ignored = {
        "type",
        "team",
        "player",
        "position",
        "possession_team",
        "play_pattern",
    }
    for key, value in event.items():
        if key in ignored or not isinstance(value, dict):
            continue
        outcome = value.get("outcome")
        if isinstance(outcome, dict) and outcome.get("name"):
            return f"{key}.{outcome['name']}"

    # In StatsBomb, a Pass without pass.outcome is normally a completed pass.
    if nested_name(event.get("type")) == "Pass":
        return "pass.Complete"
    return "-"


def event_xg(event: dict[str, Any]) -> str:
    shot = event.get("shot")
    if not isinstance(shot, dict) or shot.get("statsbomb_xg") is None:
        return "-"
    return str(shot["statsbomb_xg"])


def summary_line(number: int, event: dict[str, Any]) -> str:
    source_index = event.get("index", "-")
    period = event.get("period", "-")
    timestamp = event.get("timestamp", "-")
    minute = event.get("minute", "-")
    second = event.get("second", "-")
    event_type = nested_name(event.get("type"))
    team = nested_name(event.get("team"))
    player = nested_name(event.get("player"))
    location = format_location(event.get("location"))
    outcome = event_outcome(event)
    xg = event_xg(event)

    return (
        f"[{number:04d}] idx={source_index} P{period} {timestamp} "
        f"({minute}:{second}) | {event_type} | team={team} | player={player} "
        f"| loc={location} | outcome={outcome} | xG={xg}"
    )


def print_page(
    events: list[dict[str, Any]], offset: int, page_size: int, raw: bool
) -> None:
    page = events[offset : offset + page_size]
    first = offset + 1
    last = offset + len(page)

    print(f"\nEvents {first}-{last} of {len(events)}")
    print("=" * 80)

    if raw:
        print(json.dumps(page, ensure_ascii=False, indent=2))
        return

    for number, event in enumerate(page, start=first):
        print(summary_line(number, event))


def show_one_event(events: list[dict[str, Any]], number: int) -> None:
    if number < 1 or number > len(events):
        print(f"Event number must be between 1 and {len(events)}.")
        return
    print(json.dumps(events[number - 1], ensure_ascii=False, indent=2))


def interactive_browser(
    events: list[dict[str, Any]], start: int, page_size: int, raw: bool
) -> None:
    offset = min(start - 1, max(0, len(events) - 1))

    while True:
        print_page(events, offset, page_size, raw)
        print(
            "\nEnter/n: next | p: previous | j NUMBER: full JSON "
            "| r: toggle page raw/summary | q: quit"
        )
        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if command in {"", "n"}:
            if offset + page_size >= len(events):
                print("Already at the last page.")
            else:
                offset += page_size
        elif command == "p":
            offset = max(0, offset - page_size)
        elif command == "r":
            raw = not raw
        elif command == "q":
            return
        elif command.startswith("j "):
            try:
                show_one_event(events, int(command.split(maxsplit=1)[1]))
            except ValueError:
                print("Usage: j NUMBER, for example: j 42")
            input("Press Enter to return to the page...")
        else:
            print("Unknown command. Use n, p, j NUMBER, r, or q.")


def main() -> None:
    args = parse_args()
    event_path = args.events_dir / f"{args.match_id}.json"
    events = load_events(event_path)

    if args.event_type:
        wanted = args.event_type.casefold()
        events = [
            event
            for event in events
            if nested_name(event.get("type")).casefold() == wanted
        ]

    if not events:
        suffix = f" of type {args.event_type!r}" if args.event_type else ""
        raise SystemExit(f"No events{suffix} found in {event_path}")
    if args.start > len(events):
        raise SystemExit(
            f"--start is {args.start}, but only {len(events)} events are available."
        )

    print(f"File: {event_path}")
    if args.event_type:
        print(f"Filter: type={args.event_type}")

    if args.once or not sys.stdin.isatty():
        print_page(events, args.start - 1, args.page_size, args.raw)
        return

    interactive_browser(events, args.start, args.page_size, args.raw)


if __name__ == "__main__":
    main()
