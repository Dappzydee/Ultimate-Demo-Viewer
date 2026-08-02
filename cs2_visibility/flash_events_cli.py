"""Command-line interface for extracting flashbang detonation events."""

from __future__ import annotations

import argparse
from pathlib import Path

from .flash_events import export_flash_detonations, extract_flash_detonations
from .paths import prepare_output_path
from .progress import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="List exact flashbang_detonate events and optionally export them as JSON.")
    parser.add_argument("demo", type=Path, help="Path to a .dem file")
    parser.add_argument("--json", type=Path, help="Write normalized detonation events to JSON (relative paths are placed in out/)")
    parser.add_argument("--verbose", action="store_true", help="Show parser schema and extraction diagnostics")
    parser.add_argument("--traceback", action="store_true", help="Show the complete Python traceback for an error")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    args = parser.parse_args()
    if not args.demo.exists():
        raise SystemExit(f"Demo file not found: {args.demo}")
    configure_logging(args.verbose)
    try:
        flashes = extract_flash_detonations(args.demo, not args.no_progress)
    except ValueError as error:
        if args.traceback:
            raise
        raise SystemExit(f"Error: {error}") from error
    print("index  tick       position (X, Y, Z)                 round  thrower")
    print("-----  ---------  ----------------------------------  -----  -------")
    for flash in flashes:
        x, y, z = flash.position
        print(f"{flash.index:>5}  {flash.tick:>9}  ({x:>8.2f}, {y:>8.2f}, {z:>8.2f})  {str(flash.round_number or '-'):>5}  {flash.thrower or '-'}")
    print(f"Found {len(flashes)} flashbang detonation events.")
    if args.json:
        output_path = prepare_output_path(args.json)
        export_flash_detonations(flashes, output_path)
        print(f"Exported JSON to {output_path.resolve()}")


if __name__ == "__main__":
    main()
