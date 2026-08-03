"""Command-line interface for grenade lineup extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .grenade_lineups import (
    GRENADE_WEAPONS,
    GrenadeLineupConfig,
    export_grenade_lineups,
    export_lineup_commands,
    extract_grenade_lineups,
    lineup_command_block,
)
from .paths import prepare_output_path
from .progress import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect grenade lineups and export stand/aim points, movement metadata, and CS2 commands.",
    )
    parser.add_argument("demo", type=Path, help="Path to a .dem file")
    parser.add_argument("--json", type=Path, help="Write structured lineup JSON (relative paths go in out/)")
    parser.add_argument("--commands", type=Path, help="Write paste-ready setpos/setang blocks (relative paths go in out/)")
    parser.add_argument(
        "--grenade-type", action="append", choices=sorted(set(GRENADE_WEAPONS.values())),
        help="Only include this grenade type; repeat to include several",
    )
    parser.add_argument("--tick-rate", type=float, default=64.0, help="Demo tick rate when it is absent from the header (default: 64)")
    parser.add_argument("--lookback-seconds", type=float, default=5.0, help="Seconds of player history before each release (default: 5)")
    parser.add_argument("--stationary-speed", type=float, default=8.0, help="Maximum horizontal speed in units/s treated as stationary")
    parser.add_argument("--ramp-speed", type=float, default=20.0, help="Minimum speed in a sustained movement ramp")
    parser.add_argument("--walking-speed", type=float, default=150.0, help="Speed fallback separating walking from running")
    parser.add_argument("--angle-tolerance", type=float, default=5.0, help="Maximum pitch/yaw drift for a fixed reference")
    parser.add_argument("--collapse-distance", type=float, default=32.0, help="Distance below which reference and release collapse to one point")
    parser.add_argument("--min-stationary-seconds", type=float, default=0.125, help="Minimum stationary stretch used as a reference")
    parser.add_argument("--sustained-movement-seconds", type=float, default=0.09375, help="Required movement ramp duration after a reference")
    parser.add_argument("--verbose", action="store_true", help="Show discovered parser fields and diagnostics")
    parser.add_argument("--traceback", action="store_true", help="Show the complete Python traceback for an error")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    args = parser.parse_args()
    if not args.demo.exists():
        raise SystemExit(f"Demo file not found: {args.demo}")
    configure_logging(args.verbose)
    try:
        config = GrenadeLineupConfig(
            tick_rate=args.tick_rate,
            lookback_seconds=args.lookback_seconds,
            stationary_speed=args.stationary_speed,
            movement_speed=args.ramp_speed,
            walking_speed_threshold=args.walking_speed,
            angle_stability_degrees=args.angle_tolerance,
            collapse_distance=args.collapse_distance,
            min_stationary_seconds=args.min_stationary_seconds,
            sustained_movement_seconds=args.sustained_movement_seconds,
        )
        lineups = extract_grenade_lineups(
            args.demo, config, grenade_types=set(args.grenade_type or ()), show_progress=not args.no_progress,
        )
    except (ValueError, OSError) as error:
        if args.traceback:
            raise
        raise SystemExit(f"Error: {error}") from error

    for index, lineup in enumerate(lineups, 1):
        print(lineup_command_block(lineup, index))
        if lineup.notes:
            print("// " + "; ".join(lineup.notes))
        print()
    print(f"Found {len(lineups)} grenade throws.")
    if args.json:
        output_path = prepare_output_path(args.json)
        export_grenade_lineups(lineups, output_path)
        print(f"Exported JSON to {output_path.resolve()}")
    if args.commands:
        output_path = prepare_output_path(args.commands)
        export_lineup_commands(lineups, output_path)
        print(f"Exported commands to {output_path.resolve()}")


if __name__ == "__main__":
    main()
