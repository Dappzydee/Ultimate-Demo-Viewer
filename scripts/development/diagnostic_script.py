"""
inspect_demo.py

Quick diagnostic: lists rounds, their tick boundaries, and the corresponding
real-world seconds-into-round, so you know what values are actually valid
to pass as --round/--start/--end to vision_coloring.py.

Usage:
    python inspect_demo.py demo_dust2.dem
"""

import sys
from pathlib import Path

import polars as pl
from awpy import Demo


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python inspect_demo.py <demo.dem>")

    demo_path = Path(sys.argv[1])
    if not demo_path.exists():
        sys.exit(f"File not found: {demo_path}")

    print(f"Parsing {demo_path} ...")
    dem = Demo(str(demo_path), verbose=False)
    dem.parse(
        player_props=["X", "Y", "Z", "pitch", "yaw"],
        events=["round_start", "round_freeze_end", "round_officially_ended"],
    )

    tick_rate = dem.header.get("tick_rate") or 64
    print(f"\nMap: {dem.header.get('map_name') or dem.header.get('map')}")
    print(f"Tick rate: {tick_rate}")

    rounds_df = dem.rounds.to_pandas()
    print(f"\nColumns in dem.rounds: {list(rounds_df.columns)}")
    print(f"\nTotal rounds found: {len(rounds_df)}\n")

    # Figure out which column marks round start (varies by Awpy version)
    start_col = "freeze_end" if "freeze_end" in rounds_df.columns else "start"

    for _, row in rounds_df.iterrows():
        round_num = row.get("round_num", "?")
        start_tick = row.get(start_col)
        end_tick = row.get("official_end") or row.get("end")

        if start_tick is None or end_tick is None:
            print(f"Round {round_num}: missing tick data, skipping")
            continue

        duration_sec = (end_tick - start_tick) / tick_rate
        print(
            f"Round {round_num}: ticks {start_tick} -> {end_tick} "
            f"({duration_sec:.1f}s long). Valid --start/--end range: 0:00 to "
            f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
        )

    # Show which players exist and their overall tick coverage
    print("\nPlayers found in ticks data:")
    ticks_df = dem.ticks.to_pandas()
    for name, group in ticks_df.groupby("name"):
        print(f"  {name}: {len(group)} ticks total")


if __name__ == "__main__":
    main()