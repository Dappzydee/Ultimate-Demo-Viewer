"""
vision_coloring.py

Parses a CS2 .dem file, extracts one player's position/view-angle data for a
given time window, raycasts against the map's collision mesh, and exports a
colored .glb showing which triangles the player saw.

Usage:
    python vision_coloring.py demo.dem --player "zont1x" --start 1:00 --end 1:20 --round 1

Requires:
    pip install awpy trimesh numpy polars
    awpy get tris   (once, to download .tri files)
    A .obj export of the target map's .tri file (see tri_to_obj_converter.py)
    OR pass --tri directly and this script will convert it in-memory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import trimesh

from awpy import Demo
from awpy.data import TRIS_DIR
from awpy.visibility import VisibilityChecker


EYE_HEIGHT_OFFSET = 64.0  # CS2 stores feet/origin position; approx standing eye height


def parse_time_to_seconds(t: str) -> float:
    """Accepts 'MM:SS' or plain seconds like '80'."""
    if ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return float(t)


def tri_file_to_trimesh(tri_path: Path) -> trimesh.Trimesh:
    """Load an Awpy .tri file directly into a trimesh.Trimesh (skips the
    intermediate .obj step)."""
    tris = VisibilityChecker.read_tri_file(tri_path)

    vertices = []
    faces = []
    for i, tri in enumerate(tris):
        for p in (tri.p1, tri.p2, tri.p3):
            vertices.append([p.x, p.y, p.z])
        faces.append([i * 3, i * 3 + 1, i * 3 + 2])

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def build_view_rays(eye_pos, yaw_deg, pitch_deg, fov=90.0, grid=40):
    """Cast a grid of rays across the player's FOV cone."""
    directions = []
    half_fov = fov / 2.0
    for dy in np.linspace(-half_fov, half_fov, grid):
        for dx in np.linspace(-half_fov, half_fov, grid):
            yaw = yaw_deg + dx
            pitch = pitch_deg + dy
            yr, pr = np.radians(yaw), np.radians(pitch)
            # CS2/Source convention: yaw around Z, pitch tilts toward Z
            dir_vec = np.array([
                np.cos(pr) * np.cos(yr),
                np.cos(pr) * np.sin(yr),
                -np.sin(pr),  # positive pitch = looking down in Source engine
            ])
            directions.append(dir_vec)

    directions = np.array(directions)
    origins = np.tile(eye_pos, (len(directions), 1))
    return origins, directions


def main():
    parser = argparse.ArgumentParser(description="Color map geometry seen by a player in a CS2 demo.")
    parser.add_argument("demo", type=str, help="Path to .dem file")
    parser.add_argument("--player", type=str, required=True, help="Player name (as it appears in the demo)")
    parser.add_argument("--start", type=str, required=True, help="Start time, e.g. '1:00' or seconds")
    parser.add_argument("--end", type=str, required=True, help="End time, e.g. '1:20' or seconds")
    parser.add_argument("--round", type=int, default=1, help="Round number (default: 1)")
    parser.add_argument("--map", type=str, default=None, help="Map name override, e.g. de_dust2 (default: auto-detect from demo header)")
    parser.add_argument("--tri", type=str, default=None, help="Path to a .tri file (default: use Awpy's bundled tris dir + detected map)")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of view in degrees (default: 90)")
    parser.add_argument("--grid", type=int, default=40, help="Ray grid resolution per tick, NxN (default: 40)")
    parser.add_argument("--tick-step", type=int, default=4, help="Process every Nth tick to save time (default: 4)")
    parser.add_argument("--out", type=str, default="seen_result.glb", help="Output .glb path")
    args = parser.parse_args()

    demo_path = Path(args.demo)
    if not demo_path.exists():
        sys.exit(f"Demo file not found: {demo_path}")

    print(f"Parsing demo: {demo_path} ...")
    dem = Demo(str(demo_path), verbose=True)
    dem.parse(player_props=["X", "Y", "Z", "pitch", "yaw"], events=["round_start", "round_freeze_end", "round_officially_ended"])

    # for prop in ["X", "Y", "Z", "pitch", "yaw"]:
    #     try:
    #         dem.parse(player_props=[prop], events=[])
    #         print(f"{prop}: OK")
    #     except Exception as e:
    #         print(f"{prop}: FAILED - {e}")

    map_name = args.map or dem.header.get("map_name") or dem.header.get("map")
    if map_name is None:
        sys.exit("Could not auto-detect map name from demo header; pass --map explicitly.")
    print(f"Map detected: {map_name}")

    # Resolve tri file
    tri_path = Path(args.tri) if args.tri else TRIS_DIR / f"{map_name}.tri"
    if not tri_path.exists():
        sys.exit(f".tri file not found at {tri_path}. Run `awpy get tris` first, or pass --tri explicitly.")

    print(f"Loading collision mesh: {tri_path} ...")
    mesh = tri_file_to_trimesh(tri_path)

    # Filter ticks: correct round, correct player, correct time window
    ticks_df = dem.ticks

    rounds_df = dem.rounds
    round_row = rounds_df.filter(pl.col("round_num") == args.round)
    if round_row.is_empty():
        sys.exit(f"Round {args.round} not found in demo.")
    round_start_tick = round_row["freeze_end"][0] if "freeze_end" in round_row.columns else round_row["start"][0]

    tick_rate = dem.header.get("tick_rate") or dem.header.get("playback_ticks") or 64
    if not isinstance(tick_rate, (int, float)) or tick_rate <= 0:
        tick_rate = 64  # fallback, common for MM/community demos (use 128 for pro/FACEIT)

    start_sec = parse_time_to_seconds(args.start)
    end_sec = parse_time_to_seconds(args.end)
    start_tick = round_start_tick + int(start_sec * tick_rate)
    end_tick = round_start_tick + int(end_sec * tick_rate)

    player_ticks = ticks_df.filter(
        (pl.col("name") == args.player)
        & (pl.col("round_num") == args.round)
        & (pl.col("tick") >= start_tick)
        & (pl.col("tick") <= end_tick)
    ).sort("tick")

    if player_ticks.is_empty():
        available = ticks_df.filter(pl.col("round_num") == args.round)["name"].unique().to_list()
        sys.exit(
            f"No tick data found for player '{args.player}' in round {args.round} "
            f"between {args.start} and {args.end}.\nPlayers in this round: {available}"
        )

    print(f"Found {len(player_ticks)} ticks for {args.player}; sampling every {args.tick_step}th tick.")

    seen_triangle_ids = set()
    rows = player_ticks.to_pandas().iloc[::args.tick_step]

    for i, row in enumerate(rows.itertuples(index=False)):
        eye_pos = np.array([row.X, row.Y, row.Z + EYE_HEIGHT_OFFSET])
        yaw = row.yaw
        pitch = row.pitch

        origins, directions = build_view_rays(eye_pos, yaw, pitch, fov=args.fov, grid=args.grid)

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            origins, directions, multiple_hits=False
        )
        seen_triangle_ids.update(index_tri.tolist())

        if i % 10 == 0:
            print(f"  processed {i}/{len(rows)} sampled ticks, {len(seen_triangle_ids)} triangles seen so far")

    print(f"Total unique triangles seen: {len(seen_triangle_ids)} / {len(mesh.faces)}")

    # Color mesh: unseen = gray, seen = red
    colors = np.tile([160, 160, 160, 255], (len(mesh.faces), 1)).astype(np.uint8)
    if seen_triangle_ids:
        colors[list(seen_triangle_ids)] = [255, 0, 0, 255]
    mesh.visual.face_colors = colors

    out_path = Path(args.out)
    mesh.export(out_path)
    print(f"Exported: {out_path.resolve()}")
    print("Open this in Blender via File -> Import -> glTF 2.0, or any glTF viewer.")


if __name__ == "__main__":
    main()