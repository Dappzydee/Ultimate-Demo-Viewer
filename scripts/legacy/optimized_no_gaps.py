"""
vision_coloring.py

Parses a CS2 .dem file, extracts one player's position/view-angle data for a
given time window, and for each sampled tick tests EVERY triangle within the
player's FOV cone and max distance individually (one ray per triangle, aimed
at its center) to determine true visibility - not a fixed angular ray grid,
so thin/angled geometry (railings, staircases) doesn't fall through sampling
gaps. Exports a colored .glb showing which triangles the player saw.

Usage:
    python vision_coloring.py demo.dem --player "donk" --start 1:00 --end 1:20 --round 1

Requires:
    pip install awpy trimesh numpy polars
    awpy get tris   (once, to download .tri files)
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


def compute_forward_vector(yaw_deg, pitch_deg):
    """Source engine view direction from yaw/pitch, matching the convention
    used elsewhere in this script (positive pitch = looking down)."""
    yr, pr = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([
        np.cos(pr) * np.cos(yr),
        np.cos(pr) * np.sin(yr),
        -np.sin(pr),
    ])


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
    parser.add_argument("--max-distance", type=float, default=4000.0, help="Max distance (in map units) to test triangles at (default: 4000)")
    parser.add_argument("--tick-step", type=int, default=4, help="Process every Nth tick to save time (default: 4)")
    parser.add_argument("--out", type=str, default="seen_result.glb", help="Output .glb path")
    args = parser.parse_args()

    demo_path = Path(args.demo)
    if not demo_path.exists():
        sys.exit(f"Demo file not found: {demo_path}")

    print(f"Parsing demo: {demo_path} ...")
    dem = Demo(str(demo_path), verbose=False)
    dem.parse(player_props=["X", "Y", "Z", "pitch", "yaw"], events=["round_start", "round_freeze_end", "round_officially_ended"])

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

    # Precompute triangle centers once - these are the actual test targets,
    # one explicit visibility check per triangle, rather than hoping a
    # sparse ray grid happens to land on every triangle.
    triangle_centers = mesh.triangles_center  # shape: (num_triangles, 3)
    num_triangles = len(triangle_centers)

    half_fov_rad = np.radians(args.fov / 2.0)
    cos_half_fov = np.cos(half_fov_rad)

    seen_triangle_ids = set()
    rows = player_ticks.to_pandas().iloc[::args.tick_step]
    total_ticks = len(rows)
    tick_rows = list(rows.itertuples(index=False))

    for i, row in enumerate(tick_rows):
        eye_pos = np.array([row.X, row.Y, row.Z + EYE_HEIGHT_OFFSET])
        forward = compute_forward_vector(row.yaw, row.pitch)
        forward = forward / np.linalg.norm(forward)

        # Vector + distance from eye to every triangle center
        to_centers = triangle_centers - eye_pos  # (N, 3)
        distances = np.linalg.norm(to_centers, axis=1)
        distances_safe = np.where(distances == 0, 1e-6, distances)
        directions_unit = to_centers / distances_safe[:, None]

        # Angle-to-forward filter (FOV cone) via dot product, plus max-distance filter
        cos_angle = directions_unit @ forward
        in_fov = cos_angle >= cos_half_fov
        in_range = distances <= args.max_distance
        candidate_mask = in_fov & in_range
        candidate_indices = np.nonzero(candidate_mask)[0]

        if len(candidate_indices) == 0:
            continue

        # One explicit ray per candidate triangle, aimed at its center.
        origins = np.tile(eye_pos, (len(candidate_indices), 1))
        directions = directions_unit[candidate_indices]

        hit_tri_ids = mesh.ray.intersects_first(origins, directions)

        # A triangle only counts as seen if the ray aimed at its center
        # actually hit that SAME triangle first (i.e. nothing else, like
        # a nearer wall, blocked the line of sight).
        truly_visible = candidate_indices[hit_tri_ids == candidate_indices]
        seen_triangle_ids.update(truly_visible.tolist())

        print(
            f"  tick {i + 1}/{total_ticks}: {len(candidate_indices)} candidates in FOV/range, "
            f"{len(truly_visible)} confirmed visible, {len(seen_triangle_ids)} unique triangles seen so far"
        )

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