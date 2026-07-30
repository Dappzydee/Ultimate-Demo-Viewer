"""Command-line interface for the reusable CS2 visibility library."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import VisibilityAnalyzer, export_colored_mesh, load_demo_window, parse_time_seconds, resolve_tri_path
from .models import AnalysisConfig
from .paths import prepare_output_path
from .progress import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export static CS2 map geometry seen by a player as a colored GLB.")
    parser.add_argument("demo", help="Path to a .dem file")
    parser.add_argument("--player", required=True, help="Player name exactly as it appears in the demo")
    parser.add_argument("--start", required=True, help="Window start after freeze end: seconds or MM:SS")
    parser.add_argument("--end", required=True, help="Window end after freeze end: seconds or MM:SS")
    parser.add_argument("--round", type=int, default=1, dest="round_number", help="Round number (default: 1)")
    parser.add_argument("--map", dest="map_name", help="Map name override, e.g. de_dust2")
    parser.add_argument("--tri", type=Path, help="Explicit Awpy .tri geometry file")
    parser.add_argument("--tick-rate", type=float, help="Override the demo header tick rate when it is absent or incorrect")
    parser.add_argument("--tick-step", type=int, default=4, help="Analyze every Nth player tick (default: 4)")
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal field of view in degrees (default: 90)")
    parser.add_argument("--max-distance", type=float, default=4000.0, help="Maximum tested distance in map units")
    parser.add_argument("--samples-per-triangle", type=int, choices=(1, 4), default=4, help="Interior samples per face (default: 4)")
    parser.add_argument("--min-visible-samples", type=int, default=1, help="Samples that must be visible before a face is colored red")
    parser.add_argument("--ray-batch-size", type=int, default=250000, help="Maximum rays per CPU/GPU dispatch")
    parser.add_argument("--eye-height", type=float, default=64.0, help="Standing eye height above player origin")
    parser.add_argument("--crouch-eye-height", type=float, default=46.0, help="Crouched eye height above player origin")
    parser.add_argument("--no-gpu", action="store_true", help="Force the CPU raycaster")
    parser.add_argument("--verbose", action="store_true", help="Show raycasting and parser diagnostics")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    parser.add_argument("--out", type=Path, default=Path("out/seen_result.glb"), help="GLB output path (relative paths are placed in out/)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    demo_path = Path(args.demo)
    if not demo_path.exists():
        raise SystemExit(f"Demo file not found: {demo_path}")
    try:
        start, end = parse_time_seconds(args.start), parse_time_seconds(args.end)
        print(f"Loading player poses from {demo_path}...")
        poses, detected_map, tick_rate = load_demo_window(demo_path, args.player, args.round_number, start, end, args.tick_step, args.tick_rate, args.eye_height, args.crouch_eye_height)
        map_name = args.map_name or detected_map
        tri_path = resolve_tri_path(map_name, args.tri)
        config = AnalysisConfig(args.fov, args.max_distance, args.samples_per_triangle, args.min_visible_samples, args.ray_batch_size, not args.no_gpu)
        print(f"Loading {map_name} geometry from {tri_path}...")
        analyzer = VisibilityAnalyzer.from_tri_file(tri_path, config)
        print(f"Analyzing {len(poses)} poses at {tick_rate:g} ticks/s with {analyzer.raycaster.name}...")
        result = analyzer.analyze(poses, not args.no_progress)
        output_path = prepare_output_path(args.out)
        export_colored_mesh(analyzer.mesh, result.seen_mask, output_path)
        print(f"Exported {result.seen_mask.sum()} / {len(result.seen_mask)} seen faces to {output_path.resolve()}.")
        print(f"Processed {result.processed_poses} poses and {result.tested_rays} visibility rays using {result.backend}.")
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
