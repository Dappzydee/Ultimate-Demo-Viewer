"""Command-line interface for static-map flash coverage simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import resolve_tri_path
from .flash_coverage import FlashCoverageAnalyzer, FlashCoverageConfig, export_flash_coverage
from .flash_events import extract_flash_detonations, load_flash_detonation_json
from .geometry import load_tri_mesh
from .paths import prepare_output_path, resolve_input_path
from .progress import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate static-map coverage from one selected flashbang_detonate event.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", type=Path, help="Demo containing the flash event")
    source.add_argument("--flash-json", type=Path, help="One event object or JSON list made by cs2-flash-events")
    parser.add_argument("--flash-index", type=int, help="Event index to select from --demo or a JSON list")
    parser.add_argument("--map", dest="map_name", help="Map name override (required when the selected JSON has no map_name)")
    parser.add_argument("--tri", type=Path, help="Explicit Awpy .tri geometry file")
    parser.add_argument("--max-distance", type=float, default=1500.0, help="Coverage radius in map units (default: 1500)")
    parser.add_argument("--falloff-power", type=float, default=1.0, help="Distance falloff exponent; higher means faster fade")
    parser.add_argument("--samples-per-triangle", type=int, choices=(1, 4), default=4, help="Interior samples per face (default: 4)")
    parser.add_argument("--ray-batch-size", type=int, default=250000, help="Maximum rays per CPU/GPU dispatch")
    parser.add_argument("--marker-radius", type=float, default=24.0, help="Radius of the bright, selectable flash marker sphere (default: 24)")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU raycasting")
    parser.add_argument("--out", type=Path, default=Path("out/flash_coverage.glb"), help="GLB output path (relative paths are placed in out/)")
    parser.add_argument("--verbose", action="store_true", help="Show event selection and raycasting diagnostics")
    parser.add_argument("--traceback", action="store_true", help="Show the complete Python traceback for an error")
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    args = parser.parse_args()
    configure_logging(args.verbose)
    try:
        if args.demo:
            if args.flash_index is None:
                raise ValueError("--flash-index is required with --demo. Run cs2-flash-events first to list indexes.")
            flashes = extract_flash_detonations(args.demo, not args.no_progress)
            flash = next((item for item in flashes if item.index == args.flash_index), None)
            if flash is None:
                raise ValueError(f"Flash index {args.flash_index} was not found in {args.demo}.")
        else:
            flash_json_path = resolve_input_path(args.flash_json)
            flash = load_flash_detonation_json(flash_json_path, args.flash_index)
        map_name = args.map_name or flash.map_name
        if not map_name:
            raise ValueError("Map name is unavailable in the event JSON; supply --map.")
        tri_path = resolve_tri_path(map_name, args.tri)
        config = FlashCoverageConfig(args.max_distance, args.samples_per_triangle, args.falloff_power, args.ray_batch_size, not args.no_gpu)
        print(f"Selected flash {flash.index}: tick {flash.tick}, position {flash.position}; loading {map_name} geometry...")
        analyzer = FlashCoverageAnalyzer(load_tri_mesh(tri_path), config)
        print(f"Simulating one selected flash with {analyzer.raycaster.name}...")
        result = analyzer.analyze(flash, not args.no_progress)
        output_path = prepare_output_path(args.out)
        export_flash_coverage(analyzer.mesh, result.intensities, flash, output_path, args.marker_radius)
        print(f"Exported {int((result.intensities > 0).sum())} affected faces to {output_path.resolve()}.")
        print(f"Tested {result.tested_rays} rays using {result.backend}. This is an occlusion-and-distance simulation, not Valve-exact blind duration.")
    except (ValueError, FileNotFoundError) as error:
        if args.traceback:
            raise
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
