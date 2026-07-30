# Architecture guide

## Purpose

This repository is a collection of CS2 demo-analysis visualizations. Each visualization reads normalized demo data, runs a geometry calculation through a shared raycasting backend, and exports an inspectable artifact such as a GLB. Do not add one-off scripts containing parsing, geometry, and export logic together.

## Package layout

`cs2_visibility/` is the application library.

- `models.py`: stable public dataclasses and configuration values.
- `analysis.py`: player-vision orchestration and common demo-window helpers.
- `geometry.py`: Awpy `.tri` loading and face sample generation.
- `raycasting.py`: the common CPU/NVIDIA Warp raycasting contract.
- `flash_events.py`: normalized extraction of `flashbang_detonate` events and JSON interchange.
- `flash_coverage.py`: hypothetical static-map flash coverage calculation and intensity GLB export.
- `progress.py`: terminal progress and verbose logging helpers. Long-running calculations must use these rather than raw `print` calls.
- `cli.py` and feature-specific entry points: argument parsing only; computation belongs in the library.

## Data flow

```text
.dem -> Awpy parser -> normalized event/pose records -> visualization service
     -> MapGeometry (.tri) -> RaycastingBackend (Warp or CPU) -> GLB/JSON
```

Keep Awpy dataframe column naming inside the parsing modules. All other modules must work with typed records and NumPy arrays, so Awpy schema changes have one contained adaptation point.

## Visualization rules

1. Raycasting is shared infrastructure. A visualization must not call `trimesh.ray` or Warp directly.
2. Map faces are sampled with interior barycentric points, never exact vertices. Mesh vertices are shared edges and make face attribution unstable.
3. A visualization exports its model assumptions as CLI options and documentation. For example, flash coverage is an occlusion-and-distance simulation, not a claim of Valve-exact blind duration.
4. Long work must show progress. `--verbose` should add useful operational facts (event counts, backend, batch sizes, candidate rays), not per-ray noise.
5. Default commands operate on explicitly selected data. Bulk analysis requires an explicit future feature/flag.

## Adding a visualization

Create a focused module that exposes a configuration dataclass, an analyzer class/function, and an export function. Reuse `MapGeometry` helpers and `RaycastingBackend`. Add a thin command-line wrapper, document its inputs/assumptions, and provide a synthetic CPU smoke test when a real demo is unavailable.

## Flash coverage model

The flash-event tool reads only `dem.events["flashbang_detonate"]`. Its JSON records contain the detonation tick, position, map, optional round/thrower metadata, and a stable extraction index.

The coverage tool accepts one event index or one JSON event record. It casts lines of sight from the detonation position to interior map-face samples. Visible samples receive a configurable distance falloff; a face uses its strongest visible sample. This answers where a stationary hypothetical player could be affected by a flash, assuming no smoke/player/dynamic-prop obstruction. It does not model view direction or reproduce Valve's blind-duration formula.
