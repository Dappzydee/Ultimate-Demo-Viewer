# Architecture guide

## Purpose

This repository is a local CS2 demo-analysis application plus reusable command-line visualizations. The integrated app parses a demo once, retains normalized data and map analysis state, and sends compact result arrays to the WebGL viewer. GLB export is an optional output seam rather than the internal application transport.

## Package layout

`cs2_visibility/` is the application library.

- `models.py`: stable public dataclasses and configuration values.
- `analysis.py`: player-vision orchestration and common demo-window helpers.
- `geometry.py`: Awpy `.tri` loading and face sample generation.
- `raycasting.py`: the common CPU/NVIDIA Warp raycasting contract.
- `flash_events.py`: normalized extraction of `flashbang_detonate` events and JSON interchange.
- `flash_coverage.py`: hypothetical static-map flash coverage calculation and intensity GLB export.
- `grenade_lineups.py`: patch-aware grenade release parsing plus pin-pull/detonation matching, pure stand/aim derivation, movement paths, throw classification, and JSON/command export.
- `progress.py`: terminal progress and verbose logging helpers. Long-running calculations must use these rather than raw `print` calls.
- `session.py`: persistent normalized demo data, cached geometry/raycasting state, surface picking, and `.cs2session` interchange.
- `interchange.py`: packed binary vision timelines and quantized flash-result transport.
- `cli.py` and feature-specific entry points: argument parsing only; computation belongs in the library.

## Data flow

```text
.dem -> Awpy parser -> DemoSession -> normalized poses/events
                         |        -> cached MapAnalysisContext -> Warp or CPU
                         |        -> derived grenade lineups -> viewer markers/paths/commands
                         |        -> packed masks/intensities -> WebGL buffers
                         |        -> optional GLB snapshot
                         `--------> optional self-contained .cs2session
```

`viewer.py` owns the local HTTP API and background job state. The browser frontend never invokes a CLI subprocess and never requires a temporary GLB for an analysis. Geometry is loaded once; later results preserve its face order and update a dynamic GPU attribute.

When CUDA is available, `WarpRaycaster` also owns a fused analysis path. Canonically ordered interior samples are uploaded once per sample-count setting. Vision kernels perform candidate filtering and face ray queries, retain cumulative per-sample state on the device, and atomically pack per-face results. Poses are queued in chunks before synchronization, so CUDA work is not serialized behind CPU array construction. Flash kernels similarly combine range filtering, ray queries, falloff, and per-face reduction. The generic `visible()` contract remains the correctness reference and CPU fallback; custom sample layouts automatically use that path.

## Time and replay contracts

An instant vision analysis selects the nearest valid pose to the requested round-relative time. An interval samples ordered poses between its endpoints. Timeline output contains two packed face bitsets per frame: visibility at that frame and visibility accumulated from the interval start. This keeps playback compact and allows switching modes without rerunning raycasts.

## Session archives

`.cs2session` is a versioned ZIP container with JSON metadata plus compressed NumPy arrays for normalized poses and map geometry. It may also contain the current packed analysis result. It deliberately does not embed the original demo. A reopened session can rerun analyses that need the captured fields, but a future visualization requiring new demo properties still needs the source `.dem`.

Schema-v3 archives store already-derived lineup records, configured round clocks, and per-player life-window end ticks in manifest metadata, so previews and Lineups work without the original demo. Older archive schemas are intentionally rejected.

Keep Awpy dataframe column naming inside the parsing modules. All other modules must work with typed records and NumPy arrays, so Awpy schema changes have one contained adaptation point.

## Grenade lineup model

`weapon_fire` supplies the physical grenade release tick and grenade type. The parser adapter discovers button, walking, ground, and direct-velocity properties through `list_updated_fields`; unavailable velocity is derived from exact player-tick positions. Event-attached player properties may lag by one snapshot, so the exact tick stream owns release position and angles while the event remains authoritative for release identity and timing.

The pure derivation layer searches backward for a stationary stretch followed by sustained movement. A fixed reference survives only when pitch and wrapped yaw remain stable; otherwise it exports the lookback path and uses the release pose for static commands. Click strength, horizontal movement, and jumpthrow remain separate fields. No map geometry or raycasting is required for this analysis.

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
