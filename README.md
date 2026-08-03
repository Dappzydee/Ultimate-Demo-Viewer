# CS2 Demo Analyzer and 3D Viewer

This project provides one local application for loading a CS2 demo, running player-vision or flash-coverage simulations, replaying vision over time, and inspecting the result in 3D. It also includes a grenade-lineup extractor that produces stand/aim references and CS2 console commands. Analysis results stay in memory; a `.glb` is generated only when you explicitly export a snapshot.

It analyzes static Awpy `.tri` map geometry. It is not a literal frame-by-frame reconstruction: smoke occlusion, Valve-exact flash blinding, player models, dynamic props/doors, scope rendering, spectator state, and screen/UI obstruction are outside this analyzer's scope.

## Run the integrated app

```powershell
python viewer.py
```

The command opens a local browser window. Use **Open demo** or drop a `.dem` onto the app, then:

1. Choose the exclusive **Vision**, **Flash**, or **Lineups** task tab.
2. In Vision, select a round, player, and instant or interval/replay time.
3. In Flash, select a recorded flash or switch to **Placed flash** and enable **Move flash**.
4. Click the task-specific **Analyze vision** or **Analyze flash** button and inspect the result immediately.

The **Lineups** tab needs no Analyze button: grenade throws are detected while the demo loads. Filter them by round, grenade, player, or fixed/in-motion reference; select one to see its reference and release markers, movement path, recorded aim ray, instructions, warnings, and console commands. **View aim** enters the recorded player viewpoint, **Copy commands** copies a one-line `setpos`/`setang` sequence, and the filtered result can be downloaded as JSON or CFG. Lineups are preserved in newly saved `.cs2session` archives; older sessions explain that their original demo must be reopened.

The parsed demo, map geometry, interior face samples, and CPU/GPU raycaster remain available for repeated analyses. Identical requests reuse their existing result. With NVIDIA Warp, static samples stay in GPU memory while distance/FOV filtering, raycasting, cumulative sample tracking, face reduction, and bit packing run in fused CUDA kernels; only compact masks or intensities return to the CPU. This optimization does not reduce the configured sample count, pose cadence, range, or visibility threshold. Vision replays can switch between the player's current view and accumulated visibility while an optional position-and-facing marker follows the analyzed player. Flash analysis supports demo events and an interactive placed flash; no coordinate entry is required. A placed flash can copy any recorded pop position from the team-grouped selector, then **Move flash** keeps it centered in view and moves it relative to the camera with WASD, Q/E for height, and Shift for faster movement. The viewport shows throttled one-sample coverage previews while moving and after stopping. These previews are transient and do not fill Analysis history; the configured full-quality calculation runs only when **Analyze flash** is clicked. Selecting a recorded flash can automatically focus it, and Flash camera places the viewpoint at its exact position for look-around inspection.

Completed analyses appear in **Analysis history** and can be reopened, renamed, pinned, discarded, or exported as GLB. Discarded items move into a separate box where they can be inspected, exported, restored, or permanently deleted; they are excluded from saved sessions. The app retains at most 20 unpinned results across active and discarded history and targets a 512 MB result budget. Pinned active results are protected and produce a warning if they push history beyond the budget. Map geometry is not duplicated per result—the history contains only compact face masks/intensities and replay poses.

You can also open a demo directly:

```powershell
python viewer.py match.dem
```

**Save session** writes every non-discarded Analysis history result into a self-contained `.cs2session` automatically. The archive contains normalized poses, events, geometry, and all active compact results, reopens without the original demo, and can run new analyses using the saved fields. Existing schema-v1 sessions with one saved result remain supported. Use **Export GLB** only when you want a displayed result or history entry as a portable snapshot.

The app uses NVIDIA Warp automatically when CUDA is available, otherwise it falls back to CPU raycasting. CPU analysis of full-map or long windows can be slow.

## Command-line exports

The original focused commands remain supported:

```powershell
python vision.py match.dem --player "donk" --round 1 --start 1:00 --end 1:20 --out seen_result.glb
```

Generated artifacts are written to `out/` by default. Relative `--out` and `--json` values are also placed in `out/`; use an absolute path only when you intentionally want output elsewhere.

If a demo header has no usable tick rate, use an explicit override:

```powershell
python vision.py match.dem --player "donk" --round 1 --start 0 --end 20 --tick-rate 128
```

Before running, install dependencies and download Awpy geometry once:

```powershell
python -m pip install -r requirements.txt
awpy get tris
```

## Accuracy controls

By default the analyzer tests four **interior** points per face: its center and three points near, but never on, the corners. This avoids shared-edge ambiguity from the older vertex tests.

```powershell
# Require two different interior samples to be visible before a face is red.
python vision.py match.dem --player "donk" --round 1 --start 0 --end 20 --min-visible-samples 2
```

`--samples-per-triangle 1` is faster but less resilient to small occluders. `--samples-per-triangle 4` is the recommended default. `--ray-batch-size` controls dispatch size for the CPU and generic reference ray paths. The fused CUDA analyzer retains its working data on the GPU and does not need a larger ray batch to achieve high throughput.

The demo's `duck_amount` property is used to interpolate between `--eye-height` (64 by default) and `--crouch-eye-height` (46 by default). If the property is unavailable, the program prints a warning and uses the standing height.

## Library API

The reusable package is `cs2_visibility`; `vision.py` is the direct command-line entry point for a checkout. A UI or larger application can load poses and run an analysis without invoking a subprocess:

```python
from pathlib import Path
from cs2_visibility import AnalysisConfig, VisibilityAnalyzer, export_colored_mesh, load_demo_window

poses, map_name, _ = load_demo_window(Path("match.dem"), "donk", 1, 0, 20, 4, None, 64, 46)
analyzer = VisibilityAnalyzer.from_tri_file(Path("de_dust2.tri"), AnalysisConfig())
result = analyzer.analyze(poses)
export_colored_mesh(analyzer.mesh, result.seen_mask, Path("seen_result.glb"))
```

## Flash coverage

List the exact `flashbang_detonate` pop positions in a demo and export them for later selection:

```powershell
python flash-events.py match.dem --json flashes.json
```

Then simulate coverage for exactly one selected event—never every flash by default:

```powershell
python flash-coverage.py --flash-json flashes.json --flash-index 12 --out flash_12.glb
```

The resulting GLB uses gray for no coverage, dark red for weak coverage, and light yellow for stronger coverage. Intensity is an occlusion-and-distance simulation, not Valve's exact blind-duration calculation. Both tools include progress bars; add `--verbose` for diagnostics or `--no-progress` for log-friendly operation.

The GLB also includes a separate bright yellow object named `flash_detonation_marker` at the exact pop position, so it can be selected and focused in Blender.

## Grenade lineup extraction

The integrated viewer exposes detected throws in its **Lineups** tab. For scripts or bulk processing, detect every grenade release from the command line and print a practicable stand/aim point plus movement and throw metadata:

```powershell
python grenade-lineups.py match.dem --json lineups.json --commands lineups.cfg
```

Relative exports are written to `out/`. Each JSON record contains the exact release pose, an optional fixed reference pose, a generated movement instruction, a movement path for in-motion fallbacks, separate click/movement/jumpthrow fields, velocity, notes, and ready-to-paste `setpos`/`setang` commands. Use `--grenade-type flashbang` (repeatable) to filter the result.

The extractor discovers current demoparser fields from the demo before parsing. This checkout's current demo exposes `m_nButtonDownMaskPrev`, `m_bIsWalking`, and `m_hGroundEntity`; ordinary velocity is derived from adjacent player positions because no usable direct player-velocity property is present. If a future patch omits the button mask, click type is reported as unknown instead of guessed. Left+right medium-strength throws are reported as `both`.

The commands use the current CS2 forms `setpos X Y Z` and `setang pitch yaw roll` and require `sv_cheats`. Static commands cannot reproduce a run, walk, jump timing, or crouch state, so those remain explicit human-readable instructions. A fixed reference is rejected when no stationary setup exists or the aim drifts beyond the configured tolerance; in that case the commands intentionally use the release point and the JSON includes the approach path.

## Inspect existing GLB results locally

The repository includes a dependency-free WebGL 2 viewer, so Blender is not required for routine inspection:

```powershell
python viewer.py out/seen_result.glb
```

Open a GLB on startup or use **Open GLB** in the integrated app:

- GPU-rendered face colors with lit, flat-color, and normal views.
- Orbit, pan, zoom, and WASD/QE movement with scene/object framing.
- Named-object selection and visibility controls, including `flash_detonation_marker`.
- Perspective/orthographic cameras, optional wireframe, grid and axes.
- Triangle, vertex, file-size, bounds, camera, and frame-rate diagnostics.

GLB remains the recommended snapshot export format. `.cs2session` is the replayable application format. A current Chrome, Edge, or Firefox browser with WebGL 2 is required. Everything is served on `127.0.0.1` and remains local to the machine.

## Project layout

Supported application behavior lives in `cs2_visibility/`. `viewer/` contains the independent static viewer, while `scripts/` contains development utilities and archived prototypes that are not part of the supported application. Large demo files and generated outputs stay local and are ignored by Git. See [the project layout guide](docs/project-layout.md) before adding a new top-level file.

## Optional installed commands

`pyproject.toml` also defines `cs2-vision`, `cs2-flash-events`, `cs2-flash-coverage`, and `cs2-grenade-lineups`. They are generated only after `python -m pip install -e .` and are available only while that Python environment is activated. They intentionally have different names from the `.py` files, just like `pytest` is generated from a Python package rather than a file named `pytest.py`.
