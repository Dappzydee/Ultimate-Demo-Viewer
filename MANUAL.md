# CS2 Demo Visualization Manual

This software loads a Counter-Strike 2 `.dem` once, runs repeated analyses in memory, and displays colored 3D map geometry in its integrated local viewer. It contains two visualization tools:

- Player vision: static map faces a selected player could see during a selected time window.
- Flash coverage: static map faces that could be affected by one selected flashbang detonation.

The repository also includes a non-rendering grenade-lineup extractor. It derives stand/aim references, movement paths, throw classification, and console commands from player ticks around every grenade release.

Both tools use Awpy `.tri` map geometry and raycasts. They model static geometry only. They do not currently account for smoke, fire, player models, dynamic props/doors, or UI effects.

## Integrated application

```powershell
python viewer.py
```

Open or drop a `.dem`, then select Vision, Flash, or Lineups. Vision supports an instant timestamp or a start/end interval. Interval results include a timeline with current and accumulated visibility modes. Flash events are grouped by team; manual flashes can be entered as XYZ coordinates or placed by clicking the map. Lineups are detected automatically at load time and do not need a separate Analyze action.

In **Lineups**, filter by round, grenade, player, or reference quality and select a throw. The viewport uses blue for the stand/reference marker, orange for release, cyan for movement, and yellow for recorded aim. **Frame** fits the complete approach, **View aim** enters the player's recorded eye position and angles, and Escape exits that view. The detail card provides movement instructions, classification, speed, warnings, and copyable console commands. Filtered records can be downloaded as JSON or CFG and are retained when saving a new `.cs2session`.

Results remain in memory. **Export GLB** writes only the currently displayed snapshot. **Save session** creates a replayable `.cs2session` with normalized demo data, map geometry, and the current result; it can be reopened without the original `.dem`.

The local service listens only on `127.0.0.1`. The UI is browser-based, but parsing and raycasting run in the local Python process so Awpy and NVIDIA Warp remain available.

## Output files

Generated GLB and JSON files are placed in the project's `out/` folder. A relative `--out` or `--json` filename is automatically placed there, so `--out result.glb` becomes `out/result.glb`. Use an absolute path only when you intentionally want to write elsewhere.

## Requirements

- Python 3.11 or newer.
- A CS2 demo file (`.dem`).
- Awpy map geometry downloaded locally.
- NVIDIA Warp and a CUDA-capable NVIDIA GPU are recommended for practical performance. CPU mode works but can be very slow on full maps.

Install dependencies from the project directory:

```powershell
python -m pip install -r requirements.txt
awpy get tris
```

`awpy get tris` is needed only once. It downloads the `.tri` meshes used for static map raycasting.

## Progress and diagnostics

Long-running calculations show a terminal progress bar by default.

- Add `--verbose` to show useful diagnostics such as selected events, mesh candidates, ray batches, and the chosen GPU/CPU backend.
- Add `--no-progress` when running in a log file, CI system, or non-interactive terminal.
- Add `--no-gpu` to force the CPU backend for comparison/debugging.
- Add `--traceback` to print the full Python exception traceback, including the source file and line number, for handled errors. Unexpected programming errors already print a traceback by default.

## Player vision

Use `vision.py` to export a GLB where red faces were seen during the time window and gray faces were not.

```powershell
python vision.py match.dem --player "donk" --round 1 --start 1:00 --end 1:20 --out player_vision.glb
```

Time values are measured from the end of freeze time for the selected round. They accept seconds (`20`) or minutes and seconds (`1:20`).

### Useful player-vision options

```powershell
python vision.py match.dem --player "donk" --round 1 --start 0 --end 20 `
  --fov 90 --tick-step 4 --samples-per-triangle 4 --min-visible-samples 1 `
  --out player_vision.glb --verbose
```

- `--tick-rate 128`: overrides an absent or incorrect demo-header tick rate.
- `--tick-step 4`: analyzes every fourth tick. Lower values are more detailed but slower.
- `--samples-per-triangle 4`: tests the center plus three inset interior points. This is the recommended accuracy setting.
- `--min-visible-samples 2`: requires two different interior samples to be visible before a face is painted red. This avoids marking a large face based on one tiny visible corner.
- `--eye-height 64` and `--crouch-eye-height 46`: configure eye positions. If Awpy exposes `duck_amount`, the tool interpolates between these values.

## Flash event extraction

Use `flash-events.py` to list exact `flashbang_detonate` events from a demo. This reads the detonation event stream and reports the exact tick and X/Y/Z pop position from the demo.

```powershell
python flash-events.py match.dem --json flashes.json
```

The console output assigns each flash a stable `index`. The JSON output is designed for later processing:

```json
{
  "index": 12,
  "tick": 19974,
  "position": { "x": 1144.35, "y": 555.42, "z": 458.69 },
  "map_name": "de_dust2",
  "thrower": "example_player"
}
```

The JSON file contains a list of these objects. Keep the list if you want to select events by index later.

## Grenade lineups

Use `grenade-lineups.py` when you want to reproduce throws rather than simulate flash coverage:

```powershell
python grenade-lineups.py match.dem --json lineups.json --commands lineups.cfg
```

The tool reads grenade `weapon_fire` events as release ticks, discovers the current demo's movement properties, and loads five seconds of player ticks before each release. It uses the last stationary stretch before sustained movement as the fixed reference only when pitch and wrapped yaw remain within five degrees of the release aim. If no such stretch exists, or aim drifts too far, `reference_point` is `null`, the approach is exported in `movement_path`, and the static commands use the release point.

Classification is exported on independent axes:

- `click`: `left`, `right`, `both` (CS2's medium-strength left+right throw), or `unknown` when the button mask is unavailable.
- `movement`: `standing`, `walking`, or `running`, using `m_bIsWalking` when available and a speed threshold otherwise.
- `jumpthrow`: button- or airborne-rise detection independent of horizontal movement.

Each record includes exact release/reference poses, a generated movement direction and distance, release velocity, confidence notes, and `setpos X Y Z` / `setang pitch yaw 0`. These cheat commands place and aim the player, but cannot recreate velocity, jump timing, or crouch. Run them in a suitable practice server with `sv_cheats 1` and follow the exported comments.

Useful controls include `--grenade-type flashbang` (repeatable), `--tick-rate`, `--lookback-seconds`, `--stationary-speed`, `--ramp-speed`, `--walking-speed`, `--angle-tolerance`, and `--collapse-distance`. Add `--verbose` to print the discovered patch-dependent fields.

## Flash coverage

Use `flash-coverage.py` to simulate coverage for one explicitly selected flash. It never processes all flashes automatically.

```powershell
python flash-coverage.py --flash-json flashes.json --flash-index 12 --out flash_12.glb
```

Because generated JSON goes into `out/`, either `--flash-json out/flashes.json` or the shorter `--flash-json flashes.json` works.

You can select directly from a demo instead of JSON, but must specify the event index:

```powershell
python flash-coverage.py --demo match.dem --flash-index 12 --out flash_12.glb
```

The flash coverage GLB uses:

- Gray: no simulated coverage.
- Dark red: weak simulated effect.
- Light yellow: stronger simulated effect.

Each flash-coverage GLB also contains a separate, bright yellow sphere named `flash_detonation_marker`. In Blender's Outliner, select that object and use **Numpad `.`** (Frame Selected) to jump directly to the exact pop position. Use `--marker-radius` to make the marker larger or smaller.

The simulation casts from the exact detonation position through static map geometry. Intensity falls off with distance and is not a Valve-exact blind-duration model. It also does not know the direction a hypothetical player would be facing.

### Useful flash-coverage options

```powershell
python flash-coverage.py --flash-json flashes.json --flash-index 12 `
  --max-distance 1500 --falloff-power 1.5 --samples-per-triangle 4 `
  --out flash_12.glb --verbose
```

- `--max-distance`: simulation radius in map units; larger values test more faces and take longer.
- `--falloff-power`: higher values make intensity fade faster with distance.
- `--samples-per-triangle`: use `4` for more resilient coverage around small occluders; use `1` for a faster rough pass.
- `--map de_dust2`: required only when the selected JSON does not include `map_name`.
- `--tri C:\path\to\map.tri`: replaces Awpy's default map mesh.

## Viewing results locally

Launch the included viewer with a result file:

```powershell
python viewer.py out/player_vision.glb
```

The command opens the integrated local app with that GLB. You can open or drag another `.dem`, `.cs2session`, or GLB at any time. Left-drag orbits, middle-drag pans, the wheel zooms, `WASD` and `Q/E` move, and `F` frames the selected object or full scene. The toolbar exposes flat colors, lit colors, normals, wireframe, grid, axes, and perspective/orthographic views.

The viewer requires a current browser with WebGL 2 but has no extra Python or JavaScript dependencies. The local server listens only on `127.0.0.1`; model data is not uploaded.

## Viewing results in Blender

1. Open Blender.
2. Select **File > Import > glTF 2.0**.
3. Select the exported `.glb` file.
4. Use Material Preview or Rendered view to inspect face colors.

The mesh can be large. Give Blender time to import and avoid enabling expensive modifiers on the imported geometry.

## Testing

### Quick command checks

These commands verify that each CLI loads and exposes its options:

```powershell
python vision.py --help
python flash-events.py --help
python flash-coverage.py --help
python grenade-lineups.py --help
```

Compile all Python modules before a change:

```powershell
python -m compileall -q cs2_visibility vision.py flash-events.py flash-coverage.py grenade-lineups.py
```

### Real-demo smoke test

Use a short, known demo window and a single flash event. Start with a small radius and one sample per face to avoid a long CPU test.

```powershell
python flash-events.py demo_dust2.dem --json flashes.json
python grenade-lineups.py demo_dust2.dem --grenade-type flashbang --json lineups.json
python flash-coverage.py --flash-json flashes.json --flash-index 0 `
  --max-distance 300 --samples-per-triangle 1 --out flash_smoke_test.glb --verbose
```

Confirm that:

1. The listed event has a plausible tick, map, player, and XYZ position.
2. The lineup JSON contains plausible release/reference points, throw axes, and console commands.
3. The coverage command selects that same index and reports a raycasting backend.
4. The GLB file exists and imports into Blender.
5. The colored region is near the selected flash position.

For a GPU test, omit `--no-gpu` and check that output says `nvidia-warp`. If it says `cpu`, install `warp-lang`, verify CUDA/NVIDIA drivers, and run again.

### CPU/GPU comparison

For a small radius, export once on GPU and once on CPU. The affected geometry should be materially similar.

```powershell
python flash-coverage.py --flash-json flashes.json --flash-index 0 --max-distance 300 --out flash_gpu.glb
python flash-coverage.py --flash-json flashes.json --flash-index 0 --max-distance 300 --no-gpu --out flash_cpu.glb
```

Small differences at mesh boundaries are possible because the GPU uses single-precision calculations. Interior face samples are used to minimize shared-edge ambiguity.

## Current limitations

This is experimental software. Do not treat flash colors as the game's exact blind duration. Validate a scenario against actual `player_blind` events before relying on it for detailed gameplay conclusions.

See `docs/architecture.md` for developer-focused architecture and extension guidance, and `CHANGELOG.md` for explicitly versioned project history.
