# CS2 Static Map Visibility Analyzer

This project exports CS2 map geometry as a colored `.glb` file: red faces were visible to a selected player during a demo time window; gray faces were not.

It analyzes static Awpy `.tri` map geometry. It is not a literal frame-by-frame reconstruction: smokes, flashes, player models, dynamic props/doors, scope rendering, spectator state, and screen/UI obstruction are outside this analyzer's scope.

## Run it

```powershell
python nvidia-warp-kimi.py match.dem --player "donk" --round 1 --start 1:00 --end 1:20 --out seen_result.glb
```

The command uses NVIDIA Warp automatically when CUDA is available, otherwise it falls back to CPU raycasting. Use `--no-gpu` to force CPU mode.

If a demo header has no usable tick rate, use an explicit override:

```powershell
python nvidia-warp-kimi.py match.dem --player "donk" --round 1 --start 0 --end 20 --tick-rate 128
```

Before running, install the dependencies and download Awpy geometry once:

```powershell
pip install awpy trimesh numpy polars warp-lang
awpy get tris
```

## Accuracy controls

By default the analyzer tests four **interior** points per face: its center and three points near, but never on, the corners. This avoids shared-edge ambiguity from the older vertex tests.

```powershell
# Require two different interior samples to be visible before a face is red.
python nvidia-warp-kimi.py match.dem --player "donk" --round 1 --start 0 --end 20 --min-visible-samples 2
```

`--samples-per-triangle 1` is faster but less resilient to small occluders. `--samples-per-triangle 4` is the recommended default. `--ray-batch-size` controls GPU/CPU dispatch size; increase it only if sufficient GPU memory is available.

The demo's `duck_amount` property is used to interpolate between `--eye-height` (64 by default) and `--crouch-eye-height` (46 by default). If the property is unavailable, the program prints a warning and uses the standing height.

## Library API

The reusable package is `cs2_visibility`; `nvidia-warp-kimi.py` is only a compatibility CLI wrapper. A UI or larger application can load poses and run an analysis without invoking a subprocess:

```python
from pathlib import Path
from cs2_visibility import AnalysisConfig, VisibilityAnalyzer, export_colored_mesh, load_demo_window

poses, map_name, _ = load_demo_window(Path("match.dem"), "donk", 1, 0, 20, 4, None, 64, 46)
analyzer = VisibilityAnalyzer.from_tri_file(Path("de_dust2.tri"), AnalysisConfig())
result = analyzer.analyze(poses)
export_colored_mesh(analyzer.mesh, result.seen_mask, Path("seen_result.glb"))
```
