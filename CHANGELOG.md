# Changelog

All notable, explicitly versioned changes to this project are recorded here.

## 0.2.0 - 2026-07-30 - Experimental flash coverage

Status: untested experimental implementation; explicitly considered rough/prototype-quality ("pure slop") until it has been validated against real gameplay and CUDA/Warp runs.

- Added a reusable architecture guide and shared progress/logging utilities.
- Added `flash-events.py` to extract exact `flashbang_detonate` pop positions and export normalized JSON.
- Added `flash-coverage.py` to simulate static-map coverage for one explicitly selected flash event.
- Added distance-based per-face intensity coloring with line-of-sight occlusion checks.
- Added `--verbose` and progress-bar support to flash tools and player-vision analysis.
- Added a NumPy 2 compatibility workaround for GLB export in the installed Trimesh version.
