# Changelog

All notable, explicitly versioned changes to this project are recorded here.

## 0.4.0 - 2026-08-02 - Player replay and analysis history

- Added a toggleable player position marker and facing arrow that follow Vision replay frames.
- Added reusable Analysis History entries with open, rename, pin, delete, GLB export, and session-selection actions.
- Reused identical analysis requests and bounded unpinned history to 20 results with a 512 MB target.
- Added multi-result `.cs2session` schema v2 while retaining schema-v1 compatibility.
- Kept map geometry shared between results; history stores only compact masks, intensities, and replay poses.
- Expanded automated coverage to eight passing tests; interactive browser validation remains unavailable in the current environment.

## 0.3.0 - 2026-08-02 - Integrated demo analysis viewer

Status: integrated local application with automated server, session, and replay coverage; interactive browser validation remains pending where noted.

- Combined demo loading, repeated in-memory analysis, replay inspection, session persistence, and optional GLB snapshot export in one local viewer application.
- Added exclusive Vision and Flash task tabs with task-specific controls and Analyze actions in one shared pane.
- Added instant vision snapshots and interval replays with separate current and accumulated visibility modes.
- Replaced the flash dropdown with a selectable list of recorded flashes grouped by team.
- Added manual XYZ flash placement and map-surface placement for arbitrary flash simulations.
- Added an optional **Focus selected flash** behavior that centers a selected flash in the viewport.
- Added **Flash camera** mode, which places the viewpoint at the flash position, supports drag-to-look inspection, displays a yellow viewport frame and mode badge, and exits with Escape.
- Kept the flash marker selected for normal inspection and hid it while Flash camera is inside it so the marker cannot occlude the view.
- Documented that arbitrary flash placement currently requires an explicit analysis; continuously draggable, live coverage preview is planned but not implemented.
- Added viewer regression checks; all six automated tests pass. Interactive browser automation was unavailable for the final visual camera-mode check.

## 0.2.0 - 2026-07-30 - Experimental flash coverage

Status: untested experimental implementation; explicitly considered rough/prototype-quality ("pure slop") until it has been validated against real gameplay and CUDA/Warp runs.

- Added a reusable architecture guide and shared progress/logging utilities.
- Added `flash-events.py` to extract exact `flashbang_detonate` pop positions and export normalized JSON.
- Added `flash-coverage.py` to simulate static-map coverage for one explicitly selected flash event.
- Added distance-based per-face intensity coloring with line-of-sight occlusion checks.
- Added `--verbose` and progress-bar support to flash tools and player-vision analysis.
- Added a NumPy 2 compatibility workaround for GLB export in the installed Trimesh version.
