# Changelog

All notable, explicitly versioned changes to this project are recorded here.

## 0.9.0 - 2026-08-03 - Crouch poses and consistent round clocks

- Applied the reverse-counting configured round clock to recorded Flash events, Lineup lists and event details, Vision replay, and Vision history ranges.
- Added a generated crouching aiming asset and carried `duck_amount` through Vision previews and replay payloads so standing and crouching models switch at the recorded pose.
- Removed the two Vision selection endpoint models when an analysis result opens; they return only after the user changes the player, mode, or time selection.
- Added a cyan exact-look ray to the current Vision replay pose, using the configured Vision maximum distance.
- Hid all overlay models, markers, and analysis lines while inside Flash camera mode so they cannot obstruct the camera.
- Added separate collision-limited pin-pull and release look rays for in-motion grenade lineups while leaving the fixed-reference visualization unchanged.
- Upgraded Vision replay payloads to include crouch state while retaining decoding support for existing pose-less and standing-only result payloads; all 19 automated tests pass.

## 0.8.0 - 2026-08-03 - Player paths and pose assets

- Added immediate alive-period movement-path previews when selecting a Vision round and player.
- Replaced elapsed-second Vision controls with the demo's reverse-counting round clock and capped each player's selection at their death or the round end.
- Added simultaneous cyan start and orange end models for interval selection, with recorded position, yaw, and pitch; instant selection displays one model.
- Added generated low-poly GLB assets for aiming, grenade holding, grenade throwing, and flashbang visualization, plus a deterministic asset-generation script.
- Replaced generic replay and lineup objects with the new pose assets while retaining the established fixed and in-motion lineup color semantics.
- Introduced `.cs2session` schema v3 for configured round clocks and player life-window end ticks; older session schemas are intentionally rejected.
- Added preview API, archive, asset-serving, death-window, and viewer regression coverage; all 19 automated tests pass.

## 0.7.0 - 2026-08-03 - Grenade lineup viewer

- Added a dedicated **Lineups** analysis tab with round, grenade, player, and fixed/in-motion filters, grouped throw selection, detail cards, console-command copying, and filtered JSON/CFG export.
- Added persistent lineup data to newly saved `.cs2session` archives while keeping older sessions readable and clearly reporting when lineup records are unavailable.
- Added 3D lineup visualization with approach paths and distinct marker semantics: fixed throws retain blue reference/orange release markers, while in-motion throws show purple pin-pull and blue release markers.
- Matched flash, HE, smoke, molotov, and incendiary events back to their releases and added red recorded-detonation markers plus detonation data in JSON/session exports.
- Changed recorded aim rays to stop at the first map collision or a configurable maximum distance.
- Added a centered lineup-camera crosshair and adjustable 60-120 degree FOV with a wider 90-degree default.
- Validated the included Dust II demo with 324 lineups, 324 pin-pull poses, and 321 recorded detonations; all 19 automated tests pass.

## 0.6.0 - 2026-08-03 - Grenade lineup extraction

- Added patch-aware detection of flashbang, smoke, HE, molotov, incendiary, decoy, tag, and snowball release events from CS2 demos.
- Added pure lineup derivation for fixed stand/aim references, angle-stability rejection, standing-point collapse, and in-motion path fallbacks.
- Classified click strength, walking/running/standing movement, jumpthrows, crouch state, release velocity, and left+right medium-strength throws on independent fields.
- Added human-readable movement directions and distances plus current CS2 `setpos`/`setang` command export in JSON and paste-ready CFG formats.
- Added the `grenade-lineups.py` checkout command and `cs2-grenade-lineups` installed entry point with grenade filters and configurable detection thresholds.
- Validated 324 grenade releases across the included current-build demo and added seven synthetic lineup tests; all 18 automated tests pass.

## 0.5.0 - 2026-08-03 - Fused CUDA analysis

- Moved vision candidate filtering, raycasting, cumulative sample tracking, face reduction, and bit packing into fused NVIDIA Warp kernels.
- Moved flash range filtering, raycasting, falloff, and face reduction into a fused GPU kernel.
- Kept static face samples and cumulative visibility state in GPU memory and queued vision poses in chunks to eliminate per-ray host transfers and per-pose synchronization.
- Preserved the generic raycasting path as the CPU fallback and correctness reference, including automatic fallback for custom sample layouts.
- Matched the archived 173,385,654-ray vision result byte-for-byte and matched reference flash intensities without reducing any accuracy setting.
- Improved the measured 32-pose Dust II workload from 5.783 seconds to 0.301 seconds (19.2x) while sustaining 98-100% GPU SM utilization on an RTX 3060 Ti.
- Added fused-path routing tests and CUDA/reference equivalence coverage; all 11 automated tests pass.

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
