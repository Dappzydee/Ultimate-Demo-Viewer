"""High-level demo loading, visibility analysis, and export functions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Iterable

import numpy as np
import polars as pl
import trimesh

from awpy import Demo
from awpy.data import TRIS_DIR

from .geometry import export_glb_bytes, forward_vector, interior_sample_points, load_tri_mesh
from .models import AnalysisConfig, PlayerPose, VisibilityResult, VisibilityTimelineResult
from .progress import ProgressBar
from .raycasting import create_raycaster


def parse_time_seconds(value: str) -> float:
    """Parse ``seconds`` or ``minutes:seconds`` and reject invalid windows."""
    try:
        if ":" in value:
            minutes, seconds = value.split(":", maxsplit=1)
            result = int(minutes) * 60 + float(seconds)
        else:
            result = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid time '{value}'. Use seconds or MM:SS.") from error
    if result < 0:
        raise ValueError("Time values cannot be negative.")
    return result


def load_demo_window(
    demo_path: Path, player: str, round_number: int, start_seconds: float, end_seconds: float,
    tick_step: int, tick_rate_override: float | None, eye_height: float, crouch_eye_height: float,
) -> tuple[list[PlayerPose], str, float]:
    """Load sampled player poses. Duck amount is used when the demo exposes it."""
    if end_seconds < start_seconds:
        raise ValueError("--end must be greater than or equal to --start.")
    if tick_step < 1:
        raise ValueError("--tick-step must be at least 1.")
    demo = Demo(str(demo_path), verbose=False)
    demo.parse(
        player_props=["X", "Y", "Z", "pitch", "yaw", "duck_amount"],
        events=["round_start", "round_freeze_end", "round_officially_ended"],
    )
    map_name = demo.header.get("map_name") or demo.header.get("map")
    if not map_name:
        raise ValueError("Could not determine the map from the demo; supply --map.")
    round_rows = demo.rounds.filter(pl.col("round_num") == round_number)
    if round_rows.is_empty():
        raise ValueError(f"Round {round_number} does not exist in this demo.")
    start_tick = round_rows["freeze_end"][0] if "freeze_end" in round_rows.columns else round_rows["start"][0]
    header_rate = demo.header.get("tick_rate")
    if tick_rate_override is not None and tick_rate_override <= 0:
        raise ValueError("--tick-rate must be positive.")
    tick_rate = tick_rate_override if tick_rate_override is not None else (
        float(header_rate) if isinstance(header_rate, (int, float)) and header_rate > 0 else 64.0
    )
    lower = start_tick + round(start_seconds * tick_rate)
    upper = start_tick + round(end_seconds * tick_rate)
    rows = demo.ticks.filter(
        (pl.col("name") == player) & (pl.col("round_num") == round_number) &
        (pl.col("tick") >= lower) & (pl.col("tick") <= upper)
    ).sort("tick")[::tick_step]
    if rows.is_empty():
        names = demo.ticks.filter(pl.col("round_num") == round_number)["name"].unique().sort().to_list()
        raise ValueError(f"No ticks found for player '{player}'. Available players: {names}")
    has_duck = "duck_amount" in rows.columns
    if not has_duck:
        print("Warning: demo has no duck_amount property; using the configured standing eye height.")
    poses = []
    for row in rows.iter_rows(named=True):
        duck = float(row.get("duck_amount") or 0.0) if has_duck else 0.0
        height = eye_height + (crouch_eye_height - eye_height) * np.clip(duck, 0.0, 1.0)
        poses.append(PlayerPose(int(row["tick"]), np.array((row["X"], row["Y"], row["Z"] + height), dtype=np.float64), float(row["yaw"]), float(row["pitch"])))
    return poses, str(map_name), tick_rate


class VisibilityAnalyzer:
    """Analyze static map faces visible from a sequence of player poses."""

    def __init__(
        self, mesh: trimesh.Trimesh, config: AnalysisConfig, *,
        sample_points: np.ndarray | None = None, sample_face_ids: np.ndarray | None = None,
        raycaster: object | None = None,
    ) -> None:
        self.mesh = mesh
        self.config = config
        if sample_points is None or sample_face_ids is None:
            sample_points, sample_face_ids = interior_sample_points(mesh, config.samples_per_triangle)
        self.sample_points, self.sample_face_ids = sample_points, sample_face_ids
        self.raycaster = raycaster or create_raycaster(mesh, config.prefer_gpu)

    @classmethod
    def from_tri_file(cls, tri_path: Path, config: AnalysisConfig) -> "VisibilityAnalyzer":
        return cls(load_tri_mesh(tri_path), config)

    def analyze(self, poses: Iterable[PlayerPose], show_progress: bool = True) -> VisibilityResult:
        """Batch raycasts from multiple ticks while maintaining per-face evidence."""
        # Track individual sample locations, rather than incrementing an
        # unbounded count each tick. This is a real boolean mask: repeatedly
        # seeing the same small corner cannot satisfy a stricter threshold.
        seen_samples = np.zeros(len(self.sample_points), dtype=bool)
        cos_half_fov = np.cos(np.radians(self.config.horizontal_fov_degrees / 2))
        pending_origins: list[np.ndarray] = []
        pending_directions: list[np.ndarray] = []
        pending_faces: list[np.ndarray] = []
        pending_sample_ids: list[np.ndarray] = []
        pending_lengths: list[np.ndarray] = []
        ray_count = 0
        pending_count = 0

        def flush() -> None:
            nonlocal ray_count, pending_count
            if not pending_faces:
                return
            origins = np.concatenate(pending_origins)
            directions = np.concatenate(pending_directions)
            faces = np.concatenate(pending_faces)
            sample_ids = np.concatenate(pending_sample_ids)
            lengths = np.concatenate(pending_lengths)
            visible = self.raycaster.visible(origins, directions, faces, lengths)
            seen_samples[sample_ids[visible]] = True
            ray_count += len(faces)
            pending_origins.clear(); pending_directions.clear(); pending_faces.clear(); pending_sample_ids.clear(); pending_lengths.clear()
            pending_count = 0

        pose_list = list(poses)
        progress = ProgressBar(len(pose_list), "Raycasting player vision", show_progress)
        processed = 0
        for pose in pose_list:
            processed += 1
            offsets = self.sample_points - pose.position
            distances = np.linalg.norm(offsets, axis=1)
            valid_distance = (distances > 1e-6) & (distances <= self.config.max_distance)
            directions = np.divide(offsets, distances[:, None], out=np.zeros_like(offsets), where=distances[:, None] > 1e-6)
            candidates = valid_distance & ((directions @ forward_vector(pose.yaw_degrees, pose.pitch_degrees)) >= cos_half_fov)
            if not np.any(candidates):
                progress.update(processed)
                continue
            faces = self.sample_face_ids[candidates]
            sample_ids = np.flatnonzero(candidates)
            pending_origins.append(np.broadcast_to(pose.position, (len(faces), 3)).copy())
            pending_directions.append(directions[candidates])
            pending_faces.append(faces)
            pending_sample_ids.append(sample_ids)
            pending_lengths.append(distances[candidates] + self.config.ray_endpoint_epsilon)
            pending_count += len(faces)
            if pending_count >= self.config.ray_batch_size:
                flush()
            progress.update(processed)
        flush()
        progress.finish()
        seen_mask = seen_samples.reshape(-1, self.config.samples_per_triangle).sum(axis=1) >= self.config.min_visible_samples
        return VisibilityResult(seen_mask, processed, ray_count, self.raycaster.name)

    def analyze_timeline(
        self, poses: Iterable[PlayerPose], show_progress: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> VisibilityTimelineResult:
        """Analyze ordered poses and retain compact replayable face masks."""
        pose_list = list(poses)
        face_count = len(self.mesh.faces)
        packed_width = (face_count + 7) // 8
        instant_masks = np.zeros((len(pose_list), packed_width), dtype=np.uint8)
        cumulative_masks = np.zeros_like(instant_masks)
        cumulative_samples = np.zeros(len(self.sample_points), dtype=bool)
        cos_half_fov = np.cos(np.radians(self.config.horizontal_fov_degrees / 2))
        pending_origins: list[np.ndarray] = []
        pending_directions: list[np.ndarray] = []
        pending_faces: list[np.ndarray] = []
        pending_lengths: list[np.ndarray] = []
        pending_records: list[tuple[int, np.ndarray, int, int]] = []
        pending_count = 0
        ray_count = 0
        completed = 0
        progress = ProgressBar(len(pose_list), "Raycasting player vision timeline", show_progress)

        def flush() -> None:
            nonlocal pending_count, ray_count, completed
            if not pending_records:
                return
            if pending_count:
                origins = np.concatenate(pending_origins)
                directions = np.concatenate(pending_directions)
                faces = np.concatenate(pending_faces)
                lengths = np.concatenate(pending_lengths)
                visible = self.raycaster.visible(origins, directions, faces, lengths)
                ray_count += len(faces)
            else:
                visible = np.empty(0, dtype=bool)

            for frame_index, sample_ids, start, end in pending_records:
                visible_sample_ids = sample_ids[visible[start:end]]
                instant_counts = np.bincount(
                    self.sample_face_ids[visible_sample_ids], minlength=face_count,
                )
                instant = instant_counts >= self.config.min_visible_samples
                cumulative_samples[visible_sample_ids] = True
                cumulative = (
                    cumulative_samples.reshape(-1, self.config.samples_per_triangle).sum(axis=1)
                    >= self.config.min_visible_samples
                )
                instant_masks[frame_index] = np.packbits(instant, bitorder="little")
                cumulative_masks[frame_index] = np.packbits(cumulative, bitorder="little")
                completed += 1
                progress.update(completed)
                if progress_callback:
                    progress_callback(completed, len(pose_list))

            pending_origins.clear()
            pending_directions.clear()
            pending_faces.clear()
            pending_lengths.clear()
            pending_records.clear()
            pending_count = 0

        for frame_index, pose in enumerate(pose_list):
            offsets = self.sample_points - pose.position
            distances = np.linalg.norm(offsets, axis=1)
            valid_distance = (distances > 1e-6) & (distances <= self.config.max_distance)
            directions = np.divide(
                offsets, distances[:, None], out=np.zeros_like(offsets), where=distances[:, None] > 1e-6,
            )
            candidates = valid_distance & (
                (directions @ forward_vector(pose.yaw_degrees, pose.pitch_degrees)) >= cos_half_fov
            )
            sample_ids = np.flatnonzero(candidates)
            start = pending_count
            if len(sample_ids):
                faces = self.sample_face_ids[sample_ids]
                pending_origins.append(np.broadcast_to(pose.position, (len(faces), 3)).copy())
                pending_directions.append(directions[sample_ids])
                pending_faces.append(faces)
                pending_lengths.append(distances[sample_ids] + self.config.ray_endpoint_epsilon)
                pending_count += len(sample_ids)
            pending_records.append((frame_index, sample_ids, start, pending_count))
            if pending_count >= self.config.ray_batch_size:
                flush()

        flush()
        progress.finish()
        return VisibilityTimelineResult(
            ticks=np.asarray([pose.tick for pose in pose_list], dtype=np.uint32),
            instant_masks=instant_masks,
            cumulative_masks=cumulative_masks,
            face_count=face_count,
            processed_poses=len(pose_list),
            tested_rays=ray_count,
            backend=self.raycaster.name,
        )


def resolve_tri_path(map_name: str, override: Path | None) -> Path:
    path = override or TRIS_DIR / f"{map_name}.tri"
    if not path.exists():
        raise FileNotFoundError(f"Map geometry file not found: {path}. Run 'awpy get tris' first.")
    return path


def export_colored_mesh(mesh: trimesh.Trimesh, seen_mask: np.ndarray, output_path: Path) -> None:
    """Export a GLB with red seen faces and gray unseen faces."""
    output_path.write_bytes(export_colored_mesh_bytes(mesh, seen_mask))


def export_colored_mesh_bytes(mesh: trimesh.Trimesh, seen_mask: np.ndarray) -> bytes:
    """Build a colored vision snapshot without writing a temporary file."""
    if len(seen_mask) != len(mesh.faces):
        raise ValueError("Expected one visibility value for every mesh face.")
    colors = np.full((len(mesh.faces), 4), (160, 160, 160, 255), dtype=np.uint8)
    colors[seen_mask] = (255, 0, 0, 255)
    output_mesh = mesh.copy()
    output_mesh.visual.face_colors = colors
    return export_glb_bytes(output_mesh)
