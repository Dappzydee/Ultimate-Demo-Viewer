"""Hypothetical static-map coverage from one selected flash detonation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from collections.abc import Callable

import numpy as np
import trimesh

from .flash_events import FlashDetonation
from .geometry import export_glb_bytes, interior_sample_points
from .progress import ProgressBar
from .raycasting import create_raycaster

LOGGER = logging.getLogger("cs2_visibility")


@dataclass(frozen=True)
class FlashCoverageConfig:
    """Controls an explicitly approximate, distance-and-occlusion coverage map."""

    max_distance: float = 1500.0
    samples_per_triangle: int = 4
    falloff_power: float = 1.0
    ray_batch_size: int = 250_000
    prefer_gpu: bool = True
    ray_endpoint_epsilon: float = 0.05

    def __post_init__(self) -> None:
        if self.max_distance <= 0 or self.falloff_power <= 0 or self.ray_batch_size <= 0:
            raise ValueError("max_distance, falloff_power, and ray_batch_size must be positive.")
        if self.samples_per_triangle not in (1, 4):
            raise ValueError("samples_per_triangle must be 1 or 4.")


@dataclass(frozen=True)
class FlashCoverageResult:
    """Per-face normalized intensities, where 0 is unaffected and 1 is strongest."""

    intensities: np.ndarray
    tested_rays: int
    backend: str


class FlashCoverageAnalyzer:
    """Compute coverage for one flash event; it never enumerates other flashes."""

    def __init__(
        self, mesh: trimesh.Trimesh, config: FlashCoverageConfig, *,
        sample_points: np.ndarray | None = None, sample_face_ids: np.ndarray | None = None,
        raycaster: object | None = None,
    ) -> None:
        self.mesh = mesh
        self.config = config
        if sample_points is None or sample_face_ids is None:
            sample_points, sample_face_ids = interior_sample_points(mesh, config.samples_per_triangle)
        self.sample_points, self.sample_face_ids = sample_points, sample_face_ids
        self.raycaster = raycaster or create_raycaster(mesh, config.prefer_gpu)

    def analyze(
        self, flash: FlashDetonation, show_progress: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> FlashCoverageResult:
        origin = np.asarray(flash.position, dtype=np.float64)
        preparation = ProgressBar(2, "Preparing flash coverage", show_progress)
        preparation.update(0)
        offsets = self.sample_points - origin
        distances = np.linalg.norm(offsets, axis=1)
        candidate_mask = (distances > 1e-6) & (distances <= self.config.max_distance)
        candidate_ids = np.flatnonzero(candidate_mask)
        preparation.update(2)
        preparation.finish()
        LOGGER.debug("Flash %d at tick %d: %d/%d samples are within %.0f units", flash.index, flash.tick, len(candidate_ids), len(self.sample_points), self.config.max_distance)

        sample_intensities = np.zeros(len(self.sample_points), dtype=np.float32)
        progress = ProgressBar(len(candidate_ids), "Raycasting flash coverage", show_progress)
        for start in range(0, len(candidate_ids), self.config.ray_batch_size):
            ids = candidate_ids[start : start + self.config.ray_batch_size]
            batch_offsets = offsets[ids]
            batch_distances = distances[ids]
            directions = batch_offsets / batch_distances[:, None]
            faces = self.sample_face_ids[ids]
            origins = np.broadcast_to(origin, (len(ids), 3)).copy()
            visible = self.raycaster.visible(origins, directions, faces, batch_distances + self.config.ray_endpoint_epsilon)
            # A normalized, configurable falloff. It is deliberately not
            # presented as Valve's private blind-duration formula.
            sample_intensities[ids[visible]] = np.power(1.0 - batch_distances[visible] / self.config.max_distance, self.config.falloff_power)
            progress.update(min(start + len(ids), len(candidate_ids)))
            if progress_callback:
                progress_callback(min(start + len(ids), len(candidate_ids)), len(candidate_ids))
            LOGGER.debug("Flash ray batch %d-%d: %d/%d visible", start, start + len(ids), int(visible.sum()), len(ids))
        progress.finish()
        intensities = sample_intensities.reshape(-1, self.config.samples_per_triangle).max(axis=1)
        return FlashCoverageResult(intensities, len(candidate_ids), self.raycaster.name)


def export_flash_coverage(
    mesh: trimesh.Trimesh, intensities: np.ndarray, flash: FlashDetonation, output_path: Path, marker_radius: float = 24.0,
) -> None:
    """Export coverage plus a separately selectable detonation marker sphere."""
    output_path.write_bytes(export_flash_coverage_bytes(mesh, intensities, flash, marker_radius))


def export_flash_coverage_bytes(
    mesh: trimesh.Trimesh, intensities: np.ndarray, flash: FlashDetonation, marker_radius: float = 24.0,
) -> bytes:
    """Build an in-memory flash snapshot with a selectable detonation marker."""
    if len(intensities) != len(mesh.faces):
        raise ValueError("Expected one flash intensity for every mesh face.")
    if marker_radius <= 0:
        raise ValueError("marker_radius must be positive.")
    colors = np.full((len(mesh.faces), 4), (160, 160, 160, 255), dtype=np.uint8)
    affected = intensities > 0
    # Stronger flash coverage becomes visually lighter, as requested.
    strength = intensities[affected, None]
    weak = np.array((100, 0, 0), dtype=np.float32)
    strong = np.array((255, 245, 80), dtype=np.float32)
    colors[affected, :3] = (weak + strength * (strong - weak)).astype(np.uint8)
    output_mesh = mesh.copy()
    output_mesh.visual.face_colors = colors
    # A separate scene object lets Blender users select and focus the exact
    # flash pop location independently of the large map geometry.
    marker = trimesh.creation.uv_sphere(radius=marker_radius, count=[16, 16])
    # Translate vertices directly instead of apply_translation(): the installed
    # older Trimesh release calls a NumPy 2-removed API in that helper.
    marker.vertices += np.asarray(flash.position, dtype=np.float64)
    marker.visual.face_colors = np.tile((255, 255, 0, 255), (len(marker.faces), 1))
    scene = trimesh.Scene()
    scene.add_geometry(output_mesh, node_name="flash_coverage", geom_name="flash_coverage")
    scene.add_geometry(marker, node_name="flash_detonation_marker", geom_name="flash_detonation_marker")
    return export_glb_bytes(scene)
