"""Public data models used by the visibility-analysis library."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnalysisConfig:
    """Controls the static-map visibility approximation.

    ``min_visible_samples`` determines how much of a triangle must be visible.
    A value of 1 is inclusive and useful for "ever seen" exports; a higher
    value is stricter for large triangles with only a small visible corner.
    """

    horizontal_fov_degrees: float = 90.0
    max_distance: float = 4000.0
    samples_per_triangle: int = 4
    min_visible_samples: int = 1
    ray_batch_size: int = 250_000
    prefer_gpu: bool = True
    ray_endpoint_epsilon: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.horizontal_fov_degrees <= 360:
            raise ValueError("horizontal_fov_degrees must be in (0, 360].")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if self.samples_per_triangle not in (1, 4):
            raise ValueError("samples_per_triangle must be 1 or 4.")
        if not 1 <= self.min_visible_samples <= self.samples_per_triangle:
            raise ValueError("min_visible_samples must be between 1 and samples_per_triangle.")
        if self.ray_batch_size <= 0:
            raise ValueError("ray_batch_size must be positive.")


@dataclass(frozen=True)
class PlayerPose:
    """The player's eye position and view angles at one demo tick."""

    tick: int
    position: np.ndarray
    yaw_degrees: float
    pitch_degrees: float


@dataclass(frozen=True)
class VisibilityResult:
    """Result of one time-window analysis."""

    seen_mask: np.ndarray
    processed_poses: int
    tested_rays: int
    backend: str


@dataclass(frozen=True)
class VisibilityTimelineResult:
    """Packed current and accumulated visibility for an ordered pose sequence.

    Each row in ``instant_masks`` and ``cumulative_masks`` is a little-endian
    bitset with one bit per map face. Keeping the result packed makes replay
    practical without retaining a dense frame-by-face boolean matrix.
    """

    ticks: np.ndarray
    instant_masks: np.ndarray
    cumulative_masks: np.ndarray
    face_count: int
    processed_poses: int
    tested_rays: int
    backend: str

    @property
    def seen_mask(self) -> np.ndarray:
        """Return the final accumulated face mask."""
        if not len(self.cumulative_masks):
            return np.zeros(self.face_count, dtype=bool)
        return np.unpackbits(self.cumulative_masks[-1], bitorder="little")[: self.face_count].astype(bool)
