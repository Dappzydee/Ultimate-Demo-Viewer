"""CUDA integration checks for the fused NVIDIA Warp analysis path."""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

from cs2_visibility.analysis import VisibilityAnalyzer
from cs2_visibility.flash_coverage import FlashCoverageAnalyzer, FlashCoverageConfig
from cs2_visibility.flash_events import FlashDetonation
from cs2_visibility.geometry import interior_sample_points
from cs2_visibility.models import AnalysisConfig, PlayerPose
from cs2_visibility.raycasting import WarpRaycaster

try:
    import warp as wp
except ImportError:
    wp = None


def cuda_available() -> bool:
    return wp is not None and wp.get_cuda_device_count() > 0


class GenericWarp:
    """Expose only the pre-fusion interface as a correctness reference."""

    name = "nvidia-warp-reference"

    def __init__(self, raycaster: WarpRaycaster) -> None:
        self.raycaster = raycaster

    def visible(self, origins, directions, face_ids, lengths):
        return self.raycaster.visible(origins, directions, face_ids, lengths)


def test_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [9, -2, -2], [9, 2, -2], [9, 0, 2],
                [-2, 9, -2], [2, 9, -2], [0, 9, 2],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        process=False,
    )


@unittest.skipUnless(cuda_available(), "CUDA is required for fused Warp integration tests.")
class FusedWarpIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = test_mesh()
        self.sample_points, self.sample_face_ids = interior_sample_points(self.mesh, 4)
        self.warp = WarpRaycaster(self.mesh)
        self.reference = GenericWarp(self.warp)

    def test_vision_matches_generic_warp_path(self) -> None:
        config = AnalysisConfig(
            horizontal_fov_degrees=70,
            max_distance=20,
            samples_per_triangle=4,
            min_visible_samples=2,
        )
        poses = [
            PlayerPose(100, np.zeros(3), 0, 0),
            PlayerPose(104, np.zeros(3), 90, 0),
        ]
        reference = VisibilityAnalyzer(
            self.mesh, config,
            sample_points=self.sample_points,
            sample_face_ids=self.sample_face_ids,
            raycaster=self.reference,
        ).analyze_timeline(poses, show_progress=False)
        fused = VisibilityAnalyzer(
            self.mesh, config,
            sample_points=self.sample_points,
            sample_face_ids=self.sample_face_ids,
            raycaster=self.warp,
        ).analyze_timeline(poses, show_progress=False)

        self.assertEqual(fused.tested_rays, reference.tested_rays)
        np.testing.assert_array_equal(fused.instant_masks, reference.instant_masks)
        np.testing.assert_array_equal(fused.cumulative_masks, reference.cumulative_masks)

    def test_flash_matches_generic_warp_path(self) -> None:
        config = FlashCoverageConfig(
            max_distance=20,
            samples_per_triangle=4,
            falloff_power=1.5,
        )
        flash = FlashDetonation(0, 100, (0, 0, 0), "de_test")
        reference = FlashCoverageAnalyzer(
            self.mesh, config,
            sample_points=self.sample_points,
            sample_face_ids=self.sample_face_ids,
            raycaster=self.reference,
        ).analyze(flash, show_progress=False)
        fused = FlashCoverageAnalyzer(
            self.mesh, config,
            sample_points=self.sample_points,
            sample_face_ids=self.sample_face_ids,
            raycaster=self.warp,
        ).analyze(flash, show_progress=False)

        self.assertEqual(fused.tested_rays, reference.tested_rays)
        np.testing.assert_allclose(fused.intensities, reference.intensities, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
