"""Interchangeable CPU and NVIDIA Warp raycasting backends.

The generic :meth:`Raycaster.visible` interface remains the reference path and
is also used by the CPU backend.  The Warp backend additionally exposes fused
vision and flash operations.  Those operations keep static face samples on the
GPU and return compact per-face results instead of transferring several arrays
for every ray dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import trimesh

try:
    import warp as wp
except ImportError:  # Allows CPU-only installations to import this package.
    wp = None


class Raycaster(Protocol):
    name: str

    def visible(self, origins: np.ndarray, directions: np.ndarray, face_ids: np.ndarray, lengths: np.ndarray) -> np.ndarray: ...


class CpuRaycaster:
    name = "cpu"

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.mesh = mesh

    def visible(self, origins: np.ndarray, directions: np.ndarray, face_ids: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        # trimesh returns -1 for misses. Its built-in intersector has no
        # per-ray maximum-distance parameter; this is safe because a visible
        # target face is necessarily the first face along its ray.
        hits = self.mesh.ray.intersects_first(origins, directions, max_distance=lengths)
        return hits == face_ids


if wp is not None:
    @wp.kernel
    def _raycast_kernel(
        mesh_id: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        directions: wp.array(dtype=wp.vec3),
        face_ids: wp.array(dtype=wp.int32),
        lengths: wp.array(dtype=float),
        visible_out: wp.array(dtype=wp.int32),
    ):
        ray_id = wp.tid()
        query = wp.mesh_query_ray(mesh_id, origins[ray_id], directions[ray_id], lengths[ray_id])
        if query.result and query.face == face_ids[ray_id]:
            visible_out[ray_id] = 1
        else:
            visible_out[ray_id] = 0


    @wp.kernel
    def _vision_faces_kernel(
        mesh_id: wp.uint64,
        sample_points: wp.array(dtype=wp.vec3d),
        samples_per_face: int,
        origins: wp.array(dtype=wp.vec3d),
        forwards: wp.array(dtype=wp.vec3d),
        pose_index: int,
        output_pose_index: int,
        words_per_pose: int,
        max_distance: wp.float64,
        cos_half_fov: wp.float64,
        endpoint_epsilon: wp.float64,
        min_visible_samples: int,
        cumulative_samples: wp.array(dtype=wp.uint8),
        instant_words: wp.array(dtype=wp.uint32),
        cumulative_words: wp.array(dtype=wp.uint32),
        candidate_counts: wp.array(dtype=wp.float32),
    ):
        """Test every interior sample for one face and pack its result bits."""
        face_id = wp.tid()
        origin = origins[pose_index]
        forward = forwards[pose_index]
        maximum_squared = max_distance * max_distance
        instant_count = wp.int32(0)
        cumulative_count = wp.int32(0)
        candidate_count = wp.int32(0)

        for sample_offset in range(samples_per_face):
            sample_id = face_id * samples_per_face + sample_offset
            offset = sample_points[sample_id] - origin
            distance_squared = wp.dot(offset, offset)
            visible = wp.bool(False)
            if distance_squared > wp.float64(1.0e-12) and distance_squared <= maximum_squared:
                distance = wp.sqrt(distance_squared)
                direction_double = offset / distance
                if wp.dot(direction_double, forward) >= cos_half_fov:
                    candidate_count += 1
                    ray_origin = wp.vec3(
                        wp.float32(origin[0]), wp.float32(origin[1]), wp.float32(origin[2]),
                    )
                    ray_direction = wp.vec3(
                        wp.float32(direction_double[0]),
                        wp.float32(direction_double[1]),
                        wp.float32(direction_double[2]),
                    )
                    query = wp.mesh_query_ray(
                        mesh_id, ray_origin, ray_direction, wp.float32(distance + endpoint_epsilon),
                    )
                    visible = query.result and query.face == face_id

            if visible:
                instant_count += 1
                cumulative_samples[sample_id] = wp.uint8(1)
            if cumulative_samples[sample_id] != wp.uint8(0):
                cumulative_count += 1

        candidate_counts[face_id] = wp.float32(candidate_count)
        word_index = output_pose_index * words_per_pose + face_id // 32
        bit_mask = wp.uint32(1) << wp.uint32(face_id % 32)
        if instant_count >= min_visible_samples:
            wp.atomic_or(instant_words, word_index, bit_mask)
        if cumulative_count >= min_visible_samples:
            wp.atomic_or(cumulative_words, word_index, bit_mask)


    @wp.kernel
    def _flash_faces_kernel(
        mesh_id: wp.uint64,
        sample_points: wp.array(dtype=wp.vec3d),
        samples_per_face: int,
        origin: wp.vec3d,
        max_distance: wp.float64,
        falloff_power: wp.float64,
        endpoint_epsilon: wp.float64,
        intensities: wp.array(dtype=wp.float32),
        candidate_counts: wp.array(dtype=wp.float32),
    ):
        """Compute one face's flash intensity without materializing ray arrays."""
        face_id = wp.tid()
        maximum_squared = max_distance * max_distance
        strongest = wp.float64(0.0)
        candidate_count = wp.int32(0)

        for sample_offset in range(samples_per_face):
            sample_id = face_id * samples_per_face + sample_offset
            offset = sample_points[sample_id] - origin
            distance_squared = wp.dot(offset, offset)
            if distance_squared > wp.float64(1.0e-12) and distance_squared <= maximum_squared:
                candidate_count += 1
                distance = wp.sqrt(distance_squared)
                direction_double = offset / distance
                ray_origin = wp.vec3(
                    wp.float32(origin[0]), wp.float32(origin[1]), wp.float32(origin[2]),
                )
                ray_direction = wp.vec3(
                    wp.float32(direction_double[0]),
                    wp.float32(direction_double[1]),
                    wp.float32(direction_double[2]),
                )
                query = wp.mesh_query_ray(
                    mesh_id, ray_origin, ray_direction, wp.float32(distance + endpoint_epsilon),
                )
                if query.result and query.face == face_id:
                    intensity = wp.pow(
                        wp.float64(1.0) - distance / max_distance, falloff_power,
                    )
                    strongest = wp.max(strongest, intensity)

        intensities[face_id] = wp.float32(strongest)
        candidate_counts[face_id] = wp.float32(candidate_count)


class WarpRaycaster:
    name = "nvidia-warp"

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        if wp is None:
            raise RuntimeError("NVIDIA Warp is not installed.")
        wp.init()
        if wp.get_cuda_device_count() <= 0:
            raise RuntimeError("No CUDA device is available.")
        points = wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device="cuda")
        indices = wp.array(mesh.faces.reshape(-1).astype(np.int32), dtype=wp.int32, device="cuda")
        self.mesh = wp.Mesh(points=points, indices=indices)
        self.device = wp.get_device("cuda")
        self.face_count = len(mesh.faces)
        self._sample_points: dict[int, tuple[np.ndarray, object]] = {}

    def _gpu_samples(self, sample_points: np.ndarray, samples_per_face: int):
        expected = self.face_count * samples_per_face
        if sample_points.shape != (expected, 3):
            raise ValueError(
                f"Expected {expected} ordered sample points for {self.face_count} faces."
            )
        cached = self._sample_points.get(samples_per_face)
        if cached is None or cached[0] is not sample_points:
            gpu_points = wp.array(
                np.ascontiguousarray(sample_points, dtype=np.float64),
                dtype=wp.vec3d,
                device=self.device,
            )
            cached = (sample_points, gpu_points)
            self._sample_points[samples_per_face] = cached
        return cached[1]

    def visible(self, origins: np.ndarray, directions: np.ndarray, face_ids: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        count = len(face_ids)
        if count == 0:
            return np.empty(0, dtype=bool)
        output = wp.zeros(count, dtype=wp.int32, device="cuda")
        wp.launch(
            _raycast_kernel,
            dim=count,
            inputs=[
                self.mesh.id,
                wp.array(origins.astype(np.float32), dtype=wp.vec3, device="cuda"),
                wp.array(directions.astype(np.float32), dtype=wp.vec3, device="cuda"),
                wp.array(face_ids.astype(np.int32), dtype=wp.int32, device="cuda"),
                wp.array(lengths.astype(np.float32), dtype=float, device="cuda"),
            ],
            outputs=[output],
        )
        return output.numpy().astype(bool)

    def analyze_vision_timeline(
        self,
        sample_points: np.ndarray,
        samples_per_face: int,
        origins: np.ndarray,
        forwards: np.ndarray,
        max_distance: float,
        cos_half_fov: float,
        endpoint_epsilon: float,
        min_visible_samples: int,
        progress_callback: Callable[[int, int], None] | None = None,
        *,
        pose_chunk_size: int = 32,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Run ordered vision poses while retaining all large arrays on CUDA."""
        if len(origins) != len(forwards):
            raise ValueError("Expected one forward vector for every vision origin.")
        pose_count = len(origins)
        packed_width = (self.face_count + 7) // 8
        words_per_pose = (self.face_count + 31) // 32
        instant_masks = np.zeros((pose_count, packed_width), dtype=np.uint8)
        cumulative_masks = np.zeros_like(instant_masks)
        if not pose_count:
            return instant_masks, cumulative_masks, 0

        gpu_samples = self._gpu_samples(sample_points, samples_per_face)
        gpu_origins = wp.array(
            np.ascontiguousarray(origins, dtype=np.float64), dtype=wp.vec3d, device=self.device,
        )
        gpu_forwards = wp.array(
            np.ascontiguousarray(forwards, dtype=np.float64), dtype=wp.vec3d, device=self.device,
        )
        cumulative_samples = wp.zeros(
            len(sample_points), dtype=wp.uint8, device=self.device,
        )
        candidate_counts = wp.empty(
            self.face_count, dtype=wp.float32, device=self.device,
        )
        ray_count = 0

        for chunk_start in range(0, pose_count, pose_chunk_size):
            chunk_count = min(pose_chunk_size, pose_count - chunk_start)
            instant_words = wp.zeros(
                chunk_count * words_per_pose, dtype=wp.uint32, device=self.device,
            )
            cumulative_words = wp.zeros(
                chunk_count * words_per_pose, dtype=wp.uint32, device=self.device,
            )
            candidate_totals = wp.empty(
                chunk_count, dtype=wp.float32, device=self.device,
            )
            for local_index in range(chunk_count):
                pose_index = chunk_start + local_index
                wp.launch(
                    _vision_faces_kernel,
                    dim=self.face_count,
                    inputs=[
                        self.mesh.id,
                        gpu_samples,
                        samples_per_face,
                        gpu_origins,
                        gpu_forwards,
                        pose_index,
                        local_index,
                        words_per_pose,
                        max_distance,
                        cos_half_fov,
                        endpoint_epsilon,
                        min_visible_samples,
                        cumulative_samples,
                        instant_words,
                        cumulative_words,
                        candidate_counts,
                    ],
                    device=self.device,
                )
                wp.utils.array_sum(
                    candidate_counts,
                    out=candidate_totals[local_index : local_index + 1],
                )

            instant_host = instant_words.numpy().reshape(chunk_count, words_per_pose)
            cumulative_host = cumulative_words.numpy().reshape(chunk_count, words_per_pose)
            totals_host = candidate_totals.numpy().astype(np.int64)
            chunk_end = chunk_start + chunk_count
            instant_masks[chunk_start:chunk_end] = (
                instant_host.view(np.uint8).reshape(chunk_count, words_per_pose * 4)[:, :packed_width]
            )
            cumulative_masks[chunk_start:chunk_end] = (
                cumulative_host.view(np.uint8).reshape(chunk_count, words_per_pose * 4)[:, :packed_width]
            )
            ray_count += int(totals_host.sum(dtype=np.int64))
            if progress_callback:
                for completed in range(chunk_start + 1, chunk_end + 1):
                    progress_callback(completed, pose_count)

        return instant_masks, cumulative_masks, ray_count

    def analyze_flash(
        self,
        sample_points: np.ndarray,
        samples_per_face: int,
        origin: np.ndarray,
        max_distance: float,
        falloff_power: float,
        endpoint_epsilon: float,
    ) -> tuple[np.ndarray, int]:
        """Run fused distance, occlusion, and intensity work for one flash."""
        gpu_samples = self._gpu_samples(sample_points, samples_per_face)
        intensities = wp.empty(self.face_count, dtype=wp.float32, device=self.device)
        candidate_counts = wp.empty(self.face_count, dtype=wp.float32, device=self.device)
        candidate_total = wp.empty(1, dtype=wp.float32, device=self.device)
        origin = np.asarray(origin, dtype=np.float64)
        wp.launch(
            _flash_faces_kernel,
            dim=self.face_count,
            inputs=[
                self.mesh.id,
                gpu_samples,
                samples_per_face,
                wp.vec3d(float(origin[0]), float(origin[1]), float(origin[2])),
                max_distance,
                falloff_power,
                endpoint_epsilon,
                intensities,
                candidate_counts,
            ],
            device=self.device,
        )
        wp.utils.array_sum(candidate_counts, out=candidate_total)
        return intensities.numpy(), int(candidate_total.numpy()[0])


def create_raycaster(mesh: trimesh.Trimesh, prefer_gpu: bool) -> Raycaster:
    """Return Warp when usable; otherwise return the dependable CPU backend."""
    if prefer_gpu:
        try:
            return WarpRaycaster(mesh)
        except Exception as error:
            print(f"GPU raycasting unavailable ({error}); using CPU raycasting.")
    return CpuRaycaster(mesh)
