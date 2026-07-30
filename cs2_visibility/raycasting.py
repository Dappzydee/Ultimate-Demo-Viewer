"""Interchangeable CPU and NVIDIA Warp raycasting backends."""

from __future__ import annotations

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


def create_raycaster(mesh: trimesh.Trimesh, prefer_gpu: bool) -> Raycaster:
    """Return Warp when usable; otherwise return the dependable CPU backend."""
    if prefer_gpu:
        try:
            return WarpRaycaster(mesh)
        except Exception as error:
            print(f"GPU raycasting unavailable ({error}); using CPU raycasting.")
    return CpuRaycaster(mesh)
