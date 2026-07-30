"""Map mesh loading and robust interior sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from awpy.visibility import VisibilityChecker


def load_tri_mesh(tri_path: Path) -> trimesh.Trimesh:
    """Load an Awpy ``.tri`` file without merging or altering its faces."""
    triangles = VisibilityChecker.read_tri_file(tri_path)
    vertices = np.empty((len(triangles) * 3, 3), dtype=np.float64)
    faces = np.empty((len(triangles), 3), dtype=np.int64)
    for index, triangle in enumerate(triangles):
        base = index * 3
        vertices[base : base + 3] = (
            (triangle.p1.x, triangle.p1.y, triangle.p1.z),
            (triangle.p2.x, triangle.p2.y, triangle.p2.z),
            (triangle.p3.x, triangle.p3.y, triangle.p3.z),
        )
        faces[index] = (base, base + 1, base + 2)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def interior_sample_points(mesh: trimesh.Trimesh, samples_per_triangle: int) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly interior barycentric samples and their source face IDs.

    The old approach tested mesh vertices. Vertices are shared boundaries, so
    a ray could be attributed to a neighboring face. These samples deliberately
    stay inside each face: its center plus three points close to its corners.
    """
    triangles = mesh.triangles
    if samples_per_triangle == 1:
        weights = np.array(((1 / 3, 1 / 3, 1 / 3),), dtype=np.float64)
    elif samples_per_triangle == 4:
        weights = np.array(
            (
                (1 / 3, 1 / 3, 1 / 3),
                (0.80, 0.10, 0.10),
                (0.10, 0.80, 0.10),
                (0.10, 0.10, 0.80),
            ),
            dtype=np.float64,
        )
    else:
        raise ValueError("samples_per_triangle must be 1 or 4.")

    points = np.einsum("sv,nvj->nsj", weights, triangles).reshape(-1, 3)
    face_ids = np.repeat(np.arange(len(mesh.faces), dtype=np.int32), len(weights))
    return points, face_ids


def forward_vector(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    """Convert CS2 yaw/pitch angles into a normalized world-space direction."""
    yaw, pitch = np.radians((yaw_degrees, pitch_degrees))
    return np.array(
        (np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)),
        dtype=np.float64,
    )
