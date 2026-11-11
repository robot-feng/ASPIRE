"""Bounded sparse voxel boundary mesh for preserved observed point clouds."""

from __future__ import annotations

import numpy as np


class SparseSceneLimitError(ValueError):
    """The complete point cloud cannot fit the configured sparse-mesh budget."""


# Budgets are absolute, not proportional to the distance between points.
# One quad produces four vertices and two triangles before mesh conversion.
MAX_INPUT_POINTS = 1_500_000
MAX_OCCUPIED_VOXELS = 200_000
MAX_SURFACE_QUADS = 400_000

_RECORD = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
_FACES = (
    (0, 1, ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
    (0, -1, ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    (1, 1, ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
    (1, -1, ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0))),
    (2, 1, ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
    (2, -1, ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
)


def _records(points: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(points).view(_RECORD).reshape(-1)


def sparse_voxel_mesh(points, pitch: float, *, max_input_points=MAX_INPUT_POINTS,
                      max_voxels=MAX_OCCUPIED_VOXELS, max_quads=MAX_SURFACE_QUADS):
    """Return outward-wound mesh arrays containing every input point.

    The voxel origin and initial integer-cell assignment match the existing
    dense ``_mesh_from_pointcloud`` path. A neighbouring cell is additionally
    occupied if floating-point rounding puts a point just outside its cell.
    Limits are checked before constructing mesh arrays; on any limit or numeric
    error the entire request fails.
    """
    if isinstance(pitch, bool) or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("voxel size must be positive and finite")
    for name, limit in (("input point", max_input_points), ("occupied voxel", max_voxels),
                        ("surface quad", max_quads)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{name} limit must be a positive integer")
    raw = np.asarray(points)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
        raise ValueError("scene points must have nonempty shape (N, 3)")
    if len(raw) > max_input_points:
        raise SparseSceneLimitError(f"preserved scene exceeds input point limit {max_input_points}")
    pts = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(pts).all():
        raise ValueError("scene points must be finite")
    origin = pts.min(axis=0) - pitch
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = (pts - origin) / pitch
    # Double precision must represent the integer coordinates and neighbouring
    # corners exactly; otherwise fail instead of dropping or misplacing points.
    if (not np.isfinite(origin).all() or not np.isfinite(scaled).all()
            or np.any(scaled < 0) or np.any(scaled >= 2**52 - 2)
            or np.any(origin + pitch <= origin)):
        raise ValueError("scene voxel coordinates exceed safe numeric precision")
    cells = np.floor(scaled).astype(np.int64)
    lower = origin + cells.astype(np.float64) * pitch
    upper = origin + (cells + 1).astype(np.float64) * pitch
    if np.any(upper <= lower):
        raise ValueError("scene points cannot be reliably enclosed by voxel geometry")
    # In rare floating-point boundary cases the dense assignment's computed
    # corner falls just beyond its input point. Keep that original occupied
    # cell and add the adjacent cell that actually encloses the point.
    below, above = pts < lower, pts > upper
    outside = np.any(below | above, axis=1)
    if np.any(outside):
        extra = cells[outside] - below[outside].astype(np.int64) + above[outside].astype(np.int64)
        extra_lower = origin + extra.astype(np.float64) * pitch
        extra_upper = origin + (extra + 1).astype(np.float64) * pitch
        if (np.any(extra < 0) or np.any(extra >= 2**52 - 2)
                or np.any(pts[outside] < extra_lower) or np.any(pts[outside] > extra_upper)
                or np.any(extra_upper <= extra_lower)):
            raise ValueError("scene points cannot be reliably enclosed by voxel geometry")
        cells = np.concatenate((cells, extra), axis=0)
    voxels = np.unique(cells, axis=0)
    if len(voxels) > max_voxels:
        raise SparseSceneLimitError(f"preserved scene exceeds occupied voxel limit {max_voxels}")
    keys = _records(voxels)
    boundaries = []
    quad_count = 0
    for axis, direction, template in _FACES:
        adjacent = voxels.copy()
        adjacent[:, axis] += direction
        queries = _records(adjacent)
        indices = np.searchsorted(keys, queries)
        found = indices < len(keys)
        found[found] = keys[indices[found]] == queries[found]
        boundary = voxels[~found]
        quad_count += len(boundary)
        if quad_count > max_quads:
            raise SparseSceneLimitError(f"preserved scene exceeds surface quad limit {max_quads}")
        boundaries.append((boundary, np.asarray(template, dtype=np.float64)))
    vertices = np.empty((quad_count * 4, 3), dtype=np.float64)
    triangles = np.empty((quad_count * 2, 3), dtype=np.int64)
    start = 0
    for boundary, template in boundaries:
        count = len(boundary)
        end = start + count
        vertices[start * 4:end * 4] = (
            (boundary[:, None, :] + template[None, :, :]).reshape(-1, 3) * pitch + origin
        )
        first = np.arange(start, end, dtype=np.int64) * 4
        triangles[start * 2:end * 2] = np.column_stack(
            (first, first + 2, first + 1, first, first + 3, first + 2)
        ).reshape(-1, 3)
        start = end
    return vertices, triangles
