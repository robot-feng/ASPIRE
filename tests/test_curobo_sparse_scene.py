"""CPU-only preserved-scene voxel checks; no cuRobo or device required."""

import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest

from cap_harness.providers.curobo import service
from cap_harness.providers.curobo.sparse_scene import SparseSceneLimitError, sparse_voxel_mesh
from cap_harness.providers.wire import encode_numpy


class Mesh:
    def __init__(self, name, *, pose, vertices, faces):
        self.name = name
        self.pose = pose
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def _volume(mesh):
    triangles = mesh.vertices[mesh.faces]
    return np.einsum("ij,ij->i", triangles[:, 0],
                     np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6


@pytest.mark.parametrize("points,pitch", [
    ([[0, 0, 0]], 0.01),
    ([[-.02, -.02, -.02], [-.01, -.02, -.02], [-.01, -.02, -.02],
      [0, 0, 0], [.0101, 0, 0], [.0099, 0, 0]], .01),
    ([[-2, 1, -1], [-1.2, 1.4, -1.3], [-1.8, 1.4, -.9]], .2),
])
def test_sparse_mesh_matches_dense_output_for_small_scenes(points, pitch):
    points = np.asarray(points, dtype=np.float64)
    dense = service._voxelized_scene_mesh(points, voxel_size_m=pitch, mesh_type=Mesh)
    sparse = service._sparse_voxelized_scene_mesh(points, voxel_size_m=pitch, mesh_type=Mesh)
    np.testing.assert_array_equal(sparse.vertices, dense.vertices)
    np.testing.assert_array_equal(sparse.faces, dense.faces)
    assert sparse.name == dense.name and sparse.pose == dense.pose


def test_shared_internal_faces_removed_and_winding_outward():
    mesh = service._sparse_voxelized_scene_mesh(np.array([[0, 0, 0], [1.1, 0, 0]]),
                                               voxel_size_m=1, mesh_type=Mesh)
    assert len(mesh.vertices) == 40 and len(mesh.faces) == 20
    assert _volume(mesh) == pytest.approx(2.0)


def test_far_points_negative_coordinates_and_duplicate_boundary_are_all_represented():
    points = np.array([[-1000, -2000, -3000], [-1000, -2000, -3000],
                       [-999, -2000, -3000], [2000, 3000, 4000]], dtype=float)
    vertices, faces = sparse_voxel_mesh(points, 1.0)
    # Two adjacent voxels and one distant voxel: 10 + 6 exterior quads.
    assert vertices.shape == (64, 3) and faces.shape == (32, 3)
    for point in points:
        quads = vertices.reshape(-1, 4, 3)
        # Each cell's corners include its point on at least one exposed face.
        assert any(np.all(point >= quad.min(axis=0)) and np.all(point <= quad.max(axis=0))
                   for quad in quads)


def test_rounding_boundary_keeps_original_and_adjacent_enclosing_voxel():
    points = np.array([[0.6261305809020996, -3.4032630920410156, -13.26635456085205],
                       [2.9147400856018066, 1.4717369079589844, 0.05990157648921013]])
    pitch = .025
    origin = points.min(axis=0) - pitch
    cells = np.floor((points - origin) / pitch).astype(np.int64)
    assert np.any(points < origin + cells * pitch)  # Regression: roundoff at a face.
    vertices, faces = sparse_voxel_mesh(points, pitch)
    assert len(faces) == 32  # One isolated voxel plus two adjacent boundary voxels.
    assert np.isfinite(vertices).all()


@pytest.mark.parametrize("kwargs,limit", [
    ({"max_input_points": 1}, "input point"),
    ({"max_voxels": 1}, "occupied voxel"),
    ({"max_quads": 9}, "surface quad"),
])
def test_resource_budgets_reject_without_partial_output(kwargs, limit):
    with pytest.raises(SparseSceneLimitError, match=limit):
        sparse_voxel_mesh(np.array([[0, 0, 0], [1.1, 0, 0]]), 1.0, **kwargs)


def test_numeric_overflow_and_invalid_pitch_fail_closed():
    with pytest.raises(ValueError, match="numeric precision"):
        sparse_voxel_mesh(np.array([[0, 0, 0], [1e100, 0, 0]]), 0.01)
    for pitch in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="voxel size"):
            sparse_voxel_mesh(np.array([[0, 0, 0]]), pitch)


def test_sparse_budget_fails_before_any_python_mesh_conversion(monkeypatch):
    def reject(points, pitch):
        return sparse_voxel_mesh(points, pitch, max_voxels=1)

    monkeypatch.setattr(service, "sparse_voxel_mesh", reject)
    with pytest.raises(SparseSceneLimitError, match="occupied voxel"):
        service._sparse_voxelized_scene_mesh(np.array([[0, 0, 0], [1.1, 0, 0]]),
                                             voxel_size_m=1,
                                             mesh_type=lambda *_a, **_k: pytest.fail("constructed mesh"))


def _request(points):
    points = np.asarray(points, dtype=np.float32)
    return service.BaseRequest.model_validate({
        "base_frame": "base", "model": "agibot_g1",
        "joint_positions": {"right": [0] * 7},
        "joint_names": {"right": [f"j{i}" for i in range(7)]},
        "base_transforms": {"right": np.eye(4).tolist()},
        "end_effector_links": {"right": "tcp"}, "voxel_size_m": 1.0,
        "scene_points_base64": encode_numpy(points),
        "target_scene_partition": {"mode": "pixel_partition_v2", "snapshot_id": "test",
                                   "snapshot_fingerprint": "a" * 64,
                                   "scene_fingerprint": "b" * 64,
                                   "scene_points_sha256": hashlib.sha256(points.tobytes()).hexdigest()},
    })


def test_service_v2_handles_far_scene_without_filter_or_carve(monkeypatch):
    scene_module = ModuleType("curobo.scene")
    scene_module.Scene = lambda *, mesh: mesh
    monkeypatch.setitem(sys.modules, "curobo.scene", scene_module)
    monkeypatch.setattr(service, "_collision_disabled", lambda *_: False)
    monkeypatch.setattr(service, "_drop_distant_points", lambda *_: pytest.fail("filtered"))
    monkeypatch.setattr(service, "_carve_robot_points", lambda *_: pytest.fail("carved"))
    build = service._sparse_voxelized_scene_mesh
    monkeypatch.setattr(service, "_sparse_voxelized_scene_mesh",
                        lambda points, **kwargs: build(points, **kwargs, mesh_type=Mesh))
    output = service._scene(_request([[-100, -1, 0], [0, 0, 0], [100, 1, 0]]), object())
    mesh = output[0]
    assert mesh.vertices.min(axis=0)[0] <= -100
    assert mesh.vertices.max(axis=0)[0] >= 100
    assert len(mesh.faces) == 36


def test_service_reports_sparse_budget_as_422_before_mesh_creation(monkeypatch):
    scene_module = ModuleType("curobo.scene")
    scene_module.Scene = lambda *, mesh: mesh
    monkeypatch.setitem(sys.modules, "curobo.scene", scene_module)
    monkeypatch.setattr(service, "_collision_disabled", lambda *_: False)
    monkeypatch.setattr(service, "sparse_voxel_mesh",
                        lambda points, pitch: sparse_voxel_mesh(points, pitch, max_quads=1))
    with pytest.raises(service.HTTPException) as error:
        service._scene(_request([[0, 0, 0]]), object())
    assert error.value.status_code == 422 and "surface quad limit" in error.value.detail


def test_legacy_scene_keeps_its_existing_range_filter_and_carve(monkeypatch):
    scene_module = ModuleType("curobo.scene")
    scene_module.Scene = lambda *, mesh: mesh
    monkeypatch.setitem(sys.modules, "curobo.scene", scene_module)
    monkeypatch.setattr(service, "_collision_disabled", lambda *_: False)
    monkeypatch.setattr(service, "_sparse_voxelized_scene_mesh", lambda *_a, **_k: pytest.fail("sparse"))
    seen = []
    monkeypatch.setattr(service, "_carve_robot_points",
                        lambda _request, _robot, points: seen.append(points.copy()) or points)
    monkeypatch.setattr(service, "_voxelized_scene_mesh",
                        lambda points, **_kwargs: points.copy())
    request = _request([[0, 0, 0], [10, 0, 0]]).model_copy(update={"target_scene_partition": None})
    output = service._scene(request, object())
    np.testing.assert_array_equal(seen[0], [[0, 0, 0]])
    np.testing.assert_array_equal(output[0], [[0, 0, 0]])
