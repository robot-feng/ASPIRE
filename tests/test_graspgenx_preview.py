from __future__ import annotations

import json

import numpy as np
import pytest

from cap_harness.contracts import GraspCandidate, GraspSet, PointCloud, Pose
from cap_harness.providers.graspgenx.preview import write_grasp_preview


def _candidate(*, frame="base", direction=(0, 1, 0), direction_frame="base"):
    return GraspCandidate(
        pose=Pose(position=np.array([0.1, 0.2, 0.3]),
                  quaternion_wxyz=np.array([1, 0, 0, 0]), frame=frame),
        score=0.8,
        width_m=None,
        metadata={"grasp_frame": "tcp", "approach_direction_in_frame": direction,
                  "approach_direction_frame": direction_frame,
                  "approach_direction_semantics": "graspgenx_base_positive_z_toward_object"},
    )


def _read(path):
    lines = path.read_text().splitlines()
    header_end = lines.index("end_header")
    metadata_line = next(line for line in lines if line.startswith("comment metadata "))
    metadata = json.loads(metadata_line.removeprefix("comment metadata "))
    return metadata, np.loadtxt(lines[header_end + 1:])


def test_preview_preserves_scene_pose_and_framed_approach(tmp_path):
    scene = PointCloud(points=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]), frame="base",
                       colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]]))
    grasps = GraspSet(ok=True, grasps=[_candidate()], diagnostics={"snapshot_id": "abc"})
    path = tmp_path / "candidates.ply"
    write_grasp_preview(path, scene, grasps, snapshot_id="abc", timestamp_s=10.5)
    metadata, vertices = _read(path)

    assert metadata["frame"] == "base"
    assert metadata["snapshot_id"] == "abc"
    assert metadata["timestamp_s"] == 10.5
    assert metadata["collision_checked"] is False
    assert metadata["gripper_geometry_shown"] is False
    assert metadata["candidates"][0]["pose_frame"] == "tcp"
    assert metadata["candidates"][0]["approach_shown"] is True
    np.testing.assert_array_equal(vertices[:2, :3], scene.points)
    np.testing.assert_array_equal(vertices[:2, 3:6], [[255, 0, 0], [0, 128, 255]])
    assert np.all(vertices[:2, 6] == -1)
    # TCP +Z is not the approach here: its approach is along parent-frame +Y.
    approach = vertices[vertices[:, 7] == 4, :3]
    np.testing.assert_allclose(approach[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(approach[-1], [0.1, 0.16, 0.3])


def test_preview_rejects_unrelated_scene_and_preserves_existing_files(tmp_path):
    scene = PointCloud(points=np.zeros((1, 3)), frame="base")
    target = tmp_path / "preview.ply"
    target.write_text("existing")
    with pytest.raises(ValueError, match="frames"):
        write_grasp_preview(target, scene, GraspSet(ok=True, grasps=[_candidate(frame="camera")]),
                            snapshot_id="abc", timestamp_s=1)
    with pytest.raises(ValueError, match="snapshot"):
        write_grasp_preview(target, scene,
                            GraspSet(ok=True, grasps=[_candidate()], diagnostics={"snapshot_id": "other"}),
                            snapshot_id="abc", timestamp_s=1)
    with pytest.raises(FileExistsError):
        write_grasp_preview(target, scene, GraspSet(ok=True, grasps=[_candidate()]),
                            snapshot_id="abc", timestamp_s=1)
    assert target.read_text() == "existing"


def test_preview_omits_unknown_approach_and_downsamples_scene(tmp_path):
    scene = PointCloud(points=np.zeros((6, 3)), frame="base")
    grasps = GraspSet(ok=True, grasps=[_candidate(direction_frame="other")])
    path = tmp_path / "preview.ply"
    write_grasp_preview(path, scene, grasps, snapshot_id="xyz", timestamp_s=0,
                        max_scene_points=2)
    metadata, vertices = _read(path)
    assert metadata["scene_points_original"] == 6
    assert metadata["scene_points_shown"] == 2
    assert not metadata["candidates"][0]["approach_shown"]
    assert not np.any(vertices[:, 7] == 4)


def test_preview_rejects_bad_approach_without_creating_output(tmp_path):
    scene = PointCloud(points=np.zeros((1, 3)), frame="base")
    grasps = GraspSet(ok=True, grasps=[_candidate(direction=(0, 2, 0))])
    path = tmp_path / "preview.ply"
    with pytest.raises(ValueError, match="unit vector"):
        write_grasp_preview(path, scene, grasps, snapshot_id="xyz", timestamp_s=0)
    assert not path.exists()
