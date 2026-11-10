"""Unknown grasp widths remain unknown through selection and recording."""

import json

import pytest

from cap_harness.api import CapApi
from cap_harness.contracts import GraspCandidate, GraspSet, Pose
from cap_harness.geometry import select_top_down_grasp


def test_unknown_width_can_be_selected() -> None:
    down = Pose([0.1, 0.2, 0.3], [0, 1, 0, 0], frame="base")
    unknown = GraspCandidate(down, 0.9, None)
    numeric = GraspCandidate(down, 0.8, 0.04)
    grasps = GraspSet(ok=True, grasps=[unknown, numeric])

    assert select_top_down_grasp(grasps) is unknown
    # select_grasp is a pure helper; it does not use any adapter state.
    assert CapApi.select_grasp(None, grasps, strategy="highest_score") is unknown
    assert CapApi.select_grasp(None, grasps, strategy="top_down") is unknown


def test_unknown_width_serializes_as_null(tmp_path) -> None:
    pytest.importorskip("imageio.v2")
    from cap_harness.artifacts import Serializer

    down = Pose([0.1, 0.2, 0.3], [0, 1, 0, 0], frame="base")
    unknown = GraspCandidate(down, 0.9, None)
    numeric = GraspCandidate(down, 0.8, 0.04)
    path = tmp_path / "grasp.json"
    Serializer(tmp_path).dump(path, unknown)
    encoded = json.loads(path.read_text())
    assert encoded["width_m"] is None
    Serializer(tmp_path).dump(path, numeric)
    assert json.loads(path.read_text())["width_m"] == 0.04
