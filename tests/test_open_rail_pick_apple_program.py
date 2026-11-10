from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cap_harness.contracts import GraspCandidate, Pose, TaskContext
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor


PROGRAM = (Path(__file__).resolve().parents[1] / "examples/open_rail/pick_apple.py").read_text()


def _run(*, config, collision_aware=True, tcp=True):
    calls = []
    pose = Pose(
        position=np.array([0.4, 0.1, 0.3]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="base",
    )
    candidate = GraspCandidate(
        pose=pose,
        score=0.8,
        width_m=None,
        metadata={
            "grasp_frame": "tcp",
            "tcp_transform_applied": tcp,
            "approach_direction_frame": "base",
            "approach_direction_in_frame": [1.0, 0.0, 0.0],
            "approach_direction_semantics": "graspgenx_base_positive_z_toward_object",
        },
    )
    observation = SimpleNamespace(robot_state=SimpleNamespace(base_frame="base"))
    localization = SimpleNamespace(ok=True, diagnostics={"snapshot_id": "head@12"})
    grasps = SimpleNamespace(ok=True, grasps=(candidate,), diagnostics={"tcp_transform_applied": tcp})
    scene = object()
    trajectory = SimpleNamespace(collision_aware=collision_aware)
    registry = ToolRegistry()

    def register(name, function):
        registry.register(function, name=name, layer="shared_atomic", capability="test", public=True)

    register(
        "get_task_context",
        lambda: TaskContext("open_rail", 0, "apple", "apple", "real", metadata={"open_rail_pick_apple": config}),
    )
    register("get_observation", lambda name: observation)

    def localize(query, received, **kwargs):
        calls.append(("localize", received, kwargs))
        return localization

    def generate(received, obs, **kwargs):
        calls.append(("grasp", received, obs, kwargs))
        return grasps

    def get_scene(received):
        calls.append(("scene", received))
        return scene

    def plan(target, **kwargs):
        calls.append(("plan", target, kwargs))
        return SimpleNamespace(ok=True, trajectory=trajectory)

    register("localize_object_from_observation", localize)
    register("generate_grasps_for_localization", generate)
    register("get_planning_scene", get_scene)
    register("plan_motion", plan)
    return ProgramExecutor(registry).execute_program(PROGRAM), calls, observation, scene


def _config():
    return {
        "camera_name": "head",
        "arm": "right",
        "pregrasp_distance_m": 0.1,
        "max_camera_state_skew_s": 0.02,
        "minimum_depth_points": 20,
        "minimum_depth_fraction": 0.5,
        "max_candidates": 3,
    }


def test_pregrasp_preview_uses_one_observation_and_never_claims_grasp():
    execution, calls, observation, scene = _run(config=_config())
    assert execution.ok, execution.error
    assert execution.result["ok"] is False
    assert execution.result["stage"] == "pregrasp_preview"
    assert [call[0] for call in calls] == ["localize", "grasp", "scene", "plan"]
    assert calls[0][1] is observation and calls[1][2] is observation and calls[2][1] is observation
    assert calls[3][2]["planning_scene"] is scene
    np.testing.assert_allclose(calls[3][1].position, [0.3, 0.1, 0.3])
    assert not any(call.tool in {"execute_trajectory", "step", "open_gripper"} for call in execution.calls)


def test_unchecked_plan_and_missing_tcp_cannot_be_returned_as_preview():
    unchecked, _, _, _ = _run(config=_config(), collision_aware=False)
    missing_tcp, calls, _, _ = _run(config=_config(), tcp=False)
    assert unchecked.ok and unchecked.result["stage"] == "pregrasp_planning"
    assert missing_tcp.ok and missing_tcp.result["stage"] == "grasp_generation"
    assert [call[0] for call in calls] == ["localize", "grasp"]


def test_missing_configuration_fails_before_observation():
    execution, calls, _, _ = _run(config=None)
    assert execution.ok and execution.result["stage"] == "configuration"
    assert not calls
    assert [call.tool for call in execution.calls] == ["get_task_context"]
