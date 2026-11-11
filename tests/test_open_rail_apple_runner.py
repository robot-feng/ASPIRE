from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import requests
from test_open_rail_scene_partition import _attachment

from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder
from cap_harness.contracts import CameraObservation, Observation, Pose, RobotState, TaskContext
from cap_harness.open_rail import OpenRailAdapter, OpenRailProfile
from cap_harness.open_rail.runner import run_apple_preview
from cap_harness.providers.curobo import service
from cap_harness.providers.wire import encode_numpy

PROGRAM = Path(__file__).resolve().parents[1] / "examples/open_rail/pick_apple.py"
DEPTH_FILTER = {
    "max_depth_deviation_m": 0.05,
    "max_neighbor_depth_step_m": 0.01,
    "min_valid_pixels": 20,
    "min_valid_fraction": 0.8,
    "min_depth_inlier_fraction": 0.8,
    "min_component_fraction": 0.8,
}
SWEEP = {
    "extents_open": [0.09, 0.02, 0.05],
    "offset_open": [0, 0, 0.1],
    "extents_mid": [0.04, 0.02, 0.05],
    "offset_mid": [0, 0, 0.1],
}


class _Control:
    def __init__(self):
        names = {"right": tuple(f"joint_{i}" for i in range(7))}
        self.profile = OpenRailProfile(
            name="fake",
            robot_type="fake",
            model="agibot_g1",
            base_frame="base",
            arms=("right",),
            joint_names=names,
            end_effector_links={"right": "fake_tcp"},
            base_transforms={"right": np.eye(4)},
            arm_aliases={"primary": "right"},
        )
        pose = Pose([0, 0, 0], [1, 0, 0, 0], "base")
        state = RobotState(
            joint_positions={"right": np.zeros(7)},
            joint_velocities={"right": np.zeros(7)},
            end_effector_poses={"right": pose},
            gripper_positions={"right": 1.0},
            joint_names=names,
            base_frame="base",
            embodiment="open_rail_7dof",
            timestamp_s=12.0,
        )
        yy, xx = np.indices((8, 10))
        camera = CameraObservation(
            rgb=np.zeros((8, 10, 3), dtype=np.uint8),
            depth_m=0.6 + 0.0005 * ((xx - 4.5) ** 2 + (yy - 3.5) ** 2),
            intrinsics=np.array([[100, 0, 4], [0, 100, 3], [0, 0, 1]]),
            frame="head_optical",
            camera_pose=pose,
            timestamp_s=12.0,
        )
        self.context = TaskContext(
            "open_rail", 0, "apple", "apple", "real",
            metadata={"open_rail_pick_apple": {
                "camera_name": "head", "arm": "right", "pregrasp_distance_m": 0.1,
                "max_camera_state_skew_s": 0.02, "minimum_depth_points": 20,
                "minimum_depth_fraction": 0.8, "max_candidates": 1,
            }},
        )
        self.observation = Observation({"head": camera}, state, self.context, timestamp_s=12.0)
        self.commands = []

    def get_robot_state(self):
        return self.observation.robot_state

    def get_observation(self):
        return self.observation

    def step(self, action):
        self.commands.append(action)
        raise AssertionError("preview sent a robot command")

    execute_trajectory = set_gripper = set_grippers = step


def _recorder(tmp_path):
    return RunRecorder(
        output_root=tmp_path, benchmark="test", suite="test", task_id=0,
        task_name="apple", seed=0, max_steps=10, capture_videos=False,
    )


@pytest.mark.parametrize(
    ("verified", "expected_stage"),
    [(True, "pregrasp_preview"), (False, "pregrasp_planning")],
)
def test_preview_wires_real_clients_and_reuses_one_snapshot(
    monkeypatch, tmp_path, verified, expected_stage
):
    calls = []

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def request(_session, method, url, **kwargs):
        calls.append((url, kwargs.get("json")))
        if ":8117/" in url:
            mask = np.zeros((8, 10), dtype="<f4")
            if kwargs["json"]["query"] == "apple":
                mask[1:7, 0:4] = 1
            else:
                mask[1:7, 6:10] = 1
            return Response({"results": [{
                "label": "apple", "box_xyxy": [1, 1, 9, 7], "shape": [8, 10],
                "probabilities_base64": base64.b64encode(mask.tobytes()).decode(),
            }], "total_instances": 1, "truncated": False})
        if ":8120/" in url:
            return Response({"grasps_base64": encode_numpy(np.eye(4)[None]),
                             "scores_base64": encode_numpy(np.array([0.8])),
                             "branch_tags": [], "timing": {}})
        if ":8118/" in url:
            return Response({"joint_positions_base64": encode_numpy(np.zeros((2, 7))),
                             "dt_s": 0.1, "collision_diagnostics": {
                                 "collision_disabled": False, "scene_supplied": True,
                                 "world_collision_check_enabled": True,
                                 "self_collision_check_enabled": True,
                                 "robot_model_verified": verified,
                             }})
        raise AssertionError(url)

    monkeypatch.setattr(requests.Session, "request", request)
    control = _Control()
    adapter = OpenRailAdapter(control, control.profile, task_context=control.context,
                              controlled_arms=("right",))
    recorder = _recorder(tmp_path)
    result = run_apple_preview(
        adapter, program_path=PROGRAM, recorder=recorder, depth_filter=DEPTH_FILTER,
        gripper_sweep=SWEEP, gripper_base_to_tcp=np.eye(4), control_period_s=0.05,
    )
    assert result.ok, result.error
    assert result.result["stage"] == expected_stage
    assert result.result["ok"] is False
    assert len(calls) == 4
    assert ":8117/" in calls[0][0] and ":8117/" in calls[1][0]
    assert [calls[i][1]["query"] for i in (0, 1)] == ["apple", "plate"]
    assert ":8120/" in calls[2][0] and ":8118/" in calls[3][0]
    assert calls[3][1]["model"] == "agibot_g1"
    assert control.commands == []
    assert len(list(recorder.root.glob("trace/calls/*wevisone-segment-text/output.json"))) == 2
    preview = (recorder.root / "media/grasp-candidates.ply").read_text()
    assert '"snapshot_id":"head_optical@12.000000000"' in preview
    assert '"collision_checked":false' in preview
    recorder.finalize(program_ok=True, task_success=False, termination_reason="program_completed",
                      program_result=None)


def test_preview_rejects_motion_enabled_adapter_before_reading(tmp_path):
    control = _Control()
    adapter = OpenRailAdapter(control, control.profile, task_context=control.context,
                              allow_physical_motion=True, controlled_arms=("right",))
    recorder = _recorder(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        run_apple_preview(adapter, program_path=PROGRAM, recorder=recorder,
                          depth_filter=DEPTH_FILTER, gripper_sweep=SWEEP,
                          gripper_base_to_tcp=np.eye(4), control_period_s=0.05)
    assert control.commands == []
    recorder.finalize(program_ok=False, task_success=False, termination_reason="program_error",
                      program_result=None)


def test_chain_preview_requires_explicit_attachment_before_observation(tmp_path):
    control = _Control()
    config = dict(control.context.metadata["open_rail_pick_apple"], preview_full_chain=True,
                  preview_scene_voxel_size_m=0.01)
    control.context = replace(control.context, metadata={"open_rail_pick_apple": config})
    adapter = OpenRailAdapter(control, control.profile, task_context=control.context, controlled_arms=("right",))
    recorder = _recorder(tmp_path)
    with pytest.raises(ValueError, match="attachment"):
        run_apple_preview(adapter, program_path=PROGRAM, recorder=recorder,
                          depth_filter=DEPTH_FILTER, gripper_sweep=SWEEP,
                          gripper_base_to_tcp=np.eye(4), control_period_s=0.05,
                          chain_lift_offset_m=[0, 0, 0.1])
    assert control.commands == []
    recorder.finalize(program_ok=False, task_success=False, termination_reason="program_error", program_result=None)


@pytest.mark.parametrize("tampered_field", [None, "target_mask_sha256", "snapshot_fingerprint"])
def test_optional_chain_uses_same_mask_snapshot_and_never_commands_robot(monkeypatch, tmp_path, tampered_field):
    calls = []

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def request(_session, method, url, **kwargs):
        payload = kwargs.get("json")
        calls.append((url, payload))
        if ":8117/" in url:
            mask = np.zeros((8, 10), dtype="<f4")
            if payload["query"] == "apple":
                mask[1:7, 0:4] = 1
            else:
                mask[1:7, 6:10] = 1
            return Response({"results": [{"label": "apple", "box_xyxy": [1, 1, 9, 7],
                             "shape": [8, 10], "probabilities_base64": base64.b64encode(mask.tobytes()).decode()}],
                             "total_instances": 1, "truncated": False})
        if ":8120/" in url:
            return Response({"grasps_base64": encode_numpy(np.eye(4)[None]),
                             "scores_base64": encode_numpy(np.array([0.8])), "branch_tags": [], "timing": {}})
        if ":8118/" in url:
            req = service.PlanRequest.model_validate(payload)
            start = np.asarray(payload["joint_positions"]["right"])
            return Response({"joint_positions_base64": encode_numpy(np.stack((start, start + 0.01))),
                             "dt_s": 0.1, "attachment_diagnostics": service._attachment_diagnostics(req),
                             "scene_partition_diagnostics": service._scene_partition_diagnostics(req),
                             "collision_diagnostics": {"collision_disabled": False, "scene_supplied": True,
                                 "world_collision_check_enabled": True, "self_collision_check_enabled": True,
                                 "robot_model_verified": False}})
        raise AssertionError(url)

    monkeypatch.setattr(requests.Session, "request", request)
    if tampered_field:
        generate = CapApi.generate_grasps_for_localization

        def tamper_mask(self, localization, *args, **kwargs):
            result = generate(self, localization, *args, **kwargs)
            return replace(result, diagnostics={**result.diagnostics, tampered_field: "wrong"})

        monkeypatch.setattr(CapApi, "generate_grasps_for_localization", tamper_mask)
    control = _Control()
    config = dict(control.context.metadata["open_rail_pick_apple"], preview_full_chain=True,
                  preview_scene_voxel_size_m=0.01)
    control.context = replace(control.context, metadata={"open_rail_pick_apple": config})
    adapter = OpenRailAdapter(control, control.profile, task_context=control.context, controlled_arms=("right",))
    recorder = _recorder(tmp_path)
    kwargs = dict(program_path=PROGRAM, recorder=recorder, depth_filter=DEPTH_FILTER,
                  gripper_sweep=SWEEP, gripper_base_to_tcp=np.eye(4), control_period_s=0.05,
                  chain_lift_offset_m=[0, 0, 0.1], chain_attachment=_attachment())
    if tampered_field:
        with pytest.raises(ValueError, match="target mask and snapshot"):
            run_apple_preview(adapter, **kwargs)
        assert not any(":8118/" in url for url, _ in calls)
    else:
        result = run_apple_preview(adapter, **kwargs)
        assert result.ok and result.result["ok"] is False and result.result["stage"] == "chain_preview"
        chain, = result.result["chain_attempts"]
        assert chain.completed and all(not stage.diagnostics["collision_aware"] for stage in chain.stages)
        plans = [payload for url, payload in calls if ":8118/" in url]
        assert len(plans) == 3
        assert plans[1]["target_scene_partition"] == plans[2]["target_scene_partition"]
        assert plans[1]["joint_positions"]["right"] == list(chain.stages[0].waypoints[-1])
        assert plans[2]["joint_positions"]["right"] == list(chain.stages[1].waypoints[-1])
    assert control.commands == []
    recorder.finalize(program_ok=True, task_success=False, termination_reason="program_completed", program_result=None)
