"""CPU-only G1 cuRobo configuration and non-executable-plan checks."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from fastapi import HTTPException
import numpy as np
import pytest

from cap_harness.providers.curobo import service
from cap_harness.providers.curobo.client import _collision_acknowledged


def _request() -> service.PlanRequest:
    return service.PlanRequest(
        base_frame="base",
        model="agibot_g1",
        joint_positions={"left": [0.1] * 7, "right": [0.2] * 7},
        joint_names={
            "left": [f"l{i}" for i in range(7)],
            "right": [f"r{i}" for i in range(7)],
        },
        base_transforms={"left": np.eye(4).tolist(), "right": np.eye(4).tolist()},
        end_effector_links={"left": "left_tcp", "right": "right_tcp"},
        arm="right",
        target={
            "type": "pose",
            "position": [0.1, 0.2, 0.3],
            "quaternion_wxyz": [1, 0, 0, 0],
            "frame": "base",
        },
    )


def test_g1_config_requires_explicit_existing_absolute_file(monkeypatch, tmp_path: Path) -> None:
    req = _request()
    monkeypatch.delenv("CAP_HARNESS_CUROBO_G1_CONFIG", raising=False)
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)
    with pytest.raises(HTTPException, match="not configured"):
        service._robot_config(req)
    monkeypatch.setenv("CAP_HARNESS_CUROBO_G1_CONFIG", "g1.yml")
    with pytest.raises(HTTPException, match="absolute path"):
        service._robot_config(req)
    config = tmp_path / "g1.yml"
    config.write_text("robot_cfg: {}\n")
    monkeypatch.setenv("CAP_HARNESS_CUROBO_G1_CONFIG", str(config))
    assert service._robot_config(req) == str(config)
    monkeypatch.setenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", "1")
    with pytest.raises(HTTPException, match="requires collision"):
        service._robot_config(req)


def test_g1_requires_both_arms_and_unique_joint_names() -> None:
    req = _request()
    assert set(service._validate(req)) == {"left", "right"}
    one_arm = req.model_copy(
        update={
            "joint_positions": {"right": req.joint_positions["right"]},
            "joint_names": {"right": req.joint_names["right"]},
            "base_transforms": {"right": req.base_transforms["right"]},
            "end_effector_links": {"right": req.end_effector_links["right"]},
        }
    )
    with pytest.raises(HTTPException, match="both observed arm"):
        service._validate(one_arm)
    duplicate = req.model_copy(
        update={"joint_names": {"left": req.joint_names["left"], "right": req.joint_names["left"]}}
    )
    with pytest.raises(HTTPException, match="unique"):
        service._validate(duplicate)


def test_g1_rejects_inactive_arm_motion_at_any_waypoint() -> None:
    req = _request()
    names = req.joint_names["right"] + req.joint_names["left"]
    trajectory = np.tile(np.array([0.2] * 7 + [0.1] * 7), (3, 1))
    trajectory[:, 0] += [0, 0.05, 0.1]
    service._check_g1_inactive_arm(req, names, trajectory)
    trajectory[1, 7] += 0.01
    with pytest.raises(HTTPException, match="inactive arm"):
        service._check_g1_inactive_arm(req, names, trajectory)
    with pytest.raises(HTTPException, match="omitted inactive-arm"):
        service._check_g1_inactive_arm(req, names[:7], trajectory[:, :7])


def test_g1_explicit_sdk_to_model_mapping_preserves_order_and_inactive_arm(
    monkeypatch, tmp_path: Path
) -> None:
    req = _request()
    sdk_names = req.joint_names["left"] + req.joint_names["right"]
    mapping = {name: f"urdf_{name}" for name in sdk_names}
    map_file = tmp_path / "joint-map.json"
    map_file.write_text(json.dumps({"joint_name_map": mapping}))
    monkeypatch.setenv("CAP_HARNESS_CUROBO_G1_JOINT_MAP", str(map_file))

    # Planner order is different from both SDK order and arm insertion order.
    names = list(reversed(list(mapping.values())))
    indices = service._requested_indices(req, "right", names)
    assert [names[index] for index in indices] == [mapping[n] for n in req.joint_names["right"]]
    by_name = {
        **dict.fromkeys(req.joint_names["left"], 0.1),
        **dict.fromkeys(req.joint_names["right"], 0.2),
    }
    robot_values = {mapping[name]: value for name, value in by_name.items()}
    assert np.allclose(
        service._joint_positions_in_model_order(req, names),
        [robot_values[name] for name in names],
    )
    trajectory = np.array([[robot_values[name] for name in names]] * 2)
    service._check_g1_inactive_arm(req, names, trajectory)
    trajectory[1, names.index(mapping["l0"])] += 0.01
    with pytest.raises(HTTPException, match="inactive arm"):
        service._check_g1_inactive_arm(req, names, trajectory)

    # A mid-request file change cannot silently alter an already-bound map.
    map_file.write_text(json.dumps({"joint_name_map": {}}))
    assert service._model_joint_names(req, "left") == [mapping[n] for n in req.joint_names["left"]]
    with pytest.raises(HTTPException, match="uniquely cover"):
        service._model_joint_names(_request(), "left")


def test_g1_joint_map_rejects_duplicate_model_names(monkeypatch, tmp_path: Path) -> None:
    req = _request()
    mapping = {name: "same" for names in req.joint_names.values() for name in names}
    map_file = tmp_path / "joint-map.json"
    map_file.write_text(json.dumps({"joint_name_map": mapping}))
    monkeypatch.setenv("CAP_HARNESS_CUROBO_G1_JOINT_MAP", str(map_file))
    with pytest.raises(HTTPException, match="uniquely cover"):
        service._model_joint_names(req, "right")


def test_candidate_joint_map_references_the_installed_a2d_urdf() -> None:
    urdf = Path("/home/agi/app/bin/motion-control/share/robot_model/urdf/a2d_v01/model.urdf")
    if not urdf.is_file():
        pytest.skip("the optional local a2d_v01 description is not installed")
    candidate_path = (
        Path(__file__).resolve().parents[1]
        / "src/cap_harness/configs/open_rail/agibot_g1_a2d_v01_joint_map.candidate.json"
    )
    mapping = json.loads(candidate_path.read_text())["joint_name_map"]
    joints = {joint.attrib["name"] for joint in ET.parse(urdf).getroot().findall("joint")}
    assert len(mapping) == 14
    assert len(set(mapping.values())) == 14
    assert set(mapping.values()) <= joints
    assert {f"Joint{i}_{side}" for side in ("l", "r") for i in range(1, 8)} == set(mapping)


@pytest.mark.parametrize("legacy_model", ["panda", "yam_real", "dual_panda"])
def test_legacy_single_arm_seven_joint_fallback_is_preserved(legacy_model: str) -> None:
    request = _request().model_copy(
        update={
            "model": legacy_model,
            "joint_positions": {"right": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
            "joint_names": {"right": [f"sdk_{i}" for i in range(7)]},
        }
    )
    model_names = [f"model_{i}" for i in range(7)]
    assert np.allclose(
        service._joint_positions_in_model_order(request, model_names),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    g1 = request.model_copy(update={"model": "agibot_g1"})
    with pytest.raises(HTTPException, match="do not match"):
        service._joint_positions_in_model_order(g1, model_names)


def test_g1_rejects_tool_frames_not_in_the_configured_robot() -> None:
    class Planner:
        tool_frames = ["left_tcp", "unrelated_right_tcp"]

    req = _request()
    with pytest.raises(HTTPException, match="tool frames"):
        service._ordered_goal_poses(req, {"right": req.target}, Planner(), object())


def test_g1_collision_ack_does_not_claim_model_verification(monkeypatch) -> None:
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)
    req = _request().model_copy(update={"scene_points_base64": "present"})
    ack = service._collision_diagnostics(req)
    assert ack["world_collision_check_enabled"]
    assert ack["self_collision_check_enabled"]
    assert ack["robot_model_verified"] is False
    response = {"collision_diagnostics": ack}
    assert (
        _collision_acknowledged(
            response, attached_object=None, strict=False, require_robot_model_ack=True
        )
        is False
    )
    with pytest.raises(ValueError, match="did not verify"):
        _collision_acknowledged(
            response, attached_object=None, strict=True, require_robot_model_ack=True
        )
    assert (
        _collision_acknowledged(
            {}, attached_object=None, strict=False, require_robot_model_ack=True
        )
        is False
    )


def test_g1_offline_plan_preserves_joint_mapping_and_is_unverified(monkeypatch) -> None:
    class Planner:
        def plan_pose(self, goal, current):
            return object()

    req = _request().model_copy(update={"scene_points_base64": "present"})
    names = req.joint_names["left"] + req.joint_names["right"]
    trajectory = np.array([[0.1] * 7 + [0.2] * 7, [0.1] * 7 + [0.3] * 7])
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)
    monkeypatch.setattr(service, "_planner", lambda request: Planner())
    monkeypatch.setattr(service, "_state", lambda request, planner: object())
    monkeypatch.setattr(service, "_goal", lambda *args: object())
    monkeypatch.setattr(service, "_trajectory", lambda *args, **kwargs: (trajectory, 0.05, names))

    result = service._plan(req)

    assert np.allclose(service.decode_numpy(result.joint_positions_base64), [[0.2] * 7, [0.3] * 7])
    assert result.collision_diagnostics["robot_model_verified"] is False

    trajectory[1, 0] = 0.15
    with pytest.raises(HTTPException, match="inactive arm"):
        service._plan(req)
