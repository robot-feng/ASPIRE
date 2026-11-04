"""CPU-only tests for explicit cuRobo held-object geometry inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cap_harness.contracts import (
    PlanningScene,
    PointCloud,
    Pose,
    RobotPlanningContext,
    RobotState,
)
from cap_harness.providers.curobo import service
from cap_harness.providers.curobo.attachment import (
    AttachedObjectGeometry,
    carve_attached_object_points,
    merge_attachment_into_kinematics,
    pose7_to_matrix,
    transform_spheres_to_base,
)
from cap_harness.providers.curobo.client import CuRoboProvider, _attachment_diagnostics
from cap_harness.providers.wire import encode_numpy


def _attachment() -> AttachedObjectGeometry:
    return AttachedObjectGeometry(
        object_link_name="held_object",
        attachment_joint_name="held_object_fixed_joint",
        parent_link_name="verified_tool_frame",
        parent_to_object=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
        collision_spheres=((0.0, 0.0, 0.0, 0.04), (0.0, 0.0, 0.04, 0.03)),
        contact_link_names=("left_finger", "right_finger"),
    )


def test_attachment_patch_uses_explicit_transform_geometry_and_contacts() -> None:
    source = {
        "tool_frames": ["left_tool", "right_tool"],
        "collision_link_names": ["arm_link"],
        "collision_spheres": {"arm_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.1}]},
        "self_collision_ignore": {
            "arm_link": ["wrist_link"],
            "held_object": ["camera_mount"],
        },
    }

    configured = merge_attachment_into_kinematics(source, _attachment())

    assert configured["extra_links"]["held_object"] == {
        "fixed_transform": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        "joint_name": "held_object_fixed_joint",
        "joint_type": "FIXED",
        "link_name": "held_object",
        "parent_link_name": "verified_tool_frame",
    }
    assert configured["collision_spheres"]["held_object"] == [
        {"center": [0.0, 0.0, 0.0], "radius": 0.04},
        {"center": [0.0, 0.0, 0.04], "radius": 0.03},
    ]
    assert configured["collision_link_names"] == ["arm_link", "held_object"]
    assert configured["tool_frames"] == ["left_tool", "right_tool", "held_object"]
    assert configured["self_collision_ignore"]["held_object"] == [
        "camera_mount",
        "left_finger",
        "right_finger",
    ]
    assert "extra_links" not in source
    assert "held_object" not in source["collision_spheres"]


def test_attachment_composes_parent_frame_and_transforms_sphere_centers() -> None:
    attachment = _attachment()
    parent_pose = pose7_to_matrix(
        [0.5, -0.2, 0.3, 0.7071067811865476, 0.0, 0.7071067811865475, 0.0]
    )
    object_pose = attachment.pose_in_base(parent_pose)
    spheres = transform_spheres_to_base(object_pose, attachment.collision_spheres)

    assert np.allclose(object_pose[:3, 3], [0.6, -0.2, 0.3])
    assert np.allclose(spheres[0, :3], [0.6, -0.2, 0.3])
    assert np.allclose(spheres[1, :3], [0.64, -0.2, 0.3])
    assert np.allclose(spheres[:, 3], [0.04, 0.03])


def test_attachment_carve_removes_object_points_and_keeps_adjacent_table() -> None:
    # A target-local sphere removes only its covered points; table samples 6 cm
    # away from the fruit remain part of the collision scene.
    scene = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.15, 0.00, 0.00],
            [0.00, 0.15, 0.00],
        ]
    )
    result = carve_attached_object_points(
        scene,
        np.eye(4),
        [[0.0, 0.0, 0.0, 0.05]],
        padding_m=0.005,
    )

    assert np.array_equal(result, [[0.15, 0.0, 0.0], [0.0, 0.15, 0.0]])


@pytest.mark.parametrize(
    "pose",
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, float("nan"), 0, 0, 0],
    ],
)
def test_pose_requires_finite_normalized_quaternion(pose: list[float]) -> None:
    with pytest.raises(ValueError):
        pose7_to_matrix(pose)


def test_merge_rejects_file_backed_spheres_and_duplicate_link() -> None:
    with pytest.raises(ValueError, match="inline mapping"):
        merge_attachment_into_kinematics({"collision_spheres": "robot_spheres.yml"}, _attachment())

    source = {
        "collision_spheres": {"held_object": []},
        "tool_frames": [],
        "collision_link_names": [],
    }
    with pytest.raises(ValueError, match="already exists"):
        merge_attachment_into_kinematics(source, _attachment())


def test_attachment_rejects_missing_contacts_or_invalid_spheres() -> None:
    with pytest.raises(ValueError, match="contact_link_names"):
        AttachedObjectGeometry(
            "held_object", "fixed", "tool", (0, 0, 0, 1, 0, 0, 0), ((0, 0, 0, 0.1),), ()
        )
    with pytest.raises(ValueError, match="positive radii"):
        AttachedObjectGeometry(
            "held_object",
            "fixed",
            "tool",
            (0, 0, 0, 1, 0, 0, 0),
            ((0, 0, 0, 0),),
            ("finger",),
        )


def test_attachment_snapshots_mutable_sequences_as_immutable_tuples() -> None:
    parent_to_object = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
    collision_spheres = np.array([[0.0, 0.0, 0.0, 0.04]], dtype=np.float64)
    contact_link_names = ["left_finger", "right_finger"]
    attachment = AttachedObjectGeometry(
        object_link_name="held_object",
        attachment_joint_name="fixed",
        parent_link_name="verified_tool_frame",
        parent_to_object=parent_to_object,
        collision_spheres=collision_spheres,
        contact_link_names=contact_link_names,
    )

    parent_to_object[2] = 10.0
    collision_spheres[0, 3] = 10.0
    contact_link_names[0] = "unrelated_link"

    assert attachment.parent_to_object == (0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0)
    assert attachment.collision_spheres == ((0.0, 0.0, 0.0, 0.04),)
    assert attachment.contact_link_names == ("left_finger", "right_finger")
    assert attachment.to_payload()["collision_spheres"] == [[0.0, 0.0, 0.0, 0.04]]
    with pytest.raises(TypeError):
        attachment.collision_spheres[0][3] = 0.5  # type: ignore[index]


def test_attachment_rejects_contact_names_as_one_string() -> None:
    with pytest.raises(ValueError, match="sequence of link names"):
        AttachedObjectGeometry(
            "held_object",
            "fixed",
            "tool",
            (0, 0, 0, 1, 0, 0, 0),
            ((0, 0, 0, 0.1),),
            "finger",
        )


def _service_payload(*, attached: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "base_frame": "robot_base",
        "model": "panda",
        "joint_positions": {"primary": [0.0] * 7},
        "joint_names": {"primary": [f"joint_{index}" for index in range(7)]},
        "base_transforms": {"primary": np.eye(4).tolist()},
        "end_effector_links": {"primary": "tool"},
    }
    if attached is not None:
        payload["attached_object"] = attached
    return payload


def _collision_ack(
    *,
    disabled: bool = False,
    scene: bool = True,
    world: bool = True,
    self_collision: bool = True,
) -> dict[str, bool]:
    return {
        "collision_disabled": disabled,
        "scene_supplied": scene,
        "world_collision_check_enabled": world,
        "self_collision_check_enabled": self_collision,
    }


def test_typed_provider_serializes_optional_attachment_only_when_requested() -> None:
    attachment = _attachment()
    state = SimpleNamespace(
        base_frame="robot_base",
        joint_positions={"primary": np.zeros(7)},
        arms=("primary",),
    )
    context = SimpleNamespace(
        model="panda",
        joint_names={"primary": [f"joint_{index}" for index in range(7)]},
        base_transforms={"primary": np.eye(4)},
        end_effector_links={"primary": "tool"},
    )
    provider = CuRoboProvider()

    without_attachment = provider._base_payload(state, context)
    with_attachment = provider._base_payload(state, context, attached_object=attachment)

    assert "attached_object" not in without_attachment
    assert with_attachment["attached_object"] == attachment.to_payload()
    diagnostics = _attachment_diagnostics(attachment)
    assert diagnostics["attached_object_geometry_verified"] is False
    assert diagnostics["attached_object_sphere_coverage_verified"] is False
    assert diagnostics["attached_object_scene_carve_may_remove_support_points"] is True
    assert diagnostics["attached_object_contact_links_ignored_for_self_collision"] == [
        "left_finger",
        "right_finger",
    ]


def test_service_parses_attachment_and_builds_isolated_robot_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = _attachment()
    request = service.BaseRequest.model_validate(
        _service_payload(attached=attachment.to_payload())
        | {"scene_points_base64": "present"}
    )
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<robot name="test">
        <link name="verified_tool_frame"/>
        <link name="left_finger"/>
        <link name="right_finger"/>
        <joint name="left_finger_mount" type="fixed">
          <parent link="verified_tool_frame"/><child link="left_finger"/>
        </joint>
        <joint name="right_finger_mount" type="fixed">
          <parent link="verified_tool_frame"/><child link="right_finger"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )
    config_path = tmp_path / "robot.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "urdf_path": str(urdf_path),
                        "tool_frames": ["verified_tool_frame"],
                        "collision_link_names": ["arm_link"],
                        "collision_spheres": {
                            "arm_link": [{"center": [0, 0, 0], "radius": 0.1}]
                        },
                        "extra_links": None,
                        "self_collision_ignore": {"arm_link": ["wrist_link"]},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CAP_HARNESS_CUROBO_PANDA_CONFIG", str(config_path))
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)
    service._ATTACHED_CONFIG_PATHS.clear()

    generated = Path(service._robot_config(request))
    config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]

    assert kinematics["extra_links"]["held_object"]["parent_link_name"] == (
        "verified_tool_frame"
    )
    assert kinematics["collision_spheres"]["held_object"][0]["radius"] == pytest.approx(
        0.04
    )
    assert "held_object" in kinematics["tool_frames"]
    assert "held_object" in kinematics["collision_link_names"]
    diagnostics = service._attachment_diagnostics(request)
    assert diagnostics is not None
    assert diagnostics["geometry_verified"] is False
    assert diagnostics["sphere_coverage_verified"] is False
    assert diagnostics["scene_carve_may_remove_support_points"] is True
    diagnostics = service._attachment_diagnostics(request)
    assert diagnostics is not None
    assert diagnostics["geometry_verified"] is False
    assert diagnostics["sphere_coverage_verified"] is False
    assert diagnostics["scene_carve_may_remove_support_points"] is True


def test_service_refuses_attachment_for_non_tree_urdf(tmp_path: Path) -> None:
    urdf_path = tmp_path / "loop.urdf"
    urdf_path.write_text(
        """<robot name="loop">
        <link name="verified_tool_frame"/>
        <link name="left_finger"/>
        <link name="right_finger"/>
        <joint name="left_finger_mount" type="fixed">
          <parent link="verified_tool_frame"/><child link="left_finger"/>
        </joint>
        <joint name="right_finger_mount" type="fixed">
          <parent link="verified_tool_frame"/><child link="right_finger"/>
        </joint>
        <loop_joint name="parallel_linkage" type="spherical"/>
        </robot>""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported loop_joint"):
        service._validate_attachment_links(
            {"urdf_path": str(urdf_path)}, _attachment()
        )


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


class _JsonResponse:
    status_code = 200
    text = ""

    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def json(self) -> dict[str, object]:
        return self.body


class _JsonSession:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _JsonResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _JsonResponse(self.response)


def _unattached_plan_result(
    response_fields: dict[str, object], *, strict_collision_ack: bool = False
):
    pose = Pose([0.2, 0.0, 0.1], [1, 0, 0, 0], "robot_base")
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=pose,
        gripper_positions=0.5,
        base_frame="robot_base",
        joint_names=[f"joint_{index}" for index in range(7)],
        embodiment="robosuite",
    )
    context = RobotPlanningContext(
        embodiment="robosuite",
        model="panda",
        joint_names=state.joint_names,
        base_transforms={"primary": np.eye(4)},
        end_effector_links={"primary": "verified_tool_frame"},
    )
    scene = PlanningScene(
        PointCloud(points=np.array([[0.1, 0.0, 0.0]]), frame="robot_base")
    )
    session = _JsonSession(
        {
            "joint_positions_base64": encode_numpy(np.zeros((2, 7), dtype=np.float64)),
            "dt_s": 0.1,
        }
        | response_fields
    )
    result = CuRoboProvider(
        session=session,  # type: ignore[arg-type]
        strict_collision_ack=strict_collision_ack,
    ).plan_to_pose(state, pose, scene=scene, context=context)
    return result, session


@pytest.mark.parametrize(
    ("response_fields", "strict", "expected_ok", "expected_collision_aware"),
    [
        pytest.param({}, False, True, True, id="legacy-default-compatibility"),
        pytest.param({}, True, False, None, id="strict-rejects-missing-ack"),
        pytest.param(
            {
                "collision_diagnostics": _collision_ack(
                    disabled=True, world=False, self_collision=False
                )
            },
            False,
            True,
            False,
            id="explicit-disabled-is-not-collision-aware",
        ),
        pytest.param(
            {
                "collision_diagnostics": _collision_ack(
                    disabled=True, world=False, self_collision=False
                )
            },
            True,
            False,
            None,
            id="strict-rejects-disabled-checks",
        ),
        pytest.param(
            {"collision_diagnostics": _collision_ack(scene=False, world=False)},
            False,
            True,
            False,
            id="scene-required-for-world-collision-ack",
        ),
        pytest.param(
            {"collision_diagnostics": _collision_ack(scene=False, world=True)},
            False,
            False,
            None,
            id="rejects-world-ack-without-scene",
        ),
    ],
)
def test_plan_collision_ack_compatibility_and_strict_modes(
    response_fields: dict[str, object],
    strict: bool,
    expected_ok: bool,
    expected_collision_aware: bool | None,
) -> None:
    result, _ = _unattached_plan_result(
        response_fields, strict_collision_ack=strict
    )

    assert result.ok is expected_ok
    if expected_ok:
        assert result.trajectory is not None
        assert result.trajectory.collision_aware is expected_collision_aware
        if not result.diagnostics["collision_acknowledged"]:
            assert result.diagnostics["collision_ack_status"] == "legacy_service_unknown"
        else:
            assert result.diagnostics["collision_aware"] is expected_collision_aware
    else:
        assert result.error is not None


def test_service_collision_ack_requires_scene_and_enabled_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_without_scene = service.BaseRequest.model_validate(_service_payload())
    request_with_scene = service.BaseRequest.model_validate(
        _service_payload() | {"scene_points_base64": "present"}
    )
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)

    assert service._collision_diagnostics(request_without_scene) == {
        "collision_disabled": False,
        "scene_supplied": False,
        "world_collision_check_enabled": False,
        "self_collision_check_enabled": True,
    }
    assert service._collision_diagnostics(request_with_scene) == _collision_ack()

    monkeypatch.setenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", "1")
    assert service._collision_diagnostics(request_with_scene) == _collision_ack(
        disabled=True, world=False, self_collision=False
    )


@pytest.mark.parametrize(
    ("response_fields", "expected_ok"),
    [
        pytest.param(
            {
                "attachment_diagnostics": {
                    "object_link": "held_object",
                    "collision_model": "caller_supplied_spheres",
                },
                "collision_diagnostics": _collision_ack(),
            },
            True,
            id="matching-service-ack",
        ),
        pytest.param({}, False, id="legacy-service-omitted-ack"),
        pytest.param(
            {
                "attachment_diagnostics": {
                    "object_link": "held_object",
                    "collision_model": "caller_supplied_spheres",
                }
            },
            False,
            id="attachment-requires-collision-ack",
        ),
        pytest.param(
            {
                "attachment_diagnostics": {
                    "object_link": "held_object",
                    "collision_model": "caller_supplied_spheres",
                },
                "collision_diagnostics": _collision_ack(
                    disabled=True, world=False, self_collision=False
                ),
            },
            False,
            id="attachment-rejects-disabled-collision-checks",
        ),
        pytest.param(
            {
                "attachment_diagnostics": {
                    "object_link": "different_object",
                    "collision_model": "caller_supplied_spheres",
                }
            },
            False,
            id="wrong-object-link-ack",
        ),
        pytest.param(
            {
                "attachment_diagnostics": {
                    "object_link": "held_object",
                    "collision_model": "robot_spheres_only",
                }
            },
            False,
            id="wrong-collision-model-ack",
        ),
    ],
)
def test_typed_plan_requires_matching_attachment_service_ack(
    response_fields: dict[str, object], expected_ok: bool
) -> None:
    attachment = _attachment()
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=Pose([0, 0, 0], [1, 0, 0, 0], "robot_base"),
        gripper_positions=0.5,
        base_frame="robot_base",
        joint_names=[f"joint_{index}" for index in range(7)],
        embodiment="robosuite",
    )
    context = RobotPlanningContext(
        embodiment="robosuite",
        model="panda",
        joint_names=state.joint_names,
        base_transforms={"primary": np.eye(4)},
        end_effector_links={"primary": "verified_tool_frame"},
    )
    scene = PlanningScene(
        PointCloud(points=np.array([[0.1, 0.0, 0.0]]), frame="robot_base")
    )
    session = _JsonSession(
        {
            "joint_positions_base64": encode_numpy(np.zeros((2, 7), dtype=np.float64)),
            "dt_s": 0.1,
        }
        | response_fields
    )

    result = CuRoboProvider(session=session).plan_to_pose(  # type: ignore[arg-type]
        state,
        Pose([0.2, 0.0, 0.1], [1, 0, 0, 0], "robot_base"),
        scene=scene,
        context=context,
        attached_object=attachment,
    )

    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["attached_object"] == attachment.to_payload()
    assert result.ok is expected_ok
    if expected_ok:
        assert result.diagnostics["attached_object_geometry_verified"] is False
        assert result.diagnostics["attached_object_scene_carve_may_remove_support_points"] is True
    else:
        assert result.error is not None


def test_synchronized_attached_plan_rejects_legacy_service_response() -> None:
    attachment = _attachment()
    pose = Pose([0.2, 0.0, 0.1], [1, 0, 0, 0], "robot_base")
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=pose,
        gripper_positions=0.5,
        base_frame="robot_base",
        joint_names=[f"joint_{index}" for index in range(7)],
        embodiment="robosuite",
    )
    context = RobotPlanningContext(
        embodiment="robosuite",
        model="panda",
        joint_names=state.joint_names,
        base_transforms={"primary": np.eye(4)},
        end_effector_links={"primary": "verified_tool_frame"},
    )
    scene = PlanningScene(PointCloud(points=np.array([[0.1, 0.0, 0.0]]), frame="robot_base"))
    session = _JsonSession(
        {
            "joint_positions_base64": {
                "primary": encode_numpy(np.zeros((2, 7), dtype=np.float64))
            },
            "dt_s": 0.1,
            "attachment_diagnostics": {
                "object_link": "held_object",
                "collision_model": "caller_supplied_spheres",
            },
        }
    )

    result = CuRoboProvider(session=session).plan_synchronized_motion(  # type: ignore[arg-type]
        state,
        {"primary": pose},
        scene=scene,
        context=context,
        attached_object=attachment,
    )

    assert not result.ok
    assert result.error is not None
    assert "collision diagnostics" in str(result.error.details.get("error", ""))


def test_service_scene_carve_uses_object_fk_and_preserves_neighboring_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attachment = _attachment()
    request = service.BaseRequest.model_validate(
        _service_payload(attached=attachment.to_payload())
        | {"scene_points_base64": "present", "voxel_size_m": 0.005}
    )

    class _FakeKinematicsParams:
        def get_sphere_index_from_link_name(self, name: str) -> _FakeTensor:
            assert name == "held_object"
            return _FakeTensor([1])

    object_pose = SimpleNamespace(
        position=_FakeTensor([[0.0, 0.0, 0.0]]),
        quaternion=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    fk = SimpleNamespace(
        robot_spheres=_FakeTensor([[1.0, 0.0, 0.0, 0.02], [0.0, 0.0, 0.0, 0.04]]),
        tool_poses=SimpleNamespace(to_dict=lambda: {"held_object": object_pose}),
    )
    robot = SimpleNamespace(
        kinematics=SimpleNamespace(
            config=SimpleNamespace(
                kinematics_config=_FakeKinematicsParams(),
            )
        ),
        compute_kinematics=lambda _: fk,
    )
    monkeypatch.setattr(service, "_segmentation_kinematics", lambda _: None)
    monkeypatch.setattr(service, "_state", lambda *_: object())
    scene_points = np.array(
        [
            [0.0, 0.0, 0.0],  # attached object return
            [0.15, 0.0, 0.0],  # nearby table point
            [1.0, 0.0, 0.0],  # robot body return
        ]
    )

    filtered = service._carve_robot_points(request, robot, scene_points)

    assert np.array_equal(filtered, [[0.15, 0.0, 0.0]])


@pytest.mark.parametrize("index_mode", ["missing_config", "missing_method", "empty"])
def test_planner_attachment_carve_fails_closed_without_object_sphere_indices(
    index_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = _attachment()
    request = service.BaseRequest.model_validate(
        _service_payload(attached=attachment.to_payload())
        | {"scene_points_base64": "present"}
    )

    class _EmptyIndexParams:
        def get_sphere_index_from_link_name(self, _: str) -> _FakeTensor:
            return _FakeTensor([])

    if index_mode == "missing_config":
        kinematics = SimpleNamespace()
    elif index_mode == "missing_method":
        kinematics = SimpleNamespace(config=SimpleNamespace(kinematics_config=object()))
    else:
        kinematics = SimpleNamespace(
            config=SimpleNamespace(kinematics_config=_EmptyIndexParams())
        )
    fk = SimpleNamespace(robot_spheres=_FakeTensor([[0.0, 0.0, 0.0, 0.04]]))
    planner = SimpleNamespace(kinematics=kinematics, compute_kinematics=lambda _: fk)
    monkeypatch.setattr(service, "_segmentation_kinematics", lambda _: None)
    monkeypatch.setattr(service, "_state", lambda *_: object())

    with pytest.raises(service.HTTPException) as exc:
        service._carve_robot_points(request, planner, np.array([[0.0, 0.0, 0.0]]))
    assert exc.value.status_code == 500
    assert "attached-object" in str(exc.value.detail)


def test_yam_dense_segmentation_can_omit_attached_link_sphere_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attachment = _attachment()
    request = service.BaseRequest.model_validate(
        _service_payload(attached=attachment.to_payload())
        | {"model": "yam_real", "scene_points_base64": "present", "voxel_size_m": 0.005}
    )
    body_fk = SimpleNamespace(robot_spheres=_FakeTensor([[1.0, 0.0, 0.0, 0.02]]))
    segmentation = SimpleNamespace(compute_kinematics=lambda _: body_fk)
    object_pose = SimpleNamespace(
        position=_FakeTensor([[0.0, 0.0, 0.0]]),
        quaternion=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    planner_fk = SimpleNamespace(
        tool_poses=SimpleNamespace(to_dict=lambda: {"held_object": object_pose})
    )
    planner = SimpleNamespace(compute_kinematics=lambda _: planner_fk)
    monkeypatch.setattr(service, "_segmentation_kinematics", lambda _: segmentation)
    monkeypatch.setattr(service, "_state", lambda *_: object())
    scene_points = np.array(
        [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )

    filtered = service._carve_robot_points(request, planner, scene_points)

    assert np.array_equal(filtered, [[0.15, 0.0, 0.0]])


def test_attached_model_requires_a_scene_and_collision_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = _attachment().to_payload()
    request = service.BaseRequest.model_validate(_service_payload(attached=attachment))
    monkeypatch.setenv("CAP_HARNESS_CUROBO_PANDA_CONFIG", str(tmp_path / "missing.yml"))
    monkeypatch.delenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", raising=False)

    with pytest.raises(service.HTTPException) as exc:
        service._robot_config(request)
    assert exc.value.detail == "attached-object planning requires an observed scene point cloud"

    request.scene_points_base64 = "present"
    monkeypatch.setenv("CAP_HARNESS_CUROBO_DISABLE_COLLISION", "1")
    with pytest.raises(service.HTTPException) as exc:
        service._robot_config(request)
    assert exc.value.detail == "attached-object planning requires collision checking to be enabled"


def test_attached_object_fk_frame_is_not_added_as_a_stationary_goal() -> None:
    attachment = _attachment()
    request = service.BaseRequest.model_validate(
        _service_payload(attached=attachment.to_payload())
    )
    planner = SimpleNamespace(tool_frames=["verified_tool_frame", "held_object"])

    assert service._goal_tool_frames(request, planner) == ["verified_tool_frame"]

    request_without_attachment = service.BaseRequest.model_validate(_service_payload())
    assert service._goal_tool_frames(request_without_attachment, planner) == [
        "verified_tool_frame",
        "held_object",
    ]


def test_unattached_service_request_keeps_config_selection_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = str(tmp_path / "unused.yml")
    monkeypatch.setenv("CAP_HARNESS_CUROBO_PANDA_CONFIG", config_path)
    request = service.BaseRequest.model_validate(_service_payload())

    assert service._robot_config(request) == config_path
    assert service._attachment_diagnostics(request) is None
    assert service._attachment_diagnostics(request) is None


def test_attached_plan_refuses_to_change_gripper_position() -> None:
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=Pose([0, 0, 0], [1, 0, 0, 0], "robot_base"),
        gripper_positions=0.5,
        base_frame="robot_base",
        joint_names=[f"joint_{index}" for index in range(7)],
        embodiment="robosuite",
    )
    context = RobotPlanningContext(
        embodiment="robosuite",
        model="panda",
        joint_names=state.joint_names,
        base_transforms={"primary": np.eye(4)},
        end_effector_links={"primary": "verified_tool_frame"},
    )
    scene = PlanningScene(
        PointCloud(points=np.array([[0.1, 0.0, 0.0]]), frame="robot_base")
    )
    provider = CuRoboProvider()

    result = provider.plan_to_pose(
        state,
        Pose([0.2, 0.0, 0.1], [1, 0, 0, 0], "robot_base"),
        scene=scene,
        context=context,
        gripper_position=0.0,
        attached_object=_attachment(),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.message == "attached-object planning cannot change gripper position"
