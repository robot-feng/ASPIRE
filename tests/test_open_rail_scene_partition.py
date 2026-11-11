from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.api import CapApi
from cap_harness.contracts import (
    CameraObservation, LocalizationResult, ObjectGeometry, Observation, Pose,
    MotionStrategy, RobotPlanningContext, RobotState, Segmentation, SegmentationSet,
)
from cap_harness.geometry import mask_to_point_cloud
from cap_harness.open_rail.scene_partition import partition_target_scene
from cap_harness.open_rail.snapshot_identity import snapshot_fingerprint
from cap_harness.providers.curobo import service
from cap_harness.providers.curobo.attachment import AttachedObjectGeometry
from cap_harness.providers.curobo.client import CuRoboProvider, _validated_partition_payload
from cap_harness.providers.wire import encode_numpy
from cap_harness.registry import ToolRegistry


def _fixture():
    pose = Pose([0, 0, 0], [1, 0, 0, 0], "base")
    state = RobotState(
        joint_positions={"right": np.zeros(7)}, joint_velocities={"right": np.zeros(7)},
        end_effector_poses={"right": pose}, gripper_positions={"right": 1.0},
        base_frame="base", timestamp_s=1.0, embodiment="open_rail_7dof",
    )
    # Target and table are deliberately very close in 3-D; the former AABB or
    # a padded sphere would remove table points alongside target pixels.
    camera = CameraObservation(
        rgb=np.zeros((3, 3, 3), dtype=np.uint8), depth_m=np.ones((3, 3)),
        intrinsics=np.array([[1000, 0, 1], [0, 1000, 1], [0, 0, 1]]),
        frame="optical", camera_pose=pose, timestamp_s=1.0,
    )
    observation = Observation({"head": camera}, state, timestamp_s=1.0)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    segment = Segmentation(mask, "apple", 0.5, "head", "optical")
    target = mask_to_point_cloud(camera, segment, target_frame="base")
    localization = LocalizationResult(
        ok=True, geometry=ObjectGeometry(pose=pose, extents=np.full(3, 0.01)),
        segmentation=segment, point_cloud=target,
        diagnostics={"snapshot_id": "optical@1.000000000", "camera_name": "head",
                     "segmentation_source": "provider", "instance_count": 1,
                     "robot_state_timestamp_s": 1.0,
                     "target_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                     "snapshot_fingerprint": snapshot_fingerprint(camera, state)},
    )
    return observation, localization


def _partition(observation, localization):
    return partition_target_scene(
        observation, localization, camera_name="head", max_camera_state_skew_s=0.02,
        voxel_size_m=0.01,
    )


def test_preserves_background_pixels_inside_target_envelope():
    observation, localization = _fixture()
    result = _partition(observation, localization)
    assert len(result.target.points) == 1
    assert len(result.scene.point_cloud.points) == 8
    assert np.min(np.linalg.norm(result.scene.point_cloud.points - result.target.points[0], axis=1)) < 0.002
    assert result.scene.frame == "base"
    assert result.snapshot_id == localization.diagnostics["snapshot_id"]
    assert result.snapshot_fingerprint == localization.diagnostics["snapshot_fingerprint"]


def test_partition_accepts_actual_cap_localization():
    observation, expected = _fixture()

    class Adapter:
        embodiment = "open_rail_7dof"

        def get_robot_state(self):
            return observation.robot_state

    class Provider:
        def segment_text(self, camera, query):
            assert camera is observation.cameras["head"] and query == "apple"
            return SegmentationSet(ok=True, segmentations=(expected.segmentation,))

    api = CapApi(Adapter(), segmentation_provider=Provider(), default_camera="head")
    api.estimate_geometry = lambda cloud: expected.geometry
    localization = api.localize_object_from_observation("apple", observation, require_single_instance=True)
    assert localization.ok
    result = _partition(observation, localization)
    assert len(result.target.points) == 1 and len(result.scene.point_cloud.points) == 8


def test_other_cameras_are_preserved_conservatively():
    observation, localization = _fixture()
    camera = observation.cameras["head"]
    other = replace(camera, frame="other_optical")
    observation = replace(observation, cameras={"head": camera, "other": other})
    result = _partition(observation, localization)
    # This unsegmented view includes the target as well as eight background pixels.
    assert len(result.scene.point_cloud.points) == 17
    changed_depth = np.array(other.depth_m)
    changed_depth[0, 0] = 1.1
    changed = replace(observation, cameras={"head": camera, "other": replace(other, depth_m=changed_depth)})
    assert _partition(changed, localization).scene_fingerprint != result.scene_fingerprint


def test_rejects_changed_snapshot_and_foreign_or_tampered_target():
    observation, localization = _fixture()
    camera = observation.cameras["head"]
    changed = np.array(camera.depth_m)
    changed[0, 0] = 2
    with pytest.raises(ValueError, match="unique observation"):
        _partition(replace(observation, cameras={"head": replace(camera, depth_m=changed)}), localization)
    with pytest.raises(ValueError, match="point cloud does not match"):
        _partition(observation, replace(localization, point_cloud=replace(localization.point_cloud, points=[[5, 5, 5]])))
    with pytest.raises(ValueError, match="unique observation"):
        _partition(observation, replace(localization, diagnostics={**localization.diagnostics, "instance_count": 2}))


def test_scene_fingerprint_changes_when_target_mask_changes():
    observation, localization = _fixture()
    first = _partition(observation, localization)
    mask = np.array(localization.segmentation.mask)
    mask[1, 2] = True
    segment = replace(localization.segmentation, mask=mask)
    points = mask_to_point_cloud(observation.cameras["head"], segment, target_frame="base")
    with pytest.raises(ValueError, match="differs from the source localization"):
        _partition(observation, replace(localization, segmentation=segment, point_cloud=points))
    revised = _partition(observation, replace(
        localization, segmentation=segment, point_cloud=points,
        diagnostics={**localization.diagnostics, "target_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()},
    ))
    assert revised.snapshot_fingerprint == first.snapshot_fingerprint
    assert revised.scene_fingerprint != first.scene_fingerprint


def test_rejects_unsynchronized_views_or_empty_obstacle_scene():
    observation, localization = _fixture()
    camera = observation.cameras["head"]
    other = replace(camera, frame="other_optical", timestamp_s=1.1)
    with pytest.raises(ValueError, match="allowed skew"):
        _partition(replace(observation, cameras={"head": camera, "other": other}), localization)
    full = replace(localization.segmentation, mask=np.ones((3, 3), dtype=bool))
    full_cloud = mask_to_point_cloud(camera, full, target_frame="base")
    with pytest.raises(ValueError, match="no pixels"):
        _partition(observation, replace(
            localization, segmentation=full, point_cloud=full_cloud,
            diagnostics={**localization.diagnostics, "target_mask_sha256": hashlib.sha256(full.mask.tobytes()).hexdigest()},
        ))


def _attachment():
    return AttachedObjectGeometry(
        object_link_name="held", attachment_joint_name="fixed", parent_link_name="tcp",
        parent_to_object=(0, 0, 0, 1, 0, 0, 0), collision_spheres=((0, 0, 0, 0.05),),
        contact_link_names=("finger",),
    )


def test_service_validates_scene_bytes_and_skips_second_target_carve(monkeypatch):
    observation, localization = _fixture()
    partition = _partition(observation, localization)
    attachment = _attachment()
    proof = _validated_partition_payload(partition, partition.scene, observation.robot_state, attachment)
    scene = partition.scene.point_cloud.points.astype(np.float32)
    payload = {
        "base_frame": "base", "model": "agibot_g1", "joint_positions": {"right": [0.0] * 7},
        "joint_names": {"right": list(observation.robot_state.joint_names["right"])},
        "base_transforms": {"right": np.eye(4).tolist()}, "end_effector_links": {"right": "tcp"},
        "scene_points_base64": encode_numpy(scene), "attached_object": attachment.to_payload(),
        "target_scene_partition": proof,
    }
    request = service.BaseRequest.model_validate(payload)
    bad = dict(payload, scene_points_base64=encode_numpy(scene + 0.01))
    with pytest.raises(ValueError, match="SHA-256"):
        service.BaseRequest.model_validate(bad)

    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def __array__(self, dtype=None): return np.array([[10, 10, 10, 0.01]], dtype=dtype)

    segmentation = SimpleNamespace(compute_kinematics=lambda _: SimpleNamespace(robot_spheres=Tensor()))
    monkeypatch.setattr(service, "_segmentation_kinematics", lambda _: segmentation)
    monkeypatch.setattr(service, "_state", lambda *_: object())
    monkeypatch.setattr(service, "carve_attached_object_points", lambda *args, **kwargs: pytest.fail("second carve"))
    result = service._carve_robot_points(request, object(), scene)
    np.testing.assert_array_equal(result, scene)
    diagnostics = service._attachment_diagnostics(request)
    assert diagnostics["scene_partition"] == proof
    assert diagnostics["robot_scene_carve_may_remove_support_points"] is True
    collision = service._collision_diagnostics(request)
    assert collision["robot_model_verified"] is False
    assert collision["target_partition_safety_verified"] is False
    assert collision["robot_scene_carve_verified"] is False
    with pytest.raises(ValueError, match="attached object and scene"):
        service.BaseRequest.model_validate({**payload, "attached_object": None})


@pytest.mark.parametrize("ack", ["matching", "missing", "different"])
def test_provider_requires_service_partition_ack_and_never_marks_plan_executable(ack):
    observation, localization = _fixture()
    partition = _partition(observation, localization)
    state = observation.robot_state
    context = RobotPlanningContext(
        embodiment=state.embodiment, model="agibot_g1", joint_names=state.joint_names,
        base_transforms={"right": np.eye(4)}, end_effector_links={"right": "tcp"},
    )

    class Response:
        status_code = 200
        text = ""
        def __init__(self, payload): self.payload = payload
        def json(self): return self.payload

    class Session:
        calls = []
        def request(self, method, url, **kwargs):
            payload = kwargs["json"]
            self.calls.append(payload)
            request = service.PlanRequest.model_validate(payload)
            diagnostics = service._attachment_diagnostics(request)
            if ack == "missing":
                diagnostics.pop("scene_partition")
            if ack == "different":
                diagnostics["scene_partition"]["scene_fingerprint"] = "f" * 64
            return Response({"joint_positions_base64": encode_numpy(np.zeros((2, 7))), "dt_s": 0.1,
                             "attachment_diagnostics": diagnostics,
                             "collision_diagnostics": {"collision_disabled": False, "scene_supplied": True,
                                "world_collision_check_enabled": True, "self_collision_check_enabled": True,
                                "robot_model_verified": True}})

    session = Session()
    provider = CuRoboProvider(session=session)
    result = provider.plan_to_pose(state, Pose([0, 0, 1], [1, 0, 0, 0], "base"), arm="right",
                                   scene=partition.scene, context=context, attached_object=_attachment(),
                                   target_scene_partition=partition)
    assert len(session.calls) == 1
    if ack == "matching":
        assert result.ok and not result.trajectory.collision_aware
        assert result.diagnostics["collision_aware"] is False
        assert result.diagnostics["collision_checks_enabled"] is True
        assert result.diagnostics["robot_scene_carve_may_remove_support_points"] is True
    else:
        assert not result.ok and "invalid motion plan" in result.error.message


def test_tampered_partition_is_refused_before_service():
    observation, localization = _fixture()
    partition = _partition(observation, localization)
    tampered = replace(partition.scene, point_cloud=replace(partition.scene.point_cloud, points=[[9, 9, 9]]))
    with pytest.raises(ValueError, match="differs from its source"):
        _validated_partition_payload(partition, tampered, observation.robot_state, _attachment())
    with pytest.raises(ValueError, match="differs from its source"):
        _validated_partition_payload(replace(partition, scene_fingerprint="f" * 64), partition.scene,
                                     observation.robot_state, _attachment())
    changed_state = replace(observation.robot_state, joint_positions={"right": np.ones(7)})
    with pytest.raises(ValueError, match="planning state does not match"):
        _validated_partition_payload(partition, partition.scene, changed_state, _attachment())
    with pytest.raises(ValueError, match="requires an attached object"):
        _validated_partition_payload(partition, partition.scene, observation.robot_state, None)


@pytest.mark.parametrize("preserve,attached", [(False, True), (True, True), (True, False)])
def test_cap_api_passes_bound_partition_to_curobo_and_keeps_plan_non_executable(preserve, attached):
    observation, localization = _fixture()
    state = observation.robot_state

    class Adapter:
        embodiment = "open_rail_7dof"

        def get_robot_state(self):
            return state

        def get_planning_context(self):
            return RobotPlanningContext(
                embodiment=state.embodiment, model="agibot_g1", joint_names=state.joint_names,
                base_transforms={"right": np.eye(4)}, end_effector_links={"right": "tcp"},
            )

    class Response:
        status_code = 200
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.calls = []

        def request(self, _method, _url, **kwargs):
            payload = kwargs["json"]
            self.calls.append(payload)
            request = service.PlanRequest.model_validate(payload)
            return Response({
                "joint_positions_base64": encode_numpy(np.zeros((2, 7))), "dt_s": 0.1,
                "attachment_diagnostics": service._attachment_diagnostics(request),
                "scene_partition_diagnostics": service._scene_partition_diagnostics(request),
                "collision_diagnostics": {"collision_disabled": False, "scene_supplied": True,
                    "world_collision_check_enabled": True, "self_collision_check_enabled": True,
                    "robot_model_verified": True},
            })

    session = Session()
    provider = CuRoboProvider(session=session)
    api = CapApi(Adapter(), integrated_pose_planning_providers={"curobo-integrated": provider})
    assert "partition_target_scene" in api.register_tools(ToolRegistry()).public_tools()
    partition = api.partition_target_scene(observation, localization, camera_name="head",
                                           max_camera_state_skew_s=0.02)
    kwargs = dict(arm="right", strategy=MotionStrategy(pose_planner="curobo-integrated"),
                  planning_scene=partition.scene, robot_state=state,
                  attached_object=_attachment() if attached else None,
                  target_scene_partition=partition, preserve_observed_scene=preserve)
    result = api.plan_motion(Pose([0, 0, 1], [1, 0, 0, 0], "base"), **kwargs)
    assert result.ok and not result.trajectory.collision_aware
    assert result.diagnostics["collision_aware"] is False
    assert session.calls[0]["target_scene_partition"]["scene_fingerprint"] == partition.scene_fingerprint
    assert session.calls[0]["target_scene_partition"]["mode"] == ("pixel_partition_v2" if preserve else "pixel_partition_v1")
    altered_scene = replace(partition.scene, point_cloud=replace(partition.scene.point_cloud, points=[[9, 9, 9]]))
    rejected = api.plan_motion(Pose([0, 0, 1], [1, 0, 0, 0], "base"), **{**kwargs, "planning_scene": altered_scene})
    assert not rejected.ok and len(session.calls) == 1
    rejected = api.plan_motion(Pose([0, 0, 1], [1, 0, 0, 0], "base"), **{**kwargs, "target_scene_partition": None, "preserve_observed_scene": True})
    assert not rejected.ok and len(session.calls) == 1


@pytest.mark.parametrize("attached", [False, True])
def test_preserved_partition_skips_all_geometric_carves(monkeypatch, attached):
    import sys
    from types import ModuleType

    observation, localization = _fixture()
    partition = _partition(observation, localization)
    attachment = _attachment() if attached else None
    proof = _validated_partition_payload(
        partition, partition.scene, observation.robot_state, attachment,
        preserve_observed_scene=True,
    )
    points = partition.scene.point_cloud.points.astype(np.float32)
    payload = {
        "base_frame": "base", "model": "agibot_g1", "joint_positions": {"right": [0.0] * 7},
        "joint_names": {"right": list(observation.robot_state.joint_names["right"])},
        "base_transforms": {"right": np.eye(4).tolist()}, "end_effector_links": {"right": "tcp"},
        "scene_points_base64": encode_numpy(points), "target_scene_partition": proof,
        "attached_object": None if attachment is None else attachment.to_payload(),
    }
    request = service.BaseRequest.model_validate(payload)
    scene_module = ModuleType("curobo.scene")
    scene_module.Scene = lambda *, mesh: mesh
    monkeypatch.setitem(sys.modules, "curobo.scene", scene_module)
    monkeypatch.setattr(service, "_collision_disabled", lambda *_: False)
    monkeypatch.setattr(service, "_drop_distant_points", lambda *_: pytest.fail("background was filtered"))
    monkeypatch.setattr(service, "_carve_robot_points", lambda *_: pytest.fail("background was carved"))
    captured = []
    monkeypatch.setattr(service, "_voxelized_scene_mesh", lambda received, **_: captured.append(received.copy()))
    service._scene(request, object())
    np.testing.assert_array_equal(captured[0], points)
    diagnostics = service._scene_partition_diagnostics(request)
    assert diagnostics["scene_partition"] == proof
    assert diagnostics["robot_scene_carve_applied"] is False
    assert diagnostics["target_interaction_verified"] is False
    assert diagnostics["occluded_space_verified"] is False

    too_far = points.copy()
    too_far[0, 0] = 6.0
    payload["scene_points_base64"] = encode_numpy(too_far)
    import hashlib
    payload["target_scene_partition"] = {**proof, "scene_points_sha256": hashlib.sha256(too_far.tobytes()).hexdigest()}
    far_request = service.BaseRequest.model_validate(payload)
    with pytest.raises(service.HTTPException, match="supported range"):
        service._scene(far_request, object())
    huge_grid = points.copy()
    huge_grid[0] = [2.0, 2.0, 2.0]
    payload["scene_points_base64"] = encode_numpy(huge_grid)
    payload["voxel_size_m"] = 0.001
    payload["target_scene_partition"] = {**proof, "scene_points_sha256": hashlib.sha256(huge_grid.tobytes()).hexdigest()}
    with pytest.raises(service.HTTPException, match="voxel grid"):
        service._scene(service.BaseRequest.model_validate(payload), object())
    monkeypatch.setattr(service, "_collision_disabled", lambda *_: True)
    with pytest.raises(service.HTTPException, match="requires collision checking"):
        service._scene(request, object())


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("ack", ["matching", "missing", "unsafe"])
def test_preserved_partition_needs_service_ack_and_stays_preview_only(attached, ack):
    observation, localization = _fixture()
    partition = _partition(observation, localization)
    state = observation.robot_state
    context = RobotPlanningContext(
        embodiment=state.embodiment, model="agibot_g1", joint_names=state.joint_names,
        base_transforms={"right": np.eye(4)}, end_effector_links={"right": "tcp"},
    )

    class Response:
        status_code = 200
        text = ""
        def __init__(self, payload): self.payload = payload
        def json(self): return self.payload

    class Session:
        def request(self, _method, _url, **kwargs):
            request = service.PlanRequest.model_validate(kwargs["json"])
            diag = service._scene_partition_diagnostics(request)
            if ack == "missing":
                diag = None
            elif ack == "unsafe":
                diag["robot_scene_carve_applied"] = True
            return Response({
                "joint_positions_base64": encode_numpy(np.zeros((2, 7))), "dt_s": 0.1,
                "attachment_diagnostics": service._attachment_diagnostics(request),
                "scene_partition_diagnostics": diag,
                "collision_diagnostics": {"collision_disabled": False, "scene_supplied": True,
                    "world_collision_check_enabled": True, "self_collision_check_enabled": True,
                    "robot_model_verified": True},
            })

    provider = CuRoboProvider(session=Session())
    result = provider.plan_to_pose(
        state, Pose([0, 0, 1], [1, 0, 0, 0], "base"), arm="right",
        scene=partition.scene, context=context, attached_object=_attachment() if attached else None,
        target_scene_partition=partition, preserve_observed_scene=True,
    )
    if ack == "matching":
        assert result.ok and not result.trajectory.collision_aware
        assert result.diagnostics["collision_aware"] is False
        assert result.diagnostics["robot_scene_carve_may_remove_support_points"] is False
        assert result.diagnostics["target_interaction_verified"] is False
        assert result.diagnostics["occluded_space_verified"] is False
    else:
        assert not result.ok


def test_preserving_scene_requires_explicit_bound_partition():
    observation, _ = _fixture()
    state = observation.robot_state
    context = RobotPlanningContext(
        embodiment=state.embodiment, model="agibot_g1", joint_names=state.joint_names,
        base_transforms={"right": np.eye(4)}, end_effector_links={"right": "tcp"},
    )
    partition = _partition(*_fixture())
    result = CuRoboProvider().plan_to_joints(
        state, np.zeros(7), arm="right", scene=partition.scene, context=context,
        preserve_observed_scene=True,
    )
    assert not result.ok
    assert "requires a pixel-partitioned scene" in result.error.details["error"]
