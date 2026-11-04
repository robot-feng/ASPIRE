from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np

from cap_harness.api import CapApi
from cap_harness.contracts import (
    CameraObservation,
    GraspCandidate,
    GraspSet,
    MotionStrategy,
    ObjectGeometry,
    Observation,
    PlanningScene,
    PlanResult,
    PointCloud,
    Pose,
    RobotState,
    Segmentation,
    SegmentationSet,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import ProgramExecutor


def _pose(position=(0.0, 0.0, 0.0), frame="base") -> Pose:
    return Pose(
        position=np.asarray(position, dtype=np.float64),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame=frame,
    )


def _observation(embodiment: str = "open_rail_7dof") -> Observation:
    state = RobotState(
        joint_positions=np.zeros(7),
        joint_velocities=np.zeros(7),
        end_effector_poses=_pose(),
        gripper_positions=0.5,
        base_frame="base",
        timestamp_s=12.0,
        embodiment=embodiment,
    )
    camera = CameraObservation(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float64),
        intrinsics=np.eye(3),
        frame="head_optical",
        camera_pose=_pose(),
        timestamp_s=12.0,
    )
    return Observation(
        cameras={"head": camera},
        robot_state=state,
        task_context=TaskContext("open_rail", 0, "pick apple", "pick apple", "real"),
        timestamp_s=12.0,
    )


class _Adapter:
    def __init__(self, observation: Observation) -> None:
        self.observation = observation
        self.embodiment = observation.robot_state.embodiment
        self.observation_reads = 0

    def get_observation(self) -> Observation:
        self.observation_reads += 1
        return self.observation

    def get_robot_state(self) -> RobotState:
        return self.observation.robot_state


class _SegmentationProvider:
    def __init__(self) -> None:
        self.camera_observations: list[CameraObservation] = []

    def segment_text(self, observation: CameraObservation, query: str) -> SegmentationSet:
        self.camera_observations.append(observation)
        return SegmentationSet(
            ok=True,
            segmentations=(
                Segmentation(
                    mask=np.array([[True, False], [False, False]]),
                    label=query,
                    score=0.5,
                    camera_name=observation.frame,
                    frame=observation.frame,
                ),
            ),
        )

    def segment_points(self, *args, **kwargs) -> SegmentationSet:
        del args, kwargs
        raise AssertionError("point segmentation is not used")


class _GraspProvider:
    def __init__(self) -> None:
        self.point_cloud: PointCloud | None = None
        self.scene_point_cloud: PointCloud | None = None

    def generate_grasps(
        self, point_cloud: PointCloud, *, scene_point_cloud=None, **_kwargs
    ) -> GraspSet:
        self.point_cloud = point_cloud
        self.scene_point_cloud = scene_point_cloud
        return GraspSet(
            ok=True,
            grasps=(GraspCandidate(pose=_pose(frame=point_cloud.frame), score=0.5, width_m=0.04),),
        )


class _PosePlanner:
    def __init__(self) -> None:
        self.scene = None
        self.robot_state = None

    def plan_to_pose(self, robot_state, target_pose, *, scene=None, **_kwargs):
        del target_pose
        self.robot_state = robot_state
        self.scene = scene
        return PlanResult(
            ok=True,
            trajectory=Trajectory(
                joint_positions=robot_state.joint_positions["primary"][None],
                dt_s=0.05,
                joint_names=robot_state.joint_names["primary"],
                planner="fake",
                collision_aware=True,
                expected_start=robot_state,
                arm="primary",
                embodiment=robot_state.embodiment,
            ),
        )


class _JointPlanner:
    def plan_to_joints(self, robot_state, _goal, *, scene=None, **_kwargs):
        assert scene is not None
        return PlanResult(
            ok=True,
            trajectory=Trajectory(
                joint_positions=robot_state.joint_positions["primary"][None],
                dt_s=0.05,
                joint_names=robot_state.joint_names["primary"],
                planner="fake",
                collision_aware=True,
                expected_start=robot_state,
                arm="primary",
                embodiment=robot_state.embodiment,
            ),
        )


def test_perception_and_grasp_reuse_the_supplied_observation() -> None:
    observation = _observation()
    adapter = _Adapter(observation)
    segmentation = _SegmentationProvider()
    grasping = _GraspProvider()
    api = CapApi(
        adapter,  # type: ignore[arg-type]
        segmentation_provider=segmentation,
        grasp_providers={"graspgenx": grasping},
        default_camera="head",
    )
    api.estimate_geometry = lambda point_cloud: ObjectGeometry(  # type: ignore[method-assign]
        pose=_pose(point_cloud.points.mean(axis=0), frame=point_cloud.frame),
        extents=np.array([0.1, 0.1, 0.1]),
        point_count=len(point_cloud.points),
    )

    localized = api.localize_object_from_observation(
        "apple",
        observation,
        camera_name="head",
    )
    grasps = api.generate_grasps_for_localization(
        localized,
        observation,
        camera_name="head",
        max_camera_state_skew_s=0.05,
        backend="graspgenx",
    )

    assert localized.ok and grasps.ok
    assert adapter.observation_reads == 0
    assert segmentation.camera_observations == [observation.cameras["head"]]
    assert grasping.point_cloud is not None and grasping.scene_point_cloud is not None
    assert grasping.point_cloud.frame == "base"
    assert grasping.scene_point_cloud.frame == "base"
    assert localized.diagnostics["camera_timestamp_s"] == 12.0
    assert grasps.diagnostics["camera_timestamp_s"] == 12.0
    assert grasps.diagnostics["observation_timestamp_s"] == 12.0
    assert grasps.diagnostics["snapshot_id"] == localized.diagnostics["snapshot_id"]


def test_grasp_generation_rejects_localization_from_an_older_snapshot() -> None:
    observation = _observation()
    grasping = _GraspProvider()
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        segmentation_provider=_SegmentationProvider(),
        grasp_providers={"graspgenx": grasping},
        default_camera="head",
    )
    api.estimate_geometry = lambda cloud: ObjectGeometry(  # type: ignore[method-assign]
        pose=_pose(cloud.points.mean(axis=0), frame=cloud.frame),
        extents=np.array([0.1, 0.1, 0.1]),
    )
    localized = api.localize_object_from_observation("apple", observation)
    newer = replace(
        observation,
        cameras={"head": replace(observation.cameras["head"], timestamp_s=13.0)},
        timestamp_s=13.0,
    )

    result = api.generate_grasps_for_localization(
        localized, newer, max_camera_state_skew_s=0.05
    )

    assert not result.ok
    assert result.error is not None and result.error.details["reason"] == "snapshot_mismatch"
    assert grasping.point_cloud is None


def test_grasp_generation_rejects_ambiguous_localization() -> None:
    observation = _observation()
    grasping = _GraspProvider()
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        segmentation_provider=_SegmentationProvider(),
        grasp_providers={"graspgenx": grasping},
        default_camera="head",
    )
    api.estimate_geometry = lambda cloud: ObjectGeometry(  # type: ignore[method-assign]
        pose=_pose(cloud.points.mean(axis=0), frame=cloud.frame),
        extents=np.array([0.1, 0.1, 0.1]),
    )
    localized = api.localize_object_from_observation("apple", observation)
    ambiguous = replace(localized, diagnostics={**localized.diagnostics, "instance_count": 2})

    result = api.generate_grasps_for_localization(
        ambiguous, observation, max_camera_state_skew_s=0.05
    )

    assert not result.ok
    assert grasping.point_cloud is None


def test_grasp_generation_rejects_unsynchronized_camera_and_robot_state() -> None:
    original = _observation()
    observation = replace(
        original, robot_state=replace(original.robot_state, timestamp_s=11.8)
    )
    grasping = _GraspProvider()
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        segmentation_provider=_SegmentationProvider(),
        grasp_providers={"graspgenx": grasping},
        default_camera="head",
    )
    api.estimate_geometry = lambda cloud: ObjectGeometry(  # type: ignore[method-assign]
        pose=_pose(cloud.points.mean(axis=0), frame=cloud.frame),
        extents=np.array([0.1, 0.1, 0.1]),
    )
    localized = api.localize_object_from_observation("apple", observation)

    result = api.generate_grasps_for_localization(
        localized, observation, max_camera_state_skew_s=0.05
    )

    assert not result.ok
    assert result.error is not None and result.error.details["reason"] == "camera_state_skew"
    assert grasping.point_cloud is None


def test_localization_can_reject_ambiguous_instances_before_projection() -> None:
    observation = _observation()
    provider = _SegmentationProvider()
    first = provider.segment_text(observation.cameras["head"], "apple").segmentations[0]
    second = Segmentation(
        mask=np.array([[False, True], [False, False]]),
        label="apple",
        score=0.5,
        camera_name="head_optical",
        frame="head_optical",
    )
    provider.segment_text = lambda *_args: SegmentationSet(  # type: ignore[method-assign]
        ok=True, segmentations=(first, second)
    )
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        segmentation_provider=provider,
        default_camera="head",
    )

    result = api.localize_object_from_observation(
        "apple", observation, camera_name="head", require_single_instance=True
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.details["instance_count"] == 2
    assert result.point_cloud is None


def test_localization_counts_provider_instances_before_empty_masks_are_filtered() -> None:
    observation = _observation()
    provider = _SegmentationProvider()
    first = provider.segment_text(observation.cameras["head"], "apple").segmentations[0]
    provider.segment_text = lambda *_args: SegmentationSet(  # type: ignore[method-assign]
        ok=True, segmentations=(first,), diagnostics={"total_instances": 2}
    )
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        segmentation_provider=provider,
        default_camera="head",
    )

    result = api.localize_object_from_observation(
        "apple", observation, require_single_instance=True
    )

    assert not result.ok
    assert result.error is not None and result.error.details["instance_count"] == 2
    assert result.point_cloud is None


def test_localization_rejects_sparse_depth_in_the_target_mask() -> None:
    original = _observation()
    camera = original.cameras["head"]
    sparse = CameraObservation(
        rgb=camera.rgb,
        depth_m=np.array([[1.0, 0.0], [0.0, 0.0]]),
        intrinsics=camera.intrinsics,
        frame=camera.frame,
        camera_pose=camera.camera_pose,
        timestamp_s=camera.timestamp_s,
    )
    observation = Observation(cameras={"head": sparse}, robot_state=original.robot_state)
    mask = Segmentation(
        mask=np.ones((2, 2), dtype=bool),
        label="apple",
        score=0.5,
        camera_name="head",
        frame=sparse.frame,
    )
    api = CapApi(_Adapter(observation), default_camera="head")  # type: ignore[arg-type]

    result = api.localize_object_from_observation(
        "apple",
        observation,
        segmentation=mask,
        min_valid_depth_points=2,
        min_valid_depth_fraction=0.5,
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.details["valid_depth_points"] == 1
    assert result.error.details["valid_depth_fraction"] == 0.25


def test_explicit_planning_scene_is_reused_without_another_camera_read() -> None:
    observation = _observation()
    adapter = _Adapter(observation)
    planner = _PosePlanner()
    api = CapApi(
        adapter,  # type: ignore[arg-type]
        integrated_pose_planning_providers={"curobo-integrated": planner},
        default_camera="head",
    )
    scene = api.get_planning_scene(observation=observation)
    target = PointCloud(points=np.array([[0.0, 0.0, 1.0]]), frame="base")

    result = api.plan_motion(
        _pose(),
        strategy=MotionStrategy(pose_planner="curobo-integrated"),
        planning_scene=scene,
        target_point_cloud=target,
        robot_state=observation.robot_state,
    )

    assert result.ok
    assert adapter.observation_reads == 0
    assert planner.scene is not None
    assert planner.scene.timestamp_s == 12.0
    assert planner.robot_state is not None
    assert planner.robot_state.timestamp_s == observation.robot_state.timestamp_s
    assert not np.any(np.all(np.isclose(planner.scene.point_cloud.points, target.points[0]), axis=1))
    assert len(planner.scene.point_cloud.points) < len(scene.point_cloud.points)


def test_explicit_planning_scene_must_match_robot_base_frame() -> None:
    observation = _observation()
    api = CapApi(_Adapter(observation))  # type: ignore[arg-type]
    scene = api.get_planning_scene(observation=observation)
    camera_frame_scene = PlanningScene(
        point_cloud=PointCloud(points=np.array([[0.0, 0.0, 1.0]]), frame="camera"),
        timestamp_s=scene.timestamp_s,
    )

    result = api.plan_motion(_pose(), planning_scene=camera_frame_scene)

    assert not result.ok
    assert result.error is not None
    assert "expected 'base'" in result.error.message


def _attachment() -> dict:
    return {
        "object_link_name": "apple_link",
        "attachment_joint_name": "apple_joint",
        "parent_link_name": "tcp",
        "parent_to_object": [0, 0, 0.1, 1, 0, 0, 0],
        "collision_spheres": [[0, 0, 0, 0.05]],
        "contact_link_names": ["finger"],
    }


def test_program_passes_attachment_to_capable_pose_planner() -> None:
    class Planner(_PosePlanner):
        supports_attached_objects = True

        def plan_to_pose(self, robot_state, target_pose, **kwargs):
            self.attachment = kwargs["attached_object"]
            return super().plan_to_pose(robot_state, target_pose, **kwargs)

    observation = _observation()
    planner = Planner()
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        integrated_pose_planning_providers={"curobo-integrated": planner},
        default_camera="head",
    )
    program = f"""
target = Pose(position=[0, 0, 0], quaternion_wxyz=[1, 0, 0, 0], frame='base')
result = plan_motion(target, strategy=MotionStrategy(pose_planner='curobo-integrated'),
                     attached_object={_attachment()!r}).ok
"""

    execution = ProgramExecutor(api.register_tools(ToolRegistry())).execute_program(program)

    assert execution.ok and execution.result is True
    assert planner.attachment.object_link_name == "apple_link"


def test_attached_object_rejects_interpolation_or_unsupported_provider() -> None:
    observation = _observation()
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        integrated_pose_planning_providers={"curobo-integrated": _PosePlanner()},
        default_camera="head",
    )

    interpolation = api.plan_motion(_pose(), attached_object=_attachment())
    unsupported = api.plan_motion(
        _pose(),
        strategy=MotionStrategy(pose_planner="curobo-integrated"),
        attached_object=_attachment(),
    )

    assert not interpolation.ok and interpolation.error.code.value == "unsupported"
    assert not unsupported.ok and unsupported.error.code.value == "unsupported"


def test_joint_planner_receives_the_attached_object() -> None:
    class Planner(_JointPlanner):
        supports_attached_objects = True

        def plan_to_joints(self, robot_state, target, **kwargs):
            self.attachment = kwargs["attached_object"]
            return super().plan_to_joints(robot_state, target, **kwargs)

    planner = Planner()
    api = CapApi(
        _Adapter(_observation()),  # type: ignore[arg-type]
        trajectory_planning_providers={"curobo": planner},
        default_camera="head",
    )

    result = api.plan_motion(
        np.ones(7),
        strategy=MotionStrategy(trajectory_planner="curobo"),
        attached_object=_attachment(),
    )

    assert result.ok
    assert planner.attachment.object_link_name == "apple_link"


def test_attached_object_cannot_also_remove_a_target_bounding_box() -> None:
    api = CapApi(_Adapter(_observation()))  # type: ignore[arg-type]
    target = PointCloud(points=np.array([[0.0, 0.0, 1.0]]), frame="base")

    result = api.plan_motion(
        _pose(),
        strategy=MotionStrategy(pose_planner="curobo-integrated"),
        attached_object=_attachment(),
        target_point_cloud=target,
    )

    assert not result.ok and result.error.code.value == "invalid_request"


def test_move_to_pose_forwards_attachment_and_does_not_execute_failed_plan() -> None:
    api = CapApi(_Adapter(_observation()))  # type: ignore[arg-type]
    observed = {}

    def fail_plan(_target, **kwargs):
        observed.update(kwargs)
        return PlanResult(ok=False, error=ApiError(ErrorCode.PLANNING_FAILED, "test failure"))

    api.plan_motion = fail_plan  # type: ignore[method-assign]
    attachment = _attachment()
    result = api.move_to_pose(_pose(), attached_object=attachment)

    assert not result.ok
    assert observed["attached_object"] is attachment


def test_unconfigured_grasp_backend_does_not_acquire_a_camera_frame() -> None:
    adapter = _Adapter(_observation("robosuite"))
    api = CapApi(adapter)  # type: ignore[arg-type]

    result = api.generate_grasps("head", np.ones((2, 2), dtype=bool), backend="missing")

    assert not result.ok
    assert result.error is not None and result.error.code.value == "unsupported"
    assert adapter.observation_reads == 0


def test_legacy_perception_results_keep_their_original_diagnostics() -> None:
    observation = _observation("robosuite")
    api = CapApi(
        _Adapter(observation),  # type: ignore[arg-type]
        segmentation_provider=_SegmentationProvider(),
        grasp_providers={"graspgenx": _GraspProvider()},
        default_camera="head",
    )
    api.estimate_geometry = lambda cloud: ObjectGeometry(  # type: ignore[method-assign]
        pose=_pose(cloud.points.mean(axis=0), frame=cloud.frame),
        extents=np.array([0.1, 0.1, 0.1]),
    )

    localized = api.localize_object("apple")
    grasps = api.generate_grasps("head", localized.segmentation, backend="graspgenx")

    assert localized.ok and localized.diagnostics == {}
    assert grasps.ok and grasps.diagnostics == {}


def test_legacy_localization_checks_provider_before_missing_camera() -> None:
    api = CapApi(_Adapter(_observation("robosuite")))  # type: ignore[arg-type]

    result = api.localize_object("apple", camera_name="missing")

    assert not result.ok
    assert result.error is not None and result.error.code.value == "unsupported"


def test_default_scene_timestamp_remains_the_observation_timestamp() -> None:
    observation = replace(_observation("robosuite"), timestamp_s=None)
    api = CapApi(_Adapter(observation))  # type: ignore[arg-type]

    assert api._planning_scene().timestamp_s is None
    assert api.get_planning_scene(observation).timestamp_s == 12.0


def test_default_joint_planner_scene_keeps_existing_adapter_behavior() -> None:
    class AdapterWithSceneTransform(_Adapter):
        def planning_scene_request(self, _scene, _arm):
            raise AssertionError("default joint planning must not apply this transform")

    adapter = AdapterWithSceneTransform(_observation("robosuite"))
    api = CapApi(
        adapter,  # type: ignore[arg-type]
        trajectory_planning_providers={"curobo": _JointPlanner()},
    )

    result = api.plan_motion(np.ones(7), strategy=MotionStrategy(trajectory_planner="curobo"))

    assert result.ok


def test_snapshot_tools_are_registered_only_for_open_rail() -> None:
    names = {
        "get_planning_scene",
        "generate_grasps_for_localization",
        "generate_grasps_from_observation",
        "localize_object_from_observation",
    }
    open_rail_registry = CapApi(_Adapter(_observation())).register_tools(ToolRegistry())
    robosuite_registry = CapApi(
        _Adapter(_observation("robosuite"))
    ).register_tools(ToolRegistry())

    assert names <= set(open_rail_registry.public_tools())
    assert names.isdisjoint(robosuite_registry.public_tools())


def test_existing_perception_tool_signatures_remain_unchanged() -> None:
    assert tuple(inspect.signature(CapApi.segment_text).parameters) == (
        "self",
        "camera_name",
        "text",
    )
    assert tuple(inspect.signature(CapApi.segment_points).parameters) == (
        "self",
        "camera_name",
        "points_px",
    )
    assert tuple(inspect.signature(CapApi.generate_grasps).parameters) == (
        "self",
        "camera_name",
        "mask",
        "backend",
        "max_candidates",
    )
    assert tuple(inspect.signature(CapApi.localize_object).parameters) == (
        "self",
        "query",
        "camera_name",
        "target_frame",
    )
