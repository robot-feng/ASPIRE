"""Partition an observed target and collision scene by image pixels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np

from cap_harness.contracts import LocalizationResult, Observation, PlanningScene, PointCloud, Segmentation
from cap_harness.geometry import mask_to_point_cloud
from cap_harness.open_rail.snapshot_identity import snapshot_fingerprint


@dataclass(frozen=True, slots=True)
class TargetScenePartition:
    """Target points and non-target scene from one bound observation.

    This is not evidence that the target mask, calibration, occluded space or
    robot collision model is correct. Other camera views, if any, remain intact
    in the scene, so an unsegmented second view can still contain the target.
    """

    target: PointCloud
    scene: PlanningScene
    snapshot_id: str
    snapshot_fingerprint: str
    target_camera: str
    scene_fingerprint: str
    source_observation: Observation = field(repr=False, compare=False)
    source_localization: LocalizationResult = field(repr=False, compare=False)
    max_camera_state_skew_s: float


def partition_target_scene(
    observation: Observation,
    localization: LocalizationResult,
    *,
    camera_name: str,
    max_camera_state_skew_s: float,
    voxel_size_m: float,
) -> TargetScenePartition:
    """Project target and background pixels separately, without a 3-D carve.

    All non-target pixels with valid depth are retained, including table points
    inside the target's bounding box or collision-sphere envelope. The caller
    must validate its segmentation, calibration, wall-clock freshness and
    unobserved space before using the scene; this function does not mark a
    motion or a robot model as collision-checked.
    """
    if not isinstance(observation, Observation) or not isinstance(localization, LocalizationResult):
        raise ValueError("an observation and localization are required")
    state = observation.robot_state
    if not state.embodiment.startswith("open_rail_"):
        raise ValueError("a bound Open-RAIL observation is required")
    if not isinstance(camera_name, str) or camera_name not in observation.cameras:
        raise ValueError("target camera is unavailable")
    if not localization.ok or localization.segmentation is None or localization.point_cloud is None:
        raise ValueError("successful target localization with a mask and point cloud is required")
    if isinstance(max_camera_state_skew_s, bool) or isinstance(voxel_size_m, bool):
        raise ValueError("skew and voxel size must be finite numbers")
    try:
        max_skew, voxel_size = float(max_camera_state_skew_s), float(voxel_size_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("skew and voxel size must be finite numbers") from exc
    if not np.isfinite(max_skew) or max_skew < 0 or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("skew must be nonnegative and voxel size positive")
    camera = observation.cameras[camera_name]
    if camera.timestamp_s is None or state.timestamp_s is None:
        raise ValueError("camera and state timestamps are required")
    expected_id = f"{camera.frame}@{camera.timestamp_s:.9f}"
    fingerprint = snapshot_fingerprint(camera, state)
    info = localization.diagnostics
    if (
        info.get("snapshot_id") != expected_id
        or info.get("snapshot_fingerprint") != fingerprint
        or info.get("camera_name") != camera_name
        or info.get("robot_state_timestamp_s") != state.timestamp_s
        or info.get("segmentation_source") != "provider"
        or info.get("instance_count") != 1
    ):
        raise ValueError("target localization does not match this unique observation")
    mask = localization.segmentation
    if mask.frame != camera.frame or mask.camera_name not in (camera_name, camera.frame):
        raise ValueError("target mask does not match the selected camera")
    if mask.mask.shape != camera.depth_m.shape:
        raise ValueError("target mask shape does not match the selected camera")
    if info.get("target_mask_sha256") != hashlib.sha256(mask.mask.tobytes()).hexdigest():
        raise ValueError("target mask differs from the source localization")
    if any(
        view.timestamp_s is None or abs(view.timestamp_s - state.timestamp_s) > max_skew
        for view in observation.cameras.values()
    ):
        raise ValueError("camera and state timestamps exceed the allowed skew")
    if any(view.camera_pose.frame != state.base_frame for view in observation.cameras.values()):
        raise ValueError("all camera poses must be expressed in the robot base frame")

    target = mask_to_point_cloud(camera, mask, target_frame=state.base_frame)
    if localization.point_cloud.frame != target.frame or not np.array_equal(
        localization.point_cloud.points, target.points
    ):
        raise ValueError("localized point cloud does not match the observed target mask")

    clouds = []
    for name, view in observation.cameras.items():
        if name == camera_name:
            selection = replace(mask, mask=~mask.mask)
        else:
            selection = Segmentation(
                mask=np.ones(view.depth_m.shape, dtype=bool),
                label="unsegmented scene", score=1.0, camera_name=name, frame=view.frame,
            )
        clouds.append(mask_to_point_cloud(view, selection, target_frame=state.base_frame).points)
    points = np.concatenate(clouds, axis=0)
    scene = PlanningScene(
        point_cloud=PointCloud(points=points, frame=state.base_frame),
        voxel_size_m=voxel_size,
        timestamp_s=min(view.timestamp_s for view in observation.cameras.values()),
    )
    # Bind the entire scene, not just the selected target camera. Other views
    # may preserve additional obstacles even when they share timestamps.
    scene_identity = {
        "views": [
            [name, snapshot_fingerprint(view, state)]
            for name, view in sorted(observation.cameras.items())
        ],
        "target_camera": camera_name,
        "target_mask_sha256": hashlib.sha256(mask.mask.tobytes()).hexdigest(),
        "voxel_size_m": voxel_size,
    }
    scene_fingerprint = hashlib.sha256(
        json.dumps(scene_identity, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return TargetScenePartition(
        target, scene, expected_id, fingerprint, camera_name, scene_fingerprint,
        observation, localization, max_skew,
    )


def verify_target_scene_partition(partition: TargetScenePartition, scene: PlanningScene) -> None:
    """Reconstruct the split before it is sent to an external planner."""
    if not isinstance(partition, TargetScenePartition) or not isinstance(scene, PlanningScene):
        raise ValueError("a target scene partition and planning scene are required")
    rebuilt = partition_target_scene(
        partition.source_observation,
        partition.source_localization,
        camera_name=partition.target_camera,
        max_camera_state_skew_s=partition.max_camera_state_skew_s,
        voxel_size_m=partition.scene.voxel_size_m,
    )
    if (
        rebuilt.snapshot_id != partition.snapshot_id
        or rebuilt.snapshot_fingerprint != partition.snapshot_fingerprint
        or rebuilt.scene_fingerprint != partition.scene_fingerprint
        or scene.frame != rebuilt.scene.frame
        or scene.timestamp_s != rebuilt.scene.timestamp_s
        or scene.voxel_size_m != rebuilt.scene.voxel_size_m
        or not np.array_equal(scene.point_cloud.points, rebuilt.scene.point_cloud.points)
        or partition.target.frame != rebuilt.target.frame
        or not np.array_equal(partition.target.points, rebuilt.target.points)
        or partition.scene.frame != rebuilt.scene.frame
        or partition.scene.timestamp_s != rebuilt.scene.timestamp_s
        or not np.array_equal(partition.scene.point_cloud.points, rebuilt.scene.point_cloud.points)
    ):
        raise ValueError("scene partition differs from its source observation and target mask")


__all__ = ["TargetScenePartition", "partition_target_scene", "verify_target_scene_partition"]
