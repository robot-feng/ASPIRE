"""Non-executable, static-snapshot trajectory-chain preview.

The state after the first segment is a planning assumption, not an observation.
In particular, unchanged end-effector poses and velocities are placeholders.
No stage verifies contact, grip, target motion, calibration or occluded space.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np

from cap_harness.contracts import PlanningScene, PointCloud, Pose
from cap_harness.open_rail.scene_partition import verify_target_scene_partition
from cap_harness.open_rail.snapshot_identity import snapshot_fingerprint


def _context_id(context):
    if context is None:
        raise ValueError("preview chain requires a robot planning context")
    value = [context.embodiment, context.model, [
        [arm, list(context.joint_names[arm]), context.end_effector_links[arm],
         np.asarray(context.base_transforms[arm], dtype=np.float64).tolist()]
        for arm in sorted(context.joint_names)
    ]]
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _History:
    snapshot_fingerprint: str
    scene_fingerprint: str
    context_fingerprint: str
    arm: str
    trajectories: tuple


def _endpoint(state, trajectory, camera, arm):
    if trajectory.collision_aware:
        raise ValueError("preview history may contain only non-executable trajectories")
    if (trajectory.arm != arm or trajectory.embodiment != state.embodiment
            or tuple(trajectory.joint_names) != tuple(state.joint_names[arm])
            or snapshot_fingerprint(camera, trajectory.expected_start) != snapshot_fingerprint(camera, state)):
        raise ValueError("trajectory does not start at the preceding planning state")
    if not np.allclose(trajectory.joint_positions[0], state.joint_positions[arm], rtol=0, atol=1e-7):
        raise ValueError("first waypoint does not match planning start")
    if trajectory.gripper_positions is None or not np.allclose(
        trajectory.gripper_positions, state.gripper_positions[arm], rtol=0, atol=1e-7
    ):
        raise ValueError("preview does not model a gripper action")
    positions = dict(state.joint_positions)
    positions[arm] = trajectory.joint_positions[-1]
    return replace(state, joint_positions=positions)


def verify_preview_history(history, partition, state, context, arm):
    """Check source and every segment; do not substitute a fabricated observation."""
    if not isinstance(history, _History) or not history.trajectories:
        raise ValueError("preview history must contain a preceding trajectory")
    if (history.snapshot_fingerprint != partition.snapshot_fingerprint
            or history.scene_fingerprint != partition.scene_fingerprint
            or history.context_fingerprint != _context_id(context) or history.arm != arm):
        raise ValueError("preview history source, context or arm differs")
    source = partition.source_observation.robot_state
    camera = partition.source_observation.cameras[partition.target_camera]
    if snapshot_fingerprint(camera, source) != partition.snapshot_fingerprint:
        raise ValueError("preview source observation differs")
    predicted = source
    for trajectory in history.trajectories:
        predicted = _endpoint(predicted, trajectory, camera, arm)
    if snapshot_fingerprint(camera, predicted) != snapshot_fingerprint(camera, state):
        raise ValueError("planning start is not the previous trajectory endpoint")


@dataclass(frozen=True, slots=True)
class PreviewStage:
    """Inspection data, deliberately not a CAP Trajectory or PlanResult."""
    name: str
    ok: bool
    joint_names: tuple
    start_joints: tuple
    waypoints: tuple
    diagnostics: dict
    error: object = None


@dataclass(frozen=True, slots=True)
class PreviewChain:
    """Partial or complete preview; all trajectories are non-executable."""
    stages: tuple
    completed: bool
    assumptions: tuple = (
        "static source scene; target motion and occlusion unverified",
        "hypothetical joint states are not observations; poses and velocities are placeholders",
        "gripper actuation and target contact are not simulated",
        "lift attachment is caller-supplied and grasp success is assumed",
        "robot model, camera calibration and collision safety are unverified",
        "pregrasp scene uses the legacy planner path which may carve robot spheres",
    )


def preview_pick_chain(provider, partition, context, *, arm, pregrasp, approach, lift, attached_object):
    """Plan three poses with a shared snapshot and explicit assumed attachment.

    This offline-only helper never calls a robot. The first segment sees the
    observed target; later v2 scene partitions omit it and cannot establish
    target contact or collision safety. The first segment's legacy scene path
    may carve robot spheres. Any failed stage stops the chain.
    """
    verify_target_scene_partition(partition, partition.scene)
    if attached_object is None:
        raise ValueError("lift preview requires explicit assumed attachment geometry")
    source = partition.source_observation.robot_state
    if arm not in source.joint_positions or arm not in context.joint_names:
        raise ValueError("preview arm is missing from source or context")
    if any(not isinstance(pose, Pose) or pose.frame != source.base_frame
           for pose in (pregrasp, approach, lift)):
        raise ValueError("each preview target must be a pose in the source base frame")
    context_id = _context_id(context)
    points = np.concatenate((partition.scene.point_cloud.points, partition.target.points))
    full_scene = PlanningScene(PointCloud(points, partition.scene.frame),
                               voxel_size_m=partition.scene.voxel_size_m,
                               timestamp_s=partition.scene.timestamp_s)
    camera = partition.source_observation.cameras[partition.target_camera]
    state = source
    stages = []
    history = []
    for name, target in (("pregrasp", pregrasp), ("approach", approach), ("lift", lift)):
        if name == "pregrasp":
            result = provider.plan_to_pose(state, target, arm=arm, scene=full_scene, context=context)
        else:
            proof = _History(partition.snapshot_fingerprint, partition.scene_fingerprint,
                             context_id, arm, tuple(history))
            result = provider._plan(
                state, target, arm=arm, gripper_position=None, scene=partition.scene,
                context=context, time_dilation_factor=None, interpolation_dt_s=None,
                maximum_trajectory_dt_s=None,
                attached_object=attached_object if name == "lift" else None,
                target_scene_partition=partition, preserve_observed_scene=True,
                preview_history=proof,
            )
        if not result.ok:
            stages.append(PreviewStage(name, False, tuple(state.joint_names[arm]),
                                       tuple(state.joint_positions[arm]), (),
                                       dict(result.diagnostics), result.error))
            return PreviewChain(tuple(stages), False)
        original_flag = result.trajectory.collision_aware
        trajectory = replace(result.trajectory, collision_aware=False)
        next_state = _endpoint(state, trajectory, camera, arm)
        diagnostics = dict(result.diagnostics)
        diagnostics.update({"preview_only": True, "planner_collision_aware_before_preview": original_flag,
                            "collision_aware": False, "target_reobserved": False,
                            "target_interaction_verified": False, "occluded_space_verified": False,
                            "planning_start_is_hypothetical": name != "pregrasp",
                            "attachment_and_grasp_assumed": name == "lift",
                            "source_snapshot_fingerprint": partition.snapshot_fingerprint,
                            "source_scene_fingerprint": partition.scene_fingerprint,
                            "planning_context_fingerprint": context_id})
        stages.append(PreviewStage(name, True, tuple(trajectory.joint_names),
                                   tuple(state.joint_positions[arm]),
                                   tuple(tuple(row) for row in trajectory.joint_positions), diagnostics))
        history.append(trajectory)
        state = next_state
    return PreviewChain(tuple(stages), True)
