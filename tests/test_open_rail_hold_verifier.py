from __future__ import annotations

from dataclasses import replace

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    LocalizationResult,
    ObjectGeometry,
    Observation,
    Pose,
    RobotState,
)
from cap_harness.open_rail.verification import HoldVerifier


def _pose(x: float, z: float) -> Pose:
    return Pose(
        position=np.array([x, 0.0, z]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="base",
    )


def _sample(timestamp: float, object_x: float, object_z: float, tool_z: float):
    camera = CameraObservation(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2)),
        intrinsics=np.eye(3),
        frame="head_optical",
        camera_pose=_pose(0.0, 0.0),
        timestamp_s=timestamp,
    )
    state = RobotState(
        joint_positions={"right": np.zeros(7)},
        joint_velocities={"right": np.zeros(7)},
        end_effector_poses={"right": _pose(0.0, tool_z)},
        gripper_positions={"right": 0.5},
        base_frame="base",
        timestamp_s=timestamp,
        embodiment="open_rail_7dof",
    )
    observation = Observation(cameras={"head": camera}, robot_state=state, timestamp_s=timestamp)
    localization = LocalizationResult(
        ok=True,
        geometry=ObjectGeometry(pose=_pose(object_x, object_z), extents=np.full(3, 0.08)),
        diagnostics={
            "snapshot_id": f"head_optical@{timestamp:.9f}",
            "robot_state_timestamp_s": timestamp,
            "camera_name": "head",
            "segmentation_source": "provider",
            "instance_count": 1,
        },
    )
    return observation, localization


def _verifier() -> HoldVerifier:
    observation, localization = _sample(1.0, 0.0, 0.05, 0.10)
    return HoldVerifier(
        observation,
        localization,
        camera_name="head",
        arm="right",
        up_in_base=(0.0, 0.0, 1.0),
        minimum_lift_m=0.05,
        max_relative_error_m=0.02,
        max_sample_gap_s=1.1,
        max_state_skew_s=0.05,
    )


def test_hold_requires_lift_and_three_seconds_of_fresh_following_samples() -> None:
    verifier = _verifier()
    results = [
        verifier.observe(*_sample(t, 0.0, 0.15, 0.20)) for t in (2.0, 3.0, 4.0, 5.0)
    ]

    assert [result.status for result in results] == ["pending", "pending", "pending", "verified"]
    assert results[-1].stable_duration_s == 3.0
    assert results[-1].stable_samples == 4


def test_stationary_or_slipping_object_does_not_count_as_held() -> None:
    verifier = _verifier()
    stationary = verifier.observe(*_sample(2.0, 0.0, 0.05, 0.20))
    slipping = verifier.observe(*_sample(3.0, 0.05, 0.15, 0.20))

    assert stationary.status == "not_lifted"
    assert slipping.status == "not_following"


def test_occlusion_and_gaps_reset_hold_window() -> None:
    verifier = _verifier()
    assert verifier.observe(*_sample(2.0, 0.0, 0.15, 0.20)).status == "pending"
    observation, _ = _sample(3.0, 0.0, 0.15, 0.20)
    assert verifier.observe(observation, None).status == "unknown"
    assert verifier.observe(*_sample(4.0, 0.0, 0.15, 0.20)).stable_samples == 1
    assert verifier.observe(*_sample(6.0, 0.0, 0.15, 0.20)).stable_samples == 1
    assert verifier.observe(*_sample(7.0, 0.0, 0.15, 0.20)).status == "pending"


def test_localization_from_an_older_image_is_unknown() -> None:
    verifier = _verifier()
    observation, _ = _sample(3.0, 0.0, 0.15, 0.20)
    _, old_localization = _sample(2.0, 0.0, 0.15, 0.20)

    evidence = verifier.observe(observation, old_localization)

    assert evidence.status == "unknown"
    assert "snapshot" in evidence.reason


def test_occluded_new_frame_prevents_reuse_of_an_older_frame() -> None:
    verifier = _verifier()
    newer, _ = _sample(5.0, 0.0, 0.15, 0.20)
    assert verifier.observe(newer, None).status == "unknown"

    evidence = verifier.observe(*_sample(4.0, 0.0, 0.15, 0.20))

    assert evidence.status == "unknown"
    assert "not newer" in evidence.reason


def test_unsynchronized_robot_state_is_unknown() -> None:
    verifier = _verifier()
    observation, localization = _sample(2.0, 0.0, 0.15, 0.20)
    stale_state = replace(observation.robot_state, timestamp_s=1.7)

    evidence = verifier.observe(replace(observation, robot_state=stale_state), localization)

    assert evidence.status == "unknown"
    assert "not synchronized" in evidence.reason


def test_multiple_instances_are_unknown() -> None:
    verifier = _verifier()
    observation, localization = _sample(2.0, 0.0, 0.15, 0.20)
    ambiguous = replace(localization, diagnostics={**localization.diagnostics, "instance_count": 2})

    evidence = verifier.observe(observation, ambiguous)

    assert evidence.status == "unknown"
    assert "ambiguous" in evidence.reason


def test_relative_position_uses_the_rotating_tool_frame() -> None:
    baseline, localization = _sample(1.0, 0.03, 0.05, 0.10)
    verifier = HoldVerifier(
        baseline,
        localization,
        camera_name="head",
        arm="right",
        up_in_base=(0.0, 0.0, 1.0),
        minimum_lift_m=0.05,
        max_relative_error_m=0.01,
        max_sample_gap_s=1.1,
        max_state_skew_s=0.05,
    )
    observation, localized = _sample(2.0, 0.0, 0.15, 0.20)
    rotated_tool = Pose(
        position=np.array([0.0, 0.0, 0.20]),
        quaternion_wxyz=np.array([2**-0.5, 0.0, 0.0, 2**-0.5]),
        frame="base",
    )
    rotated_state = replace(observation.robot_state, end_effector_poses={"right": rotated_tool})
    rotated_object = ObjectGeometry(
        pose=Pose(
            position=np.array([0.0, 0.03, 0.15]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="base",
        ),
        extents=np.full(3, 0.08),
    )

    evidence = verifier.observe(
        replace(observation, robot_state=rotated_state),
        replace(localized, geometry=rotated_object),
    )

    assert evidence.status == "pending"
    assert evidence.relative_error_m is not None and evidence.relative_error_m < 1e-10
