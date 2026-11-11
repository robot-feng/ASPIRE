from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from cap_harness.contracts import (
    CameraObservation,
    LocalizationResult,
    Observation,
    Pose,
    RobotState,
    Segmentation,
)
from cap_harness.geometry import estimate_geometry, mask_to_point_cloud
from cap_harness.open_rail.verification import HoldVerifier
from cap_harness.open_rail.hold_replay import HoldReplaySample, replay_hold
from cap_harness.open_rail.snapshot_identity import snapshot_fingerprint


def _pose(x: float, z: float) -> Pose:
    return Pose(
        position=np.array([x, 0.0, z]),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="base",
    )


def _sample(timestamp: float, object_x: float, object_z: float, tool_z: float):
    camera = CameraObservation(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.array([[1.0, 1.01], [1.02, 1.04]]),
        intrinsics=np.diag([100.0, 100.0, 1.0]),
        frame="head_optical",
        camera_pose=_pose(object_x, object_z - 1.0),
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
    mask = Segmentation(np.ones((2, 2), dtype=bool), "apple", 1.0, "head", camera.frame)
    cloud = mask_to_point_cloud(camera, mask, target_frame="base")
    localization = LocalizationResult(
        ok=True,
        geometry=estimate_geometry(cloud), segmentation=mask, point_cloud=cloud,
        diagnostics={
            "snapshot_id": f"head_optical@{timestamp:.9f}",
            "robot_state_timestamp_s": timestamp,
            "camera_name": "head",
            "segmentation_source": "provider",
            "instance_count": 1,
            "snapshot_fingerprint": snapshot_fingerprint(camera, state),
            "target_mask_sha256": hashlib.sha256(mask.mask.tobytes()).hexdigest(),
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


def test_hold_rejects_changed_camera_content_with_unchanged_timestamp() -> None:
    verifier = _verifier()
    observation, localization = _sample(2.0, 0.0, 0.15, 0.20)
    camera = replace(observation.cameras["head"], depth_m=np.full((2, 2), 1.2))
    changed = replace(observation, cameras={"head": camera})

    evidence = verifier.observe(changed, localization)

    assert evidence.status == "unknown"
    assert "content" in evidence.reason


@pytest.mark.parametrize("defect", ["missing_mask", "mask", "cloud", "geometry"])
def test_hold_rejects_target_evidence_not_derived_from_the_observation(defect) -> None:
    verifier = _verifier()
    observation, localization = _sample(2.0, 0.0, 0.15, 0.20)
    if defect == "missing_mask":
        localization = replace(localization, segmentation=None)
    elif defect == "mask":
        altered = np.array(localization.segmentation.mask)
        altered[0, 0] = False
        localization = replace(localization, segmentation=replace(localization.segmentation, mask=altered))
    elif defect == "cloud":
        localization = replace(localization, point_cloud=replace(
            localization.point_cloud, points=localization.point_cloud.points + [0, 0, 0.1]
        ))
    else:
        geometry = localization.geometry
        pose = replace(geometry.pose, position=geometry.pose.position + [0, 0, 0.1])
        localization = replace(localization, geometry=replace(geometry, pose=pose))
    evidence = verifier.observe(observation, localization)
    assert evidence.status == "unknown"


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
    rotated_camera = replace(
        observation.cameras["head"],
        camera_pose=Pose([0.0, 0.03, -0.85], [2**-0.5, 0.0, 0.0, 2**-0.5], "base"),
    )
    rotated_cloud = mask_to_point_cloud(rotated_camera, localized.segmentation, target_frame="base")

    evidence = verifier.observe(
        replace(observation, cameras={"head": rotated_camera}, robot_state=rotated_state),
        replace(
            localized,
            geometry=estimate_geometry(rotated_cloud),
            point_cloud=rotated_cloud,
            diagnostics={
                **localized.diagnostics,
                "snapshot_fingerprint": snapshot_fingerprint(rotated_camera, rotated_state),
            },
        ),
    )

    assert evidence.status == "pending"
    assert evidence.relative_error_m is not None and evidence.relative_error_m < 1e-10


def _replay(samples, *, expected_samples=None, capture_clock_verified=True):
    records = [
        item if isinstance(item, HoldReplaySample) else HoldReplaySample(index, *item)
        for index, item in enumerate(samples)
    ]
    return replay_hold(
        *_sample(1.0, 0.0, 0.05, 0.10),
        records,
        expected_samples=len(records) if expected_samples is None else expected_samples,
        capture_clock_verified=capture_clock_verified,
        camera_name="head",
        arm="right",
        up_in_base=(0.0, 0.0, 1.0),
        minimum_lift_m=0.05,
        max_relative_error_m=0.02,
        max_sample_gap_s=1.1,
        max_state_skew_s=0.05,
    )


def test_replay_verifies_only_the_past_window() -> None:
    result = _replay([_sample(t, 0.0, 0.15, 0.20) for t in (2.0, 3.0, 4.0, 5.0)])

    assert result.status == "verified_history"
    assert result.evidence[-1].stable_duration_s == 3.0
    assert result.current_hold_verified is False
    assert "current hold requires" in result.reason


def test_replay_requires_verified_capture_clock() -> None:
    samples = [_sample(t, 0.0, 0.15, 0.20) for t in (2.0, 3.0, 4.0, 5.0)]
    with pytest.raises(ValueError, match="clock domain"):
        _replay(samples, capture_clock_verified=False)


def test_replay_does_not_skip_missing_localization_or_gaps() -> None:
    samples = [_sample(t, 0.0, 0.15, 0.20) for t in (2.0, 3.0, 4.0, 5.0, 6.0)]
    # Even an absent detection after a verified 3-second window invalidates the recording.
    samples[-1] = samples[-1][0], None
    missing = _replay(samples)
    assert missing.status == "unknown"
    assert "localization" in missing.reason
    assert missing.evidence[-2].status == "verified"
    assert missing.current_hold_verified is False

    gap = _replay([_sample(t, 0.0, 0.15, 0.20) for t in (2.0, 3.0, 5.0, 6.0, 7.0, 8.0)])
    assert gap.status == "unknown" and "gap" in gap.reason

    missing_last = _replay([_sample(t, 0.0, 0.15, 0.20) for t in (2.0, 3.0)], expected_samples=3)
    assert missing_last.status == "unknown" and "missing samples" in missing_last.reason
    skipped_index = _replay([
        HoldReplaySample(0, *_sample(2.0, 0.0, 0.15, 0.20)),
        HoldReplaySample(2, *_sample(3.0, 0.0, 0.15, 0.20)),
    ])
    assert skipped_index.status == "unknown" and "sequence" in skipped_index.reason


def test_replay_rejects_duplicate_reversed_and_stale_frames() -> None:
    for times in ((2.0, 3.0, 3.0, 4.0, 5.0), (2.0, 3.0, 2.5, 4.0, 5.0)):
        result = _replay([_sample(t, 0.0, 0.15, 0.20) for t in times])
        assert result.status == "unknown"
        assert "strictly increasing" in result.reason

    observation, _ = _sample(3.0, 0.0, 0.15, 0.20)
    _, stale_localization = _sample(2.0, 0.0, 0.15, 0.20)
    result = _replay([_sample(2.0, 0.0, 0.15, 0.20), (observation, stale_localization)])
    assert result.status == "unknown" and "snapshot" in result.reason


def test_replay_rejects_slip_after_an_apparent_success() -> None:
    samples = [_sample(t, 0.0, 0.15, 0.20) for t in (2.0, 3.0, 4.0, 5.0)]
    samples.append(_sample(6.0, 0.05, 0.15, 0.20))
    result = _replay(samples)

    assert result.evidence[-2].status == "verified"
    assert result.status == "not_verified"
    assert result.evidence[-1].status == "not_following"
    assert result.current_hold_verified is False
