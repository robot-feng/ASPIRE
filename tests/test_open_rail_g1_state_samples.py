from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from cap_harness.open_rail.g1_state_samples import G1StateSamples


ARMS = tuple([f"Joint{i}_l" for i in range(1, 8)] + [f"Joint{i}_r" for i in range(1, 8)])


def _message(timestamp: int, **kwargs):
    return NS(header=NS(stamp=NS(sec=timestamp // 1_000_000_000, nanosec=timestamp % 1_000_000_000)), **kwargs)


def _motor(position):
    return NS(position=position)


def test_direct_messages_keep_values_and_timestamps_from_same_sample() -> None:
    samples = G1StateSamples(ARMS)
    arm = _message(1_001_000_000, name=list(ARMS), position=list(range(14)), velocity=[0.1] * 14)
    samples.arm(arm)
    samples.head(_message(1_002_000_000, name=["joint_head_yaw", "joint_head_pitch"], motor_states=[_motor(0.2), _motor(0.3)]))
    samples.waist(_message(1_003_000_000, name=["joint_lift_body", "joint_body_pitch"], motor_states=[_motor(0.5), _motor(0.4)]))
    samples.gripper("left", _message(1_004_000_000, type=2, end_state=[_motor(30)]))
    samples.gripper("right", _message(1_005_000_000, type=2, end_state=[_motor(90)]))
    # Changing or reusing a DDS message must not mutate buffered values.
    arm.position[0] = 123.0
    arm.velocity[0] = 123.0
    state, velocity, timestamps = samples.nearest(1_000_000_000)
    np.testing.assert_allclose(state, [*range(14), 0.25, 0.75, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(velocity, [0.1] * 14)
    assert timestamps == {"arm": 1_001_000_000, "head": 1_002_000_000, "waist": 1_003_000_000,
                          "gripper_left": 1_004_000_000, "gripper_right": 1_005_000_000}


def test_nearest_uses_each_group_and_keeps_missing_velocity_missing() -> None:
    samples = G1StateSamples(ARMS)
    assert samples.nearest(1_000_000_000) is None
    samples.arm(_message(1_000_000_000, name=ARMS, position=[1] * 14, velocity=[0.1] * 14))
    samples.arm(_message(2_000_000_000, name=ARMS, position=[2] * 14, velocity=[]))
    samples.head(_message(1_000_000_000, name=["joint_head_yaw", "joint_head_pitch"], motor_states=[_motor(0), _motor(0)]))
    samples.waist(_message(1_000_000_000, name=["joint_body_pitch", "joint_lift_body"], motor_states=[_motor(0), _motor(0)]))
    for side in ("left", "right"):
        samples.gripper(side, _message(1_000_000_000, type=2, end_state=[_motor(0)]))
    state, velocity, timestamps = samples.nearest(2_000_000_000)
    assert state[0] == 2 and velocity is None
    assert timestamps["arm"] == 2_000_000_000 and timestamps["head"] == 1_000_000_000


def test_unexpected_joint_names_and_non_gripper_feedback_are_rejected() -> None:
    samples = G1StateSamples(ARMS)
    with pytest.raises(ValueError, match="arm message"):
        samples.arm(_message(1, name=ARMS[::-1], position=[0] * 14, velocity=[0] * 14))
    with pytest.raises(ValueError, match="gripper endpoint"):
        samples.gripper("right", _message(1, type=1, end_state=[_motor(1)]))
    with pytest.raises(ValueError, match="invalid DDS"):
        samples.arm(_message(0, name=ARMS, position=[0] * 14, velocity=[0] * 14))


def test_side_feedback_retains_ids_and_raw_values_without_assuming_mapping() -> None:
    samples = G1StateSamples(ARMS)
    motors = [NS(id=100 + i, position=i * 0.1, velocity=i * 0.01) for i in range(7)]
    samples.arm_side("right", _message(1_000_000_000, motor_states=motors))
    motors[0].position = 99
    side = samples.nearest_sides(1_000_000_001)
    assert set(side) == {"right"}
    assert side["right"]["motor_ids"] == list(range(100, 107))
    assert side["right"]["positions_raw"][0] == 0
    assert side["right"]["velocities_raw"][1] == pytest.approx(0.01)
    assert side["right"]["timestamp_ns"] == 1_000_000_000
