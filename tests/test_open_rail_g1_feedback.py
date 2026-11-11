from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cap_harness.open_rail.g1_feedback import G1FeedbackCalibration, G1MeasuredCommandFeedback
from cap_harness.open_rail.g1_state_samples import G1StateSamples
from cap_harness.open_rail.g1_control import G1CommandCapabilities, OpenRailG1CommandRuntime
from cap_harness.errors import ErrorCode


def _calibration(**changes):
    values = dict(
        right_motor_ids=(3, 4, 5, 6, 0, 1, 2), right_motor_order_verified=True,
        arm_position_velocity_units_verified=True, gripper_feedback_convention_verified=True,
        gripper_measurement_timestamp_verified=True,
        feedback_clock_verified=True, max_sample_age_s=1.0, max_sample_gap_s=0.05,
        stable_duration_s=0.1, minimum_samples=3,
    )
    values.update(changes)
    return G1FeedbackCalibration(**values)


def test_gripper_feedback_rejects_clock_only_without_verified_measurement_timestamps():
    # A HAL publisher can repeatedly stamp the same cached gripper position
    # with a fresh message timestamp; a synchronized clock is not enough.
    with pytest.raises(ValueError, match="gripper_measurement_timestamp_verified"):
        _calibration(gripper_measurement_timestamp_verified=False)


class _ScriptedSamples(G1StateSamples):
    def __init__(self, entries):
        super().__init__(tuple(f"joint{i}" for i in range(14)))
        self.entries = list(entries)
        self.after = []

    def _next(self, after, timeout):
        self.after.append(after)
        return self.entries.pop(0) if self.entries else None

    wait_next_right_arm = _next
    wait_next_right_gripper = _next


def _monitor(samples, calibration=None):
    # First call is the command barrier; later calls are after the samples.
    times = iter([1_000_000_000] + [1_200_000_000] * 20)
    return G1MeasuredCommandFeedback(samples, calibration or _calibration(), clock_ns=lambda: next(times))


def _arm(stamp, *, velocity=0.0, positions=None, ids=None):
    # The raw ID order differs from the verified command order.
    ids = (0, 1, 2, 3, 4, 5, 6) if ids is None else ids
    positions = (4, 5, 6, 0, 1, 2, 3) if positions is None else positions
    return (stamp, ids, positions, (velocity,) * 7)


def test_waiter_requires_post_command_stable_measured_arm_with_verified_mapping():
    samples = _ScriptedSamples([_arm(t) for t in (1_050_000_000, 1_100_000_000, 1_150_000_000)])
    feedback = _monitor(samples)
    assert feedback.wait_for_right_arm(
        np.arange(7), timeout_s=1, position_tolerance_rad=0.001, velocity_tolerance_rad_s=0.01,
    )
    assert samples.after == [1_000_000_000, 1_050_000_000, 1_100_000_000]
    assert feedback.last_evidence["terminal_timestamp_ns"] == 1_150_000_000
    assert feedback.last_evidence["stable_samples"] == 3


@pytest.mark.parametrize("defect", ["moving", "gap", "wrong_target", "too_few"])
def test_unstable_motion_never_confirms_terminal_completion(defect):
    entries = [_arm(t) for t in (1_050_000_000, 1_100_000_000, 1_150_000_000)]
    if defect == "moving":
        entries[1] = _arm(1_100_000_000, velocity=0.1)
    elif defect == "gap":
        entries[1] = _arm(1_120_000_000)
    elif defect == "wrong_target":
        entries[1] = _arm(1_100_000_000, positions=(0,) * 7)
    else:
        entries.pop()
    assert not _monitor(_ScriptedSamples(entries)).wait_for_right_arm(
        np.arange(7), timeout_s=1, position_tolerance_rad=0.001, velocity_tolerance_rad_s=0.01,
    )


def test_old_future_and_wrong_id_feedback_cannot_confirm():
    with pytest.raises(ValueError, match="strictly increasing"):
        _monitor(_ScriptedSamples([_arm(1_000_000_000)])).wait_for_right_arm(
            np.arange(7), timeout_s=1, position_tolerance_rad=0.1, velocity_tolerance_rad_s=0.1,
        )
    with pytest.raises(ValueError, match="future"):
        _monitor(_ScriptedSamples([_arm(1_300_000_000)])).wait_for_right_arm(
            np.arange(7), timeout_s=1, position_tolerance_rad=0.1, velocity_tolerance_rad_s=0.1,
        )
    with pytest.raises(ValueError, match="IDs"):
        _monitor(_ScriptedSamples([_arm(1_050_000_000, ids=(0, 1, 2, 3, 4, 5, 5))])).wait_for_right_arm(
            np.arange(7), timeout_s=1, position_tolerance_rad=0.1, velocity_tolerance_rad_s=0.1,
        )


def test_stale_samples_and_clock_regression_do_not_confirm():
    entries = [_arm(t) for t in (1_050_000_000, 1_100_000_000, 1_150_000_000)]
    assert not _monitor(_ScriptedSamples(entries), _calibration(max_sample_age_s=0.01)).wait_for_right_arm(
        np.arange(7), timeout_s=1, position_tolerance_rad=0.1, velocity_tolerance_rad_s=0.1,
    )
    times = iter([1_000_000_000, 1_100_000_000, 1_090_000_000])
    monitor = G1MeasuredCommandFeedback(
        _ScriptedSamples(entries[:2]), _calibration(), clock_ns=lambda: next(times)
    )
    with pytest.raises(ValueError, match="clock regressed"):
        monitor.wait_for_right_arm(
            np.arange(7), timeout_s=1, position_tolerance_rad=0.1, velocity_tolerance_rad_s=0.1,
        )


def test_gripper_feedback_requires_stable_native_target_and_does_not_infer_grasp():
    entries = [(t, (0.7,), None) for t in (1_050_000_000, 1_100_000_000, 1_150_000_000)]
    assert _monitor(_ScriptedSamples(entries)).wait_for_right_gripper(0.7, timeout_s=1, tolerance=0.01)
    assert not _monitor(_ScriptedSamples(entries)).wait_for_right_gripper(1.0, timeout_s=1, tolerance=0.01)


@pytest.mark.parametrize("field", ["right_motor_order_verified", "arm_position_velocity_units_verified", "gripper_feedback_convention_verified", "feedback_clock_verified"])
def test_feedback_cannot_be_constructed_from_unverified_calibration(field):
    with pytest.raises(ValueError, match="verified"):
        replace(_calibration(), **{field: False})


def test_buffer_returns_only_newer_samples_without_duplicate_counts():
    buffer = G1StateSamples(tuple(f"joint{i}" for i in range(14)))
    message = NS(header=NS(stamp=NS(sec=1, nanosec=0)), type=2,
                 end_state=[NS(position=60.0)])
    buffer.gripper("right", message)
    assert buffer.wait_next_right_gripper(999_999_999, 0)[1] == (0.5,)
    assert buffer.wait_next_right_gripper(1_000_000_000, 0) is None


def test_command_runtime_requires_measured_feedback_and_latches_invalid_samples():
    class Transport:
        def __init__(self):
            self.calls = []

        def set_joint_position_control_bounded(self, lifetime_s, groups, *, timeout_s):
            self.calls.append((lifetime_s, groups))

        def publish_right_gripper_bounded(self, position, *, timeout_s):
            raise AssertionError("uncertain session must reject later commands")

    transport = Transport()
    samples = _ScriptedSamples([_arm(1_050_000_000, ids=(0, 1, 2, 3, 4, 5, 5))])
    runtime = OpenRailG1CommandRuntime(
        transport,
        feedback=_monitor(samples),
        capabilities=G1CommandCapabilities(
            arm_command_protocol_verified=True, arm_position_limits_verified=True,
            arm_terminal_feedback_verified=True, arm_cancel_or_lease_expiry_verified=True,
        ),
        allow_physical_motion=True,
        arm_joint_limits_rad=[(-10, 10)] * 7,
        position_tolerance_rad=0.01, velocity_tolerance_rad_s=0.01,
    )
    result = runtime.set_arm("right", np.arange(7), lifetime_s=0.1, timeout_s=1)
    assert not result.ok and result.error.code is ErrorCode.CONTROLLER_FAILED
    assert runtime.health["motion_session"]["uncertain"] is True
    assert runtime.set_gripper("right", 1.0, timeout_s=1).error.code is ErrorCode.SAFETY_INTERLOCK
    assert len(transport.calls) == 1


def test_command_runtime_returns_terminal_measurement_evidence():
    class Transport:
        def set_joint_position_control_bounded(self, lifetime_s, groups, *, timeout_s):
            assert set(groups) == {"right_arm"}

    samples = _ScriptedSamples([_arm(t) for t in (1_050_000_000, 1_100_000_000, 1_150_000_000)])
    runtime = OpenRailG1CommandRuntime(
        Transport(), feedback=_monitor(samples), allow_physical_motion=True,
        capabilities=G1CommandCapabilities(
            arm_command_protocol_verified=True, arm_position_limits_verified=True,
            arm_terminal_feedback_verified=True, arm_cancel_or_lease_expiry_verified=True,
        ),
        arm_joint_limits_rad=[(-10, 10)] * 7,
        position_tolerance_rad=0.01, velocity_tolerance_rad_s=0.01,
    )
    result = runtime.set_arm("right", np.arange(7), lifetime_s=0.1, timeout_s=1)
    assert result.ok
    evidence = result.diagnostics["measured_feedback"]
    assert evidence["command_barrier_ns"] == 1_000_000_000
    assert evidence["terminal_timestamp_ns"] == 1_150_000_000
    assert evidence["terminal_sample"][1] == tuple(range(7))
