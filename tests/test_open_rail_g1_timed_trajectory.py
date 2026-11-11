"""Offline timing and interlock tests; no device or SDK transport is opened."""

from dataclasses import replace
import threading
import numpy as np
import pytest

from cap_harness.contracts import Pose, RobotState, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.open_rail import g1_control
from cap_harness.open_rail.g1_control import (
    G1CommandCapabilities,
    G1TimedTrajectoryPolicy,
    G1_RIGHT_ARM_JOINTS,
    OpenRailG1CommandRuntime,
)
from cap_harness.open_rail.g1_motion_monitor import G1MotionMonitor, G1MotionMonitorEvidence


class Clock:
    def __init__(self):
        self.monotonic_s = 10.0

    def monotonic(self):
        return self.monotonic_s

    def time(self):
        return 1000.0

    def sleep(self, duration):
        assert duration >= 0
        self.monotonic_s += duration


class Transport:
    def __init__(self, clock, delay=0.0):
        self.clock = clock
        self.delay = delay
        self.calls = []

    def set_joint_position_control(self, lifetime_s, groups):
        self.calls.append((self.clock.monotonic(), lifetime_s, dict(groups)))
        self.clock.monotonic_s += self.delay

    def set_joint_position_control_bounded(self, lifetime_s, groups, *, timeout_s):
        assert timeout_s > 0
        self.set_joint_position_control(lifetime_s, groups)

    def publish_right_gripper(self, position):
        pytest.fail("arm-only trajectory must not send a gripper command")

    def publish_right_gripper_bounded(self, position, *, timeout_s):
        self.publish_right_gripper(position)


class Feedback:
    def __init__(self, reached=True):
        self.reached = reached
        self.calls = []

    def wait_for_right_arm(self, target, **kwargs):
        self.calls.append((target.copy(), kwargs))
        return self.reached


class Monitor(G1MotionMonitor):
    def __init__(self, fault_at=None, defect=None):
        self.calls = 0
        self.fault_at = fault_at
        self.defect = defect

    def check(self, *, trajectory_token, reference_joint_positions_rad, timeout_s):
        self.calls += 1
        return G1MotionMonitorEvidence(
            trajectory_token="wrong" if self.defect == "token" else trajectory_token,
            state_timestamp_s=900.0 if self.defect == "stale" else 1000.0,
            scene_timestamp_s=1000.0,
            measured_joint_positions_rad=(np.asarray(reference_joint_positions_rad) + 0.2
                                          if self.defect == "tracking" else reference_joint_positions_rad),
            reference_joint_positions_rad=reference_joint_positions_rad,
            state_fingerprint="a" * 64, scene_fingerprint="b" * 64,
            fixed_posture_within_limits=self.calls != self.fault_at,
            scene_clear=self.defect != "scene",
        )


def _state():
    return RobotState(
        joint_positions={"right": np.zeros(7)}, joint_velocities={"right": np.zeros(7)},
        end_effector_poses={"right": Pose([0, 0, 0], [1, 0, 0, 0], "base")},
        gripper_positions={"right": 0.5}, joint_names={"right": G1_RIGHT_ARM_JOINTS},
        base_frame="base", timestamp_s=999.99, embodiment="open_rail_7dof",
    )


def _trajectory():
    return Trajectory(
        joint_positions=np.array([[0.0] * 7, [0.01] * 7, [0.02] * 7]),
        dt_s=0.1, joint_names=G1_RIGHT_ARM_JOINTS, planner="fake",
        collision_aware=True, expected_start=_state(), arm="right",
        gripper_positions=np.full(3, 0.5), embodiment="open_rail_7dof",
    )


def _policy():
    return G1TimedTrajectoryPolicy(
        control_period_s=0.1, command_lifetime_s=0.15, max_dispatch_s=0.01,
        max_lateness_s=0.02, final_feedback_timeout_s=0.5,
        max_start_age_s=0.1, start_clock_verified=True, max_start_error_rad=0.01,
        max_start_gripper_error=0.01, max_joint_velocity_rad_s=[1.0] * 7,
        max_joint_acceleration_rad_s2=[5.0] * 7, dynamic_limits_verified=True,
        max_tracking_error_rad=0.05, monitor_max_age_s=0.1, monitor_timeout_s=0.01,
    )


def _runtime(transport, feedback, *, timed=True, monitor_verified=True):
    return OpenRailG1CommandRuntime(
        transport, feedback=feedback, allow_physical_motion=True,
        capabilities=G1CommandCapabilities(
            arm_command_protocol_verified=True, arm_position_limits_verified=True,
            arm_terminal_feedback_verified=True, arm_cancel_or_lease_expiry_verified=True,
            arm_timed_dispatch_verified=timed,
            arm_live_monitor_verified=monitor_verified,
        ),
        arm_joint_limits_rad=[(-1, 1)] * 7,
        position_tolerance_rad=0.01, velocity_tolerance_rad_s=0.02,
    )


def test_dispatches_each_right_arm_waypoint_on_the_control_grid(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport, feedback = Transport(clock), Feedback()
    result = _runtime(transport, feedback).execute_timed_trajectory(
        _trajectory(), _state(), policy=_policy(), monitor=Monitor()
    )
    assert result.ok and result.steps_executed == 2
    # The measured start is time zero, not a physical waypoint dispatch.
    np.testing.assert_allclose([call[0] for call in transport.calls], [10.1, 10.2])
    np.testing.assert_allclose(transport.calls[0][2]["right_arm"], [0.01] * 7)
    assert all(set(call[2]) == {"right_arm"} for call in transport.calls)
    np.testing.assert_allclose(feedback.calls[0][0], [0.02] * 7)


def test_interrupted_wait_never_dispatches_a_waypoint_early(monkeypatch):
    class InterruptedClock(Clock):
        def __init__(self):
            super().__init__()
            self.interrupted = True

        def sleep(self, duration):
            if self.interrupted:
                self.interrupted = False
                self.monotonic_s += duration / 2
            else:
                self.monotonic_s += duration

    clock = InterruptedClock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport = Transport(clock)
    result = _runtime(transport, Feedback()).execute_timed_trajectory(_trajectory(), _state(), policy=_policy(), monitor=Monitor())
    assert result.ok
    np.testing.assert_allclose([call[0] for call in transport.calls], [10.1, 10.2])


@pytest.mark.parametrize("defect", ["unchecked", "stale", "single_point", "first_waypoint", "velocity", "acceleration", "limits", "gripper", "unverified"])
def test_invalid_or_unverified_trajectory_sends_no_commands(monkeypatch, defect):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport, feedback = Transport(clock), Feedback()
    trajectory, state = _trajectory(), _state()
    if defect == "unchecked":
        trajectory = replace(trajectory, collision_aware=False)
    elif defect == "stale":
        state = replace(state, timestamp_s=900.0)
    elif defect == "single_point":
        trajectory = replace(trajectory, joint_positions=np.array([[0.0] * 7]), gripper_positions=None)
    elif defect == "first_waypoint":
        trajectory = replace(trajectory, joint_positions=np.array([[0.03] * 7, [0.04] * 7]), gripper_positions=None)
    elif defect == "velocity":
        trajectory = replace(trajectory, joint_positions=np.array([[0.0] * 7, [0.2] * 7]), gripper_positions=None)
    elif defect == "limits":
        trajectory = replace(trajectory, joint_positions=np.array([[0.0] * 7, [1.1] * 7]), gripper_positions=None)
    elif defect == "gripper":
        trajectory = replace(trajectory, gripper_positions=np.array([0.5, 0.5, 1.0]))
    policy = replace(_policy(), max_joint_acceleration_rad_s2=[0.1] * 7) if defect == "acceleration" else _policy()
    result = _runtime(transport, feedback, timed=defect != "unverified").execute_timed_trajectory(
        trajectory, state, policy=policy, monitor=Monitor()
    )
    assert not result.ok and not transport.calls and not feedback.calls


def test_slow_dispatch_stops_sending_and_latches_unknown_motion(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport, feedback = Transport(clock, delay=0.02), Feedback()
    runtime = _runtime(transport, feedback)
    result = runtime.execute_timed_trajectory(_trajectory(), _state(), policy=_policy(), monitor=Monitor())
    assert not result.ok and result.error.code is ErrorCode.TIMEOUT
    assert result.steps_executed == 1 and len(transport.calls) == 1
    assert result.diagnostics["cancel_confirmed"] is False
    assert runtime.health["motion_session"]["uncertain"] is True
    retry = runtime.execute_timed_trajectory(_trajectory(), _state(), policy=_policy(), monitor=Monitor())
    assert not retry.ok and len(transport.calls) == 1


def test_unconfirmed_terminal_state_never_claims_success_or_retries(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport, feedback = Transport(clock), Feedback(reached=False)
    runtime = _runtime(transport, feedback)
    result = runtime.execute_timed_trajectory(_trajectory(), _state(), policy=_policy(), monitor=Monitor())
    assert not result.ok and result.error.code is ErrorCode.TIMEOUT
    assert result.steps_executed == 2 and len(transport.calls) == 2
    assert runtime.health["motion_session"]["uncertain"] is True
    assert not runtime.set_gripper("right", 0.1, timeout_s=0.2).ok


def test_first_dispatched_segment_uses_measured_not_planned_start(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport = Transport(clock)
    # The measured pose is still within the configured start tolerance, but
    # the actual first dispatched segment exceeds the configured speed limit.
    state = replace(_state(), joint_positions={"right": np.full(7, -0.009)})
    trajectory = replace(_trajectory(), joint_positions=np.array([[0.0] * 7, [0.09] * 7]),
                         gripper_positions=np.full(2, 0.5))
    policy = replace(_policy(), max_joint_velocity_rad_s=[0.95] * 7,
                     max_joint_acceleration_rad_s2=[20.0] * 7)
    result = _runtime(transport, Feedback()).execute_timed_trajectory(trajectory, state, policy=policy, monitor=Monitor())
    assert not result.ok and result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert not transport.calls


def test_small_start_mismatch_does_not_cause_instantaneous_correction(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport = Transport(clock)
    state = replace(_state(), joint_positions={"right": np.full(7, -0.005)})
    result = _runtime(transport, Feedback()).execute_timed_trajectory(
        _trajectory(), state, policy=_policy(), monitor=Monitor()
    )
    assert result.ok and result.steps_executed == 2
    np.testing.assert_allclose([call[0] for call in transport.calls], [10.1, 10.2])
    np.testing.assert_allclose(transport.calls[0][2]["right_arm"], [0.01] * 7)


def test_missing_monitor_or_stale_or_wrong_trajectory_evidence_prevents_dispatch(monkeypatch):
    monkeypatch.setattr(g1_control, "time", Clock())
    for monitor in (None, Monitor(defect="stale"), Monitor(defect="token"),
                    Monitor(defect="tracking"), Monitor(defect="scene")):
        transport = Transport(g1_control.time)
        result = _runtime(transport, Feedback()).execute_timed_trajectory(
            _trajectory(), _state(), policy=_policy(), monitor=monitor
        )
        assert not result.ok and not transport.calls
    transport = Transport(g1_control.time)
    result = _runtime(transport, Feedback(), monitor_verified=False).execute_timed_trajectory(
        _trajectory(), _state(), policy=_policy(), monitor=Monitor()
    )
    assert not result.ok and not transport.calls


def test_monitor_detecting_change_after_dispatch_stops_future_points_and_latches(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport = Transport(clock)
    runtime = _runtime(transport, Feedback())
    result = runtime.execute_timed_trajectory(
        _trajectory(), _state(), policy=_policy(), monitor=Monitor(fault_at=2)
    )
    assert not result.ok and result.steps_executed == 1 and len(transport.calls) == 1
    assert result.diagnostics["cancel_confirmed"] is False
    assert runtime.health["motion_session"]["uncertain"] is True
    assert not runtime.execute_timed_trajectory(_trajectory(), _state(), policy=_policy(), monitor=Monitor()).ok
    assert len(transport.calls) == 1


def test_monitor_timeout_after_first_dispatch_latches_without_claiming_stop(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport = Transport(clock)
    runtime = _runtime(transport, Feedback())
    release = threading.Event()

    class StalledMonitor(Monitor):
        def check(self, **kwargs):
            if self.calls == 1:
                release.wait(1)
            return super().check(**kwargs)

    try:
        result = runtime.execute_timed_trajectory(
            _trajectory(), _state(), policy=_policy(), monitor=StalledMonitor()
        )
        assert not result.ok and result.error.code is ErrorCode.TIMEOUT
        assert result.steps_executed == 1 and len(transport.calls) == 1
        assert result.diagnostics["cancel_confirmed"] is False
        assert runtime.health["motion_session"]["uncertain"] is True
    finally:
        release.set()


def test_monitor_continues_while_waiting_for_terminal_feedback(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport = Transport(clock)
    release = threading.Event()

    class BlockingFeedback(Feedback):
        def wait_for_right_arm(self, *_args, **_kwargs):
            release.wait(1)
            return True

    trajectory = replace(_trajectory(), joint_positions=np.array([[0.0] * 7, [0.01] * 7]),
                         gripper_positions=np.full(2, 0.5))
    runtime = _runtime(transport, BlockingFeedback())
    monitor = Monitor(fault_at=3)
    try:
        result = runtime.execute_timed_trajectory(trajectory, _state(), policy=_policy(), monitor=monitor)
        assert not result.ok and monitor.calls >= 3 and len(transport.calls) == 1
        assert runtime.health["motion_session"]["uncertain"] is True
        assert result.diagnostics["cancel_confirmed"] is False
    finally:
        release.set()


def test_single_arm_dispatch_consuming_deadline_latches_without_waiting_for_feedback(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(g1_control, "time", clock)
    transport, feedback = Transport(clock, delay=0.2), Feedback()
    runtime = _runtime(transport, feedback)
    result = runtime.set_arm("right", np.zeros(7), lifetime_s=0.15, timeout_s=0.1)
    assert not result.ok and result.error.code is ErrorCode.TIMEOUT
    assert len(transport.calls) == 1 and not feedback.calls
    assert runtime.health["motion_session"]["uncertain"] is True
