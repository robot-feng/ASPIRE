"""G1 control composition tested exclusively with fake transport and clocks."""

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from cap_harness.contracts import Observation, Pose, RobotState, Trajectory
from cap_harness.errors import ErrorCode
from cap_harness.open_rail import g1_control, g1_facade
from cap_harness.open_rail.g1_broadcast import G1SnapshotBroadcaster
from cap_harness.open_rail.g1_control import G1CommandCapabilities, G1TimedTrajectoryPolicy, G1_RIGHT_ARM_JOINTS
from cap_harness.open_rail.g1_control import OpenRailG1ProxyCommandTransport
from cap_harness.open_rail.g1_facade import G1ControlFacade
from cap_harness.open_rail.g1_motion_monitor import G1MotionMonitor, G1MotionMonitorEvidence
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailG1ObservationRuntime
from cap_harness.open_rail.robot_profile import OpenRailProfile


class Proxy:
    def __init__(self):
        self.calls = []
        self.closed = False

    def command_right_arm(self, positions, lifetime):
        self.calls.append(("arm", tuple(positions), lifetime))

    def command_right_gripper(self, position):
        self.calls.append(("gripper", position))

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 10.0

    def monotonic(self):
        return self.t

    def time(self):
        return 1000.0

    def sleep(self, seconds):
        self.t += seconds


class Monitor(G1MotionMonitor):
    def check(self, *, trajectory_token, reference_joint_positions_rad, timeout_s):
        return G1MotionMonitorEvidence(
            trajectory_token=trajectory_token, state_timestamp_s=1000.0, scene_timestamp_s=1000.0,
            measured_joint_positions_rad=reference_joint_positions_rad,
            reference_joint_positions_rad=reference_joint_positions_rad,
            state_fingerprint="a" * 64, scene_fingerprint="b" * 64,
            fixed_posture_within_limits=True, scene_clear=True,
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
        joint_positions=np.array([[0.0] * 7, [0.01] * 7]), dt_s=0.1,
        joint_names=G1_RIGHT_ARM_JOINTS, planner="fake", collision_aware=True,
        expected_start=_state(), arm="right", gripper_positions=np.full(2, 0.5),
        embodiment="open_rail_7dof",
    )


def _policy():
    return G1TimedTrajectoryPolicy(
        control_period_s=0.1, command_lifetime_s=0.15, max_dispatch_s=0.01,
        max_lateness_s=0.02, final_feedback_timeout_s=0.5, max_start_age_s=0.1,
        start_clock_verified=True, max_start_error_rad=0.01, max_start_gripper_error=0.01,
        max_joint_velocity_rad_s=[1] * 7, max_joint_acceleration_rad_s2=[5] * 7,
        dynamic_limits_verified=True,
        max_tracking_error_rad=0.05, monitor_max_age_s=0.1, monitor_timeout_s=0.01,
    )


def _profile():
    return OpenRailProfile(
        name="fake", robot_type="agibot_g1", model="fake", base_frame="base", arms=("right",),
        joint_names={"right": G1_RIGHT_ARM_JOINTS}, end_effector_links={"right": "tcp"},
        base_transforms={"right": np.eye(4)}, arm_aliases={"primary": "right"},
    )


def _compose(tmp_path, monkeypatch, *, guard=lambda *_: True, enabled=True, timed=True, reached=True):
    proxy = Proxy()
    runtime = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path, proxy=proxy, bound_state_sampling=True, allow_physical_motion=enabled,
    )
    broadcaster = G1SnapshotBroadcaster(runtime, capture_timeout_s=0.1, close_timeout_s=0.2)
    observation = SimpleNamespace(robot_state=_state())

    class Observations:
        def __init__(self, view, *_args, **_kwargs):
            self.view = view

        def get_observation(self):
            return observation

        def get_observation_with_snapshot(self):
            return observation, object()

        def ensure_snapshot_current(self, _snapshot):
            pass

        def close(self):
            self.view.close()

    class Feedback:
        def __init__(self, *_args):
            pass

        def wait_for_right_arm(self, *_args, **_kwargs):
            return reached

        def wait_for_right_gripper(self, *_args, **_kwargs):
            return reached

    monkeypatch.setattr(g1_facade, "G1ObservationControl", Observations)
    monkeypatch.setattr(g1_facade, "G1MeasuredCommandFeedback", Feedback)
    monkeypatch.setattr(g1_control, "time", Clock())
    caps = G1CommandCapabilities(
        arm_command_protocol_verified=True, arm_position_limits_verified=True,
        arm_terminal_feedback_verified=True, arm_cancel_or_lease_expiry_verified=True,
        arm_timed_dispatch_verified=timed, gripper_terminal_feedback_verified=True,
        arm_live_monitor_verified=True,
        gripper_stop_or_hold_verified=True,
    )
    facade = G1ControlFacade(
        runtime, broadcaster, _profile(), None, None, None,
        observation_options={}, capabilities=caps, trajectory_policy=_policy(),
        arm_joint_limits_rad=((-1, 1),) * 7, position_tolerance_rad=0.01,
        velocity_tolerance_rad_s=0.02, gripper_tolerance=0.02, gripper_timeout_s=0.5,
        trajectory_guard=guard, trajectory_guard_verified=True,
        gripper_guard=lambda *_: True, gripper_guard_verified=True,
        motion_monitor=Monitor(),
    )
    # A test-only bounded transport; the real Open-RAIL proxy has no such
    # contract and must remain blocked even if all flags are set.
    class FakeBoundedTransport:
        def set_joint_position_control(self, lifetime_s, groups):
            proxy.command_right_arm(list(groups["right_arm"]), lifetime_s)

        def set_joint_position_control_bounded(self, lifetime_s, groups, *, timeout_s):
            assert timeout_s > 0
            self.set_joint_position_control(lifetime_s, groups)

        def publish_right_gripper(self, position):
            proxy.command_right_gripper(position)

        def publish_right_gripper_bounded(self, position, *, timeout_s):
            assert timeout_s > 0
            self.publish_right_gripper(position)

    facade._command._transport = FakeBoundedTransport()
    return facade, proxy, runtime, broadcaster


def test_construction_and_close_never_send_robot_commands(tmp_path, monkeypatch):
    facade, proxy, _runtime, broadcaster = _compose(tmp_path, monkeypatch)
    assert broadcaster._thread is None
    assert proxy.calls == []
    facade.close()
    assert proxy.closed and proxy.calls == []


def test_guard_and_timed_capability_gate_before_dispatch(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch, guard=lambda *_: False)
    result = facade.execute_trajectory(_trajectory())
    assert not result.ok and result.error.code is ErrorCode.STALE_PLAN
    assert proxy.calls == []
    facade.close()

    facade, proxy, *_ = _compose(tmp_path, monkeypatch, timed=False)
    result = facade.execute_trajectory(_trajectory())
    assert not result.ok and result.error.code is ErrorCode.UNSUPPORTED
    assert proxy.calls == []
    facade.close()


def test_unbounded_proxy_refuses_commands_even_if_capability_flags_are_true(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch)
    facade._command._transport = OpenRailG1ProxyCommandTransport(proxy)
    arm = facade.execute_trajectory(_trajectory())
    gripper = facade.set_gripper(0.0, arm="right")
    assert not arm.ok and arm.error.code is ErrorCode.UNSUPPORTED
    assert not gripper.ok and gripper.error.code is ErrorCode.UNSUPPORTED
    assert proxy.calls == []
    facade.close()


def test_gripper_fresh_scene_guard_blocks_before_dispatch(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch)
    facade._gripper_guard = lambda *_: False
    result = facade.set_gripper(0.0, arm="right")
    assert not result.ok and result.error.code is ErrorCode.STALE_PLAN
    assert proxy.calls == []
    facade.close()


def test_snapshot_expiring_during_guard_blocks_arm_and_gripper(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch)

    def stale(_snapshot):
        raise OpenRailG1ObservationError("snapshot became stale")

    facade._observation.ensure_snapshot_current = stale
    arm = facade.execute_trajectory(_trajectory())
    gripper = facade.set_gripper(0.0, arm="right")
    assert arm.error.code is ErrorCode.STALE_PLAN
    assert gripper.error.code is ErrorCode.STALE_PLAN
    assert proxy.calls == []
    facade.close()

def test_fake_execution_uses_only_right_arm_and_cap_gripper_mapping(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch)
    result = facade.execute_trajectory(_trajectory())
    assert result.ok and [item[0] for item in proxy.calls] == ["arm"]
    result = facade.set_gripper(0.8, arm="right")
    assert result.ok and proxy.calls[-1][0] == "gripper"
    assert proxy.calls[-1][1] == pytest.approx(0.2)
    assert not facade.set_grippers({"left": 0.0}).ok
    assert not facade.step(object()).ok
    facade.close()


def test_unconfirmed_gripper_latches_trajectory_without_claiming_grasp(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch, reached=False)
    result = facade.set_gripper(0.0, arm="right")
    assert not result.ok and result.diagnostics["contact_or_grasp_status"] == "unknown"
    assert not facade.execute_trajectory(_trajectory()).ok
    assert [item[0] for item in proxy.calls] == ["gripper"]
    facade.close()


def test_close_cannot_interrupt_inflight_motion_or_destroy_observation_owner(tmp_path, monkeypatch):
    facade, proxy, *_ = _compose(tmp_path, monkeypatch)
    started, release = threading.Event(), threading.Event()
    original = proxy.command_right_arm

    def blocking_command(positions, lifetime):
        started.set()
        assert release.wait(2)
        original(positions, lifetime)

    proxy.command_right_arm = blocking_command
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(facade.execute_trajectory, _trajectory())
        assert started.wait(1)
        with pytest.raises(RuntimeError, match="active command"):
            facade.close()
        assert not proxy.closed
        release.set()
        assert job.result(timeout=2).ok
    facade.close()
    assert proxy.closed
