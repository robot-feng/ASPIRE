from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import pytest

from cap_harness.errors import ErrorCode
from cap_harness.open_rail.g1_control import (
    G1CommandCapabilities,
    OpenRailG1CommandRuntime,
    OpenRailG1ProxyCommandTransport,
    cap_gripper_to_g1,
    g1_feedback_to_cap,
    proxy_motion_session_state,
)


class _Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def set_joint_position_control(self, lifetime_s, joint_groups) -> None:
        self.calls.append(("arm", lifetime_s, dict(joint_groups)))

    def publish_right_gripper(self, position) -> None:
        self.calls.append(("gripper", position))

    def set_joint_position_control_bounded(self, lifetime_s, joint_groups, *, timeout_s):
        assert timeout_s > 0
        self.set_joint_position_control(lifetime_s, joint_groups)

    def publish_right_gripper_bounded(self, position, *, timeout_s):
        assert timeout_s > 0
        self.publish_right_gripper(position)


def test_concurrent_facades_share_one_proxy_latch() -> None:
    class SlowProxy:
        def __getattribute__(self, name):
            if name == "_cap_harness_g1_motion_session_state":
                value = super().__getattribute__("__dict__").get(name)
                time.sleep(0.01)
                return value
            return super().__getattribute__(name)

    proxy = SlowProxy()
    with ThreadPoolExecutor(max_workers=8) as executor:
        states = list(executor.map(lambda _: proxy_motion_session_state(proxy), range(8)))

    assert all(state is states[0] for state in states)


class _Feedback:
    def __init__(self, *, reached: bool = True) -> None:
        self.reached = reached
        self.arm_calls = []
        self.gripper_calls = []

    def wait_for_right_arm(self, target_rad, **kwargs) -> bool:
        self.arm_calls.append((target_rad.copy(), kwargs))
        return self.reached

    def wait_for_right_gripper(self, target, **kwargs) -> bool:
        self.gripper_calls.append((target, kwargs))
        return self.reached


def _verified_fake_capabilities() -> G1CommandCapabilities:
    return G1CommandCapabilities(
        arm_command_protocol_verified=True,
        arm_position_limits_verified=True,
        arm_terminal_feedback_verified=True,
        arm_cancel_or_lease_expiry_verified=True,
        gripper_terminal_feedback_verified=True,
        gripper_stop_or_hold_verified=True,
    )


def _runtime(transport, feedback, *, capabilities=None, enabled=True):
    return OpenRailG1CommandRuntime(
        transport,
        feedback=feedback,
        capabilities=capabilities or _verified_fake_capabilities(),
        allow_physical_motion=enabled,
        position_tolerance_rad=0.02,
        velocity_tolerance_rad_s=0.05,
        arm_joint_limits_rad=[(-2.0, 2.0)] * 7,
        gripper_tolerance=0.03,
    )


def test_right_arm_mapping_uses_only_scoped_sdk_group_and_explicit_lease() -> None:
    target = np.linspace(-0.2, 0.2, 7)

    lifetime, groups = OpenRailG1CommandRuntime.map_arm_target(
        "right", target, lifetime_s=0.15
    )

    assert lifetime == 0.15
    assert groups == {"right_arm": tuple(target)}
    with pytest.raises(ValueError, match="only permits the configured right arm"):
        OpenRailG1CommandRuntime.map_arm_target("left", target, lifetime_s=0.15)


def test_constructor_reset_and_default_capabilities_never_publish() -> None:
    transport = _Transport()
    runtime = OpenRailG1CommandRuntime(transport, allow_physical_motion=True)
    runtime.reset()

    arm = runtime.set_arm("right", np.zeros(7), lifetime_s=0.1, timeout_s=0.2)
    gripper = runtime.set_gripper("right", 1.0, timeout_s=0.2)

    assert not arm.ok and arm.error.code is ErrorCode.UNSUPPORTED
    assert not gripper.ok and gripper.error.code is ErrorCode.UNSUPPORTED
    assert transport.calls == []
    assert runtime.health["selected_arm"] == "right"
    assert runtime.health["motion_ready"] is False


def test_runtime_sends_only_right_arm_group_after_all_explicit_gates() -> None:
    transport = _Transport()
    feedback = _Feedback()
    runtime = _runtime(transport, feedback)
    target = np.arange(7, dtype=np.float64) / 10

    result = runtime.set_arm("right", target, lifetime_s=0.2, timeout_s=0.5)

    assert result.ok
    name, lifetime, groups = transport.calls[0]
    assert name == "arm"
    assert lifetime == 0.2
    assert set(groups) == {"right_arm"}
    assert len(groups["right_arm"]) == 7
    assert np.array_equal(feedback.arm_calls[0][0], target)


def test_timeout_is_not_reported_as_cancel_and_never_uses_global_stop() -> None:
    transport = _Transport()
    runtime = _runtime(transport, _Feedback(reached=False))

    result = runtime.set_arm("right", np.ones(7), lifetime_s=0.1, timeout_s=0.2)

    assert not result.ok and result.error.code is ErrorCode.TIMEOUT
    assert result.diagnostics["cancel_confirmed"] is False
    assert result.diagnostics["global_motion_stop_sent"] is False
    assert transport.calls[0][2].keys() == {"right_arm"}
    cancelled = runtime.cancel()
    assert not cancelled.ok and cancelled.error.code is ErrorCode.UNSUPPORTED
    assert cancelled.diagnostics["global_motion_stop_sent"] is False

    runtime.reset()
    gripper = runtime.set_gripper("right", 0.5, timeout_s=0.2)
    assert not gripper.ok and gripper.error.code is ErrorCode.SAFETY_INTERLOCK
    assert gripper.diagnostics["motion_state_uncertain"] is True
    assert runtime.health["motion_session"]["uncertain"] is True
    assert len(transport.calls) == 1


@pytest.mark.parametrize("failure_stage", ["transport", "feedback"])
def test_exception_after_dispatch_latches_all_later_commands(failure_stage) -> None:
    class FailingTransport(_Transport):
        def set_joint_position_control(self, lifetime_s, joint_groups) -> None:
            super().set_joint_position_control(lifetime_s, joint_groups)
            if failure_stage == "transport":
                raise RuntimeError("publish acknowledgement lost")

    class FailingFeedback(_Feedback):
        def wait_for_right_arm(self, target_rad, **kwargs) -> bool:
            if failure_stage == "feedback":
                raise RuntimeError("feedback stream lost")
            return super().wait_for_right_arm(target_rad, **kwargs)

    transport = FailingTransport()
    runtime = _runtime(transport, FailingFeedback())
    result = runtime.set_arm("right", np.zeros(7), lifetime_s=0.1, timeout_s=0.2)

    assert not result.ok and result.error.code is ErrorCode.CONTROLLER_FAILED
    blocked = runtime.set_gripper("right", 0.5, timeout_s=0.2)
    assert not blocked.ok and blocked.error.code is ErrorCode.SAFETY_INTERLOCK
    assert len(transport.calls) == 1


def test_arm_target_outside_verified_joint_limits_is_not_dispatched() -> None:
    transport = _Transport()
    runtime = _runtime(transport, _Feedback())

    result = runtime.set_arm("right", np.full(7, 2.1), lifetime_s=0.1, timeout_s=0.2)

    assert not result.ok and result.error.code is ErrorCode.SAFETY_INTERLOCK
    assert transport.calls == []


def test_single_gripper_endpoint_preserves_normalized_command() -> None:
    transport = _Transport()
    feedback = _Feedback()
    runtime = _runtime(transport, feedback)

    result = runtime.set_gripper("right", 0.4, timeout_s=0.5)

    assert result.ok
    assert transport.calls == [("gripper", 0.4)]
    assert feedback.gripper_calls[0][0] == 0.4
    with pytest.raises(ValueError, match="only permits the configured right gripper"):
        runtime.map_gripper_target("left", 0.4)


@pytest.mark.parametrize("cap_position,native_position", [(1.0, 0.0), (0.0, 1.0), (0.25, 0.75)])
def test_cap_gripper_boundary_converts_once_and_waits_in_native_domain(
    cap_position, native_position
) -> None:
    transport, feedback = _Transport(), _Feedback()
    runtime = _runtime(transport, feedback)

    result = runtime.set_cap_gripper(cap_position, arm="right", timeout_s=0.5)

    assert result.ok
    assert transport.calls == [("gripper", native_position)]
    assert feedback.gripper_calls[0][0] == native_position
    assert g1_feedback_to_cap(native_position, convention_verified=True) == cap_position


@pytest.mark.parametrize("value", [True, np.bool_(False), float("nan"), -0.1, 1.1, "bad"])
def test_cap_gripper_invalid_targets_are_never_published(value) -> None:
    transport = _Transport()
    runtime = _runtime(transport, _Feedback())

    assert not runtime.set_cap_gripper(value, timeout_s=0.5).ok
    assert transport.calls == []


def test_gripper_feedback_conversion_requires_explicitly_verified_convention() -> None:
    with pytest.raises(ValueError, match="feedback convention must be verified"):
        g1_feedback_to_cap(0.0)
    assert cap_gripper_to_g1(1.0) == 0.0


def test_capability_evidence_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="must be a bool"):
        G1CommandCapabilities(arm_terminal_feedback_verified="false")  # type: ignore[arg-type]


def test_non_boolean_feedback_does_not_confirm_motion() -> None:
    class InvalidFeedback(_Feedback):
        def wait_for_right_arm(self, target_rad, **kwargs):
            return "true"

    transport = _Transport()
    runtime = _runtime(transport, InvalidFeedback())

    result = runtime.set_arm("right", np.zeros(7), lifetime_s=0.1, timeout_s=0.2)
    assert not result.ok and result.error.code is ErrorCode.TIMEOUT
    assert not runtime.set_gripper("right", 0.5, timeout_s=0.2).ok
    assert len(transport.calls) == 1


def test_proxy_transport_rejects_combined_or_unscoped_joint_groups() -> None:
    class Proxy:
        def command_right_arm(self, target, lifetime):
            self.arm = (target, lifetime)

        def command_right_gripper(self, target):
            self.gripper = target

    proxy = Proxy()
    transport = OpenRailG1ProxyCommandTransport(proxy)

    with pytest.raises(ValueError, match="only the SDK right_arm"):
        transport.set_joint_position_control(0.1, {"left_arm": [0.0] * 7})
    transport.set_joint_position_control(0.1, {"right_arm": [0.0] * 7})
    transport.publish_right_gripper(0.0)

    assert proxy.arm == ([0.0] * 7, 0.1)
    assert proxy.gripper == 0.0
