from __future__ import annotations

import numpy as np
import pytest

from cap_harness.errors import ErrorCode
from cap_harness.open_rail.g1_control import G1CommandCapabilities
from cap_harness.open_rail.g1_runtime import (
    OpenRailG1ObservationError,
    OpenRailG1ObservationRuntime,
)
from cap_harness.open_rail.profile import (
    default_g1_observation_profile_path,
    load_g1_observation_profile,
)


class _Proxy:
    def close(self) -> None:
        pass


class _CommandFeedback:
    def __init__(self, *, reached: bool) -> None:
        self.reached = reached

    def wait_for_right_arm(self, target_rad, **kwargs) -> bool:
        return self.reached

    def wait_for_right_gripper(self, target, **kwargs) -> bool:
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


def _observation(*, depth: np.ndarray | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "cam.head": np.full((3, 5, 3), 17, dtype=np.uint8),
        "obs.state": np.arange(20, dtype=np.float32),
        "rgb_timestamp_ns": 1_000_000_000,
        "rgb_order": "RGB",
        "state_timestamps_ns": {
            "arm": 999_800_000,
            "gripper_left": 999_810_000,
            "gripper_right": 999_820_000,
            "head": 999_830_000,
            "waist": 999_840_000,
        },
    }
    if depth is not None:
        result["cam.head_depth"] = depth
        result["depth_timestamp_ns"] = 1_000_100_000
        result["depth_encoding"] = "RS2_FORMAT_Z16"
    return result


def test_g1_profile_exposes_public_head_camera_and_unverified_calibration() -> None:
    profile = load_g1_observation_profile(default_g1_observation_profile_path())

    assert profile.robot_type == "agibot_g1"
    assert profile.public_camera_key == "head"
    assert profile.rgb_channel == "cam.head"
    assert profile.depth_channel == "cam.head_depth"
    assert profile.selected_arm == "right"
    assert profile.state_dimension == 20
    assert profile.calibration["depth_scale_m_per_unit"] is None
    assert profile.calibration["arm_tcp_transforms"] is None


def test_runtime_preserves_raw_z16_and_only_claims_configured_timestamp_skew(
    tmp_path,
) -> None:
    adapter = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path,
        maximum_rgb_depth_skew_ms=1.0,
        proxy=_Proxy(),
    )
    raw_depth = np.array([[0, 1, 4095], [65535, 2, 3]], dtype=np.uint16)

    snapshot = adapter._decode(_observation(depth=raw_depth))

    np.testing.assert_array_equal(snapshot.depth_u16, raw_depth)
    np.testing.assert_array_equal(snapshot.state, np.arange(20, dtype=np.float32))
    assert snapshot.depth_skew_ns == 100_000
    assert snapshot.is_synchronized
    assert snapshot.metadata["public_camera_key"] == "head"
    assert snapshot.metadata["calibration"]["depth_scale_m_per_unit"] is None
    assert not snapshot.rgb.flags.writeable
    assert not snapshot.depth_u16.flags.writeable
    adapter.close()


def test_missing_depth_stays_raw_and_is_not_reported_as_synchronized(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())

    snapshot = adapter._decode(_observation())

    assert snapshot.depth_u16 is None
    assert snapshot.depth_timestamp_ns is None
    assert not snapshot.is_synchronized
    adapter.close()


def test_runtime_rejects_state_that_does_not_match_selected_profile(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    observation = _observation()
    observation["obs.state"] = np.zeros(19, dtype=np.float32)

    with pytest.raises(OpenRailG1ObservationError, match="expected 20 finite"):
        adapter._decode(observation)
    adapter.close()


def test_runtime_rejects_non_z16_depth_instead_of_converting_it(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    observation = _observation(depth=np.ones((2, 3), dtype=np.float32))

    with pytest.raises(OpenRailG1ObservationError, match="invalid raw Z16"):
        adapter._decode(observation)
    adapter.close()


def test_physical_motion_opt_in_and_reset_facade_creation_do_not_publish(tmp_path) -> None:
    class Proxy(_Proxy):
        def __init__(self) -> None:
            self.commands: list[tuple[object, ...]] = []

        def command_right_arm(self, target, lifetime) -> None:
            self.commands.append(("arm", target, lifetime))

        def command_right_gripper(self, target) -> None:
            self.commands.append(("gripper", target))

    proxy = Proxy()
    observation_runtime = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path,
        allow_physical_motion=True,
        proxy=proxy,
    )
    command_runtime = observation_runtime.command_runtime()

    result = command_runtime.set_arm(
        "right", np.zeros(7), lifetime_s=0.1, timeout_s=0.2
    )

    assert not result.ok
    assert proxy.commands == []
    observation_runtime.close()


def test_health_is_read_only_and_reports_selected_arm_camera_and_calibration_gaps(tmp_path) -> None:
    runtime = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())

    health = runtime.health

    assert health["selected_arm"] == "right"
    assert health["public_camera_key"] == "head"
    assert health["selected_arm_profile"]["joints"] == (
        "Joint1_r",
        "Joint2_r",
        "Joint3_r",
        "Joint4_r",
        "Joint5_r",
        "Joint6_r",
        "Joint7_r",
    )
    assert health["control_constraints"]["selected_arm"] == "right"
    assert health["control_constraints"]["read_only_parts"] == (
        "left_arm", "head", "waist", "torso", "base"
    )
    assert health["arm_tcp_transform"] is None
    assert (
        health["camera_calibration"]["intrinsics_file"]
        == "/data/parameters/head_intrinsic_params.json"
    )
    assert health["camera_calibration"]["depth_scale_m_per_unit"] is None
    assert health["calibration_complete"] is False
    assert "camera_intrinsics_values" in health["calibration_gaps"]
    assert "depth_scale_m_per_unit" in health["calibration_gaps"]
    assert health["proxy_worker_alive"] is None
    assert health["raw_snapshot_available"] is False
    runtime.close()


def test_runtime_rejects_unimplemented_left_arm_selection(tmp_path) -> None:
    with pytest.raises(ValueError, match="selected arm in the Open-RAIL profile"):
        OpenRailG1ObservationRuntime(
            open_rail_root=tmp_path,
            selected_arm="left",
            proxy=_Proxy(),
        )


def test_motion_uncertainty_survives_new_facade_reset_and_close(tmp_path) -> None:
    class Proxy(_Proxy):
        def __init__(self) -> None:
            self.commands: list[tuple[object, ...]] = []

        def command_right_arm(self, target, lifetime) -> None:
            self.commands.append(("arm", target, lifetime))

        def command_right_gripper(self, target) -> None:
            self.commands.append(("gripper", target))

    proxy = Proxy()
    first_owner = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path,
        allow_physical_motion=True,
        proxy=proxy,
    )
    first = first_owner.command_runtime(
        feedback=_CommandFeedback(reached=False),
        capabilities=_verified_fake_capabilities(),
        position_tolerance_rad=0.02,
        velocity_tolerance_rad_s=0.05,
        arm_joint_limits_rad=((-2.0, 2.0),) * 7,
        gripper_tolerance=0.03,
    )
    timed_out = first.set_arm("right", np.zeros(7), lifetime_s=0.1, timeout_s=0.2)
    assert not timed_out.ok and timed_out.error.code is ErrorCode.TIMEOUT
    first.reset()

    second = first_owner.command_runtime()
    blocked = second.set_gripper("right", 0.5, timeout_s=0.2)
    assert not blocked.ok and blocked.error.code is ErrorCode.SAFETY_INTERLOCK
    assert len(proxy.commands) == 1
    first_owner.close()

    reopened_owner = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path,
        allow_physical_motion=True,
        proxy=proxy,
    )
    reopened = reopened_owner.command_runtime(
        feedback=_CommandFeedback(reached=True),
        capabilities=_verified_fake_capabilities(),
        position_tolerance_rad=0.02,
        velocity_tolerance_rad_s=0.05,
        arm_joint_limits_rad=((-2.0, 2.0),) * 7,
        gripper_tolerance=0.03,
    )
    still_blocked = reopened.set_arm(
        "right", np.zeros(7), lifetime_s=0.1, timeout_s=0.2
    )
    assert not still_blocked.ok and still_blocked.error.code is ErrorCode.SAFETY_INTERLOCK
    assert reopened_owner.health["motion"]["motion_session"]["uncertain"] is True
    assert len(proxy.commands) == 1
    reopened_owner.close()
