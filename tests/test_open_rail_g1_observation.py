from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
        "rgb_packet_timestamp_ns": 1_000_000_000,
        "rgb_order": "RGB",
        "rgb_encoding_format": "UNCOMPRESSED",
        "rgb_color_format": "RGB",
        "rgb_decoder": "raw-rgb",
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rgb_channel": "cam.other"}, "rgb_channel"),
        ({"depth_channel": "cam.head"}, "depth_channel"),
        ({"depth_channel": "depth"}, "depth_channel"),
    ],
)
def test_runtime_rejects_mismatched_camera_keys(tmp_path, overrides, message) -> None:
    profile = load_g1_observation_profile(default_g1_observation_profile_path())

    with pytest.raises(ValueError, match=message):
        OpenRailG1ObservationRuntime(
            open_rail_root=tmp_path, profile=replace(profile, **overrides), proxy=_Proxy()
        )


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
    assert not snapshot.is_synchronized
    assert snapshot.metadata["public_camera_key"] == "head"
    assert snapshot.metadata["calibration"]["depth_scale_m_per_unit"] is None
    assert snapshot.metadata["state_timestamps_bind_state_values"] is False
    assert not snapshot.rgb.flags.writeable
    assert not snapshot.depth_u16.flags.writeable
    adapter.close()


def test_runtime_only_preserves_explicit_state_timestamp_binding_claim(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    observation = _observation()

    unbound = adapter._decode(observation)
    observation["state_timestamps_bind_state_values"] = True
    bound = adapter._decode(observation)

    assert unbound.metadata["state_timestamps_bind_state_values"] is False
    assert bound.metadata["state_timestamps_bind_state_values"] is True
    assert not unbound.is_synchronized
    adapter.close()


def test_synchronization_requires_bound_state_and_matching_timestamps(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path, maximum_rgb_depth_skew_ms=1.0, proxy=_Proxy()
    )
    observation = _observation(depth=np.ones((3, 5), dtype=np.uint16))
    observation["state_timestamps_bind_state_values"] = True
    assert adapter._decode(observation).is_synchronized
    broken_depth = dict(observation, depth_error="bad packet")
    assert not adapter._decode(broken_depth).is_synchronized
    observation["state_timestamps_ns"]["arm"] = 990_000_000
    assert not adapter._decode(observation).is_synchronized
    observation["state_timestamps_ns"].pop("arm")
    assert not adapter._decode(observation).is_synchronized
    adapter.close()


def test_runtime_retains_only_bound_measured_arm_velocity(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    observation = _observation()
    observation["bound_arm_sample"] = {"names": [f"joint{i}" for i in range(14)], "velocities_raw": np.arange(14, dtype=np.float32)}
    with pytest.raises(OpenRailG1ObservationError, match="bound arm message"):
        adapter._decode(observation)
    observation["state_timestamps_bind_state_values"] = True
    snapshot = adapter._decode(observation)
    np.testing.assert_array_equal(snapshot.arm_velocities_rad_s, np.arange(14))
    assert not snapshot.arm_velocities_rad_s.flags.writeable
    observation["bound_arm_sample"]["velocities_raw"] = [0] * 7
    partial = adapter._decode(observation)
    assert partial.arm_velocities_rad_s is None
    assert len(partial.arm_velocity_raw) == 7
    observation["bound_arm_sample"]["velocities_raw"] = [0] * 15
    with pytest.raises(OpenRailG1ObservationError, match="1 to 14 finite raw arm velocities"):
        adapter._decode(observation)
    adapter.close()


def test_missing_depth_stays_raw_and_is_not_reported_as_synchronized(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())

    snapshot = adapter._decode(_observation())

    assert snapshot.depth_u16 is None
    assert snapshot.depth_timestamp_ns is None
    assert not snapshot.is_synchronized
    adapter.close()


def test_require_depth_skips_rgb_only_warmup_frame(tmp_path) -> None:
    class Proxy(_Proxy):
        def __init__(self):
            self.frames = iter((_observation(), _observation(depth=np.ones((2, 3), dtype=np.uint16))))

        def retrieve_observation(self):
            return next(self.frames)

    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=Proxy())

    snapshot = adapter.read_snapshot(timeout_s=1.0, require_depth=True)

    assert snapshot.depth_u16 is not None
    assert not snapshot.is_synchronized
    adapter.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_observation_timeout_must_be_positive_finite(tmp_path, timeout) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    with pytest.raises(ValueError, match="positive and finite"):
        adapter.read_snapshot(timeout_s=timeout)
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


def test_runtime_passes_profile_camera_key_and_channel_to_fake_proxy(
    tmp_path, monkeypatch
) -> None:
    class AttrConfig(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    conf_package = ModuleType("conf")
    conf_package.__path__ = []
    robots_conf = ModuleType("conf.robots_conf")
    robots_conf.get_test_config = lambda: AttrConfig(
        camera=AttrConfig(ref="legacy", names={})
    )
    client_package = ModuleType("client")
    client_package.__path__ = []
    robots_package = ModuleType("client.robots")
    robots_package.__path__ = []
    robot_proxy_module = ModuleType("client.robots.robot_proxy")

    class FakeRobotProxy:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            pass

    robot_proxy_module.RobotProxy = FakeRobotProxy
    monkeypatch.setitem(sys.modules, "conf", conf_package)
    monkeypatch.setitem(sys.modules, "conf.robots_conf", robots_conf)
    monkeypatch.setitem(sys.modules, "client", client_package)
    monkeypatch.setitem(sys.modules, "client.robots", robots_package)
    monkeypatch.setitem(sys.modules, "client.robots.robot_proxy", robot_proxy_module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "")

    profile = replace(
        load_g1_observation_profile(default_g1_observation_profile_path()),
        config_factory="conf.robots_conf:get_test_config",
        public_camera_key="rgb_main",
        rgb_channel="cam.rgb_main",
        depth_channel="cam.z16",
    )
    runtime = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path,
        profile=profile,
        calibration_camera_keys=("head_left_fisheye",),
    )

    config = runtime._proxy.kwargs["config"]
    assert config.camera.ref == "rgb_main"
    assert config.camera.names == {
        "rgb_main": "rgb_main",
        "z16": "z16",
        "head_left_fisheye": "head_left_fisheye",
    }
    assert config.camera.depth_ref == "z16"
    assert config.camera.calibration_camera_keys == ["head_left_fisheye"]
    assert config.camera.color_order == "rgb"
    assert config.allow_physical_motion is False
    runtime.close()


def test_body_uses_configured_depth_key_and_sdk_channel_without_hardware(
    monkeypatch,
) -> None:
    class FakePacket:
        type = 1
        color_format = 2
        encoding_format = 3
        image_width = 2
        image_height = 1

        def __init__(self, *, timestamp, color_format, encoding_format, payload):
            self.send_timestamp = timestamp
            self.color_format = color_format
            self.encoding_format = encoding_format
            self.payload = payload

        def get_image_data(self):
            return self.payload

    class FakeCamera:
        def __init__(self) -> None:
            self.requested: list[tuple[str, int]] = []
            self.rgb_packet = FakePacket(
                timestamp=100,
                color_format=1,
                encoding_format=3,
                payload=bytes([1, 2, 3, 4, 5, 6]),
            )
            self.depth_packet = FakePacket(
                timestamp=101,
                color_format=2,
                encoding_format=3,
                payload=np.array([12, 34], dtype=np.uint16).tobytes(),
            )

        def get_packet_nearest(self, camera_channel, reference_timestamp):
            self.requested.append((camera_channel, reference_timestamp))
            if camera_channel == "sdk/color":
                return self.rgb_packet
            return self.depth_packet

    class Buffer:
        def read_nearest(self, _timestamp):
            return [0.0], 100

    class FakeRobot:
        _left_ee_buffer = Buffer()
        _right_ee_buffer = Buffer()
        arm_joint_names = [f"joint{i}" for i in range(14)]

        def arm_joint_states_nearest(self, timestamp):
            return [0.0] * 14, timestamp

        def head_joint_states_nearest(self, timestamp):
            return [0.0, 0.0], timestamp

        def waist_joint_states_nearest(self, timestamp):
            return [0.0, 0.0], timestamp

    class FakeOpenRailBody:
        def __init__(self, config) -> None:
            self.config = config
            self.camera = FakeCamera()
            self.robot = FakeRobot()

        def retrieve_observation(self):
            return {
                "ref_timestamp": 100,
                "cam.rgb_main": np.zeros((1, 2, 3), dtype=np.uint8),
            }

    packages = (
        "client",
        "client.robots",
        "client.robots.agibot_g1",
    )
    for name in packages:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    body_module = ModuleType("client.robots.agibot_g1.body_robot")
    body_module.RobotBody = FakeOpenRailBody
    monkeypatch.setitem(sys.modules, "client.robots.agibot_g1.body_robot", body_module)
    cosine_bus = ModuleType("cosine_bus")
    cosine_bus.PacketType = SimpleNamespace(IMAGE=1)
    cosine_bus.ColorFormat = SimpleNamespace(RGB=1, BGR=4, RS2_FORMAT_Z16=2)
    cosine_bus.EncodingFormat = SimpleNamespace(JPEG=1, PNG=2, UNCOMPRESSED=3)
    monkeypatch.setitem(sys.modules, "cosine_bus", cosine_bus)

    body_path = (
        Path(__file__).resolve().parents[1]
        / "src/cap_harness/open_rail/g1_body.py"
    )
    spec = importlib.util.spec_from_file_location("g1_body_fake_test", body_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class AttrConfig(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    config = AttrConfig(
        camera=AttrConfig(
            ref="rgb_main",
            names={
                "rgb_main": "sdk/color",
                "range_image": "sdk/range",
                "fisheye": "sdk/fisheye",
            },
            depth_ref="range_image",
            calibration_camera_keys=["fisheye"],
        )
    )
    body = module.RobotBody(config)

    observation = body.retrieve_observation()

    assert body.config.camera.ref == "rgb_main"
    assert body.config.camera.names == {
        "rgb_main": "sdk/color",
        "range_image": "sdk/range",
        "fisheye": "sdk/fisheye",
    }
    assert body.camera.requested == [("sdk/color", 100), ("sdk/range", 100)]
    np.testing.assert_array_equal(
        observation["cam.rgb_main"], np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    )
    assert observation["rgb_order"] == "RGB"
    assert observation["rgb_encoding_format"] == "UNCOMPRESSED"
    assert observation["rgb_color_format"] == "RGB"
    assert observation["rgb_decoder"] == "raw-rgb"
    assert observation["rgb_packet_timestamp_ns"] == observation["rgb_timestamp_ns"]
    # Open-RAIL's FramePacket transports only 16 non-image fields. Keep this
    # observation below that bound so later diagnostics are not silently lost.
    scalar_fields = [key for key in observation if not key.startswith("cam.")]
    assert len(scalar_fields) <= 16
    assert observation["depth_color_format"] == "2"
    np.testing.assert_array_equal(
        observation["cam.range_image"], np.array([[12, 34]], dtype=np.uint16)
    )

    class BoundSamples:
        def nearest(self, timestamp):
            assert timestamp == 100
            return (np.arange(20, dtype=np.float32), np.arange(14, dtype=np.float32),
                    {name: 99 for name in ("arm", "head", "waist", "gripper_left", "gripper_right")})

        def nearest_sides(self, timestamp):
            return {}

    body._bound_state_samples = BoundSamples()
    body.config.bound_arm_message_names = body.robot.arm_joint_names
    bound_observation = body.retrieve_observation()
    np.testing.assert_array_equal(bound_observation["obs.state"], np.arange(20))
    np.testing.assert_array_equal(bound_observation["bound_arm_sample"]["velocities_raw"], np.arange(14))
    assert bound_observation["bound_arm_sample"]["names"] == body.robot.arm_joint_names
    assert bound_observation["state_timestamps_bind_state_values"] is True
    assert len([key for key in bound_observation if not key.startswith("cam.")]) <= 16
    body._bound_state_samples = None

    body.camera.rgb_packet = FakePacket(
        timestamp=100,
        color_format=4,
        encoding_format=3,
        payload=bytes([3, 2, 1, 6, 5, 4]),
    )
    bgr_observation = body.retrieve_observation()
    np.testing.assert_array_equal(
        bgr_observation["cam.rgb_main"], np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    )
    assert bgr_observation["rgb_decoder"] == "raw-bgr-to-rgb"

    cv2 = ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    cv2.imdecode = lambda _encoded, _flags: np.array(
        [[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    body.camera.rgb_packet = FakePacket(
        timestamp=100,
        color_format=1,
        encoding_format=1,
        payload=b"jpeg",
    )
    jpeg_observation = body.retrieve_observation()
    np.testing.assert_array_equal(
        jpeg_observation["cam.rgb_main"],
        np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
    )
    assert jpeg_observation["rgb_decoder"] == "opencv-imdecode-bgr-to-rgb"
    assert jpeg_observation["rgb_color_format"] == "RGB"

    body.camera.rgb_packet = FakePacket(
        timestamp=99,
        color_format=1,
        encoding_format=3,
        payload=bytes([1, 2, 3, 4, 5, 6]),
    )
    mismatched = body.retrieve_observation()
    assert "cam.rgb_main" not in mismatched
    assert "timestamp does not match" in mismatched["rgb_error"]

    body.camera.rgb_packet = FakePacket(
        timestamp=100,
        color_format=1,
        encoding_format=999,
        payload=bytes([1, 2, 3, 4, 5, 6]),
    )
    unknown = body.retrieve_observation()
    assert "cam.rgb_main" not in unknown
    assert "unsupported RGB packet encoding_format" in unknown["rgb_error"]

    body.camera.rgb_packet = FakePacket(
        timestamp=100,
        color_format=1,
        encoding_format=3,
        payload=bytes([1, 2, 3, 4, 5, 6]),
    )
    capture = body.capture_calibration_packets(["rgb_main", "fisheye"], 100)
    assert [record["metadata"]["camera_key"] for record in capture] == [
        "rgb_main",
        "fisheye",
    ]
    assert capture[0]["metadata"]["packet_timestamp_ns"] == 100
    assert capture[1]["metadata"]["packet_timestamp_ns"] == 101
    assert capture[1]["metadata"]["sdk_channel"] == "sdk/fisheye"
    assert body.camera.requested[-2:] == [("sdk/color", 100), ("sdk/fisheye", 100)]

    # Exercise the opt-in subscriptions with message-shaped fakes; no SDK node
    # or hardware is opened. Each callback must retain its own source stamp.
    import logging

    for package_name in ("sensor_msgs_pb", "sensor_msgs_pb.msg", "genie_msgs_pb", "genie_msgs_pb.msg"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    for package, message_name in (("sensor_msgs_pb", "JointState"), ("genie_msgs_pb", "EndState"),
                                  ("genie_msgs_pb", "HeadState"), ("genie_msgs_pb", "WaistState"),
                                  ("genie_msgs_pb", "ArmState")):
        name = f"{package}.msg.{message_name}_pb2"
        fake_module = ModuleType(name)
        setattr(fake_module, message_name, type(message_name, (), {}))
        monkeypatch.setitem(sys.modules, name, fake_module)

    class FakeNode:
        def __init__(self, name):
            self.name = name
            self.callbacks = {}

        def create_subscriber(self, topic, _message_type, _qos, callback):
            self.callbacks[topic] = callback
            return callback

    names = tuple([f"Joint{i}_l" for i in range(1, 8)] + [f"Joint{i}_r" for i in range(1, 8)])
    body.robot.arm_joint_names = names
    body.robot.qos = object()
    fake_dds = ModuleType("agibotdds")
    fake_dds.Node = FakeNode
    fake_dds_package = ModuleType("cosine_bus.agibotdds_py3")
    fake_dds_package.agibotdds = fake_dds
    monkeypatch.setitem(sys.modules, "cosine_bus.agibotdds_py3", fake_dds_package)
    body.logger = logging.getLogger(__name__)
    body.config.bound_state_topics = {name: name for name in ("arm", "head", "waist", "gripper_left", "gripper_right")}
    body.config.bound_state_topics.update({"arm_left": "arm_left", "arm_right": "arm_right"})
    body.config.bound_arm_message_names = names
    body._install_bound_state_subscriptions()
    stamp = SimpleNamespace(sec=0, nanosec=100)
    header = SimpleNamespace(stamp=stamp)
    motor = lambda value: SimpleNamespace(position=value)
    assert body._bound_state_node.name == "ASPIRE_G1_State"
    callbacks = body._bound_state_node.callbacks
    callbacks["arm"](SimpleNamespace(header=header, name=names, position=list(range(14)), velocity=[0.1] * 14))
    callbacks["head"](SimpleNamespace(header=header, name=["joint_head_yaw", "joint_head_pitch"], motor_states=[motor(0), motor(0)]))
    callbacks["waist"](SimpleNamespace(header=header, name=["joint_body_pitch", "joint_lift_body"], motor_states=[motor(0), motor(0)]))
    for side in ("left", "right"):
        callbacks[f"gripper_{side}"](SimpleNamespace(header=header, type=2, end_state=[motor(60)]))
        callbacks[f"arm_{side}"](SimpleNamespace(
            header=header,
            motor_states=[SimpleNamespace(id=i, position=i / 10, velocity=i / 100) for i in range(7)],
        ))
    bound_observation = body.retrieve_observation()
    assert bound_observation["state_timestamps_bind_state_values"] is True
    np.testing.assert_allclose(bound_observation["obs.state"], [*range(14), 0.5, 0.5, 0, 0, 0, 0])
    np.testing.assert_allclose(bound_observation["bound_arm_sample"]["velocities_raw"], [0.1] * 14)
    assert bound_observation["bound_arm_sample"]["sides"]["right"]["motor_ids"] == list(range(7))
