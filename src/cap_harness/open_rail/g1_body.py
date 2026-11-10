"""Open-RAIL G1 observation extension that keeps raw depth timestamps."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from client.robots.agibot_g1.body_robot import RobotBody as _OpenRailG1Body
except ImportError as exc:  # pragma: no cover - depends on the Open-RAIL runtime
    raise ModuleNotFoundError(
        "Open-RAIL must be on PYTHONPATH to load the AgiBot G1 runtime"
    ) from exc


def _rgb_packet(packet: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode one matched Cosine color packet to RGB using packet metadata."""
    import cosine_bus

    if packet.type != cosine_bus.PacketType.IMAGE:
        raise ValueError(f"unexpected RGB packet type: {packet.type!r}")
    encoding = packet.encoding_format
    color_format = packet.color_format
    width = int(packet.image_width)
    height = int(packet.image_height)
    if width <= 0 or height <= 0:
        raise ValueError("RGB packet has invalid dimensions")
    if color_format == cosine_bus.ColorFormat.RGB:
        color_name = "RGB"
    elif color_format == cosine_bus.ColorFormat.BGR:
        color_name = "BGR"
    else:
        raise ValueError(f"unsupported RGB packet color_format: {color_format!r}")

    payload = packet.get_image_data()
    if payload is None:
        raise ValueError("RGB packet has no image payload")
    if isinstance(payload, np.ndarray):
        if payload.dtype != np.uint8:
            raise ValueError("RGB packet payload array is not uint8")
        encoded = np.ascontiguousarray(payload).reshape(-1)
    else:
        try:
            encoded = np.frombuffer(payload, dtype=np.uint8)
        except TypeError as exc:
            raise ValueError("RGB packet payload is not a byte buffer") from exc

    if encoding in (cosine_bus.EncodingFormat.JPEG, cosine_bus.EncodingFormat.PNG):
        import cv2

        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None or decoded_bgr.shape != (height, width, 3):
            raise ValueError("compressed RGB packet failed to decode at declared dimensions")
        rgb = np.ascontiguousarray(decoded_bgr[:, :, ::-1])
        encoding_name = "JPEG" if encoding == cosine_bus.EncodingFormat.JPEG else "PNG"
        decoder = "opencv-imdecode-bgr-to-rgb"
    elif encoding == cosine_bus.EncodingFormat.UNCOMPRESSED:
        if encoded.size != height * width * 3:
            raise ValueError("uncompressed RGB payload length does not match its dimensions")
        raw = encoded.reshape(height, width, 3)
        rgb = np.ascontiguousarray(raw if color_name == "RGB" else raw[:, :, ::-1])
        encoding_name = "UNCOMPRESSED"
        decoder = "raw-rgb" if color_name == "RGB" else "raw-bgr-to-rgb"
    else:
        raise ValueError(f"unsupported RGB packet encoding_format: {encoding!r}")

    return rgb, {
        "rgb_order": "RGB",
        "rgb_encoding_format": encoding_name,
        "rgb_color_format": color_name,
        "rgb_decoder": decoder,
    }


class RobotBody(_OpenRailG1Body):
    """Use one Open-RAIL camera receiver for synchronized RGB, depth and joints.

    This class is loaded by Open-RAIL's ``RobotProxy`` worker through its
    ``robot_module`` hook. Its constructor only creates SDK subscriptions and
    state readers; it does not call any motion method.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._rgb_camera_key = str(self.config.camera.ref)
        self._camera_names = dict(self.config.camera.names)
        if self._rgb_camera_key not in self._camera_names:
            raise ValueError("configured RGB camera ref is missing from camera.names")
        depth_key = self.config.camera.get("depth_ref", "head_depth")
        self._depth_camera_key = depth_key if depth_key in self._camera_names else None
        self._depth_camera_channel = (
            self._camera_names[self._depth_camera_key]
            if self._depth_camera_key is not None
            else None
        )
        # The upstream receiver can return the raw Z16 image, but its common
        # image path assumes every secondary camera is three-channel color.
        # Read depth by packet below so it remains uint16 and is not color-flipped.
        self._non_ref_cameras = []
        self._right_arm_controller = None
        self._bound_state_samples = None
        self._bound_state_node = None
        self._bound_state_subscribers = []
        if self.config.get("bound_state_sampling", False) is True:
            self._install_bound_state_subscriptions()

    def _install_bound_state_subscriptions(self) -> None:
        """Use a separate read-only DDS node; the SDK node rejects duplicate topics."""
        from cosine_bus.agibotdds_py3 import agibotdds
        from sensor_msgs_pb.msg.JointState_pb2 import JointState
        from genie_msgs_pb.msg.EndState_pb2 import EndState
        from genie_msgs_pb.msg.HeadState_pb2 import HeadState
        from genie_msgs_pb.msg.WaistState_pb2 import WaistState
        from genie_msgs_pb.msg.ArmState_pb2 import ArmState
        from cap_harness.open_rail.g1_state_samples import G1StateSamples

        samples = G1StateSamples(tuple(self.config.bound_arm_message_names))

        def guarded(callback):
            reported_errors = set()
            def on_message(message):
                try:
                    callback(message)
                except (AttributeError, IndexError, TypeError, ValueError) as exc:
                    error = str(exc)
                    if error not in reported_errors:
                        reported_errors.add(error)
                        self.logger.warning("discarding invalid bound G1 state sample: %s", error)
            return on_message

        topics = self.config.bound_state_topics
        subscriptions = [
            (topics["arm"], JointState, samples.arm),
            (topics["head"], HeadState, samples.head),
            (topics["waist"], WaistState, samples.waist),
            (topics["gripper_left"], EndState, lambda message: samples.gripper("left", message)),
            (topics["gripper_right"], EndState, lambda message: samples.gripper("right", message)),
        ]
        for side in ("left", "right"):
            topic = topics.get(f"arm_{side}")
            if topic:
                subscriptions.append((topic, ArmState, lambda message, side=side: samples.arm_side(side, message)))
        node = agibotdds.Node("ASPIRE_G1_State")
        subscribers = [
            node.create_subscriber(topic, message_type, self.robot.qos, guarded(callback))
            for topic, message_type, callback in subscriptions
        ]
        if any(subscriber is None for subscriber in subscribers):
            raise RuntimeError("failed to subscribe to all G1 state feedback topics")
        self._bound_state_node = node
        self._bound_state_subscribers = subscribers
        self._bound_state_samples = samples

    def close(self) -> None:
        # No publishers are created on this node. Clear references before the
        # upstream closes its own robot/camera connection.
        self._bound_state_subscribers = []
        self._bound_state_node = None
        self._bound_state_samples = None
        super().close()

    def _require_physical_motion(self) -> None:
        if self.config.get("allow_physical_motion", False) is not True:
            raise PermissionError("G1 physical commands are disabled by the runtime profile")

    def command_right_arm(self, positions_rad: list[float], lifetime_s: float) -> None:
        """Publish only the WBC right-arm joint group; never compose a 14-joint target."""
        self._require_physical_motion()
        target = np.asarray(positions_rad, dtype=np.float64)
        lifetime = float(lifetime_s)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("right-arm target must contain seven finite radians")
        if not np.isfinite(lifetime) or lifetime <= 0.0:
            raise ValueError("joint command lifetime must be positive and finite")
        if self._right_arm_controller is None:
            from a2d_sdk.robot import RobotController

            self._right_arm_controller = RobotController()
        self._right_arm_controller.set_joint_position_control(
            lifetime, {"right_arm": target.tolist()}
        )

    def command_right_gripper(self, position: float) -> None:
        """Publish one normalized command to the dedicated right endpoint topic."""
        self._require_physical_motion()
        if isinstance(position, bool):
            raise ValueError("right gripper target must be finite and in [0, 1]")
        value = float(position)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("right gripper target must be finite and in [0, 1]")
        from sensor_msgs_pb.msg.JointState_pb2 import JointState

        command = JointState()
        command.position.append(value)
        self.robot._right_ee_cmd_publisher.publish(command)

    def execute_action(self, data: Any) -> None:
        raise PermissionError("generic Open-RAIL action can command unscoped robot parts")

    def control_robot(self, data: Any) -> None:
        raise PermissionError("generic Open-RAIL control can command unscoped robot parts")

    def _control_robot(self, data: Any) -> None:
        raise PermissionError("manual Open-RAIL control is disabled for the G1 profile")

    def reset_robot(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("physical G1 reset is disabled; adapter reset is observational")

    def capture_calibration_packets(
        self, camera_keys: list[str], reference_timestamp_ns: int
    ) -> list[dict[str, Any]]:
        """Fetch raw packets via this body's existing Cosine camera owner only."""
        if self.config.get("allow_physical_motion", False) is not False:
            raise PermissionError("raw calibration capture requires a read-only G1 runtime")
        configured_keys = self.config.camera.get("calibration_camera_keys", [])
        allowed_keys = {self._rgb_camera_key, *configured_keys}
        if any(key not in allowed_keys for key in camera_keys):
            raise ValueError("calibration request includes a camera not enabled in this profile")

        import cosine_bus

        from cap_harness.open_rail.g1_calibration_capture import capture_raw_camera_packets

        records = capture_raw_camera_packets(
            self.camera,
            self._camera_names,
            camera_keys,
            reference_timestamp_ns,
            image_packet_type=cosine_bus.PacketType.IMAGE,
        )
        for record in records:
            metadata = record["metadata"]
            if (
                metadata["camera_key"] == self._rgb_camera_key
                and metadata["packet_timestamp_ns"] != reference_timestamp_ns
            ):
                raise ValueError("head RGB packet timestamp differs from the snapshot reference")
        return records

    def retrieve_observation(self) -> dict[str, Any] | None:
        observation = super().retrieve_observation()
        if observation is None:
            return None

        rgb_timestamp_ns = int(observation["ref_timestamp"])
        observation["rgb_timestamp_ns"] = rgb_timestamp_ns
        bound = (
            None if self._bound_state_samples is None
            else self._bound_state_samples.nearest(rgb_timestamp_ns)
        )
        if bound is None:
            observation["state_timestamps_ns"] = self._state_timestamps(rgb_timestamp_ns)
        else:
            state, velocity, timestamps = bound
            observation["obs.state"] = state
            observation["state_timestamps_ns"] = timestamps
            observation["state_timestamps_bind_state_values"] = True
            # Keep names and velocities in one IPC field: Open-RAIL transports
            # at most 16 non-camera fields per FramePacket.
            observation["bound_arm_sample"] = {
                "names": list(self.config.bound_arm_message_names), "velocities_raw": velocity,
                "sides": self._bound_state_samples.nearest_sides(rgb_timestamp_ns),
            }

        rgb_key = f"cam.{self._rgb_camera_key}"
        observation.pop(rgb_key, None)
        try:
            packet = self.camera.get_packet_nearest(
                self._camera_names[self._rgb_camera_key], rgb_timestamp_ns
            )
            if packet is None:
                raise ValueError("no RGB packet at the reference timestamp")
            packet_timestamp_ns = int(packet.send_timestamp)
            observation["rgb_packet_timestamp_ns"] = packet_timestamp_ns
            if packet_timestamp_ns != rgb_timestamp_ns:
                raise ValueError(
                    "RGB packet timestamp does not match the image reference timestamp"
                )
            rgb, rgb_metadata = _rgb_packet(packet)
            observation[rgb_key] = rgb
            observation.update(rgb_metadata)
        except (AttributeError, TypeError, ValueError) as exc:
            observation["rgb_order"] = "unknown"
            observation["rgb_error"] = str(exc)

        if self._depth_camera_channel is None:
            observation["depth_timestamp_ns"] = None
            observation["depth_error"] = "no depth camera is configured"
            return observation

        packet = self.camera.get_packet_nearest(
            self._depth_camera_channel, rgb_timestamp_ns
        )
        if packet is None:
            observation["depth_timestamp_ns"] = None
            observation["depth_error"] = "no depth packet near the RGB timestamp"
            return observation

        import cosine_bus

        if packet.type != cosine_bus.PacketType.IMAGE:
            observation["depth_timestamp_ns"] = None
            observation["depth_error"] = f"unexpected depth packet type {packet.type!r}"
            return observation
        if packet.color_format != cosine_bus.ColorFormat.RS2_FORMAT_Z16:
            observation["depth_timestamp_ns"] = int(packet.send_timestamp)
            observation["depth_error"] = f"depth packet is not Z16: {packet.color_format!r}"
            return observation

        width = int(packet.image_width)
        height = int(packet.image_height)
        payload = packet.get_image_data()
        if payload is None or len(payload) != width * height * np.dtype(np.uint16).itemsize:
            observation["depth_timestamp_ns"] = int(packet.send_timestamp)
            observation["depth_error"] = "Z16 payload length does not match its dimensions"
            return observation

        depth = np.frombuffer(payload, dtype=np.uint16).reshape(height, width).copy()
        depth_timestamp_ns = int(packet.send_timestamp)
        observation[f"cam.{self._depth_camera_key}"] = depth
        # Open-RAIL's FramePacket passes at most 16 scalar fields. Frame shapes
        # and depth skew are derivable from the images and timestamps; avoid
        # duplicating them here so packet encoding/color diagnostics survive.
        observation["depth_timestamp_ns"] = depth_timestamp_ns
        observation["depth_encoding"] = "RS2_FORMAT_Z16"
        observation["depth_encoding_format"] = str(packet.encoding_format)
        observation["depth_color_format"] = str(packet.color_format)
        return observation

    def _state_timestamps(self, reference_ns: int) -> dict[str, Any]:
        timestamps: dict[str, Any] = {}
        for group in ("arm", "head", "waist"):
            _, timestamp = getattr(self.robot, f"{group}_joint_states_nearest")(reference_ns)
            timestamps[group] = None if timestamp is None else int(timestamp)

        left, left_timestamp = self.robot._left_ee_buffer.read_nearest(reference_ns)
        right, right_timestamp = self.robot._right_ee_buffer.read_nearest(reference_ns)
        timestamps["gripper_left"] = None if left is None else int(left_timestamp)
        timestamps["gripper_right"] = None if right is None else int(right_timestamp)
        return timestamps


__all__ = ["RobotBody"]
