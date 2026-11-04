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

    def retrieve_observation(self) -> dict[str, Any] | None:
        observation = super().retrieve_observation()
        if observation is None:
            return None

        rgb_timestamp_ns = int(observation["ref_timestamp"])
        observation["rgb_timestamp_ns"] = rgb_timestamp_ns
        observation["rgb_order"] = "RGB"
        observation["state_timestamps_ns"] = self._state_timestamps(rgb_timestamp_ns)

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
        observation["depth_timestamp_ns"] = depth_timestamp_ns
        observation["depth_width"] = width
        observation["depth_height"] = height
        observation["depth_encoding"] = "RS2_FORMAT_Z16"
        observation["depth_skew_ns"] = depth_timestamp_ns - rgb_timestamp_ns
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
