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
        config.camera.ref = "head"
        config.camera.names = {"head": "head", "head_depth": "head_depth"}
        super().__init__(config)
        # The upstream receiver can return the raw Z16 image, but its common
        # image path assumes every secondary camera is three-channel color.
        # Read depth by packet below so it remains uint16 and is not color-flipped.
        self._non_ref_cameras = []

    def retrieve_observation(self) -> dict[str, Any] | None:
        observation = super().retrieve_observation()
        if observation is None:
            return None

        rgb_timestamp_ns = int(observation["ref_timestamp"])
        observation["rgb_timestamp_ns"] = rgb_timestamp_ns
        observation["rgb_order"] = "RGB"
        observation["state_timestamps_ns"] = self._state_timestamps(rgb_timestamp_ns)

        packet = self.camera.get_packet_nearest("head_depth", rgb_timestamp_ns)
        if packet is None:
            observation["depth_timestamp_ns"] = None
            observation["depth_error"] = "no head depth packet near the RGB timestamp"
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
        observation["cam.head_depth"] = depth
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
