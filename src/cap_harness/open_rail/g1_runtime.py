"""Open-RAIL G1 raw observation and scoped command boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import threading
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from cap_harness.open_rail.g1_control import (
    G1CommandCapabilities,
    G1CommandFeedback,
    OpenRailG1CommandRuntime,
    OpenRailG1ProxyCommandTransport,
    proxy_motion_session_state,
)
from cap_harness.open_rail.profile import (
    G1ObservationProfile,
    default_g1_observation_profile_path,
    load_g1_observation_profile,
)


class OpenRailG1ObservationError(RuntimeError):
    """Open-RAIL G1 observation or configuration failure."""


def _plain(value: Any) -> Any:
    """Convert immutable profile mappings to ordinary artifact metadata."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class OpenRailSnapshot:
    """One raw observation, before unverified calibration is applied."""

    rgb: np.ndarray
    depth_u16: np.ndarray | None
    state: np.ndarray
    rgb_timestamp_ns: int
    depth_timestamp_ns: int | None
    state_timestamps_ns: dict[str, int | None]
    depth_error: str | None
    metadata: dict[str, Any]

    @property
    def depth_skew_ns(self) -> int | None:
        if self.depth_timestamp_ns is None:
            return None
        return self.depth_timestamp_ns - self.rgb_timestamp_ns

    @property
    def is_synchronized(self) -> bool:
        threshold_ms = self.metadata.get("maximum_rgb_depth_skew_ms")
        if threshold_ms is None or self.depth_timestamp_ns is None:
            return False
        if not self.state_timestamps_ns or any(
            value is None for value in self.state_timestamps_ns.values()
        ):
            return False
        maximum_skew_ns = max(
            abs(self.depth_timestamp_ns - self.rgb_timestamp_ns),
            *(abs(int(value) - self.rgb_timestamp_ns) for value in self.state_timestamps_ns.values()),
        )
        return maximum_skew_ns <= float(threshold_ms) * 1_000_000


class OpenRailG1ObservationRuntime:
    """Own one G1 Open-RAIL worker and read raw camera/state observations.

    Open-RAIL's worker and A2D DDS subscriptions are the sole robot/camera
    owner. The returned snapshot preserves native timestamps and raw Z16 values;
    it is not converted to ASPIRE's calibrated ``CameraObservation`` until the
    missing depth scale, registration and camera/base transform are verified.
    ``command_runtime`` supplies a scoped facade, interlocked by default.
    """

    def __init__(
        self,
        *,
        open_rail_root: str | Path,
        profile: G1ObservationProfile | str | Path | None = None,
        maximum_rgb_depth_skew_ms: float | None = None,
        allow_physical_motion: bool = False,
        selected_arm: str | None = None,
        calibration_camera_keys: Sequence[str] = (),
        proxy: Any | None = None,
    ) -> None:
        self.open_rail_root = Path(open_rail_root).resolve()
        if profile is None:
            profile = default_g1_observation_profile_path()
        self.profile = (
            load_g1_observation_profile(profile)
            if isinstance(profile, str | Path)
            else profile
        )
        if self.profile.rgb_channel != f"cam.{self.profile.public_camera_key}":
            raise ValueError("rgb_channel must match cam.<public_camera_key>")
        if (
            not self.profile.depth_channel.startswith("cam.")
            or not self.profile.depth_channel.removeprefix("cam.")
            or self.profile.depth_channel == self.profile.rgb_channel
        ):
            raise ValueError("depth_channel must name a distinct cam.<key>")
        self.maximum_rgb_depth_skew_ms = maximum_rgb_depth_skew_ms
        if type(allow_physical_motion) is not bool:
            raise ValueError("allow_physical_motion must be a bool")
        configured_selected_arm = self.profile.selected_arm
        if selected_arm is not None and selected_arm != configured_selected_arm:
            raise ValueError(
                "selected_arm must match the selected arm in the Open-RAIL profile"
            )
        if configured_selected_arm != "right":
            raise ValueError("this G1 runtime only supports selected_arm='right'")
        if isinstance(calibration_camera_keys, (str, bytes)):
            raise ValueError("calibration_camera_keys must be a sequence of camera keys")
        camera_keys = tuple(calibration_camera_keys)
        available_channels = self.profile.camera.get("calibration_channels", {})
        if not isinstance(available_channels, Mapping):
            raise ValueError("profile camera.calibration_channels must be a mapping")
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError("calibration_camera_keys must be unique")
        for key in camera_keys:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("calibration_camera_keys must contain non-empty strings")
            if key not in available_channels:
                raise ValueError(f"camera {key!r} is not enabled by this G1 profile")
            channel = available_channels[key]
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError(f"profile camera channel for {key!r} must be non-empty")
        self.calibration_camera_keys = tuple(key.strip() for key in camera_keys)
        self.allow_physical_motion = allow_physical_motion
        self.selected_arm = configured_selected_arm
        self._last_snapshot: OpenRailSnapshot | None = None
        self._command_runtime: OpenRailG1CommandRuntime | None = None
        if maximum_rgb_depth_skew_ms is not None and maximum_rgb_depth_skew_ms <= 0:
            raise ValueError("maximum_rgb_depth_skew_ms must be positive")
        self._closed = False
        self._proxy = proxy or self._create_proxy()
        self._motion_session_state = proxy_motion_session_state(self._proxy)
        self._track_proxy_shared_memory()

    def _create_proxy(self) -> Any:
        if not self.open_rail_root.is_dir():
            raise OpenRailG1ObservationError(f"Open-RAIL checkout not found: {self.open_rail_root}")
        root_text = str(self.open_rail_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        pythonpath = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if root_text not in pythonpath:
            os.environ["PYTHONPATH"] = os.pathsep.join([root_text, *filter(None, pythonpath)])

        try:
            robot_proxy = importlib.import_module("client.robots.robot_proxy").RobotProxy
            module_name, factory_name = self.profile.config_factory.split(":", 1)
            config_factory = getattr(importlib.import_module(module_name), factory_name)
        except (ImportError, AttributeError) as exc:
            raise OpenRailG1ObservationError("Open-RAIL G1 dependencies are not importable") from exc

        config = config_factory()
        config.camera.ref = self.profile.public_camera_key
        config.camera.names = {
            self.profile.public_camera_key: self.profile.rgb_channel.removeprefix("cam."),
            self.profile.depth_channel.removeprefix("cam."): self.profile.depth_channel.removeprefix("cam."),
        }
        calibration_channels = self.profile.camera.get("calibration_channels", {})
        for key in self.calibration_camera_keys:
            channel = calibration_channels[key]
            if key in config.camera.names or channel in config.camera.names.values():
                raise OpenRailG1ObservationError(
                    f"calibration camera {key!r} duplicates an existing camera mapping"
                )
            config.camera.names[key] = channel
        config.camera.calibration_camera_keys = list(self.calibration_camera_keys)
        config.camera.depth_ref = self.profile.depth_channel.removeprefix("cam.")
        # G1RobotBody returns RGB only after checking packet encoding and
        # color_format. Keep downstream Open-RAIL image encoding aligned.
        config.camera.color_order = "rgb"
        config.allow_physical_motion = self.allow_physical_motion
        return robot_proxy(
            robot_type=self.profile.robot_type,
            config=config,
            robot_module=self.profile.robot_module,
        )

    def _track_proxy_shared_memory(self) -> None:
        """Track all ring slots from Open-RAIL's FramePacket before shutdown.

        The pinned RobotProxy version's ``_track_remote_shm`` handles the old
        descriptor dict format but not ``FramePacket``. If its worker has to be
        terminated, the parent must still know every shared-memory slot so it
        can unlink them.
        """
        proxy = self._proxy
        original = getattr(proxy, "_track_remote_shm", None)
        if not callable(original) or not hasattr(proxy, "_remote_shm_names"):
            return

        def track(observation: Any) -> None:
            cameras = getattr(observation, "camera_metadata", None)
            if cameras is None:
                original(observation)
                return
            for camera in cameras:
                for shared_memory in getattr(camera, "shm", ()):
                    name = getattr(shared_memory, "name", None)
                    if name:
                        proxy._remote_shm_names.add(name)

        proxy._track_remote_shm = track

    def read_snapshot(
        self, *, timeout_s: float = 5.0, require_depth: bool = False
    ) -> OpenRailSnapshot:
        if self._closed:
            raise OpenRailG1ObservationError("G1 observation runtime is closed")
        if type(require_depth) is not bool:
            raise ValueError("require_depth must be a bool")
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_s must be positive and finite") from exc
        if isinstance(timeout_s, bool) or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be positive and finite")
        deadline = time.monotonic() + timeout
        while True:
            observation = self._proxy.retrieve_observation()
            if observation is not None:
                snapshot = self._decode(observation)
                self._last_snapshot = snapshot
                if not require_depth or snapshot.depth_u16 is not None:
                    return snapshot
            if time.monotonic() >= deadline:
                needed = "RGB-D frame" if require_depth else "frame"
                raise TimeoutError(f"no new Open-RAIL G1 {needed} within {timeout:.1f}s")
            threading.Event().wait(0.005)

    def capture_calibration_packets(self, *, timeout_s: float = 5.0) -> list[dict[str, Any]]:
        """Capture raw head RGB plus selected fisheye packets from this worker."""
        if self._closed:
            raise OpenRailG1ObservationError("G1 observation runtime is closed")
        if self.allow_physical_motion:
            raise OpenRailG1ObservationError(
                "camera calibration capture requires a read-only runtime"
            )
        if not self.calibration_camera_keys:
            raise OpenRailG1ObservationError(
                "select at least one calibration_camera_key when creating the runtime"
            )
        snapshot = self.read_snapshot(timeout_s=timeout_s)
        capture = getattr(self._proxy, "capture_calibration_packets", None)
        if not callable(capture):
            raise OpenRailG1ObservationError(
                "Open-RAIL worker does not expose raw calibration packet capture"
            )
        camera_keys = [self.profile.public_camera_key, *self.calibration_camera_keys]
        return capture(camera_keys, snapshot.rgb_timestamp_ns)

    def command_runtime(
        self,
        *,
        feedback: G1CommandFeedback | None = None,
        capabilities: G1CommandCapabilities | None = None,
        position_tolerance_rad: float | None = None,
        velocity_tolerance_rad_s: float | None = None,
        arm_joint_limits_rad: tuple[tuple[float, float], ...] | None = None,
        gripper_tolerance: float | None = None,
    ) -> OpenRailG1CommandRuntime:
        """Return a scoped command facade over this existing worker only."""
        if self._closed:
            raise OpenRailG1ObservationError("G1 observation runtime is closed")
        self._command_runtime = OpenRailG1CommandRuntime(
            OpenRailG1ProxyCommandTransport(self._proxy),
            feedback=feedback,
            capabilities=capabilities,
            allow_physical_motion=self.allow_physical_motion,
            selected_arm=self.selected_arm,
            position_tolerance_rad=position_tolerance_rad,
            velocity_tolerance_rad_s=velocity_tolerance_rad_s,
            arm_joint_limits_rad=arm_joint_limits_rad,
            gripper_tolerance=gripper_tolerance,
            session_state=self._motion_session_state,
        )
        return self._command_runtime

    @property
    def health(self) -> MappingProxyType:
        """Read-only profile, camera, calibration, and worker health evidence."""
        process = getattr(self._proxy, "_process", None)
        alive_fn = getattr(process, "is_alive", None)
        worker_alive = None if not callable(alive_fn) else bool(alive_fn())
        calibration = dict(self.profile.calibration)
        camera = dict(self.profile.camera)
        arm_profiles = self.profile.arms
        control_profile = self.profile.control
        selected_arm_profile = arm_profiles.get(self.selected_arm)
        if not isinstance(selected_arm_profile, Mapping):
            selected_arm_profile = MappingProxyType({})
        camera_calibration = MappingProxyType(
            {
                "intrinsics_file": calibration.get("camera_intrinsics")
                or camera.get("color_intrinsics_file"),
                "intrinsics_values": camera.get("camera_intrinsics"),
                "intrinsics_resolution": camera.get("intrinsics_resolution"),
                "extrinsics_file": calibration.get("camera_extrinsics")
                or camera.get("camera_extrinsics_file"),
                "extrinsics_semantics": camera.get("camera_extrinsics_semantics"),
                "depth_scale_m_per_unit": camera.get("depth_scale_m_per_unit")
                if camera.get("depth_scale_m_per_unit") is not None
                else calibration.get("depth_scale_m_per_unit"),
                "depth_registered_to_rgb": camera.get("depth_registered_to_rgb")
                if camera.get("depth_registered_to_rgb") is not None
                else calibration.get("depth_registered_to_rgb"),
                "depth_to_rgb_calibration": camera.get("depth_to_rgb_calibration")
                if camera.get("depth_to_rgb_calibration") is not None
                else calibration.get("depth_to_rgb_calibration"),
            }
        )
        tcp_transform = selected_arm_profile.get("tcp_transform")
        if tcp_transform is None:
            tcp_transform = self.profile.gripper.get("tcp_transform")
        calibration_values = {
            "camera_intrinsics_values": camera_calibration["intrinsics_values"],
            "camera_intrinsics_resolution": camera_calibration["intrinsics_resolution"],
            "camera_extrinsics_semantics": camera_calibration["extrinsics_semantics"],
            "depth_scale_m_per_unit": camera_calibration["depth_scale_m_per_unit"],
            "depth_registered_to_rgb": camera_calibration["depth_registered_to_rgb"],
            "depth_to_rgb_calibration": camera_calibration["depth_to_rgb_calibration"],
            "base_frame": calibration.get("base_frame"),
            "base_pose": calibration.get("base_pose"),
            "arm_tcp_transform": tcp_transform,
        }
        calibration_gaps = tuple(
            key for key, value in calibration_values.items() if value is None
        )
        calibration_verified = calibration.get("verified") is True
        if not calibration_verified:
            calibration_gaps += ("calibration_verification",)
        snapshot = self._last_snapshot
        command_health = (
            self._command_runtime.health
            if self._command_runtime is not None
            else MappingProxyType(
                {
                    "selected_arm": self.selected_arm,
                    "allowed_commands": ("right_arm_joint_position", "right_gripper_position"),
                    "physical_motion_authorized": self.allow_physical_motion,
                    "motion_ready": False,
                    "blockers": ("command feedback/capabilities have not been configured",),
                    "global_motion_stop_used": False,
                    "motion_session": self._motion_session_state.health,
                }
            )
        )
        return MappingProxyType(
            {
                "profile": self.profile.name,
                "robot_type": self.profile.robot_type,
                "proxy_worker_alive": worker_alive,
                "runtime_closed": self._closed,
                "selected_arm": self.selected_arm,
                "public_camera_key": self.profile.public_camera_key,
                "control_constraints": MappingProxyType(
                    {
                        "selected_arm": self.selected_arm,
                        "allowed_commands": tuple(control_profile["allowed_commands"]),
                        "read_only_parts": tuple(control_profile["read_only_parts"]),
                        "right_arm_group": "right_arm",
                        "right_gripper_topic": "/wbc/right_ee_command",
                    }
                ),
                "selected_arm_profile": selected_arm_profile,
                "arm_tcp_transform": tcp_transform,
                "camera_calibration": camera_calibration,
                "raw_snapshot_available": snapshot is not None,
                "depth_available": snapshot is not None and snapshot.depth_u16 is not None,
                "calibration_complete": calibration_verified and not calibration_gaps,
                "calibration_gaps": calibration_gaps,
                "calibration": MappingProxyType(calibration),
                "motion": command_health,
            }
        )

    def _decode(self, observation: dict[str, Any]) -> OpenRailSnapshot:
        rgb_error = observation.get("rgb_error")
        if rgb_error is not None:
            raise OpenRailG1ObservationError(f"RGB packet rejected: {rgb_error}")
        rgb_timestamp = observation.get("rgb_timestamp_ns")
        if rgb_timestamp is None:
            rgb_timestamp = observation.get("ref_timestamp")
        if rgb_timestamp is None:
            raise OpenRailG1ObservationError("observation has no RGB source timestamp")
        rgb_timestamp = int(rgb_timestamp)
        rgb_packet_timestamp = observation.get("rgb_packet_timestamp_ns")
        if rgb_packet_timestamp is None or int(rgb_packet_timestamp) != rgb_timestamp:
            raise OpenRailG1ObservationError(
                "RGB packet timestamp does not match the decoded frame timestamp"
            )
        rgb_encoding = observation.get("rgb_encoding_format")
        rgb_color_format = observation.get("rgb_color_format")
        rgb_decoder = observation.get("rgb_decoder")
        valid_rgb_decoders = {
            ("JPEG", "opencv-imdecode-bgr-to-rgb"),
            ("PNG", "opencv-imdecode-bgr-to-rgb"),
            ("UNCOMPRESSED", "raw-rgb"),
            ("UNCOMPRESSED", "raw-bgr-to-rgb"),
        }
        if (
            observation.get("rgb_order") != "RGB"
            or rgb_color_format not in {"RGB", "BGR"}
            or (rgb_encoding, rgb_decoder) not in valid_rgb_decoders
        ):
            raise OpenRailG1ObservationError(
                "RGB packet encoding/color_format has no verified RGB decoder"
            )
        rgb = np.asarray(observation.get(self.profile.rgb_channel))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise OpenRailG1ObservationError(f"invalid RGB frame: shape={rgb.shape}, dtype={rgb.dtype}")
        state = np.asarray(observation.get("obs.state"), dtype=np.float32).reshape(-1)
        if state.shape != (self.profile.state_dimension,) or not np.all(np.isfinite(state)):
            raise OpenRailG1ObservationError(
                f"expected {self.profile.state_dimension} finite RAIL state values, got {state.shape}"
            )

        depth_value = observation.get(self.profile.depth_channel)
        depth = None
        if depth_value is not None:
            candidate = np.asarray(depth_value)
            if candidate.ndim == 3 and candidate.shape[-1] == 1:
                candidate = candidate[..., 0]
            if candidate.ndim != 2 or candidate.dtype != np.uint16:
                raise OpenRailG1ObservationError(
                    f"invalid raw Z16 frame: shape={candidate.shape}, dtype={candidate.dtype}"
                )
            depth = np.array(candidate, copy=True)

        state_stamps = {
            str(name): None if stamp is None else int(stamp)
            for name, stamp in (observation.get("state_timestamps_ns") or {}).items()
        }
        raw_depth_stamp = observation.get("depth_timestamp_ns")
        depth_timestamp = None if raw_depth_stamp is None else int(raw_depth_stamp)
        calibration = _plain(self.profile.calibration)
        metadata = {
            "profile": self.profile.name,
            "robot_type": self.profile.robot_type,
            "public_camera_key": self.profile.public_camera_key,
            "rgb_order": str(observation.get("rgb_order", "unknown")),
            "rgb_encoding_format": rgb_encoding,
            "rgb_color_format": rgb_color_format,
            "rgb_decoder": rgb_decoder,
            "rgb_packet_timestamp_ns": int(rgb_packet_timestamp),
            "rgb_shape": list(rgb.shape),
            "depth_shape": None if depth is None else list(depth.shape),
            "depth_encoding": observation.get("depth_encoding"),
            "depth_encoding_format": observation.get("depth_encoding_format"),
            "depth_color_format": observation.get("depth_color_format"),
            "depth_error": observation.get("depth_error"),
            "depth_scale_m_per_unit": calibration.get("depth_scale_m_per_unit"),
            "depth_registered_to_rgb": calibration.get("depth_registered_to_rgb"),
            "state_layout": dict(self.profile.state_layout),
            # False by default: a body may set this only when the state values
            # and their per-group timestamps came from the same DDS samples.
            "state_timestamps_bind_state_values": (
                observation.get("state_timestamps_bind_state_values") is True
            ),
            "calibration": calibration,
            "maximum_rgb_depth_skew_ms": self.maximum_rgb_depth_skew_ms,
        }
        rgb = np.array(rgb, copy=True)
        state = np.array(state, copy=True)
        rgb.setflags(write=False)
        state.setflags(write=False)
        if depth is not None:
            depth.setflags(write=False)
        return OpenRailSnapshot(
            rgb=rgb,
            depth_u16=depth,
            state=state,
            rgb_timestamp_ns=rgb_timestamp,
            depth_timestamp_ns=depth_timestamp,
            state_timestamps_ns=state_stamps,
            depth_error=observation.get("depth_error"),
            metadata=metadata,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stop_observe = getattr(self._proxy, "stop_observe", None)
        if callable(stop_observe):
            try:
                stop_observe()
            except Exception:
                # Closing still sends the worker stop command if observation was
                # already stopped or its read loop has failed.
                pass
        self._proxy.close()

    def __enter__(self) -> OpenRailG1ObservationRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


G1Snapshot = OpenRailSnapshot

__all__ = [
    "G1Snapshot",
    "OpenRailG1ObservationError",
    "OpenRailG1ObservationRuntime",
    "OpenRailSnapshot",
]
