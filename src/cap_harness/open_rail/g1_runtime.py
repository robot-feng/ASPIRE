"""Read-only Open-RAIL embodiment adapter and raw observation boundary."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

from cap_harness.open_rail.profile import (
    G1ObservationProfile,
    default_g1_observation_profile_path,
    load_g1_observation_profile,
)


class OpenRailG1ObservationError(RuntimeError):
    """Open-RAIL G1 observation or configuration failure."""


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

    This initial runtime has no actuation methods by design. Open-RAIL's worker
    and A2D DDS subscriptions are the sole robot/camera owner. The returned
    snapshot preserves native timestamps and raw Z16 values; it is not converted
    to ASPIRE's calibrated ``CameraObservation`` until the missing depth scale,
    registration and camera/base transform have been verified.
    """

    def __init__(
        self,
        *,
        open_rail_root: str | Path,
        profile: G1ObservationProfile | str | Path | None = None,
        maximum_rgb_depth_skew_ms: float | None = None,
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
        self.maximum_rgb_depth_skew_ms = maximum_rgb_depth_skew_ms
        if maximum_rgb_depth_skew_ms is not None and maximum_rgb_depth_skew_ms <= 0:
            raise ValueError("maximum_rgb_depth_skew_ms must be positive")
        self._closed = False
        self._proxy = proxy or self._create_proxy()
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

    def read_snapshot(self, *, timeout_s: float = 5.0) -> OpenRailSnapshot:
        if self._closed:
            raise OpenRailG1ObservationError("G1 observation runtime is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        while True:
            observation = self._proxy.retrieve_observation()
            if observation is not None:
                return self._decode(observation)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no new Open-RAIL G1 frame within {timeout_s:.1f}s")
            threading.Event().wait(0.005)

    def _decode(self, observation: dict[str, Any]) -> OpenRailSnapshot:
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
        raw_rgb_timestamp = observation.get("rgb_timestamp_ns")
        if raw_rgb_timestamp is None:
            raw_rgb_timestamp = observation.get("ref_timestamp")
        if raw_rgb_timestamp is None:
            raise OpenRailG1ObservationError("observation has no RGB source timestamp")
        rgb_timestamp = int(raw_rgb_timestamp)
        raw_depth_stamp = observation.get("depth_timestamp_ns")
        depth_timestamp = None if raw_depth_stamp is None else int(raw_depth_stamp)
        calibration = dict(self.profile.calibration)
        metadata = {
            "profile": self.profile.name,
            "robot_type": self.profile.robot_type,
            "public_camera_key": self.profile.public_camera_key,
            "rgb_order": str(observation.get("rgb_order", "unknown")),
            "rgb_shape": list(rgb.shape),
            "depth_shape": None if depth is None else list(depth.shape),
            "depth_encoding": observation.get("depth_encoding"),
            "depth_error": observation.get("depth_error"),
            "depth_scale_m_per_unit": calibration.get("depth_scale_m_per_unit"),
            "depth_registered_to_rgb": calibration.get("depth_registered_to_rgb"),
            "state_layout": dict(self.profile.state_layout),
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
