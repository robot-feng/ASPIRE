"""Evidence that a localized object rises with an arm and remains held."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cap_harness.contracts import LocalizationResult, Observation, Pose


@dataclass(frozen=True, slots=True)
class HoldEvidence:
    status: Literal["pending", "unknown", "not_lifted", "not_following", "verified"]
    reason: str
    timestamp_s: float | None = None
    object_lift_m: float | None = None
    tool_lift_m: float | None = None
    relative_error_m: float | None = None
    stable_duration_s: float = 0.0
    stable_samples: int = 0


class HoldVerifier:
    """Compare an object's observed motion with the tool over a bounded sample window.

    The baseline must be a synchronized, localized observation taken after the
    gripper closes and before lifting. The caller separately establishes that the
    object was on the table at the baseline and that each observation is current.
    ``verified`` means that the visible
    object rose with the tool and stayed at a consistent tool-relative position
    for at least three seconds, with no observation gap above the configured limit.
    A missing, ambiguous, stale, or unsynchronized localization resets the window.
    """

    def __init__(
        self,
        observation: Observation,
        localization: LocalizationResult,
        *,
        camera_name: str,
        arm: str,
        up_in_base: object,
        minimum_lift_m: float,
        max_relative_error_m: float,
        max_sample_gap_s: float,
        max_state_skew_s: float,
        stable_duration_s: float = 3.0,
        min_samples: int = 3,
    ) -> None:
        self.camera_name = camera_name
        self.arm = arm
        up = np.asarray(up_in_base, dtype=np.float64)
        length = float(np.linalg.norm(up)) if up.shape == (3,) else 0.0
        if (
            up.shape != (3,)
            or not np.isfinite(up).all()
            or not np.isfinite(length)
            or length <= 0.0
        ):
            raise ValueError("up_in_base must be a finite nonzero 3-vector")
        self.up = up / length
        self.minimum_lift_m = self._positive(minimum_lift_m, "minimum_lift_m")
        self.max_relative_error_m = self._positive(max_relative_error_m, "max_relative_error_m")
        self.max_sample_gap_s = self._positive(max_sample_gap_s, "max_sample_gap_s")
        self.max_state_skew_s = self._positive(max_state_skew_s, "max_state_skew_s")
        self.required_duration_s = self._positive(stable_duration_s, "stable_duration_s")
        if self.required_duration_s < 3.0:
            raise ValueError("stable_duration_s must be at least 3 seconds")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 3:
            raise ValueError("min_samples must be an integer of at least 3")
        self.min_samples = min_samples
        if self.max_sample_gap_s > self.required_duration_s / (self.min_samples - 1):
            raise ValueError("max_sample_gap_s is too large for the requested evidence window")
        stamp, obj, tool = self._sample(observation, localization)
        self._baseline_object = obj.copy()
        self._baseline_tool = tool.position.copy()
        self._baseline_relative = self._tool_relative(obj, tool)
        self._last_timestamp_s = stamp
        self._window_start_s: float | None = None
        self._window_last_s: float | None = None
        self._window_samples = 0

    def observe(
        self, observation: Observation, localization: LocalizationResult | None
    ) -> HoldEvidence:
        camera = (
            observation.cameras.get(self.camera_name)
            if isinstance(observation, Observation)
            else None
        )
        stamp = None if camera is None else camera.timestamp_s
        if stamp is not None:
            if stamp <= self._last_timestamp_s:
                self._reset_window()
                return HoldEvidence(status="unknown", reason="observation timestamp is not newer")
            self._last_timestamp_s = stamp
        try:
            stamp, obj, tool = self._sample(observation, localization)
        except (TypeError, ValueError) as exc:
            self._reset_window()
            return HoldEvidence(status="unknown", reason=str(exc))

        object_lift = float(np.dot(obj - self._baseline_object, self.up))
        tool_lift = float(np.dot(tool.position - self._baseline_tool, self.up))
        relative_error = float(
            np.linalg.norm(self._tool_relative(obj, tool) - self._baseline_relative)
        )
        values = {
            "timestamp_s": stamp,
            "object_lift_m": object_lift,
            "tool_lift_m": tool_lift,
            "relative_error_m": relative_error,
        }
        if tool_lift < self.minimum_lift_m:
            self._reset_window()
            return HoldEvidence(status="pending", reason="tool has not lifted far enough", **values)
        if object_lift < self.minimum_lift_m:
            self._reset_window()
            return HoldEvidence(status="not_lifted", reason="object did not rise", **values)
        if relative_error > self.max_relative_error_m:
            self._reset_window()
            return HoldEvidence(
                status="not_following", reason="object moved relative to the tool", **values
            )

        if self._window_last_s is None or stamp - self._window_last_s > self.max_sample_gap_s:
            self._window_start_s = stamp
            self._window_samples = 0
        self._window_last_s = stamp
        self._window_samples += 1
        assert self._window_start_s is not None
        duration = stamp - self._window_start_s
        verified = duration >= self.required_duration_s and self._window_samples >= self.min_samples
        return HoldEvidence(
            status="verified" if verified else "pending",
            reason="object followed the tool" if verified else "collecting hold evidence",
            stable_duration_s=duration,
            stable_samples=self._window_samples,
            **values,
        )

    def _sample(
        self, observation: Observation, localization: LocalizationResult | None
    ) -> tuple[float, np.ndarray, Pose]:
        if not isinstance(observation, Observation):
            raise ValueError("observation is missing")
        camera = observation.cameras.get(self.camera_name)
        state = observation.robot_state
        if camera is None or camera.timestamp_s is None or state.timestamp_s is None:
            raise ValueError("camera or robot state timestamp is missing")
        if abs(camera.timestamp_s - state.timestamp_s) > self.max_state_skew_s:
            raise ValueError("camera and robot state are not synchronized")
        if localization is None or not localization.ok or localization.geometry is None:
            raise ValueError("object localization is unavailable or ambiguous")
        if (
            localization.diagnostics.get("segmentation_source") != "provider"
            or localization.diagnostics.get("instance_count") != 1
            or localization.diagnostics.get("camera_name") != self.camera_name
        ):
            raise ValueError("object localization is unavailable or ambiguous")
        snapshot_id = f"{camera.frame}@{camera.timestamp_s:.9f}"
        if localization.diagnostics.get("snapshot_id") != snapshot_id:
            raise ValueError("localization does not belong to this camera snapshot")
        if localization.diagnostics.get("robot_state_timestamp_s") != state.timestamp_s:
            raise ValueError("localization does not belong to this robot state")
        tool = state.end_effector_poses.get(self.arm)
        if tool is None or tool.frame != state.base_frame:
            raise ValueError("tool pose is unavailable in the robot base frame")
        if localization.geometry.frame != state.base_frame:
            raise ValueError("object localization is not in the robot base frame")
        return camera.timestamp_s, localization.geometry.pose.position, tool

    @staticmethod
    def _tool_relative(point: np.ndarray, tool: Pose) -> np.ndarray:
        return tool.as_matrix()[:3, :3].T @ (point - tool.position)

    def _reset_window(self) -> None:
        self._window_start_s = None
        self._window_last_s = None
        self._window_samples = 0

    @staticmethod
    def _positive(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be positive and finite")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be positive and finite") from exc
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return number


__all__ = ["HoldEvidence", "HoldVerifier"]
