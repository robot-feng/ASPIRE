"""Bound position, velocity and timestamp samples from G1 DDS messages.

This module has no SDK or hardware dependency. Callbacks copy scalar data
directly from each message, avoiding the SDK's unlocked, position-only buffer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import threading
import time
from typing import Any

import numpy as np


class G1StateSamples:
    """Small, thread-safe per-topic history; missing samples stay missing."""

    def __init__(self, arm_names: tuple[str, ...], *, capacity: int = 500) -> None:
        if len(arm_names) != 14 or len(set(arm_names)) != 14 or capacity < 1:
            raise ValueError("expected 14 distinct arm joint names and a positive capacity")
        self._arm_names = tuple(arm_names)
        self._lock = threading.Lock()
        self._updated = threading.Condition(self._lock)
        self._feedback_failure: str | None = None
        self._groups = {
            name: deque(maxlen=capacity)
            for name in ("arm", "head", "waist", "gripper_left", "gripper_right")
        }
        self._side_groups = {side: deque(maxlen=capacity) for side in ("left", "right")}

    @staticmethod
    def _timestamp(message: Any) -> int:
        stamp = message.header.stamp
        sec, nsec = int(stamp.sec), int(stamp.nanosec)
        if sec < 0 or not 0 <= nsec < 1_000_000_000 or (sec == 0 and nsec == 0):
            raise ValueError("invalid DDS message timestamp")
        return sec * 1_000_000_000 + nsec

    def _append(self, group: str, timestamp: int, values: tuple[float, ...], velocity=None) -> None:
        if not np.isfinite(values).all() or (velocity is not None and not np.isfinite(velocity).all()):
            raise ValueError("state message contains non-finite values")
        with self._updated:
            self._groups[group].append((timestamp, values, velocity))
            self._updated.notify_all()

    def arm(self, message: Any) -> None:
        if tuple(message.name) != self._arm_names or len(message.position) != 14:
            raise ValueError(
                "arm message names or position dimension do not match the profile: "
                f"names={list(message.name)!r}, positions={len(message.position)}, "
                f"velocities={len(message.velocity)}"
            )
        values = tuple(float(x) for x in message.position)
        velocity = tuple(float(x) for x in message.velocity) if message.velocity else None
        self._append("arm", self._timestamp(message), values, velocity)

    def head(self, message: Any) -> None:
        if tuple(message.name) != ("joint_head_yaw", "joint_head_pitch") or len(message.motor_states) != 2:
            raise ValueError("head message names or dimension do not match the profile")
        values = tuple(float(motor.position) for motor in message.motor_states)
        self._append("head", self._timestamp(message), values)

    def waist(self, message: Any) -> None:
        names = tuple(message.name)
        expected = ("joint_body_pitch", "joint_lift_body")
        if len(names) != 2 or set(names) != set(expected) or len(message.motor_states) != 2:
            raise ValueError("waist message names or dimension do not match the profile")
        values = tuple(float(message.motor_states[names.index(name)].position) for name in expected)
        self._append("waist", self._timestamp(message), values)

    def gripper(self, side: str, message: Any) -> None:
        if side not in ("left", "right") or int(message.type) != 2 or len(message.end_state) != 1:
            raise ValueError("expected one G1 gripper endpoint state")
        value = float(message.end_state[0].position)
        if not np.isfinite(value) or not 0.0 <= value <= 120.0:
            raise ValueError("gripper feedback lies outside the SDK's 0..120 range")
        self._append(f"gripper_{side}", self._timestamp(message), (value / 120.0,))

    def arm_side(self, side: str, message: Any) -> None:
        """Retain source motor IDs, position and velocity without assigning joint order."""
        if side not in self._side_groups or len(message.motor_states) != 7:
            raise ValueError("arm feedback must identify one side with seven motors")
        motors = tuple(message.motor_states)
        ids = tuple(int(motor.id) for motor in motors)
        positions = tuple(float(motor.position) for motor in motors)
        velocities = tuple(float(motor.velocity) for motor in motors)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("arm feedback contains non-finite values")
        sample = (self._timestamp(message), ids, positions, velocities)
        with self._updated:
            self._side_groups[side].append(sample)
            self._updated.notify_all()

    def wait_next_right_arm(self, after_ns: int, timeout_s: float):
        """Wait for a newer raw right-arm DDS sample without acquiring a camera."""
        return self._wait_next(self._side_groups["right"], after_ns, timeout_s)

    def wait_next_right_gripper(self, after_ns: int, timeout_s: float):
        """Wait for a newer native-normalized right-gripper DDS sample."""
        return self._wait_next(self._groups["gripper_right"], after_ns, timeout_s)

    def publish_bound_right_feedback(
        self, *, gripper_timestamp_ns: int, gripper_native: float,
        arm_feedback: Mapping[str, object] | None,
    ) -> None:
        """Copy already-bound feedback from one raw proxy snapshot, without inference."""
        if isinstance(gripper_timestamp_ns, bool) or not isinstance(gripper_timestamp_ns, int) or gripper_timestamp_ns <= 0:
            raise ValueError("gripper feedback requires a positive source timestamp")
        value = float(gripper_native)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("gripper feedback must be native-normalized in [0, 1]")
        grip = (gripper_timestamp_ns, (value,), None)
        arm = None
        if arm_feedback is not None:
            if not isinstance(arm_feedback, Mapping):
                raise ValueError("right-arm feedback must be a mapping")
            stamp = arm_feedback.get("timestamp_ns")
            ids = arm_feedback.get("motor_ids")
            positions = arm_feedback.get("positions_raw")
            velocities = arm_feedback.get("velocities_raw")
            if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
                raise ValueError("right-arm feedback requires a positive source timestamp")
            if not isinstance(ids, (list, tuple)) or len(ids) != 7 or any(
                isinstance(item, bool) or not isinstance(item, int) for item in ids
            ) or len(set(ids)) != 7:
                raise ValueError("right-arm feedback requires seven unique motor IDs")
            positions = np.asarray(positions)
            velocities = np.asarray(velocities)
            if positions.shape != (7,) or velocities.shape != (7,) or positions.dtype.kind not in "iuf" or velocities.dtype.kind not in "iuf" or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise ValueError("right-arm feedback requires seven finite positions and velocities")
            arm = (stamp, tuple(ids), tuple(float(x) for x in positions), tuple(float(x) for x in velocities))
        with self._updated:
            if self._feedback_failure is not None:
                raise RuntimeError(self._feedback_failure)
            additions = [(self._groups["gripper_right"], grip)]
            if arm is not None:
                additions.append((self._side_groups["right"], arm))
            for group, sample in additions:
                if group and (sample[0] < group[-1][0] or (sample[0] == group[-1][0] and sample != group[-1])):
                    raise ValueError("feedback source timestamp regressed or changed contents")
            for group, sample in additions:
                if not group or sample[0] != group[-1][0]:
                    group.append(sample)
            self._updated.notify_all()

    def fail_feedback(self, reason: str) -> None:
        """Wake feedback waiters after the sole broadcaster fails or closes."""
        with self._updated:
            if self._feedback_failure is None:
                self._feedback_failure = str(reason)
            self._updated.notify_all()

    def _wait_next(self, group, after_ns: int, timeout_s: float):
        if isinstance(after_ns, bool) or not isinstance(after_ns, int) or after_ns < 0:
            raise ValueError("after_ns must be a nonnegative integer")
        if isinstance(timeout_s, (bool, np.bool_)):
            raise ValueError("timeout_s must be finite and nonnegative")
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        with self._updated:
            while True:
                if self._feedback_failure is not None:
                    raise RuntimeError(self._feedback_failure)
                newer = [sample for sample in group if sample[0] > after_ns]
                if newer:
                    return min(newer, key=lambda sample: sample[0])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._updated.wait(remaining)

    def nearest_sides(self, timestamp_ns: int) -> dict[str, dict[str, object]]:
        """Return per-side raw feedback; it is not merged into the CAP state."""
        with self._lock:
            chosen = {
                side: min(group, key=lambda sample: abs(sample[0] - timestamp_ns))
                for side, group in self._side_groups.items() if group
            }
        return {
            side: {"timestamp_ns": sample[0], "motor_ids": list(sample[1]),
                   "positions_raw": list(sample[2]), "velocities_raw": list(sample[3])}
            for side, sample in chosen.items()
        }

    def nearest(self, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, int]] | None:
        """Return 20 positions and optional 14 arm velocities from bound messages."""
        with self._lock:
            if any(not group for group in self._groups.values()):
                return None
            selected = {
                name: min(group, key=lambda sample: abs(sample[0] - timestamp_ns))
                for name, group in self._groups.items()
            }
        state = np.asarray(
            selected["arm"][1] + selected["gripper_left"][1] + selected["gripper_right"][1]
            + selected["head"][1] + selected["waist"][1], dtype=np.float32,
        )
        arm_velocity = selected["arm"][2]
        velocity = None if arm_velocity is None else np.asarray(arm_velocity, dtype=np.float32)
        return state, velocity, {name: sample[0] for name, sample in selected.items()}
