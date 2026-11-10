"""Measured right-arm and gripper completion from bound G1 DDS samples.

No subscriptions or control connections are created here. The supplied sample
buffer belongs to the existing G1 observation worker. These checks establish
only a measured postcondition; they do not verify stop, lease or cancel behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import time

import numpy as np

from cap_harness.open_rail.g1_state_samples import G1StateSamples


@dataclass(frozen=True, slots=True)
class G1FeedbackCalibration:
    """Verified feedback identity, units, clock and temporal tolerances.

    ``right_motor_ids`` is in the exact command joint order. Gripper feedback
    must already be normalized in the native Open-RAIL command convention.
    These declarations need physical verification; no defaults are provided.
    """

    right_motor_ids: Sequence[int]
    right_motor_order_verified: bool
    arm_position_velocity_units_verified: bool
    gripper_feedback_convention_verified: bool
    feedback_clock_verified: bool
    max_sample_age_s: float
    max_sample_gap_s: float
    stable_duration_s: float
    minimum_samples: int

    def __post_init__(self) -> None:
        ids = tuple(self.right_motor_ids)
        if (
            len(ids) != 7
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0 for x in ids)
            or len(set(ids)) != 7
        ):
            raise ValueError("right_motor_ids must contain seven unique nonnegative integers")
        for name in (
            "right_motor_order_verified", "arm_position_velocity_units_verified",
            "gripper_feedback_convention_verified", "feedback_clock_verified",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"feedback requires verified {name}")
        for name in ("max_sample_age_s", "max_sample_gap_s", "stable_duration_s"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if isinstance(self.minimum_samples, bool) or not isinstance(self.minimum_samples, int) or self.minimum_samples < 2:
            raise ValueError("minimum_samples must be an integer of at least 2")
        if self.max_sample_gap_s > self.stable_duration_s / (self.minimum_samples - 1):
            raise ValueError("max_sample_gap_s is too large for the stable feedback window")
        object.__setattr__(self, "right_motor_ids", ids)


class G1MeasuredCommandFeedback:
    """Wait on bound state messages for a post-command stable measurement.

    Each wait begins with a current timestamp from a verified shared clock,
    after the transport has returned. Only strictly newer samples count. A
    sample from the future, malformed IDs or a clock regression is an error;
    stale, discontinuous, or out-of-tolerance samples reset the stable window.
    Timeout never implies cancellation or a stopped robot. This class must run
    where its existing ``G1StateSamples`` receives DDS callbacks.
    """

    def __init__(
        self,
        samples: G1StateSamples,
        calibration: G1FeedbackCalibration,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(samples, G1StateSamples) or not isinstance(calibration, G1FeedbackCalibration):
            raise ValueError("bound G1 samples and verified feedback calibration are required")
        if not callable(clock_ns) or not callable(monotonic):
            raise ValueError("feedback clocks must be callable")
        self._samples = samples
        self._calibration = calibration
        self._clock_ns = clock_ns
        self._monotonic = monotonic
        self._last_evidence: dict[str, object] | None = None

    @property
    def last_evidence(self) -> dict[str, object] | None:
        """Latest confirmed measurements, or None after a failure or timeout."""
        return None if self._last_evidence is None else dict(self._last_evidence)

    def wait_for_right_arm(
        self, target_rad: np.ndarray, *, timeout_s: float,
        position_tolerance_rad: float, velocity_tolerance_rad_s: float,
    ) -> bool:
        self._last_evidence = None
        target = np.asarray(target_rad, dtype=np.float64)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("arm target must have seven finite positions")
        position_tolerance = self._positive(position_tolerance_rad, "position_tolerance_rad")
        velocity_tolerance = self._positive(velocity_tolerance_rad_s, "velocity_tolerance_rad_s")

        def matches(sample) -> bool:
            if len(sample) != 4:
                raise ValueError("right-arm feedback sample is malformed")
            _, ids, positions, velocities = sample
            if len(ids) != 7 or len(set(ids)) != 7 or set(ids) != set(self._calibration.right_motor_ids):
                raise ValueError("right-arm feedback IDs do not match the verified mapping")
            positions = np.asarray(positions, dtype=np.float64)
            velocities = np.asarray(velocities, dtype=np.float64)
            if positions.shape != (7,) or velocities.shape != (7,) or not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise ValueError("right-arm feedback values are malformed")
            order = [ids.index(motor_id) for motor_id in self._calibration.right_motor_ids]
            return bool(
                np.all(np.abs(positions[order] - target) <= position_tolerance)
                and np.all(np.abs(velocities[order]) <= velocity_tolerance)
            )

        return self._wait("right_arm", self._samples.wait_next_right_arm, timeout_s, matches)

    def wait_for_right_gripper(self, target: float, *, timeout_s: float, tolerance: float) -> bool:
        self._last_evidence = None
        if isinstance(target, (bool, np.bool_)) or not math.isfinite(float(target)) or not 0 <= float(target) <= 1:
            raise ValueError("gripper target must be in [0, 1]")
        target = float(target)
        tolerance = self._positive(tolerance, "tolerance")

        def matches(sample) -> bool:
            if len(sample) != 3 or len(sample[1]) != 1:
                raise ValueError("right-gripper feedback sample is malformed")
            value = float(sample[1][0])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("right-gripper feedback is outside [0, 1]")
            return abs(value - target) <= tolerance

        return self._wait("right_gripper", self._samples.wait_next_right_gripper, timeout_s, matches)

    def _wait(self, stream: str, read_next, timeout_s: float, matches) -> bool:
        self._last_evidence = None
        timeout = self._positive(timeout_s, "timeout_s")
        barrier = self._now()
        previous_clock = barrier
        cursor = barrier
        deadline = self._monotonic() + timeout
        start = last = None
        count = 0
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return False
            sample = read_next(cursor, remaining)
            if sample is None or self._monotonic() >= deadline:
                return False
            if not isinstance(sample, tuple) or not sample or isinstance(sample[0], bool) or not isinstance(sample[0], int):
                raise ValueError("feedback sample has an invalid timestamp")
            stamp = sample[0]
            if stamp <= cursor:
                raise ValueError("feedback timestamp is not strictly increasing")
            cursor = stamp
            now = self._now()
            if now < previous_clock or stamp > now:
                raise ValueError("feedback clock regressed or sample is in the future")
            previous_clock = now
            if now - stamp > self._calibration.max_sample_age_s * 1e9 or not matches(sample):
                start = last = None
                count = 0
                continue
            if last is None or stamp - last > self._calibration.max_sample_gap_s * 1e9:
                start, count = stamp, 0
            last = stamp
            count += 1
            if count >= self._calibration.minimum_samples and stamp - start >= self._calibration.stable_duration_s * 1e9:
                self._last_evidence = {
                    "stream": stream,
                    "command_barrier_ns": barrier,
                    "stable_start_ns": start,
                    "terminal_timestamp_ns": stamp,
                    "stable_samples": count,
                    "terminal_sample": sample,
                }
                return True

    def _now(self) -> int:
        value = self._clock_ns()
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("shared feedback clock must return a positive integer nanosecond timestamp")
        return value

    @staticmethod
    def _positive(value: float, name: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be positive and finite")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return result


__all__ = ["G1FeedbackCalibration", "G1MeasuredCommandFeedback"]
