"""Broadcast one G1 proxy's observations and bound feedback without feedback RPCs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import threading
import time

import numpy as np

from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailG1ObservationRuntime, OpenRailSnapshot
from cap_harness.open_rail.g1_state_samples import G1StateSamples


class G1SnapshotBroadcaster:
    """Own the existing proxy frame reader and share each bound snapshot.

    Construction starts no threads. ``start`` explicitly reserves the one
    runtime reader and launches one daemon that copies frames to observation
    consumers and measured feedback to ``feedback_samples``. Feedback waits
    use a parent-process condition rather than the worker's 30-second RPC; the
    worker's camera loop is never blocked waiting for terminal feedback.
    Closing or failure wakes waiters but does not stop, cancel or confirm any
    physical command. A caller must keep the normal G1 motion interlocks.
    """

    def __init__(
        self,
        runtime: OpenRailG1ObservationRuntime,
        *,
        capture_timeout_s: float,
        close_timeout_s: float,
    ) -> None:
        if not isinstance(runtime, OpenRailG1ObservationRuntime):
            raise ValueError("broadcaster requires the sole OpenRailG1ObservationRuntime")
        if runtime.bound_state_sampling is not True:
            raise ValueError("broadcaster requires explicitly enabled bound state sampling")
        self._runtime = runtime
        self.allow_physical_motion = runtime.allow_physical_motion
        self.capture_timeout_s = self._positive(capture_timeout_s, "capture_timeout_s")
        self.close_timeout_s = self._positive(close_timeout_s, "close_timeout_s")
        self.feedback_samples = G1StateSamples(tuple(runtime.profile.arm_message_names))
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: OpenRailSnapshot | None = None
        self._version = 0
        self._failure: str | None = None
        self._closed = False

    def start(self) -> None:
        """Explicitly start the sole reader; never start a motion or policy."""
        with self._condition:
            if self._closed or self._thread is not None:
                raise OpenRailG1ObservationError("G1 broadcaster is closed or already started")
            thread = threading.Thread(target=self._run, name="g1-snapshot-broadcast", daemon=True)
            self._runtime.claim_snapshot_reader(thread)
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._runtime.release_snapshot_reader(thread)
                raise

    def _run(self) -> None:
        previous_rgb = None
        try:
            while not self._stop.is_set():
                snapshot = self._runtime.read_snapshot(timeout_s=self.capture_timeout_s)
                if not isinstance(snapshot, OpenRailSnapshot):
                    raise ValueError("G1 runtime returned an invalid raw snapshot")
                if previous_rgb is not None and snapshot.rgb_timestamp_ns <= previous_rgb:
                    raise ValueError("G1 broadcaster received a repeated or older RGB snapshot")
                previous_rgb = snapshot.rgb_timestamp_ns
                if snapshot.metadata.get("state_timestamps_bind_state_values") is True:
                    self._publish_feedback(snapshot)
                # Isolate mutable metadata between runtime, broker, and readers.
                snapshot = self._copy_snapshot(snapshot)
                with self._condition:
                    self._latest = snapshot
                    self._version += 1
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._failure = f"G1 snapshot broadcaster failed: {type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                self.feedback_samples.fail_feedback(self._failure)
        finally:
            try:
                self._runtime.release_snapshot_reader(threading.current_thread())
            except Exception as exc:
                with self._condition:
                    self._failure = f"G1 snapshot reader release failed: {type(exc).__name__}: {exc}"
                    self._condition.notify_all()
                self.feedback_samples.fail_feedback(self._failure)

    def _publish_feedback(self, snapshot: OpenRailSnapshot) -> None:
        if snapshot.metadata.get("robot_type") != "agibot_g1":
            raise ValueError("bound feedback is not from a G1 snapshot")
        layout = snapshot.metadata.get("state_layout")
        if not isinstance(layout, dict) or tuple(layout.get("right_gripper_normalized", ())) != (15, 16):
            raise ValueError("G1 right-gripper feedback layout is not established")
        state = np.asarray(snapshot.state)
        if state.shape != (20,) or not np.isfinite(state).all():
            raise ValueError("bound G1 snapshot has an invalid state")
        sides = snapshot.metadata.get("arm_side_feedback")
        if sides is not None and not isinstance(sides, dict):
            raise ValueError("G1 arm side feedback is invalid")
        self.feedback_samples.publish_bound_right_feedback(
            gripper_timestamp_ns=snapshot.state_timestamps_ns.get("gripper_right"),
            gripper_native=state[15],
            arm_feedback=None if sides is None else sides.get("right"),
        )

    def read_snapshot(self, *, timeout_s: float = 5.0, require_depth: bool = False) -> OpenRailSnapshot:
        """Wait for a new broadcast snapshot, never consuming another reader's frame."""
        timeout = self._positive(timeout_s, "timeout_s")
        if type(require_depth) is not bool:
            raise ValueError("require_depth must be a bool")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._thread is None:
                raise OpenRailG1ObservationError("G1 broadcaster has not been started")
            observed_version = self._version
            while True:
                if self._failure is not None:
                    raise OpenRailG1ObservationError(self._failure)
                if self._closed:
                    raise OpenRailG1ObservationError("G1 broadcaster is closed")
                if self._version > observed_version and self._latest is not None:
                    if not require_depth or self._latest.depth_u16 is not None:
                        return self._copy_snapshot(self._latest)
                    observed_version = self._version
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no new broadcast G1 snapshot before timeout")
                self._condition.wait(remaining)

    def close(self) -> None:
        """Stop the reader, then close its runtime; this is not a motion stop."""
        with self._condition:
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
            thread = self._thread
        self.feedback_samples.fail_feedback("G1 snapshot broadcaster closed; motion state is unknown")
        if thread is not None:
            thread.join(self.close_timeout_s)
            if thread.is_alive():
                raise TimeoutError("G1 snapshot reader did not stop; runtime remains open")
        self._runtime.close()

    @staticmethod
    def _copy_snapshot(snapshot: OpenRailSnapshot) -> OpenRailSnapshot:
        return replace(snapshot, metadata=deepcopy(snapshot.metadata),
                       state_timestamps_ns=dict(snapshot.state_timestamps_ns))

    @staticmethod
    def _positive(value: float, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be positive and finite")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return result


__all__ = ["G1SnapshotBroadcaster"]
