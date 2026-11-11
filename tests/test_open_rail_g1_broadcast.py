from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from queue import Empty, Queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.errors import ErrorCode
from cap_harness.open_rail.g1_broadcast import G1SnapshotBroadcaster
from cap_harness.open_rail.g1_control import G1CommandCapabilities, OpenRailG1CommandRuntime
from cap_harness.open_rail.g1_feedback import G1FeedbackCalibration, G1MeasuredCommandFeedback
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailG1ObservationRuntime
from cap_harness.open_rail.g1_state_samples import G1StateSamples


class _Proxy:
    def __init__(self):
        self.frames = Queue()
        self.closed = False
        self.readers = set()

    def retrieve_observation(self):
        self.readers.add(threading.current_thread().name)
        try:
            return self.frames.get_nowait()
        except Empty:
            return None

    def close(self):
        self.closed = True


def _runtime(tmp_path, proxy, *, motion=False):
    return OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path, proxy=proxy, bound_state_sampling=True,
        maximum_rgb_depth_skew_ms=50, allow_physical_motion=motion,
    )


def _frame(runtime, stamp, *, gripper=0.5, arm=True, depth=True):
    names = list(runtime.profile.arm_message_names)
    state = np.zeros(20, dtype=np.float32)
    state[15] = gripper
    side = {
        "timestamp_ns": stamp, "motor_ids": list(range(7)),
        "positions_raw": [0.0] * 7, "velocities_raw": [0.0] * 7,
    }
    payload = {
        "cam.head": np.zeros((3, 5, 3), dtype=np.uint8),
        "obs.state": state,
        "rgb_timestamp_ns": stamp, "rgb_packet_timestamp_ns": stamp,
        "rgb_order": "RGB", "rgb_encoding_format": "UNCOMPRESSED",
        "rgb_color_format": "RGB", "rgb_decoder": "raw-rgb",
        "state_timestamps_bind_state_values": True,
        "state_timestamps_ns": {key: stamp for key in ("arm", "head", "waist", "gripper_left", "gripper_right")},
        "bound_arm_sample": {"names": names, "velocities_raw": None,
                             "sides": {"right": side} if arm else {}},
    }
    if depth:
        payload["cam.head_depth"] = np.ones((3, 5), dtype=np.uint16)
        payload["depth_timestamp_ns"] = stamp
    return payload


def _calibration():
    return G1FeedbackCalibration(
        right_motor_ids=tuple(range(7)), right_motor_order_verified=True,
        arm_position_velocity_units_verified=True, gripper_feedback_convention_verified=True,
        feedback_clock_verified=True, max_sample_age_s=1, max_sample_gap_s=0.05,
        stable_duration_s=0.1, minimum_samples=3,
    )


def _publish_ipc_frames(queue, ready, start, names):
    # This process stands in for Open-RAIL's observation worker; no SDK is loaded.
    runtime = SimpleNamespace(profile=SimpleNamespace(arm_message_names=names))
    ready.set()
    if not start.wait(3):
        return
    for _ in range(9):
        queue.put(_frame(runtime, time.time_ns()))
        time.sleep(0.05)


def test_one_reader_broadcasts_same_snapshot_and_bound_feedback(tmp_path, monkeypatch):
    proxy = _Proxy()
    runtime = _runtime(tmp_path, proxy)
    broker = G1SnapshotBroadcaster(runtime, capture_timeout_s=0.5, close_timeout_s=1.5)
    assert broker._thread is None and not proxy.readers
    broker.start()
    with pytest.raises(OpenRailG1ObservationError, match="reader"):
        runtime.read_snapshot(timeout_s=0.1)
    with pytest.raises(OpenRailG1ObservationError, match="broadcaster|reader"):
        runtime.close()
    waiting = threading.Event()
    original_wait = broker._condition.wait
    waiting_count = 0
    def signal_wait(timeout=None):
        nonlocal waiting_count
        waiting_count += 1
        if waiting_count >= 2:
            waiting.set()
        return original_wait(timeout)
    monkeypatch.setattr(broker._condition, "wait", signal_wait)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(broker.read_snapshot, timeout_s=0.5, require_depth=True)
        second = executor.submit(broker.read_snapshot, timeout_s=0.5, require_depth=True)
        assert waiting.wait(1)
        proxy.frames.put(_frame(runtime, 1_000_000_000))
        snapshot = first.result(timeout=1)
        other_snapshot = second.result(timeout=1)
    arm = broker.feedback_samples.wait_next_right_arm(0, 0)
    gripper = broker.feedback_samples.wait_next_right_gripper(0, 0)
    assert arm[0] == snapshot.state_timestamps_ns["gripper_right"]
    assert gripper[0] == arm[0]
    assert snapshot.metadata["arm_side_feedback"]["right"]["timestamp_ns"] == arm[0]
    assert proxy.readers == {"g1-snapshot-broadcast"}
    snapshot.metadata["arm_side_feedback"]["right"]["motor_ids"][0] = 99
    assert other_snapshot.metadata["arm_side_feedback"]["right"]["motor_ids"][0] == 0
    assert broker._latest.metadata["arm_side_feedback"]["right"]["motor_ids"][0] == 0
    broker.close()
    assert proxy.closed


def test_feedback_timeout_is_independent_of_a_stalled_capture_and_latches(tmp_path):
    class StalledProxy(_Proxy):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def retrieve_observation(self):
            self.entered.set()
            self.release.wait(2)
            return None

    class Transport:
        def __init__(self):
            self.calls = []

        def set_joint_position_control_bounded(self, lifetime_s, groups, *, timeout_s):
            self.calls.append((lifetime_s, groups))

        def publish_right_gripper_bounded(self, value, *, timeout_s):
            raise AssertionError("uncertain command must prevent another dispatch")

    proxy = StalledProxy()
    runtime = _runtime(tmp_path, proxy, motion=True)
    broker = G1SnapshotBroadcaster(runtime, capture_timeout_s=0.5, close_timeout_s=0.03)
    broker.start()
    assert proxy.entered.wait(1)
    transport = Transport()
    feedback = G1MeasuredCommandFeedback(broker.feedback_samples, _calibration())
    command = OpenRailG1CommandRuntime(
        transport, feedback=feedback, allow_physical_motion=True,
        capabilities=G1CommandCapabilities(
            arm_command_protocol_verified=True, arm_position_limits_verified=True,
            arm_terminal_feedback_verified=True, arm_cancel_or_lease_expiry_verified=True,
        ), arm_joint_limits_rad=[(-1, 1)] * 7,
        position_tolerance_rad=0.01, velocity_tolerance_rad_s=0.01,
    )
    start = time.monotonic()
    result = command.set_arm("right", np.zeros(7), lifetime_s=0.1, timeout_s=0.03)
    assert time.monotonic() - start < 0.25
    assert not result.ok and result.error.code is ErrorCode.TIMEOUT
    assert result.diagnostics["cancel_confirmed"] is False
    assert command.health["motion_session"]["uncertain"] is True
    assert command.set_gripper("right", 0.5, timeout_s=0.1).error.code is ErrorCode.SAFETY_INTERLOCK
    with pytest.raises(TimeoutError, match="remains open"):
        broker.close()
    assert not proxy.closed
    proxy.release.set()
    broker._thread.join(1)
    broker.close()
    assert proxy.closed and len(transport.calls) == 1


def test_invalid_or_closed_broadcast_wakes_waiters_and_never_claims_stop(tmp_path):
    proxy = _Proxy()
    runtime = _runtime(tmp_path, proxy)
    broker = G1SnapshotBroadcaster(runtime, capture_timeout_s=1, close_timeout_s=1)
    broker.start()
    invalid = _frame(runtime, 1_000_000_000)
    invalid["bound_arm_sample"]["sides"]["right"]["motor_ids"] = [0] * 7
    proxy.frames.put(invalid)
    with pytest.raises(RuntimeError, match="seven unique"):
        broker.feedback_samples.wait_next_right_arm(0, 1)
    with pytest.raises(OpenRailG1ObservationError, match="broadcaster failed"):
        broker.read_snapshot(timeout_s=0.1)
    broker.close()
    assert proxy.closed


def test_duplicate_dds_samples_do_not_count_twice_and_conflicts_fail_closed():
    buffer = G1StateSamples(tuple(f"joint{i}" for i in range(14)))
    arm = {"timestamp_ns": 100, "motor_ids": list(range(7)),
           "positions_raw": [0.0] * 7, "velocities_raw": [0.0] * 7}
    for _ in range(2):
        buffer.publish_bound_right_feedback(gripper_timestamp_ns=100, gripper_native=0.5,
                                            arm_feedback=arm)
    assert buffer.wait_next_right_arm(0, 0)[0] == 100
    assert buffer.wait_next_right_arm(100, 0) is None
    changed = {**arm, "velocities_raw": [0.1] + [0.0] * 6}
    with pytest.raises(ValueError, match="changed contents"):
        buffer.publish_bound_right_feedback(gripper_timestamp_ns=101, gripper_native=0.5,
                                            arm_feedback=changed)
    # Validation is atomic: the newer gripper value was not accepted on error.
    assert buffer.wait_next_right_gripper(100, 0) is None


def test_ipc_frames_feed_observation_and_terminal_feedback_without_rpc(tmp_path, monkeypatch):
    ctx = mp.get_context("spawn")
    proxy = _Proxy()
    proxy.frames = ctx.Queue()
    runtime = _runtime(tmp_path, proxy, motion=True)
    ready, start = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_publish_ipc_frames,
                          args=(proxy.frames, ready, start, runtime.profile.arm_message_names))
    process.start()
    assert ready.wait(3)
    broker = G1SnapshotBroadcaster(runtime, capture_timeout_s=2, close_timeout_s=3)
    broker.start()
    waiting = threading.Event()
    original_wait = broker._condition.wait
    def signal_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)
    monkeypatch.setattr(broker._condition, "wait", signal_wait)

    class Transport:
        def set_joint_position_control_bounded(self, lifetime_s, groups, *, timeout_s):
            assert groups == {"right_arm": (0.0,) * 7}
            start.set()

    calibration = G1FeedbackCalibration(
        right_motor_ids=tuple(range(7)), right_motor_order_verified=True,
        arm_position_velocity_units_verified=True, gripper_feedback_convention_verified=True,
        feedback_clock_verified=True, max_sample_age_s=1, max_sample_gap_s=0.15,
        stable_duration_s=0.3, minimum_samples=3,
    )
    command = OpenRailG1CommandRuntime(
        Transport(), feedback=G1MeasuredCommandFeedback(broker.feedback_samples, calibration),
        allow_physical_motion=True,
        capabilities=G1CommandCapabilities(
            arm_command_protocol_verified=True, arm_position_limits_verified=True,
            arm_terminal_feedback_verified=True, arm_cancel_or_lease_expiry_verified=True,
        ), arm_joint_limits_rad=[(-1, 1)] * 7,
        position_tolerance_rad=0.01, velocity_tolerance_rad_s=0.01,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            observation = executor.submit(broker.read_snapshot, timeout_s=2, require_depth=True)
            assert waiting.wait(1)
            action = executor.submit(command.set_arm, "right", np.zeros(7), lifetime_s=0.1, timeout_s=2)
            frame = observation.result(timeout=1)
            assert frame.metadata["state_timestamps_bind_state_values"] is True
            assert not action.done(), "camera observation should proceed while feedback waits"
            result = action.result(timeout=2)
        assert result.ok
        assert result.diagnostics["measured_feedback"]["stable_samples"] >= 3
        assert proxy.readers == {"g1-snapshot-broadcast"}
    finally:
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(1)
        broker.close()
        proxy.frames.close()


def test_short_gripper_travel_does_not_prove_grasp_or_failure(tmp_path):
    ctx = mp.get_context("spawn")
    proxy = _Proxy()
    proxy.frames = ctx.Queue()
    runtime = _runtime(tmp_path, proxy, motion=True)
    ready, start = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_publish_ipc_frames,
                          args=(proxy.frames, ready, start, runtime.profile.arm_message_names))
    process.start()
    assert ready.wait(3)
    broker = G1SnapshotBroadcaster(runtime, capture_timeout_s=2, close_timeout_s=3)
    broker.start()

    class Transport:
        def publish_right_gripper_bounded(self, value, *, timeout_s):
            assert value == 1.0
            start.set()

    command = OpenRailG1CommandRuntime(
        Transport(), feedback=G1MeasuredCommandFeedback(broker.feedback_samples, _calibration()),
        allow_physical_motion=True, gripper_tolerance=0.01,
        capabilities=G1CommandCapabilities(
            gripper_terminal_feedback_verified=True, gripper_stop_or_hold_verified=True,
        ),
    )
    try:
        # Fake frames show a stable 0.5 opening although 1.0 was requested.
        # It could be contact, not evidence of either a successful or failed grasp.
        result = command.set_gripper("right", 1.0, timeout_s=0.3)
        assert not result.ok and result.error.code is ErrorCode.TIMEOUT
        assert result.diagnostics["contact_or_grasp_status"] == "unknown"
        assert result.diagnostics["cancel_confirmed"] is False
        assert command.health["motion_session"]["uncertain"] is True
        assert command.set_gripper("right", 1.0, timeout_s=0.1).error.code is ErrorCode.SAFETY_INTERLOCK
    finally:
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(1)
        broker.close()
        proxy.frames.close()
