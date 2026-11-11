from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "open_rail_observe_example", Path(__file__).resolve().parents[1] / "examples/open_rail/observe.py"
)
assert spec is not None and spec.loader is not None
observe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observe)


def test_rgbd_calibration_samples_share_runtime_and_preserve_raw_values(tmp_path, monkeypatch):
    snapshots = iter(
        SimpleNamespace(
            rgb=np.full((2, 3, 3), value, dtype=np.uint8),
            depth_u16=np.full((2, 3), 500 + value, dtype=np.uint16),
            state=np.array([value], dtype=np.float32),
            arm_velocity_raw=np.arange(7, dtype=np.float32),
            rgb_timestamp_ns=value,
            depth_timestamp_ns=value + 1,
            depth_skew_ns=1,
            state_timestamps_ns={"head": value + 2},
            is_synchronized=False,
            depth_error=None,
            metadata={"depth_scale_m_per_unit": None},
        )
        for value in (10, 20)
    )
    events = []

    class FakeRuntime:
        def __init__(self, **kwargs):
            events.append(("init", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append(("close",))

        def read_snapshot(self, **kwargs):
            events.append(("read", kwargs))
            return next(snapshots)

    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", FakeRuntime)
    monkeypatch.setattr(observe.time, "sleep", lambda value: events.append(("interval", value)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["observe", "--output", str(tmp_path), "--samples", "2", "--interval", "1", "--require-depth", "--bound-state-sampling"],
    )

    assert observe.main() == 0
    assert [event[0] for event in events] == ["init", "read", "interval", "read", "close"]
    assert events[0][1]["bound_state_sampling"] is True
    assert events[1][1]["require_depth"] is True
    for index, value in enumerate((10, 20), 1):
        sample = tmp_path / f"sample_{index:04d}"
        assert np.all(np.asarray(Image.open(sample / "head_rgb.png")) == value)
        assert np.all(np.load(sample / "head_depth_u16.npy") == 500 + value)
        np.testing.assert_array_equal(np.load(sample / "arm_velocity_raw.npy"), np.arange(7))
        metadata = json.loads((sample / "snapshot.json").read_text())
        assert metadata["rgb_timestamp_ns"] == value
        assert metadata["depth_timestamp_ns"] == value + 1
        assert metadata["state_timestamps_ns"] == {"head": value + 2}
        assert metadata["depth_scale_m_per_unit"] is None
        assert metadata["is_synchronized"] is False
        assert metadata["arm_velocity_raw_file"] == "arm_velocity_raw.npy"


def test_rgbd_capture_rejects_duplicate_timestamps_without_overwriting(tmp_path, monkeypatch):
    sample = SimpleNamespace(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8), depth_u16=None,
        state=np.zeros(1), rgb_timestamp_ns=100, depth_timestamp_ns=None,
        depth_skew_ns=None, state_timestamps_ns={}, is_synchronized=False,
        depth_error="unavailable", metadata={},
    )

    class FakeRuntime:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read_snapshot(self, **kwargs):
            return sample

    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", FakeRuntime)
    monkeypatch.setattr(observe.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--samples", "2"])

    with pytest.raises(RuntimeError, match="timestamp did not advance"):
        observe.main()
    assert (tmp_path / "sample_0001" / "snapshot.json").is_file()
    assert not (tmp_path / "sample_0002").exists()


def test_capture_refuses_to_mix_a_new_snapshot_with_an_old_depth_file(tmp_path):
    old_depth = tmp_path / "head_depth_u16.npy"
    old_depth.write_bytes(b"old evidence")
    with pytest.raises(FileExistsError, match="snapshot output already exists"):
        observe.save_snapshot(SimpleNamespace(), tmp_path)
    assert old_depth.read_bytes() == b"old evidence"
    assert not (tmp_path / "snapshot.json").exists()


def test_broadcast_requires_bound_sampling_before_opening_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--broadcast"])
    with pytest.raises(SystemExit):
        observe.main()
    assert list(tmp_path.iterdir()) == []


def test_capture_uses_one_broadcaster_and_closes_it(tmp_path, monkeypatch):
    from cap_harness.open_rail import g1_broadcast

    snapshot = SimpleNamespace(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8), depth_u16=None,
        state=np.zeros(1), rgb_timestamp_ns=100, depth_timestamp_ns=None,
        depth_skew_ns=None, state_timestamps_ns={}, is_synchronized=False,
        depth_error="unavailable", metadata={},
    )
    events = []

    class Runtime:
        def __init__(self, **kwargs):
            events.append(("runtime", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append(("runtime_exit",))

    class Broadcaster:
        def __init__(self, runtime, **kwargs):
            events.append(("broadcast", runtime, kwargs))
            self.feedback_samples = SimpleNamespace(
                wait_next_right_arm=lambda *_: (100, (0, 1, 2, 3, 4, 5, 6), (0,) * 7, (0,) * 7),
                wait_next_right_gripper=lambda *_: (100, (0.5,), None),
            )

        def start(self):
            events.append(("start",))

        def read_snapshot(self, **kwargs):
            events.append(("read", kwargs))
            return snapshot

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", Runtime)
    monkeypatch.setattr(g1_broadcast, "G1SnapshotBroadcaster", Broadcaster)
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--broadcast", "--bound-state-sampling"])
    assert observe.main() == 0
    assert [event[0] for event in events] == ["runtime", "broadcast", "start", "read", "close", "runtime_exit"]
    assert (tmp_path / "snapshot.json").is_file()
    assert json.loads((tmp_path / "broadcast_feedback.json").read_text())["right_arm"][0] == 100


def _mutable_snapshot():
    return SimpleNamespace(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth_u16=np.zeros((2, 3), dtype=np.uint16),
        state=np.zeros(1, dtype=np.float32), arm_velocity_raw=None,
        rgb_timestamp_ns=0, depth_timestamp_ns=0, depth_skew_ns=0,
        state_timestamps_ns={"head": 0}, is_synchronized=False,
        depth_error=None, metadata={"nested": {"value": 0}},
    )


def test_buffered_mode_copies_each_snapshot_before_next_read_and_defers_writes(tmp_path, monkeypatch):
    shared = _mutable_snapshot()
    events = []

    class Runtime:
        def __init__(self, **_kwargs):
            self.index = 0
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            events.append("close")
        def read_snapshot(self, **_kwargs):
            self.index += 1
            value = self.index * 10
            events.append(f"read{value}")
            shared.rgb.fill(value)
            shared.depth_u16.fill(500 + value)
            shared.state.fill(value)
            shared.rgb_timestamp_ns = shared.depth_timestamp_ns = value
            shared.state_timestamps_ns["head"] = value
            shared.metadata["nested"]["value"] = value
            return shared

    save = observe.save_snapshot
    def record_save(snapshot, output):
        events.append(f"save{snapshot.rgb_timestamp_ns}")
        assert not snapshot.rgb.flags.writeable
        return save(snapshot, output)

    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", Runtime)
    monkeypatch.setattr(observe, "save_snapshot", record_save)
    monkeypatch.setattr(observe.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--samples", "2", "--buffered"])
    assert observe.main() == 0
    assert events == ["read10", "read20", "save10", "save20", "close"]
    for index, value in enumerate((10, 20), 1):
        sample = tmp_path / f"sample_{index:04d}"
        assert np.all(np.asarray(Image.open(sample / "head_rgb.png")) == value)
        assert np.all(np.load(sample / "head_depth_u16.npy") == 500 + value)
        assert np.load(sample / "robot_state.npy")[0] == value
        metadata = json.loads((sample / "snapshot.json").read_text())
        assert metadata["nested"]["value"] == value
        assert metadata["state_timestamps_ns"]["head"] == value


@pytest.mark.parametrize("failure", ["read", "duplicate", "limit"])
def test_buffered_mode_flushes_completed_samples_on_failure(tmp_path, monkeypatch, failure):
    sample = _mutable_snapshot()
    sample.rgb_timestamp_ns = sample.depth_timestamp_ns = 10
    events = []
    class Runtime:
        def __init__(self, **_kwargs):
            self.calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            events.append("close")
        def read_snapshot(self, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                if failure == "read":
                    raise TimeoutError("capture failed")
                if failure == "limit":
                    sample.rgb_timestamp_ns = sample.depth_timestamp_ns = 20
            return sample

    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", Runtime)
    monkeypatch.setattr(observe.time, "sleep", lambda _: None)
    argv = ["observe", "--output", str(tmp_path), "--samples", "2", "--buffered"]
    if failure == "limit":
        argv += ["--buffer-limit-mib", str((observe.buffered_size(sample) + 1) / (1024 * 1024))]
    monkeypatch.setattr(sys, "argv", argv)
    error = {"read": TimeoutError, "duplicate": RuntimeError, "limit": MemoryError}[failure]
    with pytest.raises(error):
        observe.main()
    assert events == ["close"]
    assert json.loads((tmp_path / "sample_0001" / "snapshot.json").read_text())["rgb_timestamp_ns"] == 10
    assert not (tmp_path / "sample_0002").exists()


def test_buffered_count_limit_rejected_before_runtime_creation(tmp_path, monkeypatch):
    def no_runtime(**_kwargs):
        pytest.fail("invalid request must not start a robot runtime")
    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", no_runtime)
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--samples", "201", "--buffered"])
    with pytest.raises(SystemExit):
        observe.main()
    assert not list(tmp_path.iterdir())


def test_buffered_existing_destination_rejected_before_runtime_creation(tmp_path, monkeypatch):
    existing = tmp_path / "sample_0002"
    existing.mkdir()
    (existing / "snapshot.json").write_text("original")
    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", lambda **_: pytest.fail("unexpected runtime"))
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--samples", "2", "--buffered"])
    with pytest.raises(FileExistsError, match="destination"):
        observe.main()
    assert (existing / "snapshot.json").read_text() == "original"
    assert not (tmp_path / "sample_0001").exists()


def test_buffered_sample_survives_broadcaster_close_failure(tmp_path, monkeypatch):
    from cap_harness.open_rail import g1_broadcast
    sample = _mutable_snapshot()
    sample.rgb_timestamp_ns = sample.depth_timestamp_ns = 10

    class Runtime:
        def __init__(self, **_kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass

    class Broadcaster:
        def __init__(self, *_args, **_kwargs):
            self.feedback_samples = SimpleNamespace(
                wait_next_right_arm=lambda *_: None,
                wait_next_right_gripper=lambda *_: None,
            )
        def start(self):
            pass
        def read_snapshot(self, **_kwargs):
            return sample
        def close(self):
            raise TimeoutError("close failed")

    monkeypatch.setattr(observe, "OpenRailG1ObservationRuntime", Runtime)
    monkeypatch.setattr(g1_broadcast, "G1SnapshotBroadcaster", Broadcaster)
    monkeypatch.setattr(sys, "argv", ["observe", "--output", str(tmp_path), "--buffered",
                                    "--broadcast", "--bound-state-sampling"])
    with pytest.raises(TimeoutError, match="close failed"):
        observe.main()
    assert json.loads((tmp_path / "snapshot.json").read_text())["rgb_timestamp_ns"] == 10
