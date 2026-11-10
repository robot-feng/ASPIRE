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
        ["observe", "--output", str(tmp_path), "--samples", "2", "--interval", "1", "--require-depth"],
    )

    assert observe.main() == 0
    assert [event[0] for event in events] == ["init", "read", "interval", "read", "close"]
    assert events[1][1]["require_depth"] is True
    for index, value in enumerate((10, 20), 1):
        sample = tmp_path / f"sample_{index:04d}"
        assert np.all(np.asarray(Image.open(sample / "head_rgb.png")) == value)
        assert np.all(np.load(sample / "head_depth_u16.npy") == 500 + value)
        metadata = json.loads((sample / "snapshot.json").read_text())
        assert metadata["rgb_timestamp_ns"] == value
        assert metadata["depth_timestamp_ns"] == value + 1
        assert metadata["state_timestamps_ns"] == {"head": value + 2}
        assert metadata["depth_scale_m_per_unit"] is None
        assert metadata["is_synchronized"] is False


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
