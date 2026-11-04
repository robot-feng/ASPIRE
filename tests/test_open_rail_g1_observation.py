from __future__ import annotations

import numpy as np
import pytest

from cap_harness.open_rail.g1_runtime import (
    OpenRailG1ObservationError,
    OpenRailG1ObservationRuntime,
)
from cap_harness.open_rail.profile import (
    default_g1_observation_profile_path,
    load_g1_observation_profile,
)


class _Proxy:
    def close(self) -> None:
        pass


def _observation(*, depth: np.ndarray | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "cam.head": np.full((3, 5, 3), 17, dtype=np.uint8),
        "obs.state": np.arange(20, dtype=np.float32),
        "rgb_timestamp_ns": 1_000_000_000,
        "rgb_order": "RGB",
        "state_timestamps_ns": {
            "arm": 999_800_000,
            "gripper_left": 999_810_000,
            "gripper_right": 999_820_000,
            "head": 999_830_000,
            "waist": 999_840_000,
        },
    }
    if depth is not None:
        result["cam.head_depth"] = depth
        result["depth_timestamp_ns"] = 1_000_100_000
        result["depth_encoding"] = "RS2_FORMAT_Z16"
    return result


def test_g1_profile_exposes_public_head_camera_and_unverified_calibration() -> None:
    profile = load_g1_observation_profile(default_g1_observation_profile_path())

    assert profile.robot_type == "agibot_g1"
    assert profile.public_camera_key == "head"
    assert profile.rgb_channel == "cam.head"
    assert profile.depth_channel == "cam.head_depth"
    assert profile.state_dimension == 20
    assert profile.calibration["depth_scale_m_per_unit"] is None
    assert profile.calibration["arm_tcp_transforms"] is None


def test_runtime_preserves_raw_z16_and_only_claims_configured_timestamp_skew(
    tmp_path,
) -> None:
    adapter = OpenRailG1ObservationRuntime(
        open_rail_root=tmp_path,
        maximum_rgb_depth_skew_ms=1.0,
        proxy=_Proxy(),
    )
    raw_depth = np.array([[0, 1, 4095], [65535, 2, 3]], dtype=np.uint16)

    snapshot = adapter._decode(_observation(depth=raw_depth))

    np.testing.assert_array_equal(snapshot.depth_u16, raw_depth)
    np.testing.assert_array_equal(snapshot.state, np.arange(20, dtype=np.float32))
    assert snapshot.depth_skew_ns == 100_000
    assert snapshot.is_synchronized
    assert snapshot.metadata["public_camera_key"] == "head"
    assert snapshot.metadata["calibration"]["depth_scale_m_per_unit"] is None
    assert not snapshot.rgb.flags.writeable
    assert not snapshot.depth_u16.flags.writeable
    adapter.close()


def test_missing_depth_stays_raw_and_is_not_reported_as_synchronized(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())

    snapshot = adapter._decode(_observation())

    assert snapshot.depth_u16 is None
    assert snapshot.depth_timestamp_ns is None
    assert not snapshot.is_synchronized
    adapter.close()


def test_runtime_rejects_state_that_does_not_match_selected_profile(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    observation = _observation()
    observation["obs.state"] = np.zeros(19, dtype=np.float32)

    with pytest.raises(OpenRailG1ObservationError, match="expected 20 finite"):
        adapter._decode(observation)
    adapter.close()


def test_runtime_rejects_non_z16_depth_instead_of_converting_it(tmp_path) -> None:
    adapter = OpenRailG1ObservationRuntime(open_rail_root=tmp_path, proxy=_Proxy())
    observation = _observation(depth=np.ones((2, 3), dtype=np.float32))

    with pytest.raises(OpenRailG1ObservationError, match="invalid raw Z16"):
        adapter._decode(observation)
    adapter.close()
