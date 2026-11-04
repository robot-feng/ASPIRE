from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cap_harness.open_rail.g1_camera import (
    G1CameraCalibration,
    G1DynamicCameraCalibration,
    G1HeadKinematics,
    camera_observation_from_snapshot,
)
from cap_harness.open_rail.g1_runtime import (
    OpenRailG1ObservationError,
    OpenRailSnapshot,
)


def _snapshot(
    *,
    depth_shape: tuple[int, int] = (3, 4),
    head_yaw_pitch: tuple[float, float] = (0.0, 0.0),
    waist_pitch_lift: tuple[float, float] = (0.0, 0.0),
    rgb_timestamp_ns: int = 1_000_000_000,
    depth_timestamp_ns: int | None = 1_000_100_000,
    head_timestamp_ns: int | None = 999_950_000,
    waist_timestamp_ns: int | None = 1_000_080_000,
) -> OpenRailSnapshot:
    state = np.zeros(20, dtype=np.float32)
    state[16:18] = head_yaw_pitch
    state[18:20] = waist_pitch_lift
    return OpenRailSnapshot(
        rgb=np.full((3, 4, 3), 25, dtype=np.uint8),
        depth_u16=np.full(depth_shape, 1000, dtype=np.uint16),
        state=state,
        rgb_timestamp_ns=rgb_timestamp_ns,
        depth_timestamp_ns=depth_timestamp_ns,
        state_timestamps_ns={"head": head_timestamp_ns, "waist": waist_timestamp_ns},
        depth_error=None,
        metadata={
            "robot_type": "agibot_g1",
            "public_camera_key": "head",
            "rgb_order": "RGB",
            "rgb_packet_timestamp_ns": rgb_timestamp_ns,
            "rgb_encoding_format": "UNCOMPRESSED",
            "rgb_color_format": "RGB",
            "rgb_decoder": "raw-rgb",
            "depth_encoding": "RS2_FORMAT_Z16",
            "state_timestamps_bind_state_values": True,
            "state_layout": {
                "head_yaw_pitch_rad": [16, 18],
                "waist_pitch_rad_lift_m": [18, 20],
            },
        },
    )


def _calibration(**changes) -> G1CameraCalibration:
    values = {
        "depth_scale_m_per_unit": 0.001,
        "depth_scale_verified": True,
        "depth_registered_to_rgb": True,
        "depth_registration_verified": True,
        "intrinsics": np.array(
            [[100.0, 0.0, 2.0], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "intrinsics_resolution_hw": (3, 4),
        "intrinsics_verified": True,
        "rgb_depth_rectified_for_pinhole": True,
        "rectification_verified": True,
        "base_from_camera": np.array(
            [
                [1.0, 0.0, 0.0, 0.4],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.7],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "camera_to_base_verified": True,
        "public_camera_key": "head",
        "base_frame": "base_link",
        "camera_frame": "head_color_optical",
        "maximum_snapshot_skew_ms": 0.5,
        "synchronization_threshold_verified": True,
        "calibration_head_yaw_pitch_rad": (0.0, 0.0),
        "head_yaw_pitch_tolerance_rad": (0.01, 0.01),
        "calibration_waist_pitch_lift_m": (0.0, 0.0),
        "waist_pitch_lift_tolerance": (0.01, 0.01),
        "mount_posture_verified": True,
        "invalid_depth_values": (0, 65535),
        "invalid_depth_values_verified": True,
    }
    values.update(changes)
    return G1CameraCalibration(**values)


def _dynamic_calibration(**changes) -> G1DynamicCameraCalibration:
    values = {
        "depth_scale_m_per_unit": 0.001,
        "depth_scale_verified": True,
        "depth_registered_to_rgb": True,
        "depth_registration_verified": True,
        "intrinsics": np.array(
            [[100.0, 0.0, 2.0], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "intrinsics_resolution_hw": (3, 4),
        "intrinsics_verified": True,
        "rgb_depth_rectified_for_pinhole": True,
        "rectification_verified": True,
        "mount_from_camera": np.array(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "mount_from_camera_verified": True,
        "public_camera_key": "head",
        "base_frame": "base_link",
        "mount_frame": "verified_head_mount",
        "camera_frame": "head_color_optical",
        "maximum_snapshot_skew_ms": 0.5,
        "synchronization_threshold_verified": True,
        "timestamps_share_clock_verified": True,
        "state_layout_verified": True,
        "state_units_verified": True,
        "fk_model_id": "verified-g1-model-sha256",
        "invalid_depth_values": (0, 65535),
        "invalid_depth_values_verified": True,
    }
    values.update(changes)
    return G1DynamicCameraCalibration(**values)


def _head_kinematics(
    *, base_from_mount=None, **changes
) -> G1HeadKinematics:
    values = {
        "base_frame": "base_link",
        "mount_frame": "verified_head_mount",
        "model_id": "verified-g1-model-sha256",
        "verified": True,
        "state_layout_verified": True,
        "state_units_verified": True,
        "base_from_mount": base_from_mount
        or (lambda head, waist: np.array(
            [
                [1.0, 0.0, 0.0, head[0]],
                [0.0, 1.0, 0.0, head[1]],
                [0.0, 0.0, 1.0, waist[1]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )),
    }
    values.update(changes)
    return G1HeadKinematics(**values)


def test_verified_raw_snapshot_converts_to_camera_observation_only() -> None:
    camera = camera_observation_from_snapshot(_snapshot(), _calibration())

    assert camera.frame == "head_color_optical"
    assert camera.camera_pose.frame == "base_link"
    np.testing.assert_allclose(camera.camera_pose.position, (0.4, 0.2, 0.7))
    np.testing.assert_allclose(camera.depth_m, np.full((3, 4), 1.0))
    np.testing.assert_allclose(
        camera.intrinsics,
        [[100.0, 0.0, 2.0], [0.0, 101.0, 1.0], [0.0, 0.0, 1.0]],
    )
    assert camera.timestamp_s == 1.0
    assert not camera.rgb.flags.writeable
    assert not camera.depth_m.flags.writeable


def test_dynamic_camera_pose_uses_snapshot_head_and_waist_fk() -> None:
    calls = []

    def fake_fk(head, waist):
        calls.append((head, waist))
        return np.array(
            [
                [1.0, 0.0, 0.0, head[0]],
                [0.0, 1.0, 0.0, head[1]],
                [0.0, 0.0, 1.0, waist[1]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    snapshot = _snapshot(
        head_yaw_pitch=(0.2, -0.1), waist_pitch_lift=(0.05, 0.4)
    )
    camera = camera_observation_from_snapshot(
        snapshot,
        _dynamic_calibration(),
        head_kinematics=_head_kinematics(base_from_mount=fake_fk),
    )

    assert calls[0][0] == pytest.approx((0.2, -0.1))
    assert calls[0][1] == pytest.approx((0.05, 0.4))
    np.testing.assert_allclose(camera.camera_pose.position, (0.3, 0.1, 0.7))
    assert camera.timestamp_s == 1.0


def test_dynamic_camera_pose_tracks_changed_head_pose() -> None:
    calibration = _dynamic_calibration()
    kinematics = _head_kinematics()
    first = camera_observation_from_snapshot(
        _snapshot(head_yaw_pitch=(0.0, 0.0)), calibration, head_kinematics=kinematics
    )
    second = camera_observation_from_snapshot(
        _snapshot(head_yaw_pitch=(0.4, -0.2)), calibration, head_kinematics=kinematics
    )

    np.testing.assert_allclose(first.camera_pose.position, (0.1, 0.2, 0.3))
    np.testing.assert_allclose(second.camera_pose.position, (0.5, 0.0, 0.3), atol=1e-7)


def test_dynamic_camera_requires_verified_fk_clock_and_matching_mount() -> None:
    with pytest.raises(OpenRailG1ObservationError, match="G1HeadKinematics provider"):
        camera_observation_from_snapshot(_snapshot(), _dynamic_calibration())

    with pytest.raises(OpenRailG1ObservationError, match="timestamps_share_clock_verified"):
        camera_observation_from_snapshot(
            _snapshot(),
            _dynamic_calibration(timestamps_share_clock_verified=False),
            head_kinematics=_head_kinematics(),
        )

    with pytest.raises(OpenRailG1ObservationError, match="kinematics.verified"):
        camera_observation_from_snapshot(
            _snapshot(),
            _dynamic_calibration(),
            head_kinematics=_head_kinematics(verified=False),
        )

    with pytest.raises(OpenRailG1ObservationError, match="state_units_verified"):
        camera_observation_from_snapshot(
            _snapshot(),
            _dynamic_calibration(state_units_verified=False),
            head_kinematics=_head_kinematics(),
        )

    with pytest.raises(OpenRailG1ObservationError, match="kinematics.state_units_verified"):
        camera_observation_from_snapshot(
            _snapshot(),
            _dynamic_calibration(),
            head_kinematics=_head_kinematics(state_units_verified=False),
        )

    with pytest.raises(OpenRailG1ObservationError, match="mount frame"):
        camera_observation_from_snapshot(
            _snapshot(),
            _dynamic_calibration(mount_frame="some_other_link"),
            head_kinematics=_head_kinematics(),
        )

    with pytest.raises(OpenRailG1ObservationError, match="model identity"):
        camera_observation_from_snapshot(
            _snapshot(),
            _dynamic_calibration(fk_model_id="different-model"),
            head_kinematics=_head_kinematics(),
        )

    unbound_state = replace(
        _snapshot(),
        metadata={
            **_snapshot().metadata,
            "state_timestamps_bind_state_values": False,
        },
    )
    with pytest.raises(OpenRailG1ObservationError, match="state values bound"):
        camera_observation_from_snapshot(
            unbound_state,
            _dynamic_calibration(),
            head_kinematics=_head_kinematics(),
        )


def test_dynamic_camera_rejects_state_timestamp_outside_verified_limit() -> None:
    snapshot = _snapshot(head_timestamp_ns=1_002_000_000)
    with pytest.raises(OpenRailG1ObservationError, match="synchronization threshold"):
        camera_observation_from_snapshot(
            snapshot, _dynamic_calibration(), head_kinematics=_head_kinematics()
        )


def test_verified_invalid_depth_sentinels_become_zero_depth() -> None:
    snapshot = _snapshot()
    raw_depth = np.array(
        [[0, 65535, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]], dtype=np.uint16
    )
    snapshot = replace(snapshot, depth_u16=raw_depth)

    camera = camera_observation_from_snapshot(snapshot, _calibration())

    np.testing.assert_array_equal(camera.depth_m[0], [0.0, 0.0, 0.001, 0.002])


def test_invalid_depth_values_require_explicit_verified_uint16_values() -> None:
    with pytest.raises(OpenRailG1ObservationError, match="requires verified"):
        camera_observation_from_snapshot(
            _snapshot(),
            _calibration(invalid_depth_values=(0, 65535), invalid_depth_values_verified=False),
        )

    with pytest.raises(OpenRailG1ObservationError, match="explicitly verified sequence"):
        camera_observation_from_snapshot(
            _snapshot(),
            _calibration(invalid_depth_values=None, invalid_depth_values_verified=True),
        )

    with pytest.raises(OpenRailG1ObservationError, match="uint16 integer values"):
        camera_observation_from_snapshot(
            _snapshot(),
            _calibration(invalid_depth_values=(0, 65536), invalid_depth_values_verified=True),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("depth_scale_verified", False),
        ("depth_registered_to_rgb", False),
        ("depth_registration_verified", False),
        ("intrinsics_verified", False),
        ("rgb_depth_rectified_for_pinhole", False),
        ("rectification_verified", False),
        ("camera_to_base_verified", False),
        ("synchronization_threshold_verified", False),
        ("mount_posture_verified", False),
    ],
)
def test_unverified_calibration_is_rejected(field: str, value: bool) -> None:
    with pytest.raises(OpenRailG1ObservationError, match="requires verified"):
        camera_observation_from_snapshot(_snapshot(), _calibration(**{field: value}))


def test_rejects_nonmatching_rgb_depth_grid_and_intrinsics_resolution() -> None:
    with pytest.raises(OpenRailG1ObservationError, match="matching dimensions"):
        camera_observation_from_snapshot(_snapshot(depth_shape=(2, 4)), _calibration())
    with pytest.raises(OpenRailG1ObservationError, match="intrinsics resolution"):
        camera_observation_from_snapshot(
            _snapshot(), _calibration(intrinsics_resolution_hw=(4, 4))
        )


def test_rejects_snapshot_outside_sync_threshold_or_missing_relevant_timestamps() -> None:
    late_depth = _snapshot(depth_timestamp_ns=1_001_000_000)
    with pytest.raises(OpenRailG1ObservationError, match="synchronization threshold"):
        camera_observation_from_snapshot(late_depth, _calibration())

    missing_head_time = _snapshot(head_timestamp_ns=None)
    with pytest.raises(OpenRailG1ObservationError, match="head timestamp"):
        camera_observation_from_snapshot(missing_head_time, _calibration())


def test_rejects_head_or_waist_pose_outside_static_camera_calibration_posture() -> None:
    head_moved = _snapshot(head_yaw_pitch=(0.02, 0.0))
    with pytest.raises(OpenRailG1ObservationError, match="head posture"):
        camera_observation_from_snapshot(head_moved, _calibration())

    waist_moved = _snapshot(waist_pitch_lift=(0.0, 0.02))
    with pytest.raises(OpenRailG1ObservationError, match="waist posture"):
        camera_observation_from_snapshot(waist_moved, _calibration())


def test_rejects_non_g1_or_non_rgb_z16_snapshots() -> None:
    snapshot = _snapshot()
    bad_robot = replace(snapshot, metadata={**snapshot.metadata, "robot_type": "other"})
    with pytest.raises(OpenRailG1ObservationError, match="AgiBot G1"):
        camera_observation_from_snapshot(bad_robot, _calibration())

    with pytest.raises(OpenRailG1ObservationError, match="key does not match"):
        camera_observation_from_snapshot(
            _snapshot(), _calibration(public_camera_key="other_camera")
        )

    bad_order = replace(snapshot, metadata={**snapshot.metadata, "rgb_order": "BGR"})
    with pytest.raises(OpenRailG1ObservationError, match="RGB channel order"):
        camera_observation_from_snapshot(bad_order, _calibration())

    bad_packet_timestamp = replace(
        snapshot,
        metadata={**snapshot.metadata, "rgb_packet_timestamp_ns": snapshot.rgb_timestamp_ns + 1},
    )
    with pytest.raises(OpenRailG1ObservationError, match="not bound"):
        camera_observation_from_snapshot(bad_packet_timestamp, _calibration())

    bad_rgb_decoder = replace(
        snapshot,
        metadata={**snapshot.metadata, "rgb_encoding_format": "unknown"},
    )
    with pytest.raises(OpenRailG1ObservationError, match="no verified RGB decoder"):
        camera_observation_from_snapshot(bad_rgb_decoder, _calibration())

    bad_encoding = replace(
        snapshot, metadata={**snapshot.metadata, "depth_encoding": "unknown"}
    )
    with pytest.raises(OpenRailG1ObservationError, match="RS2_FORMAT_Z16"):
        camera_observation_from_snapshot(bad_encoding, _calibration())


def test_rejects_invalid_scale_transform_and_g1_layout() -> None:
    with pytest.raises(OpenRailG1ObservationError, match="depth_scale_m_per_unit"):
        camera_observation_from_snapshot(
            _snapshot(), _calibration(depth_scale_m_per_unit=0.0)
        )

    with pytest.raises(OpenRailG1ObservationError, match="rigid transform"):
        camera_observation_from_snapshot(
            _snapshot(), _calibration(base_from_camera=np.zeros((4, 4)))
        )

    snapshot = _snapshot()
    wrong_layout = replace(
        snapshot,
        metadata={
            **snapshot.metadata,
            "state_layout": {
                "head_yaw_pitch_rad": [0, 2],
                "waist_pitch_rad_lift_m": [18, 20],
            },
        },
    )
    with pytest.raises(OpenRailG1ObservationError, match="verified G1 head/waist"):
        camera_observation_from_snapshot(wrong_layout, _calibration())
