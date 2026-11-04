"""Calibration-gated conversion of a raw Open-RAIL G1 snapshot.

This module never opens a camera or robot connection. It only converts one
already captured raw snapshot after the caller supplies verified calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cap_harness.contracts import CameraObservation
from cap_harness.geometry import matrix_to_pose
from cap_harness.open_rail.g1_runtime import (
    OpenRailG1ObservationError,
    OpenRailSnapshot,
)


@dataclass(frozen=True, slots=True)
class G1CameraCalibration:
    """Caller-supplied, explicitly verified calibration for one G1 camera.

    ``base_from_camera`` is the transform from the named camera optical frame
    into ``base_frame``. Head/waist reference posture and tolerances bind that
    transform to the snapshot when the camera is mounted on moving joints.
    """

    depth_scale_m_per_unit: float
    depth_scale_verified: bool
    depth_registered_to_rgb: bool
    depth_registration_verified: bool
    intrinsics: np.ndarray
    intrinsics_resolution_hw: tuple[int, int]
    intrinsics_verified: bool
    rgb_depth_rectified_for_pinhole: bool
    rectification_verified: bool
    base_from_camera: np.ndarray
    camera_to_base_verified: bool
    public_camera_key: str
    base_frame: str
    camera_frame: str
    maximum_snapshot_skew_ms: float
    synchronization_threshold_verified: bool
    calibration_head_yaw_pitch_rad: tuple[float, float]
    head_yaw_pitch_tolerance_rad: tuple[float, float]
    calibration_waist_pitch_lift_m: tuple[float, float]
    waist_pitch_lift_tolerance: tuple[float, float]
    mount_posture_verified: bool
    invalid_depth_values: tuple[int, ...] | None = None
    invalid_depth_values_verified: bool = False


@dataclass(frozen=True, slots=True)
class G1DynamicCameraCalibration:
    """Verified RGB-D calibration for a camera rigidly attached to a moving link.

    ``mount_from_camera`` maps points in ``camera_frame`` into the explicitly
    named ``mount_frame``. It is composed with a per-snapshot FK result; it is
    never treated as a static base transform.
    """

    depth_scale_m_per_unit: float
    depth_scale_verified: bool
    depth_registered_to_rgb: bool
    depth_registration_verified: bool
    intrinsics: np.ndarray
    intrinsics_resolution_hw: tuple[int, int]
    intrinsics_verified: bool
    rgb_depth_rectified_for_pinhole: bool
    rectification_verified: bool
    mount_from_camera: np.ndarray
    mount_from_camera_verified: bool
    public_camera_key: str
    base_frame: str
    mount_frame: str
    camera_frame: str
    maximum_snapshot_skew_ms: float
    synchronization_threshold_verified: bool
    timestamps_share_clock_verified: bool
    state_layout_verified: bool
    state_units_verified: bool
    fk_model_id: str
    invalid_depth_values: tuple[int, ...] | None = None
    invalid_depth_values_verified: bool = False


@dataclass(frozen=True, slots=True)
class G1HeadKinematics:
    """Caller-supplied, verified FK for the camera's actual mounting link.

    The callable receives the snapshot's ``(head_yaw, head_pitch)`` and
    ``(waist_pitch, lift_m)`` values and returns ``base_from_mount``. The
    provider must be backed by a verified model, joint ordering, units, and
    mount-link identity. It must be a pure calculation from those supplied
    values; this module does not choose a URDF, infer a link, or access hardware.
    """

    base_frame: str
    mount_frame: str
    model_id: str
    verified: bool
    state_layout_verified: bool
    state_units_verified: bool
    base_from_mount: Callable[[tuple[float, float], tuple[float, float]], Any]


def _require_verified(calibration: G1CameraCalibration) -> None:
    checks = (
        ("depth_scale_verified", calibration.depth_scale_verified),
        ("depth_registered_to_rgb", calibration.depth_registered_to_rgb),
        ("depth_registration_verified", calibration.depth_registration_verified),
        ("intrinsics_verified", calibration.intrinsics_verified),
        ("rgb_depth_rectified_for_pinhole", calibration.rgb_depth_rectified_for_pinhole),
        ("rectification_verified", calibration.rectification_verified),
        ("camera_to_base_verified", calibration.camera_to_base_verified),
        ("synchronization_threshold_verified", calibration.synchronization_threshold_verified),
        ("mount_posture_verified", calibration.mount_posture_verified),
        ("invalid_depth_values_verified", calibration.invalid_depth_values_verified),
    )
    for name, value in checks:
        if type(value) is not bool or value is not True:
            raise OpenRailG1ObservationError(f"camera conversion requires verified {name}")


def _require_dynamic_verified(
    calibration: G1DynamicCameraCalibration,
    kinematics: G1HeadKinematics | None,
) -> G1HeadKinematics:
    checks = (
        ("depth_scale_verified", calibration.depth_scale_verified),
        ("depth_registered_to_rgb", calibration.depth_registered_to_rgb),
        ("depth_registration_verified", calibration.depth_registration_verified),
        ("intrinsics_verified", calibration.intrinsics_verified),
        ("rgb_depth_rectified_for_pinhole", calibration.rgb_depth_rectified_for_pinhole),
        ("rectification_verified", calibration.rectification_verified),
        ("mount_from_camera_verified", calibration.mount_from_camera_verified),
        ("synchronization_threshold_verified", calibration.synchronization_threshold_verified),
        ("timestamps_share_clock_verified", calibration.timestamps_share_clock_verified),
        ("state_layout_verified", calibration.state_layout_verified),
        ("state_units_verified", calibration.state_units_verified),
        ("invalid_depth_values_verified", calibration.invalid_depth_values_verified),
    )
    for name, value in checks:
        if type(value) is not bool or value is not True:
            raise OpenRailG1ObservationError(f"dynamic camera conversion requires verified {name}")
    if kinematics is None or not isinstance(kinematics, G1HeadKinematics):
        raise OpenRailG1ObservationError(
            "dynamic camera conversion requires an explicitly verified G1HeadKinematics provider"
        )
    for name, value in (
        ("kinematics.verified", kinematics.verified),
        ("kinematics.state_layout_verified", kinematics.state_layout_verified),
        ("kinematics.state_units_verified", kinematics.state_units_verified),
    ):
        if type(value) is not bool or value is not True:
            raise OpenRailG1ObservationError(f"dynamic camera conversion requires {name}")
    if not callable(kinematics.base_from_mount):
        raise OpenRailG1ObservationError("kinematics.base_from_mount must be callable")
    for name, value in (
        ("calibration.base_frame", calibration.base_frame),
        ("calibration.mount_frame", calibration.mount_frame),
        ("calibration.camera_frame", calibration.camera_frame),
        ("calibration.fk_model_id", calibration.fk_model_id),
        ("kinematics.base_frame", kinematics.base_frame),
        ("kinematics.mount_frame", kinematics.mount_frame),
        ("kinematics.model_id", kinematics.model_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise OpenRailG1ObservationError(f"dynamic camera conversion requires explicit {name}")
        if value != value.strip():
            raise OpenRailG1ObservationError(f"dynamic camera conversion requires normalized {name}")
    if calibration.base_frame != kinematics.base_frame:
        raise OpenRailG1ObservationError("FK base frame does not match camera calibration")
    if calibration.mount_frame != kinematics.mount_frame:
        raise OpenRailG1ObservationError("FK mount frame does not match camera calibration")
    if calibration.fk_model_id != kinematics.model_id:
        raise OpenRailG1ObservationError("FK model identity does not match camera calibration")
    if calibration.base_frame == calibration.camera_frame:
        raise OpenRailG1ObservationError("camera and base frames must be distinct")
    if calibration.mount_frame == calibration.camera_frame:
        raise OpenRailG1ObservationError("camera and mount frames must be distinct")
    return kinematics


def _validated_rigid_transform(value: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
            raise ValueError(f"{name} is not numeric")
        transform = raw.astype(np.float64)
        matrix_to_pose(transform, frame="validation")
    except (TypeError, ValueError) as exc:
        raise OpenRailG1ObservationError(f"{name} must be a valid rigid transform") from exc
    return transform


def _invalid_depth_values(
    calibration: G1CameraCalibration | G1DynamicCameraCalibration,
) -> tuple[int, ...]:
    values = calibration.invalid_depth_values
    if values is None or isinstance(values, (str, bytes, Mapping)):
        raise OpenRailG1ObservationError(
            "invalid_depth_values must be an explicitly verified sequence of uint16 values"
        )
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise OpenRailG1ObservationError(
            "invalid_depth_values must be an explicitly verified sequence of uint16 values"
        ) from exc
    normalized: list[int] = []
    for value in raw_values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= np.iinfo(np.uint16).max
        ):
            raise OpenRailG1ObservationError(
                "invalid_depth_values must contain only uint16 integer values"
            )
        normalized.append(int(value))
    if len(normalized) != len(set(normalized)):
        raise OpenRailG1ObservationError("invalid_depth_values must not contain duplicates")
    return tuple(normalized)


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise OpenRailG1ObservationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OpenRailG1ObservationError(f"{name} must be a finite number") from exc
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise OpenRailG1ObservationError(f"{name} is outside its valid range")
    return result


def _pair(value: Any, name: str, *, minimum: float | None = None) -> np.ndarray:
    if isinstance(value, (str, bytes, Mapping)):
        raise OpenRailG1ObservationError(f"{name} must contain two numbers")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise OpenRailG1ObservationError(f"{name} must contain two numbers") from exc
    if len(raw) != 2:
        raise OpenRailG1ObservationError(f"{name} must contain two numbers")
    return np.asarray(
        [_finite_number(item, name, minimum=minimum) for item in raw],
        dtype=np.float64,
    )


def _timestamp_ns(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise OpenRailG1ObservationError(f"snapshot is missing a valid {name} timestamp")
    result = int(value)
    if result < 0:
        raise OpenRailG1ObservationError(f"snapshot has an invalid {name} timestamp")
    return result


def _state_pair(snapshot: OpenRailSnapshot, key: str) -> np.ndarray:
    layout = snapshot.metadata.get("state_layout")
    if not isinstance(layout, Mapping):
        raise OpenRailG1ObservationError("snapshot has no verified G1 state layout")
    bounds = layout.get(key)
    if (
        not isinstance(bounds, (list, tuple))
        or len(bounds) != 2
        or any(isinstance(index, bool) or not isinstance(index, int) for index in bounds)
    ):
        raise OpenRailG1ObservationError(f"snapshot state layout is missing {key}")
    start, stop = bounds
    if stop - start != 2 or start < 0 or stop > snapshot.state.size:
        raise OpenRailG1ObservationError(f"snapshot state layout has an invalid {key} range")
    values = np.asarray(snapshot.state[start:stop], dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise OpenRailG1ObservationError(f"snapshot contains invalid state values for {key}")
    return values


def camera_observation_from_snapshot(
    snapshot: OpenRailSnapshot,
    calibration: G1CameraCalibration | G1DynamicCameraCalibration,
    *,
    head_kinematics: G1HeadKinematics | None = None,
) -> CameraObservation:
    """Convert raw RGB/Z16 to a calibrated pinhole ``CameraObservation``.

    ``G1CameraCalibration`` retains the fixed-camera behavior and checks that
    head/waist posture matches the calibration reference. For a moving head
    camera, pass ``G1DynamicCameraCalibration`` plus a matching, verified
    ``head_kinematics`` provider; its per-snapshot FK is composed with the
    calibrated mount-to-camera transform. Every calibration and synchronization
    claim is required explicitly. This does not construct a full ASPIRE
    ``Observation``: that also requires a verified arm model and end-effector
    state derived from this same snapshot.
    """
    if not isinstance(snapshot, OpenRailSnapshot):
        raise OpenRailG1ObservationError("snapshot must be an OpenRailSnapshot")
    if isinstance(calibration, G1CameraCalibration):
        if head_kinematics is not None:
            raise OpenRailG1ObservationError(
                "head_kinematics is only accepted with G1DynamicCameraCalibration"
            )
        _require_verified(calibration)
    elif isinstance(calibration, G1DynamicCameraCalibration):
        head_kinematics = _require_dynamic_verified(calibration, head_kinematics)
    else:
        raise OpenRailG1ObservationError(
            "calibration must be a G1CameraCalibration or G1DynamicCameraCalibration"
        )

    if snapshot.metadata.get("robot_type") != "agibot_g1":
        raise OpenRailG1ObservationError("camera conversion requires an AgiBot G1 snapshot")
    if isinstance(calibration, G1DynamicCameraCalibration) and (
        snapshot.metadata.get("state_timestamps_bind_state_values") is not True
    ):
        raise OpenRailG1ObservationError(
            "dynamic camera conversion requires state values bound to their reported DDS timestamps"
        )
    if snapshot.metadata.get("rgb_order") != "RGB":
        raise OpenRailG1ObservationError("snapshot RGB channel order is not verified as RGB")
    if snapshot.metadata.get("rgb_packet_timestamp_ns") != snapshot.rgb_timestamp_ns:
        raise OpenRailG1ObservationError("RGB packet is not bound to the snapshot timestamp")
    rgb_encoding = snapshot.metadata.get("rgb_encoding_format")
    rgb_color_format = snapshot.metadata.get("rgb_color_format")
    rgb_decoder = snapshot.metadata.get("rgb_decoder")
    if (
        rgb_color_format not in {"RGB", "BGR"}
        or (rgb_encoding, rgb_decoder)
        not in {
            ("JPEG", "opencv-imdecode-bgr-to-rgb"),
            ("PNG", "opencv-imdecode-bgr-to-rgb"),
            ("UNCOMPRESSED", "raw-rgb"),
            ("UNCOMPRESSED", "raw-bgr-to-rgb"),
        }
    ):
        raise OpenRailG1ObservationError(
            "snapshot RGB packet encoding/color_format has no verified RGB decoder"
        )
    if (
        not isinstance(calibration.public_camera_key, str)
        or not calibration.public_camera_key.strip()
        or snapshot.metadata.get("public_camera_key") != calibration.public_camera_key.strip()
    ):
        raise OpenRailG1ObservationError("camera calibration key does not match the snapshot")
    if snapshot.depth_u16 is None or snapshot.depth_error is not None:
        raise OpenRailG1ObservationError("snapshot has no valid raw depth frame")
    if snapshot.metadata.get("depth_encoding") != "RS2_FORMAT_Z16":
        raise OpenRailG1ObservationError("snapshot depth is not marked RS2_FORMAT_Z16")
    invalid_depth_values = _invalid_depth_values(calibration)

    rgb = np.asarray(snapshot.rgb)
    depth_u16 = np.asarray(snapshot.depth_u16)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise OpenRailG1ObservationError("snapshot RGB must be uint8 with shape (H, W, 3)")
    if depth_u16.ndim != 2 or depth_u16.dtype != np.uint16:
        raise OpenRailG1ObservationError("snapshot depth must be a 2D uint16 Z16 image")
    if depth_u16.shape != rgb.shape[:2]:
        raise OpenRailG1ObservationError("RGB and depth grids do not have matching dimensions")

    scale = _finite_number(
        calibration.depth_scale_m_per_unit,
        "depth_scale_m_per_unit",
        minimum=np.finfo(np.float64).tiny,
    )
    try:
        resolution = tuple(calibration.intrinsics_resolution_hw)
    except TypeError as exc:
        raise OpenRailG1ObservationError(
            "intrinsics_resolution_hw must be a positive (H, W)"
        ) from exc
    if (
        isinstance(calibration.intrinsics_resolution_hw, (str, bytes))
        or len(resolution) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in resolution
        )
    ):
        raise OpenRailG1ObservationError("intrinsics_resolution_hw must be a positive (H, W)")
    if tuple(map(int, resolution)) != rgb.shape[:2]:
        raise OpenRailG1ObservationError("intrinsics resolution does not match RGB/depth grid")
    try:
        raw_intrinsics = np.asarray(calibration.intrinsics)
        if not np.issubdtype(raw_intrinsics.dtype, np.number) or np.issubdtype(
            raw_intrinsics.dtype, np.bool_
        ):
            raise ValueError("intrinsics are not numeric")
        intrinsics = raw_intrinsics.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise OpenRailG1ObservationError("intrinsics must be a numeric 3x3 matrix") from exc
    if (
        intrinsics.shape != (3, 3)
        or not np.all(np.isfinite(intrinsics))
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-6)
    ):
        raise OpenRailG1ObservationError("intrinsics must be a valid pinhole matrix")

    skew_limit_ms = _finite_number(
        calibration.maximum_snapshot_skew_ms,
        "maximum_snapshot_skew_ms",
        minimum=0.0,
    )
    rgb_ns = _timestamp_ns(snapshot.rgb_timestamp_ns, "RGB")
    depth_ns = _timestamp_ns(snapshot.depth_timestamp_ns, "depth")
    timestamps = snapshot.state_timestamps_ns
    if not isinstance(timestamps, Mapping):
        raise OpenRailG1ObservationError("snapshot has no state timestamp map")
    state_ns = {
        part: _timestamp_ns(timestamps.get(part), part)
        for part in ("head", "waist")
    }
    maximum_skew_ns = max(
        abs(depth_ns - rgb_ns),
        *(abs(stamp - rgb_ns) for stamp in state_ns.values()),
    )
    if maximum_skew_ns > skew_limit_ms * 1_000_000.0:
        raise OpenRailG1ObservationError(
            f"snapshot exceeds verified synchronization threshold: "
            f"{maximum_skew_ns / 1_000_000.0:.3f} ms > {skew_limit_ms:.3f} ms"
        )

    try:
        state = np.asarray(snapshot.state, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise OpenRailG1ObservationError("snapshot state must be numeric") from exc
    if state.ndim != 1 or state.shape != (20,):
        raise OpenRailG1ObservationError("snapshot does not contain the expected G1 state vector")
    if not np.all(np.isfinite(state)):
        raise OpenRailG1ObservationError("snapshot state contains non-finite values")
    layout = snapshot.metadata.get("state_layout")
    if not isinstance(layout, Mapping):
        raise OpenRailG1ObservationError("snapshot does not use the verified G1 state layout")
    head_layout = layout.get("head_yaw_pitch_rad")
    waist_layout = layout.get("waist_pitch_rad_lift_m")
    if (
        not isinstance(head_layout, (list, tuple))
        or not isinstance(waist_layout, (list, tuple))
        or tuple(head_layout) != (16, 18)
        or tuple(waist_layout) != (18, 20)
    ):
        raise OpenRailG1ObservationError(
            "snapshot does not use the verified G1 head/waist state layout"
        )
    current_head = _state_pair(snapshot, "head_yaw_pitch_rad")
    current_waist = _state_pair(snapshot, "waist_pitch_rad_lift_m")
    if isinstance(calibration, G1CameraCalibration):
        reference_head = _pair(
            calibration.calibration_head_yaw_pitch_rad, "calibration head posture"
        )
        head_tolerance = _pair(
            calibration.head_yaw_pitch_tolerance_rad,
            "head posture tolerance",
            minimum=0.0,
        )
        reference_waist = _pair(
            calibration.calibration_waist_pitch_lift_m,
            "calibration waist pitch/lift",
        )
        waist_tolerance = _pair(
            calibration.waist_pitch_lift_tolerance,
            "waist posture tolerance",
            minimum=0.0,
        )
        head_error = np.array(
            [
                abs((current_head[0] - reference_head[0] + np.pi) % (2.0 * np.pi) - np.pi),
                abs(current_head[1] - reference_head[1]),
            ]
        )
        waist_error = np.abs(current_waist - reference_waist)
        if np.any(head_error > head_tolerance):
            raise OpenRailG1ObservationError(
                "head posture differs from camera calibration posture"
            )
        if np.any(waist_error > waist_tolerance):
            raise OpenRailG1ObservationError(
                "waist posture differs from camera calibration posture"
            )

    base_frame = calibration.base_frame.strip() if isinstance(calibration.base_frame, str) else ""
    camera_frame = (
        calibration.camera_frame.strip() if isinstance(calibration.camera_frame, str) else ""
    )
    if not base_frame or not camera_frame or base_frame == camera_frame:
        raise OpenRailG1ObservationError("camera and base frames must be explicit and distinct")
    if isinstance(calibration, G1CameraCalibration):
        try:
            raw_transform = np.asarray(calibration.base_from_camera)
            if not np.issubdtype(raw_transform.dtype, np.number) or np.issubdtype(
                raw_transform.dtype, np.bool_
            ):
                raise ValueError("base_from_camera is not numeric")
            camera_pose = matrix_to_pose(raw_transform.astype(np.float64), frame=base_frame)
        except (TypeError, ValueError) as exc:
            raise OpenRailG1ObservationError(
                "base_from_camera must be a valid rigid transform"
            ) from exc
    else:
        assert head_kinematics is not None
        if not isinstance(calibration.mount_frame, str) or not calibration.mount_frame.strip():
            raise OpenRailG1ObservationError(
                "dynamic camera calibration requires an explicit mount frame"
            )
        try:
            base_from_mount = head_kinematics.base_from_mount(
                tuple(map(float, current_head)), tuple(map(float, current_waist))
            )
        except Exception as exc:
            raise OpenRailG1ObservationError(
                "verified FK provider failed to compute base_from_mount"
            ) from exc
        base_from_mount = _validated_rigid_transform(base_from_mount, "base_from_mount")
        mount_from_camera = _validated_rigid_transform(
            calibration.mount_from_camera, "mount_from_camera"
        )
        try:
            base_from_camera = base_from_mount @ mount_from_camera
            camera_pose = matrix_to_pose(base_from_camera, frame=base_frame)
        except ValueError as exc:
            raise OpenRailG1ObservationError(
                "dynamic base_from_camera composition is not a valid rigid transform"
            ) from exc

    depth_m = depth_u16.astype(np.float64) * scale
    if invalid_depth_values:
        depth_m[np.isin(depth_u16, invalid_depth_values)] = 0.0

    return CameraObservation(
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        frame=camera_frame,
        camera_pose=camera_pose,
        timestamp_s=rgb_ns / 1_000_000_000.0,
    )


__all__ = [
    "G1CameraCalibration",
    "G1DynamicCameraCalibration",
    "G1HeadKinematics",
    "camera_observation_from_snapshot",
]
