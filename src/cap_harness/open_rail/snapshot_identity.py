"""Content identity for a camera observation and its associated robot state."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from cap_harness.contracts import CameraObservation, RobotState


def snapshot_fingerprint(camera: CameraObservation, state: RobotState) -> str:
    """Bind perception to RGB-D, calibration and the complete measured arm state."""
    digest = hashlib.sha256()

    def add(value: object) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            header = json.dumps([array.dtype.str, array.shape]).encode("utf-8")
            payload = array.tobytes()
        else:
            header = b"json"
            payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    add([camera.frame, camera.camera_pose.frame, camera.timestamp_s])
    for value in (camera.rgb, camera.depth_m, camera.intrinsics, camera.camera_pose.as_matrix()):
        add(value)
    add([state.base_frame, state.embodiment, state.timestamp_s])
    for arm in state.arms:
        add([arm, list(state.joint_names[arm]), state.gripper_positions[arm]])
        add(state.joint_positions[arm])
        add(state.joint_velocities[arm])
        add(state.end_effector_poses[arm].as_matrix())
    return digest.hexdigest()


__all__ = ["snapshot_fingerprint"]
