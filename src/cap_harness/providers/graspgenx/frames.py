"""Frame transforms for GraspGenX poses and calibrated robot TCP targets."""

from __future__ import annotations

import numpy as np

from cap_harness.contracts import Pose
from cap_harness.geometry import invert_transform, matrix_to_pose


def _gripper_base_to_tcp(transform: object) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("gripper_base_to_tcp must have shape (4, 4)")
    # Validate finiteness, homogeneous form, and a proper rigid rotation.
    matrix_to_pose(value, frame="gripper_base")
    return value


def gripper_base_pose_to_tcp(
    grasp_base_pose: Pose,
    gripper_base_to_tcp: object,
) -> Pose:
    """Convert a GraspGenX gripper-base pose to a measured TCP pose.

    ``gripper_base_to_tcp`` is ``T_gripper_base_tcp``: the TCP pose expressed
    in the GraspGenX gripper-base frame. The input and output poses remain in
    the same parent frame.
    """
    if not isinstance(grasp_base_pose, Pose):
        raise ValueError("grasp_base_pose must be a Pose")
    base_to_tcp = _gripper_base_to_tcp(gripper_base_to_tcp)
    return matrix_to_pose(
        grasp_base_pose.as_matrix() @ base_to_tcp,
        frame=grasp_base_pose.frame,
    )


def tcp_pregrasp_pose(
    grasp_tcp_pose: Pose,
    gripper_base_to_tcp: object,
    *,
    distance_m: float,
) -> Pose:
    """Return a pregrasp TCP pose backing away along GraspGenX local +Z.

    GraspGenX documents local +Z as the approach axis; its OBB candidate
    builder points +Z into the object, so this retreats by ``distance_m`` along
    local -Z. The retreat is computed in gripper-base coordinates before the
    measured tool transform is reapplied, preserving that axis even when the
    TCP has a rotational offset.
    """
    if not isinstance(grasp_tcp_pose, Pose):
        raise ValueError("grasp_tcp_pose must be a Pose")
    if isinstance(distance_m, bool):
        raise ValueError("distance_m must be a positive finite number")
    distance = float(distance_m)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be a positive finite number")

    base_to_tcp = _gripper_base_to_tcp(gripper_base_to_tcp)
    grasp_base = grasp_tcp_pose.as_matrix() @ invert_transform(base_to_tcp)
    retreat = np.eye(4, dtype=np.float64)
    retreat[2, 3] = -distance
    pregrasp_tcp = grasp_base @ retreat @ base_to_tcp
    return matrix_to_pose(pregrasp_tcp, frame=grasp_tcp_pose.frame)


__all__ = ["gripper_base_pose_to_tcp", "tcp_pregrasp_pose"]
