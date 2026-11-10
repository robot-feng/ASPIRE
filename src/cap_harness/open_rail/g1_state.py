"""Convert a bound G1 state sample using explicitly verified kinematics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cap_harness.contracts import Pose, RobotState
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailSnapshot
from cap_harness.open_rail.robot_profile import OpenRailProfile


@dataclass(frozen=True, slots=True)
class G1StateKinematics:
    """FK and feedback calibration verified against the actual G1.

    ``end_effector_pose`` calculates the configured link/TCP pose in the
    profile's base frame from arm joints and the same sample's head and waist.
    The callback must be pure and must not request a newer robot state.
    ``wire_joint_names`` lists the actual DDS names in the same per-arm order
    as the profile's joint names; the correspondence must be verified.
    """

    model: str
    verified: bool
    joint_order_verified: bool
    position_velocity_units_verified: bool
    gripper_feedback_verified: bool
    synchronization_verified: bool
    timestamps_share_clock_verified: bool
    wire_joint_names: Mapping[str, Sequence[str]]
    end_effector_pose: Callable[[str, str, np.ndarray, np.ndarray, np.ndarray], Pose]


def robot_state_from_snapshot(
    snapshot: OpenRailSnapshot,
    profile: OpenRailProfile,
    kinematics: G1StateKinematics,
) -> RobotState:
    """Produce measured CAP state, or fail rather than inventing missing data."""
    if not isinstance(snapshot, OpenRailSnapshot) or snapshot.metadata.get("robot_type") != "agibot_g1":
        raise OpenRailG1ObservationError("robot state conversion requires a G1 snapshot")
    if not isinstance(profile, OpenRailProfile) or profile.robot_type != "agibot_g1":
        raise OpenRailG1ObservationError("robot state conversion requires a G1 profile")
    if not isinstance(kinematics, G1StateKinematics):
        raise OpenRailG1ObservationError("robot state conversion requires verified G1 kinematics")
    for name in ("verified", "joint_order_verified", "position_velocity_units_verified",
                 "gripper_feedback_verified", "synchronization_verified", "timestamps_share_clock_verified"):
        if getattr(kinematics, name) is not True:
            raise OpenRailG1ObservationError(f"robot state conversion requires verified {name}")
    if kinematics.model != profile.model or not callable(kinematics.end_effector_pose):
        raise OpenRailG1ObservationError("verified FK model and callback must match the G1 profile")
    if set(profile.arms) != {"left", "right"} or profile.joint_dimension != 7:
        raise OpenRailG1ObservationError("G1 state requires configured left and right seven-joint arms")
    sampled_names = snapshot.metadata.get("arm_joint_names")
    wire_names = kinematics.wire_joint_names
    if not isinstance(wire_names, Mapping) or set(wire_names) != {"left", "right"}:
        raise OpenRailG1ObservationError("verified wire joint mapping must name both arms")
    if any(
        not isinstance(wire_names[arm], Sequence)
        or isinstance(wire_names[arm], (str, bytes))
        or len(wire_names[arm]) != 7
        for arm in profile.arms
    ):
        raise OpenRailG1ObservationError("verified wire joint mapping must contain seven names per arm")
    expected_names = tuple(wire_names["left"]) + tuple(wire_names["right"])
    if any(not isinstance(name, str) or not name for name in expected_names) or len(set(expected_names)) != 14:
        raise OpenRailG1ObservationError("verified wire joint mapping must contain unique names")
    if not isinstance(sampled_names, (list, tuple)) or tuple(sampled_names) != expected_names:
        raise OpenRailG1ObservationError("sampled G1 joint names do not match the robot profile")
    expected_layout = {
        "left_arm_rad": (0, 7), "right_arm_rad": (7, 14),
        "left_gripper_normalized": (14, 15), "right_gripper_normalized": (15, 16),
        "head_yaw_pitch_rad": (16, 18), "waist_pitch_rad_lift_m": (18, 20),
    }
    layout = snapshot.metadata.get("state_layout")
    if not isinstance(layout, dict) or any(tuple(layout.get(name, ())) != limits for name, limits in expected_layout.items()):
        raise OpenRailG1ObservationError("G1 state layout does not match the configured format")
    if not snapshot.is_synchronized:
        raise OpenRailG1ObservationError("G1 state and RGB-D snapshot are not synchronized")
    values = np.asarray(snapshot.state, dtype=np.float64)
    velocity = np.asarray(snapshot.arm_velocities_rad_s, dtype=np.float64)
    if values.shape != (20,) or not np.all(np.isfinite(values)):
        raise OpenRailG1ObservationError("G1 snapshot requires 20 finite state values")
    if velocity.shape != (14,) or not np.all(np.isfinite(velocity)):
        raise OpenRailG1ObservationError("G1 snapshot requires 14 measured arm velocities")
    if np.any(values[14:16] < 0.0) or np.any(values[14:16] > 1.0):
        raise OpenRailG1ObservationError("G1 gripper feedback is outside its normalized range")
    positions = {"left": values[:7].copy(), "right": values[7:14].copy()}
    velocities = {"left": velocity[:7].copy(), "right": velocity[7:14].copy()}
    poses = {}
    for arm in profile.arms:
        pose = kinematics.end_effector_pose(
            arm, profile.end_effector_links[arm], positions[arm].copy(),
            values[16:18].copy(), values[18:20].copy(),
        )
        if not isinstance(pose, Pose) or pose.frame != profile.base_frame:
            raise OpenRailG1ObservationError(f"FK returned an invalid {arm} pose or frame")
        poses[arm] = pose
    # Open-RAIL's G1 feedback uses 0=open, 1=closed; CAP uses the inverse.
    return RobotState(
        joint_positions=positions,
        joint_velocities=velocities,
        end_effector_poses=poses,
        gripper_positions={"left": 1.0 - values[14], "right": 1.0 - values[15]},
        base_frame=profile.base_frame,
        joint_names=profile.joint_names,
        timestamp_s=snapshot.state_timestamps_ns["arm"] / 1_000_000_000.0,
        embodiment=profile.embodiment,
    )
