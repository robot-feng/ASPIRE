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


@dataclass(frozen=True, slots=True)
class G1ArmSideFeedbackCalibration:
    """Verified mapping and tolerances for the separate arm feedback streams.

    The per-side mappings go from physical motor ID to the *profile joint name*.
    Units must have been independently established as radians and radians/second;
    no conversion, signs, offsets, ordering or thresholds are inferred.
    """

    motor_id_to_joint: Mapping[str, Mapping[int, str]]
    motor_id_mapping_verified: bool
    position_unit: str
    velocity_unit: str
    units_verified: bool
    timestamps_share_clock_verified: bool
    max_timestamp_skew_ns: int
    max_sample_age_ns: int
    max_position_error_rad: float


def _side_velocities(
    snapshot: OpenRailSnapshot,
    profile: OpenRailProfile,
    values: np.ndarray,
    calibration: G1ArmSideFeedbackCalibration,
    reference_timestamp_ns: int | None,
) -> np.ndarray:
    if not isinstance(calibration, G1ArmSideFeedbackCalibration):
        raise OpenRailG1ObservationError("side feedback requires explicit calibration")
    for name in ("motor_id_mapping_verified", "units_verified", "timestamps_share_clock_verified"):
        if getattr(calibration, name) is not True:
            raise OpenRailG1ObservationError(f"side feedback requires verified {name}")
    if calibration.position_unit != "rad" or calibration.velocity_unit != "rad/s":
        raise OpenRailG1ObservationError("side feedback units must be verified rad and rad/s")
    for name in ("max_timestamp_skew_ns", "max_sample_age_ns"):
        value = getattr(calibration, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise OpenRailG1ObservationError(f"side feedback requires nonnegative integer {name}")
    tolerance = calibration.max_position_error_rad
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(tolerance) or tolerance < 0
    ):
        raise OpenRailG1ObservationError("side feedback requires a finite position tolerance in radians")
    if (
        isinstance(reference_timestamp_ns, (bool, np.bool_))
        or not isinstance(reference_timestamp_ns, (int, np.integer))
        or reference_timestamp_ns <= 0
    ):
        raise OpenRailG1ObservationError("side feedback requires a current shared-clock reference timestamp")
    mapping = calibration.motor_id_to_joint
    sides = snapshot.metadata.get("arm_side_feedback")
    if not isinstance(mapping, Mapping) or set(mapping) != {"left", "right"}:
        raise OpenRailG1ObservationError("side feedback requires explicit motor mappings for both arms")
    if not isinstance(sides, Mapping) or set(sides) != {"left", "right"}:
        raise OpenRailG1ObservationError("side feedback requires samples for both arms")

    stamps = [snapshot.rgb_timestamp_ns, snapshot.state_timestamps_ns["arm"]]
    velocities = []
    for index, side in enumerate(("left", "right")):
        side_map = mapping[side]
        if (
            not isinstance(side_map, Mapping) or len(side_map) != 7
            or any(isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)) or key < 0 for key in side_map)
            or any(not isinstance(joint, str) for joint in side_map.values())
            or set(side_map.values()) != set(profile.joint_names[side])
        ):
            raise OpenRailG1ObservationError(f"{side} motor mapping must bijectively cover the profile joints")
        feedback = sides[side]
        if not isinstance(feedback, Mapping):
            raise OpenRailG1ObservationError(f"{side} side feedback is missing")
        timestamp = feedback.get("timestamp_ns")
        if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, (int, np.integer)) or timestamp <= 0:
            raise OpenRailG1ObservationError(f"{side} side feedback has an invalid timestamp")
        stamps.append(int(timestamp))
        ids = feedback.get("motor_ids")
        if (
            not isinstance(ids, (list, tuple)) or len(ids) != 7
            or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in ids)
            or len(set(ids)) != 7 or set(ids) != set(side_map)
        ):
            raise OpenRailG1ObservationError(f"{side} side feedback must contain the seven mapped unique IDs")
        try:
            raw_positions = np.asarray(feedback.get("positions_raw"))
            raw_velocities = np.asarray(feedback.get("velocities_raw"))
            if (
                raw_positions.shape != (7,) or raw_velocities.shape != (7,)
                or raw_positions.dtype.kind not in "iuf" or raw_velocities.dtype.kind not in "iuf"
                or not np.all(np.isfinite(raw_positions)) or not np.all(np.isfinite(raw_velocities))
            ):
                raise ValueError("expected seven finite numeric positions and velocities")
        except (TypeError, ValueError) as exc:
            raise OpenRailG1ObservationError(f"{side} side feedback has invalid positions or velocities") from exc
        id_by_joint = {joint: motor_id for motor_id, joint in side_map.items()}
        indices = [ids.index(id_by_joint[joint]) for joint in profile.joint_names[side]]
        ordered_positions = raw_positions[indices].astype(np.float64)
        if np.any(np.abs(ordered_positions - values[index * 7 : index * 7 + 7]) > tolerance):
            raise OpenRailG1ObservationError(f"{side} side positions disagree with the combined state")
        velocities.append(raw_velocities[indices].astype(np.float64))

    if max(stamps) - min(stamps) > calibration.max_timestamp_skew_ns:
        raise OpenRailG1ObservationError("side feedback timestamps exceed the verified skew")
    observed_stamps = [*stamps, snapshot.depth_timestamp_ns, *snapshot.state_timestamps_ns.values()]
    if any(timestamp > reference_timestamp_ns or reference_timestamp_ns - timestamp > calibration.max_sample_age_ns for timestamp in observed_stamps):
        raise OpenRailG1ObservationError("side feedback or observation is stale or in the future")
    return np.concatenate(velocities)


def robot_state_from_snapshot(
    snapshot: OpenRailSnapshot,
    profile: OpenRailProfile,
    kinematics: G1StateKinematics,
    *,
    side_feedback: G1ArmSideFeedbackCalibration | None = None,
    reference_timestamp_ns: int | None = None,
) -> RobotState:
    """Produce measured CAP state, or fail rather than inventing missing data.

    The default path retains the 14-value combined velocity sample. The
    separate-stream path is selected explicitly with ``side_feedback`` and
    requires ``reference_timestamp_ns`` to be a current reading from the
    verified common clock, not a timestamp copied from the snapshot itself.
    """
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
    if values.shape != (20,) or not np.all(np.isfinite(values)):
        raise OpenRailG1ObservationError("G1 snapshot requires 20 finite state values")
    velocity = (
        np.asarray(snapshot.arm_velocities_rad_s, dtype=np.float64)
        if side_feedback is None
        else _side_velocities(snapshot, profile, values, side_feedback, reference_timestamp_ns)
    )
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
