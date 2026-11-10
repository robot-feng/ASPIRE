from dataclasses import replace

import numpy as np
import pytest

from cap_harness.contracts import Pose
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailSnapshot
from cap_harness.open_rail.g1_state import G1StateKinematics, robot_state_from_snapshot
from cap_harness.open_rail.robot_profile import OpenRailProfile


def _profile():
    names = {side: tuple(f"Joint{i}_{side[0]}" for i in range(1, 8)) for side in ("left", "right")}
    return OpenRailProfile(
        name="test", robot_type="agibot_g1", model="test-g1", base_frame="base",
        arms=("left", "right"), joint_names=names,
        end_effector_links={"left": "left_tcp", "right": "right_tcp"},
        base_transforms={side: np.eye(4) for side in names}, arm_aliases={"primary": "right"},
    )


def _snapshot():
    return OpenRailSnapshot(
        rgb=np.zeros((1, 1, 3), dtype=np.uint8), depth_u16=np.ones((1, 1), dtype=np.uint16),
        state=np.array([*range(14), 0.2, 0.7, 0.4, 0.5, 0.6, 0.7]),
        arm_velocities_rad_s=np.arange(14) / 10,
        rgb_timestamp_ns=1_000_000_000, depth_timestamp_ns=1_000_100_000,
        state_timestamps_ns={name: 1_000_000_000 for name in ("arm", "head", "waist", "gripper_left", "gripper_right")},
        depth_error=None,
        metadata={"robot_type": "agibot_g1", "state_timestamps_bind_state_values": True,
                  "maximum_rgb_depth_skew_ms": 1.0,
                  "arm_joint_names": [*(_profile().joint_names["left"]), *(_profile().joint_names["right"])],
                  "state_layout": {"left_arm_rad": [0, 7], "right_arm_rad": [7, 14],
                                   "left_gripper_normalized": [14, 15], "right_gripper_normalized": [15, 16],
                                   "head_yaw_pitch_rad": [16, 18], "waist_pitch_rad_lift_m": [18, 20]}},
    )


def _kinematics():
    def fk(arm, link, positions, head, waist):
        assert link == f"{arm}_tcp"
        return Pose(position=np.array([positions[0], head[0], waist[1]]),
                    quaternion_wxyz=np.array([1, 0, 0, 0]), frame="base")
    return G1StateKinematics(model="test-g1", verified=True, joint_order_verified=True,
                             position_velocity_units_verified=True, gripper_feedback_verified=True,
                             synchronization_verified=True, timestamps_share_clock_verified=True,
                             wire_joint_names=_profile().joint_names,
                             end_effector_pose=fk)


def test_robot_state_uses_bound_positions_velocities_and_kinematics() -> None:
    state = robot_state_from_snapshot(_snapshot(), _profile(), _kinematics())
    np.testing.assert_array_equal(state.joint_positions["right"], np.arange(7, 14))
    np.testing.assert_allclose(state.joint_velocities["right"], np.arange(7, 14) / 10)
    np.testing.assert_allclose(state.end_effector_poses["right"].position, [7, 0.4, 0.7])
    assert state.gripper_positions["left"] == pytest.approx(0.8)
    assert state.gripper_positions["right"] == pytest.approx(0.3)
    assert state.timestamp_s == 1.0


def test_robot_state_does_not_invent_missing_velocity_or_accept_unverified_fk() -> None:
    with pytest.raises(OpenRailG1ObservationError, match="14 measured arm velocities"):
        robot_state_from_snapshot(replace(_snapshot(), arm_velocities_rad_s=None), _profile(), _kinematics())
    with pytest.raises(OpenRailG1ObservationError, match="verified verified"):
        robot_state_from_snapshot(_snapshot(), _profile(), replace(_kinematics(), verified=False))
    with pytest.raises(OpenRailG1ObservationError, match="not synchronized"):
        robot_state_from_snapshot(replace(_snapshot(), state_timestamps_ns={"arm": 1_000_000_000}), _profile(), _kinematics())
    with pytest.raises(OpenRailG1ObservationError, match="FK model"):
        robot_state_from_snapshot(_snapshot(), _profile(), replace(_kinematics(), model="wrong"))
    snapshot = _snapshot()
    mismatched = replace(snapshot, metadata={**snapshot.metadata, "arm_joint_names": list(reversed(snapshot.metadata["arm_joint_names"]))})
    with pytest.raises(OpenRailG1ObservationError, match="joint names"):
        robot_state_from_snapshot(mismatched, _profile(), _kinematics())


def test_robot_state_accepts_only_explicitly_verified_wire_to_profile_order() -> None:
    wire = {side: tuple(f"{side}_arm_joint{i}" for i in range(1, 8)) for side in ("left", "right")}
    snapshot = _snapshot()
    snapshot = replace(snapshot, metadata={**snapshot.metadata, "arm_joint_names": [*wire["left"], *wire["right"]]})
    with pytest.raises(OpenRailG1ObservationError, match="joint names"):
        robot_state_from_snapshot(snapshot, _profile(), _kinematics())
    state = robot_state_from_snapshot(snapshot, _profile(), replace(_kinematics(), wire_joint_names=wire))
    assert state.joint_names["right"] == _profile().joint_names["right"]
    np.testing.assert_array_equal(state.joint_positions["right"], np.arange(7, 14))
