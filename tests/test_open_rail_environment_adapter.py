from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.open_rail import OpenRailAdapter, OpenRailProfile


def _profile() -> OpenRailProfile:
    joints = {
        "left": tuple(f"left_joint_{i}" for i in range(7)),
        "right": tuple(f"right_joint_{i}" for i in range(7)),
    }
    return OpenRailProfile(
        name="fake_open_rail",
        robot_type="fake",
        model="fake_dual_arm",
        base_frame="base",
        arms=("left", "right"),
        joint_names=joints,
        end_effector_links={"left": "left_tcp", "right": "right_tcp"},
        base_transforms={arm: np.eye(4) for arm in joints},
        arm_aliases={"primary": "right", "secondary": "left"},
    )


@pytest.mark.parametrize("diagonal", [(2.0, 1.0, 1.0), (-1.0, 1.0, 1.0)])
def test_profile_rejects_nonrigid_base_transforms(diagonal) -> None:
    profile = _profile()
    transform = np.eye(4)
    transform[:3, :3] = np.diag(diagonal)

    with pytest.raises(ValueError, match="proper rotation"):
        replace(profile, base_transforms={**profile.base_transforms, "right": transform})


def _state(profile: OpenRailProfile) -> RobotState:
    pose = Pose(
        position=np.zeros(3),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame=profile.base_frame,
    )
    return RobotState(
        joint_positions={arm: np.zeros(7) for arm in profile.arms},
        joint_velocities={arm: np.zeros(7) for arm in profile.arms},
        end_effector_poses=dict.fromkeys(profile.arms, pose),
        gripper_positions=dict.fromkeys(profile.arms, 0.5),
        joint_names=profile.joint_names,
        base_frame=profile.base_frame,
        embodiment=profile.embodiment,
    )


def _observation(profile: OpenRailProfile) -> Observation:
    state = _state(profile)
    camera = CameraObservation(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float64),
        intrinsics=np.eye(3),
        frame="camera",
        camera_pose=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame=profile.base_frame,
        ),
    )
    return Observation(
        cameras={"head": camera},
        robot_state=state,
        task_context=TaskContext("open_rail", 0, "test", "test", "real"),
    )


class _Control:
    def __init__(self, profile: OpenRailProfile) -> None:
        self.observation = _observation(profile)
        self.commands: list[object] = []
        self.closed = False

    def get_observation(self) -> Observation:
        return self.observation

    def get_robot_state(self) -> RobotState:
        return self.observation.robot_state

    def step(self, action: RobotAction):
        self.commands.append(action)

    def execute_trajectory(self, trajectory: Trajectory | SynchronizedTrajectory):
        self.commands.append(trajectory)
        return ExecutionResult(ok=True, steps_executed=1)

    def set_gripper(self, position: float, *, arm: str):
        self.commands.append((arm, position))
        return ExecutionResult(ok=True, steps_executed=1)

    def set_grippers(self, positions):
        self.commands.append(dict(positions))
        return ExecutionResult(ok=True, steps_executed=1)

    def close(self) -> None:
        self.closed = True


def test_reset_and_reads_do_not_send_open_rail_commands() -> None:
    profile = _profile()
    control = _Control(profile)
    adapter = OpenRailAdapter(control, profile, task_context=control.observation.task_context)

    assert adapter.reset(seed=3) is control.observation
    assert adapter.get_robot_state() is control.observation.robot_state
    assert adapter.get_planning_context().embodiment == "open_rail_7dof"
    assert control.commands == []


def test_motion_is_interlocked_until_explicitly_enabled() -> None:
    profile = _profile()
    control = _Control(profile)
    adapter = OpenRailAdapter(control, profile)
    state = control.observation.robot_state
    action = RobotAction(
        {
            "right": ArmCommand(
                "joint_position",
                np.ones(7),
                embodiment=profile.embodiment,
            )
        }
    )
    trajectory = Trajectory(
        joint_positions=np.ones((1, 7)),
        dt_s=0.05,
        joint_names=profile.joint_names["right"],
        planner="test",
        collision_aware=True,
        expected_start=state,
        arm="right",
        embodiment=profile.embodiment,
    )

    assert adapter.step(action).error.code.value == "safety_interlock"
    assert adapter.execute_trajectory(trajectory).error.code.value == "safety_interlock"
    assert adapter.set_gripper(0.0).error.code.value == "safety_interlock"
    assert control.commands == []

    adapter.close()
    adapter.close()
    assert control.closed


def test_enabled_control_resolves_aliases_and_validates_gripper_range() -> None:
    profile = _profile()
    control = _Control(profile)
    adapter = OpenRailAdapter(control, profile, allow_physical_motion=True)

    result = adapter.set_gripper(0.25, arm="primary")
    invalid = adapter.set_gripper(1.1, arm="primary")

    assert result.ok
    assert control.commands == [("right", 0.25)]
    assert not invalid.ok


def test_cap_open_close_reach_native_g1_endpoint_through_explicit_bridge() -> None:
    from cap_harness.api import CapApi
    from cap_harness.open_rail.g1_control import (
        G1CommandCapabilities,
        OpenRailG1CommandRuntime,
    )

    # Test doubles only: this does not establish any physical gripper convention.
    class Transport:
        def __init__(self):
            self.targets = []

        def publish_right_gripper_bounded(self, position, *, timeout_s):
            self.targets.append(position)

    class Feedback:
        def __init__(self):
            self.targets = []

        def wait_for_right_gripper(self, target, **kwargs):
            self.targets.append(target)
            return True

    profile = _profile()
    transport, feedback = Transport(), Feedback()
    runtime = OpenRailG1CommandRuntime(
        transport,
        feedback=feedback,
        capabilities=G1CommandCapabilities(
            gripper_terminal_feedback_verified=True,
            gripper_stop_or_hold_verified=True,
        ),
        allow_physical_motion=True,
        gripper_tolerance=0.03,
    )

    class Control(_Control):
        def set_gripper(self, position: float, *, arm: str):
            return runtime.set_cap_gripper(position, arm=arm, timeout_s=0.5)

    adapter = OpenRailAdapter(Control(profile), profile, allow_physical_motion=True,
                              controlled_arms=("right",))
    api = CapApi(adapter)

    assert api.open_gripper().ok
    assert api.close_gripper().ok
    assert api.set_gripper(0.25).ok
    assert transport.targets == [0.0, 1.0, 0.75]
    assert feedback.targets == transport.targets


def test_unchecked_trajectory_is_rejected_before_transport() -> None:
    profile = _profile()
    control = _Control(profile)
    adapter = OpenRailAdapter(control, profile, allow_physical_motion=True, controlled_arms=("right",))
    trajectory = Trajectory(
        joint_positions=np.zeros((1, 7)),
        dt_s=0.05,
        joint_names=profile.joint_names["right"],
        planner="test",
        collision_aware=False,
        expected_start=control.observation.robot_state,
        arm="right",
        embodiment=profile.embodiment,
    )

    result = adapter.execute_trajectory(trajectory)

    assert result.error.code.value == "safety_interlock"
    assert control.commands == []


def test_single_arm_scope_rejects_other_arm_before_transport() -> None:
    profile = _profile()
    control = _Control(profile)
    adapter = OpenRailAdapter(
        control, profile, allow_physical_motion=True, controlled_arms=("primary",)
    )
    left_action = RobotAction(
        {"left": ArmCommand("joint_position", np.ones(7), embodiment=profile.embodiment)}
    )
    left_trajectory = Trajectory(
        joint_positions=np.ones((1, 7)),
        dt_s=0.05,
        joint_names=profile.joint_names["left"],
        planner="test",
        collision_aware=True,
        expected_start=control.observation.robot_state,
        arm="left",
        embodiment=profile.embodiment,
    )

    assert adapter.arms == ("left", "right")
    assert adapter.controlled_arms == ("right",)
    assert adapter.step(left_action).error.code.value == "safety_interlock"
    assert adapter.execute_trajectory(left_trajectory).error.code.value == "safety_interlock"
    assert adapter.set_gripper(0.5, arm="left").error.code.value == "safety_interlock"
    assert adapter.set_grippers({"right": 0.5, "left": 0.5}).error.code.value == "safety_interlock"
    assert control.commands == []
    assert adapter.set_gripper(0.5, arm="right").ok
    assert control.commands == [("right", 0.5)]


@pytest.mark.parametrize("arms", [(), ("right", "primary"), ("waist",), "right"])
def test_invalid_control_scope_is_rejected(arms) -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="controlled_arms"):
        OpenRailAdapter(_Control(profile), profile, controlled_arms=arms)


def test_motion_enable_requires_a_boolean() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="allow_physical_motion"):
        OpenRailAdapter(_Control(profile), profile, allow_physical_motion="false")  # type: ignore[arg-type]


def test_state_frame_and_trajectory_joint_order_must_match_profile() -> None:
    profile = _profile()
    control = _Control(profile)
    adapter = OpenRailAdapter(control, profile, allow_physical_motion=True)
    state = control.observation.robot_state
    other_pose = Pose(
        position=np.zeros(3),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        frame="other_base",
    )
    wrong_frame = RobotState(
        joint_positions=state.joint_positions,
        joint_velocities=state.joint_velocities,
        end_effector_poses=dict.fromkeys(profile.arms, other_pose),
        gripper_positions=state.gripper_positions,
        joint_names=state.joint_names,
        base_frame="other_base",
        embodiment=profile.embodiment,
    )
    wrong_names = {
        "left": profile.joint_names["left"],
        "right": tuple(reversed(profile.joint_names["right"])),
    }
    wrong_start = RobotState(
        joint_positions=state.joint_positions,
        joint_velocities=state.joint_velocities,
        end_effector_poses=state.end_effector_poses,
        gripper_positions=state.gripper_positions,
        joint_names=wrong_names,
        base_frame=profile.base_frame,
        embodiment=profile.embodiment,
    )
    trajectory = Trajectory(
        joint_positions=np.ones((1, 7)),
        dt_s=0.05,
        joint_names=tuple(reversed(profile.joint_names["right"])),
        planner="test",
        collision_aware=True,
        expected_start=wrong_start,
        arm="right",
        embodiment=profile.embodiment,
    )
    with pytest.raises(ValueError, match="does not match profile"):
        adapter._validate_state(wrong_frame)
    result = adapter.execute_trajectory(trajectory)

    assert not result.ok
    assert control.commands == []
