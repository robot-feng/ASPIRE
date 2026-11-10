"""ASPIRE environment boundary for robot drivers managed by Open-RAIL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from cap_harness.contracts import (
    ExecutionResult,
    Observation,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.open_rail.robot_profile import OpenRailProfile


class OpenRailControl(Protocol):
    """Typed control surface implemented by an Open-RAIL robot runtime.

    The concrete backend owns Open-RAIL transport and robot-specific conversion.
    Commands are limited to the configured arm joints and grippers; torso, waist,
    head, and base controls are outside this interface. Every command must return
    only after its terminal feedback is available.

    Gripper values in every shared contract use ASPIRE opening semantics:
    1.0 is open and 0.0 is closed. A robot-specific implementation must convert
    its own native commands and independently verified feedback, including
    action and trajectory gripper fields, at its transport boundary.
    """

    def get_observation(self) -> Observation: ...

    def get_robot_state(self) -> RobotState: ...

    def step(self, action: RobotAction) -> StepResult: ...

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult: ...

    def set_gripper(self, position: float, *, arm: str) -> ExecutionResult: ...

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult: ...

    def close(self) -> None: ...


class OpenRailAdapter:
    """Wrap one Open-RAIL robot runtime using ASPIRE's shared contracts.

    ``reset`` is observational for real hardware: it selects task metadata but
    never homes or moves the robot. Physical commands are denied by default;
    a launcher must opt in explicitly after constructing a verified profile.
    Actions can address only the profile's arm joints and grippers.
    """

    def __init__(
        self,
        control: OpenRailControl,
        profile: OpenRailProfile,
        *,
        task_context: TaskContext | None = None,
        allow_physical_motion: bool = False,
        controlled_arms: Sequence[str] | None = None,
    ) -> None:
        self._control = control
        self.profile = profile
        self._task_context = task_context
        if type(allow_physical_motion) is not bool:
            raise ValueError("allow_physical_motion must be a bool")
        self._allow_physical_motion = allow_physical_motion
        if isinstance(controlled_arms, str | bytes):
            raise ValueError("controlled_arms must be a sequence of arm names")
        names = profile.arms if controlled_arms is None else controlled_arms
        selected = tuple(profile.resolve_arm(arm) for arm in names)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or set(selected) - set(profile.arms)
        ):
            raise ValueError("controlled_arms must name distinct arms in the profile")
        self._controlled_arms = selected
        self._closed = False

    @property
    def embodiment(self) -> str:
        return self.profile.embodiment

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.profile.arms)

    def resolve_arm(self, arm: str) -> str:
        """Resolve shared ``primary``/``secondary`` aliases to robot arm names."""
        return self.profile.resolve_arm(arm)

    @property
    def controlled_arms(self) -> tuple[str, ...]:
        """Arms allowed to receive commands; other arms remain observable."""
        return self._controlled_arms

    @property
    def physical_motion_authorized(self) -> bool:
        return self._allow_physical_motion

    def get_planning_context(self) -> RobotPlanningContext:
        return self.profile.planning_context()

    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Start an episode without changing physical robot or scene state."""
        if isinstance(task_ref, TaskContext):
            self._task_context = task_ref
        elif task_ref is not None and self._task_context is None:
            raise ValueError("pass a TaskContext for real Open-RAIL tasks")
        return self.get_observation()

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise RuntimeError("no task context set; pass one to the adapter or reset()")
        return self._task_context

    def get_observation(self) -> Observation:
        observation = self._control.get_observation()
        self._validate_state(observation.robot_state)
        return observation

    def get_robot_state(self) -> RobotState:
        state = self._control.get_robot_state()
        self._validate_state(state)
        return state

    def step(self, action: RobotAction) -> StepResult:
        error = self._validate_action(action)
        if error is not None:
            return StepResult(ok=False, error=error)
        denied = self._motion_denied()
        if denied is not None:
            return StepResult(ok=False, error=denied)
        result = self._control.step(action)
        if result.observation is not None:
            self._validate_state(result.observation.robot_state)
        return result

    def execute_trajectory(
        self, trajectory: Trajectory | SynchronizedTrajectory
    ) -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        if not trajectory.collision_aware:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.SAFETY_INTERLOCK,
                    "Open-RAIL trajectories require collision checking",
                    recoverable=False,
                ),
            )
        if trajectory.embodiment != self.embodiment:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "trajectory embodiment does not match the Open-RAIL profile",
                    details={"expected": self.embodiment, "received": trajectory.embodiment},
                ),
            )
        commanded_arms = (
            {trajectory.arm}
            if isinstance(trajectory, Trajectory)
            else set(trajectory.joint_positions)
        )
        unknown = commanded_arms - set(self.arms)
        if unknown:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"trajectory names arms absent from profile: {sorted(unknown)}",
                ),
            )
        denied_arms = commanded_arms - set(self.controlled_arms)
        if denied_arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=self._arm_scope_error(denied_arms),
            )
        names_by_arm = (
            {trajectory.arm: tuple(trajectory.joint_names)}
            if isinstance(trajectory, Trajectory)
            else {arm: tuple(names) for arm, names in trajectory.joint_names.items()}
        )
        mismatched = {
            arm: names
            for arm, names in names_by_arm.items()
            if names != tuple(self.profile.joint_names[arm])
        }
        if mismatched:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "trajectory joint names do not match the Open-RAIL profile",
                    details={"mismatched_arms": sorted(mismatched)},
                ),
            )
        result = self._control.execute_trajectory(trajectory)
        if result.final_observation is not None:
            self._validate_state(result.final_observation.robot_state)
        return result

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        resolved_arm = self.profile.resolve_arm(arm)
        if resolved_arm not in self.arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"unknown arm {arm!r}",
                    details={"known_arms": list(self.arms)},
                ),
            )
        if resolved_arm not in self.controlled_arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=self._arm_scope_error({resolved_arm}),
            )
        value = self._normalized_gripper_position(position)
        if value is None:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "gripper position must be finite and in [0, 1]",
                ),
            )
        return self._control.set_gripper(value, arm=resolved_arm)

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        denied = self._motion_denied()
        if denied is not None:
            return ExecutionResult(ok=False, steps_executed=0, error=denied)
        resolved: dict[str, float] = {}
        for arm, raw_value in positions.items():
            resolved_arm = self.profile.resolve_arm(arm)
            if resolved_arm in resolved:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.INVALID_REQUEST,
                        f"arm aliases resolve to duplicate gripper target {resolved_arm!r}",
                    ),
                )
            value = self._normalized_gripper_position(raw_value)
            if value is None:
                return ExecutionResult(
                    ok=False,
                    steps_executed=0,
                    error=ApiError(
                        ErrorCode.INVALID_REQUEST,
                        f"gripper position for {arm!r} must be finite and in [0, 1]",
                    ),
                )
            resolved[resolved_arm] = value
        unknown = set(resolved) - set(self.arms)
        if unknown:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    f"unknown arms: {sorted(unknown)}",
                    details={"known_arms": list(self.arms)},
                ),
            )
        denied_arms = set(resolved) - set(self.controlled_arms)
        if denied_arms:
            return ExecutionResult(
                ok=False,
                steps_executed=0,
                error=self._arm_scope_error(denied_arms),
            )
        return self._control.set_grippers(resolved)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._control.close()

    def _validate_action(self, action: RobotAction) -> ApiError | None:
        if action.embodiment != self.embodiment:
            return ApiError(
                ErrorCode.INVALID_REQUEST,
                "action embodiment does not match the Open-RAIL profile",
                details={"expected": self.embodiment, "received": action.embodiment},
            )
        unknown = set(action.arms) - set(self.arms)
        if unknown:
            return ApiError(
                ErrorCode.INVALID_REQUEST,
                f"action names arms absent from profile: {sorted(unknown)}",
                details={"known_arms": list(self.arms)},
            )
        denied_arms = set(action.arms) - set(self.controlled_arms)
        if denied_arms:
            return self._arm_scope_error(denied_arms)
        return None

    def _arm_scope_error(self, arms: set[str]) -> ApiError:
        return ApiError(
            ErrorCode.SAFETY_INTERLOCK,
            f"commands for arms outside the configured control scope: {sorted(arms)}",
            details={"controlled_arms": list(self.controlled_arms)},
            recoverable=False,
        )

    def _validate_state(self, state: RobotState) -> None:
        if state.embodiment != self.embodiment:
            raise ValueError(
                f"runtime state embodiment {state.embodiment!r} does not match "
                f"profile {self.embodiment!r}"
            )
        if state.base_frame != self.profile.base_frame:
            raise ValueError(
                f"runtime base frame {state.base_frame!r} does not match "
                f"profile {self.profile.base_frame!r}"
            )
        if set(state.arms) != set(self.arms):
            raise ValueError("runtime state arm names do not match the Open-RAIL profile")
        for arm in self.arms:
            if tuple(state.joint_names[arm]) != tuple(self.profile.joint_names[arm]):
                raise ValueError(f"runtime joint names for {arm!r} do not match profile")

    def _motion_denied(self) -> ApiError | None:
        if self._allow_physical_motion:
            return None
        return ApiError(
            ErrorCode.SAFETY_INTERLOCK,
            "physical motion is not authorized for this Open-RAIL adapter",
            recoverable=False,
        )

    @staticmethod
    def _normalized_gripper_position(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            position = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(position) or not 0.0 <= position <= 1.0:
            return None
        return position


__all__ = ["OpenRailAdapter", "OpenRailControl"]
