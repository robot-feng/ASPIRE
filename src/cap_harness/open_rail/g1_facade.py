"""Compose G1 observation and explicitly gated right-arm execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from cap_harness.contracts import ExecutionResult, Observation, StepResult, Trajectory
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.open_rail.g1_broadcast import G1SnapshotBroadcaster
from cap_harness.open_rail.g1_camera import G1CameraCalibration, G1DynamicCameraCalibration
from cap_harness.open_rail.g1_control import G1CommandCapabilities, G1TimedTrajectoryPolicy, cap_gripper_to_g1
from cap_harness.open_rail.g1_feedback import G1FeedbackCalibration, G1MeasuredCommandFeedback
from cap_harness.open_rail.g1_motion_monitor import G1MotionMonitor
from cap_harness.open_rail.g1_observation import G1ObservationControl
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailG1ObservationRuntime, OpenRailSnapshot
from cap_harness.open_rail.g1_state import G1StateKinematics
from cap_harness.open_rail.robot_profile import OpenRailProfile


class _ReadOnlyBroadcastView:
    """Expose only observation methods while sharing the command proxy's reader."""

    allow_physical_motion = False

    def __init__(self, broadcaster: G1SnapshotBroadcaster) -> None:
        self._broadcaster = broadcaster

    def read_snapshot(self, **kwargs):
        return self._broadcaster.read_snapshot(**kwargs)

    def close(self) -> None:
        self._broadcaster.close()


class G1ControlFacade:
    """Expose one G1's measured observation, right gripper and timed arm path.

    Constructing this object never starts the broadcaster or issues commands.
    A caller must explicitly start its sole broadcaster and configure verified
    calibration and command capabilities. A trajectory also requires an
    independently verified guard which compares its original plan/scene with
    a fresh observation and the exact source snapshot, including object and
    fixed head/waist state. The snapshot has no base pose: the guard must use
    an independently verified base source, and must refuse if unavailable.
    The guard is a prerequisite, not proof supplied by this class. Present
    hardware has no verified bounded dispatch or scene guard; it remains
    interlocked.
    """

    def __init__(
        self,
        runtime: OpenRailG1ObservationRuntime,
        broadcaster: G1SnapshotBroadcaster,
        profile: OpenRailProfile,
        camera_calibration: G1CameraCalibration | G1DynamicCameraCalibration,
        state_kinematics: G1StateKinematics,
        feedback_calibration: G1FeedbackCalibration,
        *,
        observation_options: Mapping[str, object],
        capabilities: G1CommandCapabilities,
        trajectory_policy: G1TimedTrajectoryPolicy,
        arm_joint_limits_rad: tuple[tuple[float, float], ...],
        position_tolerance_rad: float,
        velocity_tolerance_rad_s: float,
        gripper_tolerance: float,
        gripper_timeout_s: float,
        trajectory_guard: Callable[[Trajectory, Observation, OpenRailSnapshot], bool],
        trajectory_guard_verified: bool,
        gripper_guard: Callable[[float, str, Observation, OpenRailSnapshot], bool],
        gripper_guard_verified: bool,
        motion_monitor: G1MotionMonitor | None = None,
    ) -> None:
        if not isinstance(runtime, OpenRailG1ObservationRuntime):
            raise ValueError("a single G1 observation runtime is required")
        if not isinstance(broadcaster, G1SnapshotBroadcaster) or broadcaster._runtime is not runtime:
            raise ValueError("broadcaster must own the same G1 runtime as the command facade")
        if not isinstance(trajectory_policy, G1TimedTrajectoryPolicy):
            raise ValueError("a verified timed trajectory policy is required")
        if trajectory_guard_verified is not True or not callable(trajectory_guard):
            raise ValueError("a verified fresh-scene and fixed-posture trajectory guard is required")
        if gripper_guard_verified is not True or not callable(gripper_guard):
            raise ValueError("a verified fresh-scene gripper guard is required")
        self._gripper_timeout_s = self._positive(gripper_timeout_s)
        self._broadcaster = broadcaster
        self._observation = G1ObservationControl(
            _ReadOnlyBroadcastView(broadcaster), profile, camera_calibration, state_kinematics,
            **dict(observation_options),
        )
        feedback = G1MeasuredCommandFeedback(broadcaster.feedback_samples, feedback_calibration)
        self._command = runtime.command_runtime(
            feedback=feedback, capabilities=capabilities,
            position_tolerance_rad=position_tolerance_rad,
            velocity_tolerance_rad_s=velocity_tolerance_rad_s,
            arm_joint_limits_rad=arm_joint_limits_rad,
            gripper_tolerance=gripper_tolerance,
        )
        self._policy = trajectory_policy
        self._trajectory_guard = trajectory_guard
        self._gripper_guard = gripper_guard
        self._motion_monitor = motion_monitor
        self._closed = False

    @staticmethod
    def _positive(value: float) -> float:
        import math

        if isinstance(value, bool):
            raise ValueError("gripper timeout must be positive and finite")
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("gripper timeout must be positive and finite")
        return result

    def get_observation(self) -> Observation:
        return self._observation.get_observation()

    def get_robot_state(self):
        return self.get_observation().robot_state

    def step(self, _action: object) -> StepResult:
        return StepResult(
            ok=False,
            error=ApiError(ErrorCode.UNSUPPORTED, "G1 single-tick action semantics are not verified"),
        )

    def execute_trajectory(self, trajectory: object) -> ExecutionResult:
        if self._closed:
            return self._failure(ErrorCode.SAFETY_INTERLOCK, "G1 control facade is closed")
        if not isinstance(trajectory, Trajectory):
            return self._failure(ErrorCode.UNSUPPORTED, "only one right-arm trajectory is supported")
        with self._command._session_state.command_lock:
            if self._closed:
                return self._failure(ErrorCode.SAFETY_INTERLOCK, "G1 control facade is closed")
            if self._command._session_state.uncertain:
                return self._command._uncertain_state_failure()
            try:
                observation, snapshot = self._observation.get_observation_with_snapshot()
            except OpenRailG1ObservationError as exc:
                return self._failure(ErrorCode.STALE_PLAN, str(exc))
            if self._trajectory_guard(trajectory, observation, snapshot) is not True:
                return self._failure(ErrorCode.STALE_PLAN, "fresh scene or fixed-posture guard failed")
            try:
                self._observation.ensure_snapshot_current(snapshot)
            except OpenRailG1ObservationError as exc:
                return self._failure(ErrorCode.STALE_PLAN, str(exc))
            return self._command.execute_timed_trajectory(
                trajectory, observation.robot_state, policy=self._policy, monitor=self._motion_monitor,
            )

    def set_gripper(self, position: float, *, arm: str) -> ExecutionResult:
        if arm != "right":
            return self._failure(ErrorCode.INVALID_REQUEST, "only the right gripper is supported")
        try:
            normalized = 1.0 - cap_gripper_to_g1(position)
        except ValueError as exc:
            return self._failure(ErrorCode.INVALID_REQUEST, str(exc))
        with self._command._session_state.command_lock:
            if self._closed:
                return self._failure(ErrorCode.SAFETY_INTERLOCK, "G1 control facade is closed")
            if self._command._session_state.uncertain:
                return self._command._uncertain_state_failure()
            try:
                observation, snapshot = self._observation.get_observation_with_snapshot()
            except OpenRailG1ObservationError as exc:
                return self._failure(ErrorCode.STALE_PLAN, str(exc))
            if self._gripper_guard(normalized, arm, observation, snapshot) is not True:
                return self._failure(ErrorCode.STALE_PLAN, "fresh-scene gripper guard failed")
            try:
                self._observation.ensure_snapshot_current(snapshot)
            except OpenRailG1ObservationError as exc:
                return self._failure(ErrorCode.STALE_PLAN, str(exc))
            return self._command.set_cap_gripper(normalized, arm=arm, timeout_s=self._gripper_timeout_s)

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        if not isinstance(positions, Mapping) or set(positions) != {"right"}:
            return self._failure(ErrorCode.INVALID_REQUEST, "only the right gripper is supported")
        return self.set_gripper(positions["right"], arm="right")

    def close(self) -> None:
        if self._closed:
            return
        lock = self._command._session_state.command_lock
        if not lock.acquire(blocking=False):
            raise RuntimeError("cannot close the G1 observation owner during an active command")
        try:
            if self._command._session_state.health["in_flight_command"] is not None:
                raise RuntimeError("cannot close the G1 observation owner during an active command")
            # Call the broadcaster directly: it allows a later close retry if
            # its capture thread is still blocked. ObservationControl marks
            # itself closed before attempting its delegated close.
            self._broadcaster.close()
            self._closed = True
        finally:
            lock.release()

    @staticmethod
    def _failure(code: ErrorCode, message: str) -> ExecutionResult:
        return ExecutionResult(ok=False, steps_executed=0, error=ApiError(code, message))


__all__ = ["G1ControlFacade"]
