"""Fail-closed, arm-scoped command mapping for the AgiBot G1 SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import threading
import time
from types import MappingProxyType
from typing import Any, Protocol
import uuid

import numpy as np

from cap_harness.contracts import ExecutionResult, RobotState, Trajectory
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.open_rail.g1_motion_monitor import G1MotionMonitor, G1MotionMonitorEvidence

G1_RIGHT_ARM = "right"
G1_RIGHT_ARM_GROUP = "right_arm"
G1_RIGHT_GRIPPER_TOPIC = "/wbc/right_ee_command"
G1_RIGHT_ARM_JOINTS = tuple(f"Joint{index}_r" for index in range(1, 8))
_SESSION_STATE_ATTRIBUTE = "_cap_harness_g1_motion_session_state"
_SESSION_CREATION_LOCK = threading.Lock()


def _normalized_gripper(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("gripper position must be finite and in [0, 1]")
    try:
        position = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gripper position must be finite and in [0, 1]") from exc
    if not math.isfinite(position) or not 0.0 <= position <= 1.0:
        raise ValueError("gripper position must be finite and in [0, 1]")
    return position


def cap_gripper_to_g1(position: float) -> float:
    """Map ASPIRE opening (1=open) to Open-RAIL's G1 command (0=open)."""
    return 1.0 - _normalized_gripper(position)


def g1_feedback_to_cap(position: float, *, convention_verified: bool = False) -> float:
    """Map a verified normalized G1 feedback value to ASPIRE's opening domain.

    This mapping is valid only after the physical feedback convention and
    normalization are verified. Raw snapshots remain in the Open-RAIL domain.
    """
    if convention_verified is not True:
        raise ValueError("G1 gripper feedback convention must be verified")
    return 1.0 - _normalized_gripper(position)


class G1MotionSessionState:
    """Shared command uncertainty latch for every facade over one G1 proxy."""

    def __init__(self) -> None:
        self.command_lock = threading.RLock()
        self._uncertain_command: str | None = None
        self._uncertain_reason: str | None = None
        self._in_flight_command: str | None = None

    def mark_in_flight(self, command: str) -> None:
        self._in_flight_command = command

    def mark_uncertain(self, command: str, reason: str) -> None:
        self._uncertain_command = command
        self._uncertain_reason = reason
        self._in_flight_command = None

    def mark_complete(self) -> None:
        self._in_flight_command = None

    @property
    def uncertain(self) -> bool:
        return self._uncertain_command is not None

    @property
    def health(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "uncertain": self.uncertain,
                "uncertain_command": self._uncertain_command,
                "uncertain_reason": self._uncertain_reason,
                "in_flight_command": self._in_flight_command,
            }
        )


def proxy_motion_session_state(proxy: Any) -> G1MotionSessionState:
    """Get/create the latch attached to the sole Open-RAIL RobotProxy owner."""
    with _SESSION_CREATION_LOCK:
        state = getattr(proxy, _SESSION_STATE_ATTRIBUTE, None)
        if state is None:
            state = G1MotionSessionState()
            try:
                setattr(proxy, _SESSION_STATE_ATTRIBUTE, state)
            except (AttributeError, TypeError) as exc:
                raise TypeError("G1 RobotProxy must retain its motion uncertainty latch") from exc
    if not isinstance(state, G1MotionSessionState):
        raise TypeError("Open-RAIL proxy has an incompatible G1 motion session latch")
    return state


class G1CommandTransport(Protocol):
    """Scoped sends with an enforced dispatch deadline.

    Returning from these methods acknowledges only publisher dispatch, not
    robot motion. A timeout leaves the command state uncertain. The current
    Open-RAIL proxy cannot satisfy this contract.
    """

    def set_joint_position_control_bounded(
        self, lifetime_s: float, joint_groups: Mapping[str, Sequence[float]], *, timeout_s: float
    ) -> None: ...

    def publish_right_gripper_bounded(self, position: float, *, timeout_s: float) -> None: ...


class G1CommandFeedback(Protocol):
    """Measured-state waiter supplied by an integration with calibrated state."""

    def wait_for_right_arm(
        self,
        target_rad: np.ndarray,
        *,
        timeout_s: float,
        position_tolerance_rad: float,
        velocity_tolerance_rad_s: float,
    ) -> bool: ...

    def wait_for_right_gripper(
        self,
        target: float,
        *,
        timeout_s: float,
        tolerance: float,
    ) -> bool: ...


class OpenRailG1ProxyCommandTransport:
    """Route only the two scoped body methods through an existing RobotProxy.

    The proxy RPC reply confirms that the local DDS publish method returned; it
    is not physical execution or completion evidence.
    """

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy

    def set_joint_position_control(
        self, lifetime_s: float, joint_groups: Mapping[str, Sequence[float]]
    ) -> None:
        if set(joint_groups) != {G1_RIGHT_ARM_GROUP}:
            raise ValueError("only the SDK right_arm joint group is allowed")
        self._proxy.command_right_arm(list(joint_groups[G1_RIGHT_ARM_GROUP]), lifetime_s)

    def publish_right_gripper(self, position: float) -> None:
        self._proxy.command_right_gripper(position)


@dataclass(frozen=True, slots=True)
class G1CommandCapabilities:
    """Evidence gates for returning terminal success from physical commands.

    The SDK exposes right-arm/right-tool groups and separate gripper topics, but
    the checked-in runtime does not prove per-command terminal acknowledgement,
    command expiry behavior, or arm-scoped cancellation. These defaults keep
    all motion disabled until an integration verifies those semantics.
    """

    right_arm_group_scoped: bool = True
    arm_command_protocol_verified: bool = False
    arm_position_limits_verified: bool = False
    right_gripper_topic_scoped: bool = True
    arm_terminal_feedback_verified: bool = False
    arm_cancel_or_lease_expiry_verified: bool = False
    arm_timed_dispatch_verified: bool = False
    arm_live_monitor_verified: bool = False
    gripper_terminal_feedback_verified: bool = False
    gripper_stop_or_hold_verified: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class G1TimedTrajectoryPolicy:
    """Explicit verified limits for one right-arm timed trajectory.

    The command transport must have separately verified bounded per-waypoint
    dispatch, command supersession (no stale queued targets), and lease
    semantics. No numeric defaults are inferred from a G1 model or controller
    name.
    """

    control_period_s: float
    command_lifetime_s: float
    max_dispatch_s: float
    max_lateness_s: float
    final_feedback_timeout_s: float
    max_start_age_s: float
    start_clock_verified: bool
    max_start_error_rad: float
    max_start_gripper_error: float
    max_joint_velocity_rad_s: Sequence[float]
    max_joint_acceleration_rad_s2: Sequence[float]
    dynamic_limits_verified: bool
    max_tracking_error_rad: float
    monitor_max_age_s: float
    monitor_timeout_s: float

    def __post_init__(self) -> None:
        for name in (
            "control_period_s", "command_lifetime_s", "max_dispatch_s", "max_lateness_s",
            "final_feedback_timeout_s", "max_start_age_s", "max_start_error_rad",
            "max_start_gripper_error",
            "max_tracking_error_rad", "monitor_max_age_s", "monitor_timeout_s",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be positive and finite")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be positive and finite") from exc
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, numeric)
        if self.start_clock_verified is not True:
            raise ValueError("trajectory start clock must be explicitly verified")
        if self.dynamic_limits_verified is not True:
            raise ValueError("trajectory dynamic limits must be explicitly verified")
        for name in ("max_joint_velocity_rad_s", "max_joint_acceleration_rad_s2"):
            limits = np.asarray(getattr(self, name), dtype=np.float64)
            if limits.shape != (7,) or not np.isfinite(limits).all() or np.any(limits <= 0):
                raise ValueError(f"{name} must contain seven positive finite limits")
            object.__setattr__(self, name, tuple(float(value) for value in limits))
        if self.max_dispatch_s >= self.control_period_s or self.max_lateness_s >= self.control_period_s:
            raise ValueError("dispatch and lateness limits must be less than one control period")
        if self.monitor_timeout_s > self.max_lateness_s:
            raise ValueError("monitor timeout must fit inside the dispatch lateness bound")
        if self.command_lifetime_s < self.control_period_s + self.max_lateness_s + self.max_dispatch_s:
            raise ValueError("command lifetime must cover the next scheduled dispatch")


class OpenRailG1CommandRuntime:
    """Translate right-arm/right-gripper requests without unsafe fallbacks.

    Construction and ``reset`` are non-actuating. The runtime has no route for
    the left arm, head, waist, torso, or base. It refuses dispatch by default:
    A2D publishers return no command receipt, and the available stop topic is a
    whole-controller stop rather than a right-arm-only cancel.
    """

    def __init__(
        self,
        transport: G1CommandTransport,
        *,
        feedback: G1CommandFeedback | None = None,
        capabilities: G1CommandCapabilities | None = None,
        allow_physical_motion: bool = False,
        selected_arm: str = G1_RIGHT_ARM,
        position_tolerance_rad: float | None = None,
        velocity_tolerance_rad_s: float | None = None,
        arm_joint_limits_rad: Sequence[Sequence[float]] | None = None,
        gripper_tolerance: float | None = None,
        session_state: G1MotionSessionState | None = None,
    ) -> None:
        if type(allow_physical_motion) is not bool:
            raise ValueError("allow_physical_motion must be a bool")
        if selected_arm != G1_RIGHT_ARM:
            raise ValueError("this G1 runtime only supports selected_arm='right'")
        self._transport = transport
        self._feedback = feedback
        if session_state is None:
            proxy = getattr(transport, "_proxy", None)
            session_state = (
                proxy_motion_session_state(proxy)
                if proxy is not None
                else G1MotionSessionState()
            )
        self._session_state = session_state
        if capabilities is not None and not isinstance(capabilities, G1CommandCapabilities):
            raise ValueError("capabilities must be G1CommandCapabilities")
        self.capabilities = capabilities or G1CommandCapabilities()
        self.allow_physical_motion = allow_physical_motion
        self.selected_arm = selected_arm
        self.position_tolerance_rad = self._positive_or_none(
            position_tolerance_rad, "position_tolerance_rad"
        )
        self.velocity_tolerance_rad_s = self._positive_or_none(
            velocity_tolerance_rad_s, "velocity_tolerance_rad_s"
        )
        self.arm_joint_limits_rad = self._joint_limits_or_none(arm_joint_limits_rad)
        self.gripper_tolerance = self._positive_or_none(gripper_tolerance, "gripper_tolerance")

    @property
    def health(self) -> Mapping[str, object]:
        """Read-only command readiness without querying or commanding the SDK."""
        blockers: list[str] = []
        if not self.allow_physical_motion:
            blockers.append("physical motion authorization is disabled")
        if not self.capabilities.right_arm_group_scoped:
            blockers.append("right-arm SDK group scoping is unverified")
        if not self.capabilities.right_gripper_topic_scoped:
            blockers.append("right-gripper endpoint scoping is unverified")
        if not self.capabilities.arm_command_protocol_verified:
            blockers.append("right-arm command header/lease semantics are unverified")
        if not callable(getattr(self._transport, "set_joint_position_control_bounded", None)):
            blockers.append("right-arm dispatch has no bounded interface")
        if not callable(getattr(self._transport, "publish_right_gripper_bounded", None)):
            blockers.append("right-gripper dispatch has no bounded interface")
        if (
            not self.capabilities.arm_position_limits_verified
            or self.arm_joint_limits_rad is None
        ):
            blockers.append("verified right-arm joint limits are not configured")
        if not self.capabilities.arm_terminal_feedback_verified:
            blockers.append("right-arm terminal feedback is unverified")
        if not self.capabilities.arm_cancel_or_lease_expiry_verified:
            blockers.append("right-arm scoped cancel/lease expiry behavior is unverified")
        if not self.capabilities.gripper_terminal_feedback_verified:
            blockers.append("right-gripper terminal feedback is unverified")
        if not self.capabilities.gripper_stop_or_hold_verified:
            blockers.append("right-gripper stop/hold behavior is unverified")
        if self._feedback is None:
            blockers.append("measured terminal feedback reader is not configured")
        if self._session_state.uncertain:
            blockers.append("a prior command has unresolved motion state")
        if self.position_tolerance_rad is None or self.velocity_tolerance_rad_s is None:
            blockers.append("verified arm tolerances are not configured")
        if self.gripper_tolerance is None:
            blockers.append("verified gripper tolerance is not configured")
        return MappingProxyType(
            {
                "selected_arm": self.selected_arm,
                "allowed_commands": ("right_arm_joint_position", "right_gripper_position"),
                "physical_motion_authorized": self.allow_physical_motion,
                "motion_ready": not blockers,
                "blockers": tuple(blockers),
                "global_motion_stop_used": False,
                "motion_session": self._session_state.health,
            }
        )

    @staticmethod
    def _positive_or_none(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} must be positive and finite")
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return result

    @staticmethod
    def _joint_limits_or_none(
        limits: Sequence[Sequence[float]] | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if limits is None:
            return None
        if len(limits) != 7:
            raise ValueError("arm_joint_limits_rad must provide seven [min, max] pairs")
        normalized: list[tuple[float, float]] = []
        for pair in limits:
            if len(pair) != 2 or any(isinstance(value, bool) for value in pair):
                raise ValueError("arm joint limits must be finite [min, max] pairs")
            lower, upper = float(pair[0]), float(pair[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("arm joint limits must be finite with min < max")
            normalized.append((lower, upper))
        return tuple(normalized)

    @staticmethod
    def map_arm_target(
        arm: str, positions_rad: Sequence[float], *, lifetime_s: float
    ) -> tuple[float, dict[str, tuple[float, ...]]]:
        """Map one joint target to the SDK's right-arm control group only."""
        if arm != G1_RIGHT_ARM:
            raise ValueError("G1 command runtime only permits the configured right arm")
        values = np.asarray(positions_rad, dtype=np.float64)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("right arm target must contain 7 finite joint angles in radians")
        if isinstance(lifetime_s, bool):
            raise ValueError("joint command lifetime must be positive and finite")
        lifetime = float(lifetime_s)
        if not math.isfinite(lifetime) or lifetime <= 0.0:
            raise ValueError("joint command lifetime must be positive and finite")
        return lifetime, {G1_RIGHT_ARM_GROUP: tuple(float(x) for x in values)}

    @staticmethod
    def map_gripper_target(arm: str, position: float) -> float:
        """Preserve Open-RAIL's normalized gripper command for one endpoint."""
        if arm != G1_RIGHT_ARM:
            raise ValueError("G1 command runtime only permits the configured right gripper")
        if isinstance(position, bool):
            raise ValueError("right gripper position must be finite and in [0, 1]")
        value = float(position)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("right gripper position must be finite and in [0, 1]")
        return value

    def reset(self) -> None:
        """Reset no state and never invoke RobotDds.reset() or clear the latch."""

    def execute_timed_trajectory(
        self,
        trajectory: Trajectory,
        measured_start: RobotState,
        *,
        policy: G1TimedTrajectoryPolicy,
        monitor: G1MotionMonitor | None = None,
    ) -> ExecutionResult:
        """Dispatch a scoped trajectory at its verified control rate.

        This does not verify the scene, head/waist lock, TCP, collision model,
        or current workspace. Those are separate caller prerequisites. A
        The measured start anchors time zero; the first planned point is checked
        against it but is not dispatched as an instantaneous correction. Later
        points are dispatched at their planned intervals. A waypoint publish
        is not completion: only a terminal position/velocity
        measurement can return success. Any uncertain publish, deadline miss,
        or feedback failure latches the shared proxy session; no later command
        or global stop is sent automatically. Current G1 capabilities do not
        verify timed dispatch, so this path is disabled by default.
        """
        with self._session_state.command_lock:
            if self._session_state.uncertain:
                return self._uncertain_state_failure()
            if not isinstance(trajectory, Trajectory) or not isinstance(measured_start, RobotState):
                return self._failure(ErrorCode.INVALID_REQUEST, "trajectory and measured start are required")
            if not isinstance(policy, G1TimedTrajectoryPolicy):
                return self._failure(ErrorCode.INVALID_REQUEST, "a timed trajectory policy is required")
            if not self.capabilities.arm_live_monitor_verified:
                return self._failure(ErrorCode.UNSUPPORTED, "G1 in-flight monitor has not been verified")
            if not isinstance(monitor, G1MotionMonitor):
                return self._failure(ErrorCode.UNSUPPORTED, "a live G1 trajectory monitor is required")
            if trajectory.arm != G1_RIGHT_ARM or trajectory.embodiment != "open_rail_7dof":
                return self._failure(ErrorCode.INVALID_REQUEST, "only a G1 right-arm trajectory is supported")
            if not trajectory.collision_aware:
                return self._failure(ErrorCode.SAFETY_INTERLOCK, "trajectory collision checking is required")
            if tuple(trajectory.joint_names) != G1_RIGHT_ARM_JOINTS:
                return self._failure(ErrorCode.INVALID_REQUEST, "G1 trajectory joint order does not match the SDK")
            if not self.capabilities.arm_timed_dispatch_verified:
                return self._failure(ErrorCode.UNSUPPORTED, "bounded G1 timed dispatch is unverified")
            dispatch = getattr(self._transport, "set_joint_position_control_bounded", None)
            if not callable(dispatch):
                return self._failure(ErrorCode.UNSUPPORTED, "G1 transport does not provide bounded timed dispatch")
            if not math.isclose(trajectory.dt_s, policy.control_period_s, rel_tol=0.0, abs_tol=1e-9):
                return self._failure(ErrorCode.INVALID_REQUEST, "trajectory interval must match the verified control period")
            if len(trajectory.joint_positions) < 2:
                return self._failure(ErrorCode.INVALID_REQUEST, "trajectory must contain a timed motion segment")
            if measured_start.embodiment != trajectory.embodiment or set(measured_start.arms) != set(trajectory.expected_start.arms):
                return self._failure(ErrorCode.INVALID_REQUEST, "measured robot does not match trajectory start")
            if measured_start.timestamp_s is None:
                return self._failure(ErrorCode.STALE_PLAN, "measured start timestamp is missing")
            age_s = time.time() - measured_start.timestamp_s
            if not math.isfinite(age_s) or age_s < 0 or age_s > policy.max_start_age_s:
                return self._failure(ErrorCode.STALE_PLAN, "measured start is stale or from the future")
            for arm in trajectory.expected_start.arms:
                if tuple(measured_start.joint_names[arm]) != tuple(trajectory.expected_start.joint_names[arm]):
                    return self._failure(ErrorCode.STALE_PLAN, "measured joint order differs from planned start")
                if np.max(np.abs(measured_start.joint_positions[arm] - trajectory.expected_start.joint_positions[arm])) > policy.max_start_error_rad:
                    return self._failure(ErrorCode.STALE_PLAN, "measured joint state differs from planned start")
                if abs(measured_start.gripper_positions[arm] - trajectory.expected_start.gripper_positions[arm]) > policy.max_start_gripper_error:
                    return self._failure(ErrorCode.STALE_PLAN, "measured gripper differs from planned start")
            if np.max(np.abs(trajectory.joint_positions[0] - measured_start.joint_positions[G1_RIGHT_ARM])) > policy.max_start_error_rad:
                return self._failure(ErrorCode.STALE_PLAN, "first waypoint differs from measured start")
            if trajectory.gripper_positions is not None and np.any(
                np.abs(trajectory.gripper_positions - measured_start.gripper_positions[G1_RIGHT_ARM]) > policy.max_start_gripper_error
            ):
                return self._failure(ErrorCode.UNSUPPORTED, "timed arm trajectories cannot change the gripper")
            blocked = self._arm_blocker(trajectory.joint_positions[0])
            if blocked is not None:
                return blocked
            assert self.arm_joint_limits_rad is not None
            limits = np.asarray(self.arm_joint_limits_rad, dtype=np.float64)
            if np.any(trajectory.joint_positions < limits[:, 0]) or np.any(trajectory.joint_positions > limits[:, 1]):
                return self._failure(ErrorCode.SAFETY_INTERLOCK, "trajectory exceeds configured arm joint limits")
            if np.any(np.abs(measured_start.joint_velocities[G1_RIGHT_ARM]) > self.velocity_tolerance_rad_s):
                return self._failure(ErrorCode.STALE_PLAN, "right arm is moving at trajectory start")
            # The measured pose is the actual t=0 anchor. The planned initial
            # point is checked above but never sent as an instantaneous jump.
            # Validate precisely the setpoint path that will be dispatched.
            points = np.vstack((measured_start.joint_positions[G1_RIGHT_ARM], trajectory.joint_positions[1:]))
            path_velocities = np.diff(points, axis=0) / trajectory.dt_s
            if np.any(np.abs(path_velocities) > np.asarray(policy.max_joint_velocity_rad_s)):
                return self._failure(ErrorCode.SAFETY_INTERLOCK, "trajectory exceeds configured arm velocity limits")
            velocities = np.vstack((measured_start.joint_velocities[G1_RIGHT_ARM], path_velocities, np.zeros(7)))
            if np.any(np.abs(np.diff(velocities, axis=0)) > np.asarray(policy.max_joint_acceleration_rad_s2) * trajectory.dt_s):
                return self._failure(ErrorCode.SAFETY_INTERLOCK, "trajectory exceeds configured arm acceleration limits")

            self._session_state.mark_in_flight("right_arm_timed_trajectory")
            start = time.monotonic()
            attempted = 0
            token = uuid.uuid4().hex
            reference = np.array(measured_start.joint_positions[G1_RIGHT_ARM], copy=True)
            try:
                for index, target in enumerate(trajectory.joint_positions[1:], start=1):
                    scheduled = start + index * trajectory.dt_s
                    delay = scheduled - time.monotonic()
                    while delay > 0:
                        time.sleep(delay)
                        delay = scheduled - time.monotonic()
                    self._check_motion_monitor(monitor, token, reference, policy)
                    dispatch_start = time.monotonic()
                    if dispatch_start - scheduled > policy.max_lateness_s:
                        raise TimeoutError("trajectory dispatch missed its scheduled deadline")
                    attempted += 1
                    lifetime, groups = self.map_arm_target(
                        G1_RIGHT_ARM, target, lifetime_s=policy.command_lifetime_s
                    )
                    dispatch(lifetime, groups, timeout_s=policy.max_dispatch_s)
                    if time.monotonic() - dispatch_start > policy.max_dispatch_s:
                        raise TimeoutError("trajectory dispatch exceeded its verified duration")
                    reference = np.array(target, copy=True)
                self._wait_for_terminal_while_monitoring(monitor, token, reference, policy)
            except BaseException as exc:
                if attempted:
                    self._session_state.mark_uncertain("right_arm_timed_trajectory", str(exc))
                else:
                    self._session_state.mark_complete()
                if not isinstance(exc, Exception):
                    raise
                return self._failure(
                    ErrorCode.TIMEOUT if isinstance(exc, TimeoutError) else ErrorCode.CONTROLLER_FAILED,
                    f"G1 trajectory execution is unconfirmed: {exc}", steps_executed=attempted,
                    diagnostics={"cancel_confirmed": False, "global_motion_stop_sent": False},
                )
            self._session_state.mark_complete()
            return ExecutionResult(ok=True, steps_executed=attempted, diagnostics=self._feedback_diagnostics())

    @staticmethod
    def _check_motion_monitor(monitor: G1MotionMonitor, token: str, reference: np.ndarray,
                              policy: G1TimedTrajectoryPolicy) -> None:
        # A stalled monitor must not allow more points through or block the
        # caller indefinitely. Its daemon may finish later; it is read-only.
        result: dict[str, object] = {}
        done = threading.Event()

        def sample() -> None:
            try:
                result["value"] = monitor.check(
                    trajectory_token=token,
                    reference_joint_positions_rad=tuple(reference),
                    timeout_s=policy.monitor_timeout_s,
                )
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=sample, name="g1-motion-monitor-check", daemon=True).start()
        if not done.wait(policy.monitor_timeout_s):
            raise TimeoutError("G1 in-flight monitor timed out")
        if "error" in result:
            raise RuntimeError(f"G1 in-flight monitor failed: {result['error']}")
        evidence = result.get("value")
        if not isinstance(evidence, G1MotionMonitorEvidence) or evidence.trajectory_token != token:
            raise ValueError("G1 monitor evidence does not belong to this trajectory")
        now = time.time()
        for stamp in (evidence.state_timestamp_s, evidence.scene_timestamp_s):
            if isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or not 0 <= now - stamp <= policy.monitor_max_age_s:
                raise ValueError("G1 monitor evidence is stale or has an invalid timestamp")
        for fingerprint in (evidence.state_fingerprint, evidence.scene_fingerprint):
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
                raise ValueError("G1 monitor evidence lacks source fingerprints")
        measured = np.asarray(evidence.measured_joint_positions_rad, dtype=np.float64)
        reported_reference = np.asarray(evidence.reference_joint_positions_rad, dtype=np.float64)
        if measured.shape != (7,) or reported_reference.shape != (7,) or not np.isfinite(measured).all() or not np.isfinite(reported_reference).all() or not np.array_equal(reported_reference, reference):
            raise ValueError("G1 monitor joint state or trajectory reference is invalid")
        if np.any(np.abs(measured - reference) > policy.max_tracking_error_rad):
            raise ValueError("G1 arm tracking error exceeded its configured limit")
        if evidence.fixed_posture_within_limits is not True or evidence.scene_clear is not True:
            raise ValueError("G1 fixed posture or observed scene is no longer valid")

    def _wait_for_terminal_while_monitoring(self, monitor: G1MotionMonitor, token: str,
                                            reference: np.ndarray, policy: G1TimedTrajectoryPolicy) -> None:
        # Terminal feedback can take longer than one control period. Keep
        # checking the read-only monitor; no waypoint is sent while waiting.
        result: dict[str, object] = {}
        done = threading.Event()

        def wait_for_feedback() -> None:
            try:
                result["reached"] = self._feedback.wait_for_right_arm(
                    reference, timeout_s=policy.final_feedback_timeout_s,
                    position_tolerance_rad=self.position_tolerance_rad,
                    velocity_tolerance_rad_s=self.velocity_tolerance_rad_s,
                )
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        deadline = time.monotonic() + policy.final_feedback_timeout_s
        threading.Thread(target=wait_for_feedback, name="g1-terminal-feedback", daemon=True).start()
        while True:
            self._check_motion_monitor(monitor, token, reference, policy)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("final trajectory feedback timed out")
            if done.wait(min(remaining, policy.control_period_s)):
                if "error" in result:
                    raise RuntimeError(f"G1 terminal feedback failed: {result['error']}")
                if result.get("reached") is not True:
                    raise TimeoutError("final trajectory target was not confirmed by measured feedback")
                self._check_motion_monitor(monitor, token, reference, policy)
                return

    def set_arm(
        self,
        arm: str,
        positions_rad: Sequence[float],
        *,
        lifetime_s: float,
        timeout_s: float,
    ) -> ExecutionResult:
        with self._session_state.command_lock:
            return self._set_arm_locked(
                arm, positions_rad, lifetime_s=lifetime_s, timeout_s=timeout_s
            )

    def _set_arm_locked(
        self,
        arm: str,
        positions_rad: Sequence[float],
        *,
        lifetime_s: float,
        timeout_s: float,
    ) -> ExecutionResult:
        if self._session_state.uncertain:
            return self._uncertain_state_failure()
        try:
            lifetime, groups = self.map_arm_target(arm, positions_rad, lifetime_s=lifetime_s)
        except (TypeError, ValueError) as exc:
            return self._failure(ErrorCode.INVALID_REQUEST, str(exc))
        timeout = self._validated_timeout(timeout_s)
        if timeout is None:
            return self._failure(ErrorCode.INVALID_REQUEST, "timeout_s must be positive and finite")
        blocked = self._arm_blocker(np.asarray(groups[G1_RIGHT_ARM_GROUP]))
        if blocked is not None:
            return blocked

        target = np.asarray(groups[G1_RIGHT_ARM_GROUP], dtype=np.float64)
        self._session_state.mark_in_flight("right_arm_joint_position")
        try:
            started = time.monotonic()
            self._transport.set_joint_position_control_bounded(lifetime, groups, timeout_s=timeout)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("arm dispatch consumed the command deadline")
            reached = self._feedback.wait_for_right_arm(
                target,
                timeout_s=remaining,
                position_tolerance_rad=self.position_tolerance_rad,  # type: ignore[arg-type]
                velocity_tolerance_rad_s=self.velocity_tolerance_rad_s,  # type: ignore[arg-type]
            )
        except BaseException as exc:
            self._session_state.mark_uncertain(
                "right_arm_joint_position", f"command/feedback exception: {exc}"
            )
            if not isinstance(exc, Exception):
                raise
            return self._failure(
                ErrorCode.TIMEOUT if isinstance(exc, TimeoutError) else ErrorCode.CONTROLLER_FAILED,
                f"right-arm command/feedback failed: {exc}",
                steps_executed=1,
                diagnostics={
                    "arm": G1_RIGHT_ARM,
                    "cancel_confirmed": False,
                    "command_lifetime_s": lifetime,
                    "global_motion_stop_sent": False,
                },
            )
        if reached is not True:
            self._session_state.mark_uncertain(
                "right_arm_joint_position", "measured target was not confirmed before timeout"
            )
            return self._failure(
                ErrorCode.TIMEOUT,
                "right-arm target was not confirmed by measured position and velocity",
                steps_executed=1,
                diagnostics={
                    "arm": G1_RIGHT_ARM,
                    "cancel_confirmed": False,
                    "command_lifetime_s": lifetime,
                    "global_motion_stop_sent": False,
                },
            )
        self._session_state.mark_complete()
        return ExecutionResult(ok=True, steps_executed=1, diagnostics=self._feedback_diagnostics())

    def set_gripper(
        self,
        arm: str,
        position: float,
        *,
        timeout_s: float,
    ) -> ExecutionResult:
        with self._session_state.command_lock:
            return self._set_gripper_locked(arm, position, timeout_s=timeout_s)

    def set_cap_gripper(
        self,
        position: float,
        *,
        arm: str = G1_RIGHT_ARM,
        timeout_s: float,
    ) -> ExecutionResult:
        """Translate an ASPIRE opening target before the native G1 command.

        ``set_gripper`` stays in Open-RAIL's native command domain. This explicit
        entry point is intended for a future ``OpenRailControl`` implementation;
        it does not itself implement observations, steps, or trajectories.
        """
        try:
            native_position = cap_gripper_to_g1(position)
        except ValueError as exc:
            return self._failure(ErrorCode.INVALID_REQUEST, str(exc))
        return self.set_gripper(arm, native_position, timeout_s=timeout_s)

    def _set_gripper_locked(
        self, arm: str, position: float, *, timeout_s: float
    ) -> ExecutionResult:
        if self._session_state.uncertain:
            return self._uncertain_state_failure()
        try:
            value = self.map_gripper_target(arm, position)
        except (TypeError, ValueError) as exc:
            return self._failure(ErrorCode.INVALID_REQUEST, str(exc))
        timeout = self._validated_timeout(timeout_s)
        if timeout is None:
            return self._failure(ErrorCode.INVALID_REQUEST, "timeout_s must be positive and finite")
        blocked = self._gripper_blocker()
        if blocked is not None:
            return blocked

        self._session_state.mark_in_flight("right_gripper_position")
        try:
            started = time.monotonic()
            self._transport.publish_right_gripper_bounded(value, timeout_s=timeout)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("gripper dispatch consumed the command deadline")
            reached = self._feedback.wait_for_right_gripper(
                value,
                timeout_s=remaining,
                tolerance=self.gripper_tolerance,  # type: ignore[arg-type]
            )
        except BaseException as exc:
            self._session_state.mark_uncertain(
                "right_gripper_position", f"command/feedback exception: {exc}"
            )
            if not isinstance(exc, Exception):
                raise
            return self._failure(
                ErrorCode.TIMEOUT if isinstance(exc, TimeoutError) else ErrorCode.CONTROLLER_FAILED,
                f"right-gripper command/feedback failed: {exc}",
                steps_executed=1,
                diagnostics={
                    "arm": G1_RIGHT_ARM,
                    "cancel_confirmed": False,
                    "gripper_topic": G1_RIGHT_GRIPPER_TOPIC,
                    "global_motion_stop_sent": False,
                    "contact_or_grasp_status": "unknown",
                },
            )
        if reached is not True:
            self._session_state.mark_uncertain(
                "right_gripper_position", "measured target was not confirmed before timeout"
            )
            return self._failure(
                ErrorCode.TIMEOUT,
                "right-gripper target was not confirmed by measured feedback",
                steps_executed=1,
                diagnostics={
                    "arm": G1_RIGHT_ARM,
                    "cancel_confirmed": False,
                    "gripper_topic": G1_RIGHT_GRIPPER_TOPIC,
                    "global_motion_stop_sent": False,
                    "contact_or_grasp_status": "unknown",
                },
            )
        self._session_state.mark_complete()
        return ExecutionResult(ok=True, steps_executed=1, diagnostics=self._feedback_diagnostics())

    def _feedback_diagnostics(self) -> dict[str, object]:
        try:
            evidence = getattr(self._feedback, "last_evidence", None)
        except Exception:
            # A diagnostic accessor cannot turn a terminal feedback result into
            # an ambiguous action exception.
            return {}
        return {"measured_feedback": dict(evidence)} if isinstance(evidence, Mapping) else {}

    def _uncertain_state_failure(self) -> ExecutionResult:
        state = self._session_state.health
        return self._failure(
            ErrorCode.SAFETY_INTERLOCK,
            "a prior G1 command has unresolved motion state; no command is sent",
            diagnostics={
                "motion_state_uncertain": True,
                "uncertain_command": state["uncertain_command"],
                "uncertain_reason": state["uncertain_reason"],
                "global_motion_stop_sent": False,
            },
        )

    def cancel(self, arm: str = G1_RIGHT_ARM) -> ExecutionResult:
        """Refuse cancellation because the SDK stop endpoint is whole-body."""
        if arm != G1_RIGHT_ARM:
            return self._failure(ErrorCode.INVALID_REQUEST, "only the right arm is configured")
        return self._failure(
            ErrorCode.UNSUPPORTED,
            "no right-arm scoped cancel is available; global motion_stop is not sent",
            diagnostics={
                "arm": arm,
                "cancel_confirmed": False,
                "available_stop_scope": "global",
                "global_motion_stop_sent": False,
            },
        )

    @staticmethod
    def _validated_timeout(value: float) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return None
        return timeout if math.isfinite(timeout) and timeout > 0.0 else None

    def _arm_blocker(self, target_rad: np.ndarray) -> ExecutionResult | None:
        if not self.allow_physical_motion:
            return self._failure(
                ErrorCode.SAFETY_INTERLOCK,
                "physical G1 motion is disabled",
            )
        if not callable(getattr(self._transport, "set_joint_position_control_bounded", None)):
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "G1 arm transport has no bounded dispatch interface",
            )
        if not self.capabilities.right_arm_group_scoped:
            return self._failure(ErrorCode.UNSUPPORTED, "SDK right-arm command group is unverified")
        if not self.capabilities.arm_command_protocol_verified:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "SDK right-arm command header and lease behavior are unverified",
            )
        if (
            not self.capabilities.arm_position_limits_verified
            or self.arm_joint_limits_rad is None
        ):
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "verified right-arm joint limits are required before dispatch",
            )
        limits = np.asarray(self.arm_joint_limits_rad, dtype=np.float64)
        if np.any(target_rad < limits[:, 0]) or np.any(target_rad > limits[:, 1]):
            return self._failure(
                ErrorCode.SAFETY_INTERLOCK,
                "right-arm target is outside the configured joint limits",
            )
        if not self.capabilities.arm_terminal_feedback_verified:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "SDK publishes right-arm targets without verified terminal feedback",
            )
        if not self.capabilities.arm_cancel_or_lease_expiry_verified:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "SDK has no verified right-arm cancel or lease-expiry stop/hold behavior",
                diagnostics={"available_stop_scope": "global"},
            )
        if (
            self._feedback is None
            or self.position_tolerance_rad is None
            or self.velocity_tolerance_rad_s is None
        ):
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "measured arm feedback and verified position/velocity tolerances are required",
            )
        return None

    def _gripper_blocker(self) -> ExecutionResult | None:
        if not self.allow_physical_motion:
            return self._failure(ErrorCode.SAFETY_INTERLOCK, "physical G1 motion is disabled")
        if not callable(getattr(self._transport, "publish_right_gripper_bounded", None)):
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "G1 gripper transport has no bounded dispatch interface",
            )
        if not self.capabilities.right_gripper_topic_scoped:
            return self._failure(ErrorCode.UNSUPPORTED, "right-gripper endpoint is unverified")
        if not self.capabilities.gripper_terminal_feedback_verified:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "SDK publishes gripper targets without verified terminal feedback",
            )
        if not self.capabilities.gripper_stop_or_hold_verified:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "SDK has no verified right-gripper stop/hold behavior",
                diagnostics={"available_stop_scope": "unavailable"},
            )
        if self._feedback is None or self.gripper_tolerance is None:
            return self._failure(
                ErrorCode.UNSUPPORTED,
                "measured gripper feedback and a verified tolerance are required",
            )
        return None

    @staticmethod
    def _failure(
        code: ErrorCode,
        message: str,
        *,
        steps_executed: int = 0,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            ok=False,
            steps_executed=steps_executed,
            error=ApiError(code, message, recoverable=False),
            diagnostics=diagnostics or {},
        )


__all__ = [
    "G1_RIGHT_ARM",
    "G1_RIGHT_ARM_GROUP",
    "G1_RIGHT_ARM_JOINTS",
    "G1_RIGHT_GRIPPER_TOPIC",
    "G1CommandCapabilities",
    "G1CommandFeedback",
    "G1CommandTransport",
    "G1TimedTrajectoryPolicy",
    "OpenRailG1CommandRuntime",
    "OpenRailG1ProxyCommandTransport",
    "cap_gripper_to_g1",
    "g1_feedback_to_cap",
    "proxy_motion_session_state",
]
