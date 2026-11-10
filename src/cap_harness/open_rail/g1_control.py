"""Fail-closed, arm-scoped command mapping for the AgiBot G1 SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import threading
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from cap_harness.contracts import ExecutionResult
from cap_harness.errors import ApiError, ErrorCode

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
    """The two SDK calls this scoped driver may issue."""

    def set_joint_position_control(
        self, lifetime_s: float, joint_groups: Mapping[str, Sequence[float]]
    ) -> None: ...

    def publish_right_gripper(self, position: float) -> None: ...


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
    gripper_terminal_feedback_verified: bool = False
    gripper_stop_or_hold_verified: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")


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
            self._transport.set_joint_position_control(lifetime, groups)
            reached = self._feedback.wait_for_right_arm(
                target,
                timeout_s=timeout,
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
                ErrorCode.CONTROLLER_FAILED,
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
            self._transport.publish_right_gripper(value)
            reached = self._feedback.wait_for_right_gripper(
                value,
                timeout_s=timeout,
                tolerance=self.gripper_tolerance,  # type: ignore[arg-type]
            )
        except BaseException as exc:
            self._session_state.mark_uncertain(
                "right_gripper_position", f"command/feedback exception: {exc}"
            )
            if not isinstance(exc, Exception):
                raise
            return self._failure(
                ErrorCode.CONTROLLER_FAILED,
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
    "OpenRailG1CommandRuntime",
    "OpenRailG1ProxyCommandTransport",
    "cap_gripper_to_g1",
    "g1_feedback_to_cap",
    "proxy_motion_session_state",
]
