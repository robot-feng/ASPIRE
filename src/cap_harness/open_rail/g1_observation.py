"""Read-only bridge from a single G1 raw snapshot to ASPIRE observations."""

from __future__ import annotations

from collections.abc import Callable
import math

from cap_harness.contracts import ExecutionResult, Observation, StepResult, TaskContext
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.open_rail.g1_camera import (
    G1CameraCalibration,
    G1DynamicCameraCalibration,
    G1HeadKinematics,
    G1RgbdRectification,
    camera_observation_from_snapshot,
)
from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError, OpenRailSnapshot
from cap_harness.open_rail.g1_state import (
    G1ArmSideFeedbackCalibration,
    G1StateKinematics,
    robot_state_from_snapshot,
)
from cap_harness.open_rail.robot_profile import OpenRailProfile


class G1ObservationControl:
    """Adapt one caller-owned G1 observation runtime without enabling motion.

    Camera and state are converted from the same raw snapshot. The caller must
    provide verified camera/robot calibration and a verified common-clock read;
    this class does not infer calibration from a profile name or a source file.
    It owns the supplied runtime and closes it when the adapter closes.
    """

    def __init__(
        self,
        runtime: object,
        profile: OpenRailProfile,
        camera_calibration: G1CameraCalibration | G1DynamicCameraCalibration,
        state_kinematics: G1StateKinematics,
        *,
        clock_now_ns: Callable[[], int],
        clock_verified: bool,
        max_snapshot_age_ns: int,
        head_kinematics: G1HeadKinematics | None = None,
        side_feedback: G1ArmSideFeedbackCalibration | None = None,
        rectification: G1RgbdRectification | None = None,
        task_context: TaskContext | None = None,
    ) -> None:
        if getattr(runtime, "allow_physical_motion", None) is not False:
            raise ValueError("G1 observation runtime must be explicitly read-only")
        if not callable(getattr(runtime, "read_snapshot", None)) or not callable(
            getattr(runtime, "close", None)
        ):
            raise ValueError("runtime must support snapshot reads and close")
        if not isinstance(profile, OpenRailProfile) or profile.robot_type != "agibot_g1":
            raise ValueError("profile must describe an AgiBot G1")
        if not isinstance(camera_calibration, (G1CameraCalibration, G1DynamicCameraCalibration)):
            raise ValueError("camera calibration is required")
        if camera_calibration.base_frame != profile.base_frame:
            raise ValueError("camera calibration and robot profile must share the base frame")
        if not isinstance(state_kinematics, G1StateKinematics):
            raise ValueError("G1 state kinematics are required")
        if clock_verified is not True or not callable(clock_now_ns):
            raise ValueError("a verified common-clock reference is required")
        if isinstance(max_snapshot_age_ns, bool) or not isinstance(max_snapshot_age_ns, int) or max_snapshot_age_ns <= 0:
            raise ValueError("max_snapshot_age_ns must be a positive integer")
        self._runtime = runtime
        self._profile = profile
        self._camera_calibration = camera_calibration
        self._state_kinematics = state_kinematics
        self._head_kinematics = head_kinematics
        self._side_feedback = side_feedback
        if rectification is not None and not isinstance(rectification, G1RgbdRectification):
            raise ValueError("rectification must be a G1RgbdRectification")
        self._rectification = rectification
        self._clock_now_ns = clock_now_ns
        self._max_age_ns = max_snapshot_age_ns
        self._task_context = task_context
        self._closed = False

    def get_observation(self) -> Observation:
        if self._closed:
            raise OpenRailG1ObservationError("G1 observation control is closed")
        snapshot = self._runtime.read_snapshot(require_depth=True)  # type: ignore[attr-defined]
        if not isinstance(snapshot, OpenRailSnapshot) or not snapshot.is_synchronized:
            raise OpenRailG1ObservationError("G1 raw snapshot is missing or not synchronized")
        now = self._clock_now_ns()
        if isinstance(now, bool) or not isinstance(now, int) or now <= 0:
            raise OpenRailG1ObservationError("common-clock reference must be a positive integer")
        timestamps = (snapshot.rgb_timestamp_ns, snapshot.depth_timestamp_ns, *snapshot.state_timestamps_ns.values())
        if any(stamp is None or stamp > now or now - stamp > self._max_age_ns for stamp in timestamps):
            raise OpenRailG1ObservationError("G1 snapshot is stale or from the future")
        skew_ms = self._camera_calibration.maximum_snapshot_skew_ms
        if isinstance(skew_ms, bool) or not isinstance(skew_ms, (float, int)) or not math.isfinite(skew_ms) or skew_ms < 0:
            raise OpenRailG1ObservationError("camera synchronization threshold must be finite and nonnegative")
        if any(abs(stamp - snapshot.rgb_timestamp_ns) > skew_ms * 1_000_000 for stamp in timestamps):
            raise OpenRailG1ObservationError("robot and camera exceed the verified synchronization threshold")
        state = robot_state_from_snapshot(
            snapshot, self._profile, self._state_kinematics,
            side_feedback=self._side_feedback, reference_timestamp_ns=now,
        )
        camera = camera_observation_from_snapshot(
            snapshot, self._camera_calibration, head_kinematics=self._head_kinematics,
            rectification=self._rectification,
        )
        return Observation(
            cameras={self._camera_calibration.public_camera_key: camera},
            robot_state=state,
            task_context=self._task_context,
            timestamp_s=camera.timestamp_s,
        )

    def get_robot_state(self):
        return self.get_observation().robot_state

    @staticmethod
    def _denied() -> ApiError:
        return ApiError(ErrorCode.SAFETY_INTERLOCK, "G1 observation control cannot move the robot")

    def step(self, _action: object) -> StepResult:
        return StepResult(ok=False, error=self._denied())

    def execute_trajectory(self, _trajectory: object) -> ExecutionResult:
        return ExecutionResult(ok=False, steps_executed=0, error=self._denied())

    def set_gripper(self, _position: float, *, arm: str) -> ExecutionResult:
        del arm
        return ExecutionResult(ok=False, steps_executed=0, error=self._denied())

    def set_grippers(self, _positions: object) -> ExecutionResult:
        return ExecutionResult(ok=False, steps_executed=0, error=self._denied())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._runtime.close()  # type: ignore[attr-defined]


__all__ = ["G1ObservationControl"]
