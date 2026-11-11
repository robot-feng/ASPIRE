"""Contract for independent, read-only G1 in-flight motion monitoring.

No hardware monitor is supplied. Implementations must bind each report to
fresh measured arm, head, waist, base and scene sources, compare them against
the verified trajectory and fixed-posture baseline, and never issue commands.
An implementation's report is not a substitute for validating its sensors,
calibration, scene checker, timing, coverage and physical stop behavior.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class G1MotionMonitorEvidence:
    """A single trajectory-bound check of live robot and scene observations."""

    trajectory_token: str
    state_timestamp_s: float
    scene_timestamp_s: float
    measured_joint_positions_rad: Sequence[float]
    reference_joint_positions_rad: Sequence[float]
    state_fingerprint: str
    scene_fingerprint: str
    fixed_posture_within_limits: bool
    scene_clear: bool


class G1MotionMonitor(ABC):
    """Read-only monitor of one trajectory; reports must use the shared clock.

    ``reference_joint_positions_rad`` is the last dispatched target (or the
    measured start before the first dispatch). The fixed-posture assessment
    must include the independently observed base, head and waist compared to
    their preplanned baselines; the scene check must use the relevant current
    scene. Each assessment is associated with its own fresh source timestamp
    and fingerprint. No current monitor satisfies this contract on G1.
    """

    @abstractmethod
    def check(self, *, trajectory_token: str, reference_joint_positions_rad: Sequence[float],
              timeout_s: float) -> G1MotionMonitorEvidence:
        """Return an independently measured report for this trajectory."""


__all__ = ["G1MotionMonitor", "G1MotionMonitorEvidence"]
