"""Replay synchronized observations as evidence of a past object hold.

This module does not acquire frames, run a model, or control hardware. It
allows observations captured by a single camera owner to be localized later,
so model inference latency does not change the recorded capture intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from cap_harness.contracts import LocalizationResult, Observation
from cap_harness.open_rail.verification import HoldEvidence, HoldVerifier


@dataclass(frozen=True, slots=True)
class HoldReplaySample:
    """One scheduled capture attempt in its original zero-based order."""

    sequence_number: int
    observation: Observation | None
    localization: LocalizationResult | None


@dataclass(frozen=True, slots=True)
class HoldReplayResult:
    """A conclusion about recorded evidence, never the current physical state."""

    status: Literal["verified_history", "not_verified", "unknown"]
    reason: str
    evidence: tuple[HoldEvidence, ...]
    current_hold_verified: bool = field(default=False, init=False)


def replay_hold(
    baseline: Observation,
    baseline_localization: LocalizationResult,
    samples: Sequence[HoldReplaySample],
    *,
    expected_samples: int,
    capture_clock_verified: bool,
    camera_name: str,
    arm: str,
    up_in_base: object,
    minimum_lift_m: float,
    max_relative_error_m: float,
    max_sample_gap_s: float,
    max_state_skew_s: float,
    stable_duration_s: float = 3.0,
    min_samples: int = 3,
) -> HoldReplayResult:
    """Evaluate a complete ordered recording of a lift and post-lift hold.

    The caller establishes that the baseline was captured after closure and
    before lifting, with the object on its support. ``capture_clock_verified``
    must be true only after the capture timestamps' common clock domain has
    been verified. The capture manifest must supply one numbered record for
    every scheduled attempt and its independently recorded expected count.
    Each localization must describe its exact
    RGB-D/state snapshot. This function creates a fresh verifier per replay and
    checks every supplied sample, including those after the first verified
    window. A missing/ambiguous localization, non-increasing capture timestamp,
    or inter-sample gap above the configured bound invalidates the entire
    replay; it is never silently skipped or bridged. The time between the
    pre-lift baseline and first post-lift sample is not a hold sampling gap.

    A positive result establishes only a historical window in this recording.
    It does not prove the object is still held when inference finishes; that
    requires a separate current, synchronized observation and assessment.
    """
    verifier = HoldVerifier(
        baseline,
        baseline_localization,
        camera_name=camera_name,
        arm=arm,
        up_in_base=up_in_base,
        minimum_lift_m=minimum_lift_m,
        max_relative_error_m=max_relative_error_m,
        max_sample_gap_s=max_sample_gap_s,
        max_state_skew_s=max_state_skew_s,
        stable_duration_s=stable_duration_s,
        min_samples=min_samples,
    )
    if capture_clock_verified is not True:
        raise ValueError("the capture clock domain must be verified")
    if isinstance(expected_samples, bool) or not isinstance(expected_samples, int) or expected_samples < 1:
        raise ValueError("expected_samples must be a positive integer from the capture manifest")
    if not isinstance(samples, Sequence):
        raise ValueError("samples must be a finite sequence of recorded frames")
    if len(samples) != expected_samples:
        return HoldReplayResult("unknown", "capture manifest has missing samples", ())

    evidence: list[HoldEvidence] = []
    previous_stamp: float | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, HoldReplaySample):
            return HoldReplayResult("unknown", f"sample {index} is missing", tuple(evidence))
        if type(sample.sequence_number) is not int or sample.sequence_number != index:
            return HoldReplayResult("unknown", f"sample {index} sequence is incomplete", tuple(evidence))
        observation, localization = sample.observation, sample.localization
        if not isinstance(observation, Observation):
            return HoldReplayResult("unknown", f"sample {index} observation is missing", tuple(evidence))
        camera = observation.cameras.get(camera_name)
        if camera is None or camera.timestamp_s is None:
            return HoldReplayResult("unknown", f"sample {index} timestamp is missing", tuple(evidence))
        stamp = camera.timestamp_s
        if previous_stamp is not None:
            if stamp <= previous_stamp:
                return HoldReplayResult(
                    "unknown", f"sample {index} timestamp is not strictly increasing", tuple(evidence)
                )
            if stamp - previous_stamp > verifier.max_sample_gap_s:
                return HoldReplayResult(
                    "unknown", f"sample {index} exceeds the maximum sampling gap", tuple(evidence)
                )
        previous_stamp = stamp
        result = verifier.observe(observation, localization)
        evidence.append(result)
        if result.status == "unknown":
            return HoldReplayResult("unknown", f"sample {index}: {result.reason}", tuple(evidence))
        if result.status in ("not_lifted", "not_following"):
            return HoldReplayResult("not_verified", f"sample {index}: {result.reason}", tuple(evidence))

    if not evidence or evidence[-1].status != "verified":
        return HoldReplayResult("not_verified", "recording has no complete hold window", tuple(evidence))
    return HoldReplayResult(
        "verified_history",
        "historical hold window verified; current hold requires a separate fresh assessment",
        tuple(evidence),
    )


__all__ = ["HoldReplayResult", "HoldReplaySample", "replay_hold"]
