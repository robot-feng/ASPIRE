"""Compose the read-only apple preview with the existing CAP runner.

The caller owns the robot connection, calibration, model services and recorder.
This entry does not start an Open-RAIL policy or turn on physical control.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, TracedProvider
from cap_harness.open_rail.adapter import OpenRailAdapter
from cap_harness.providers.curobo.client import CuRoboProvider, DEFAULT_CUROBO_URL
from cap_harness.providers.depth_refined import DepthRefinedSegmentationProvider
from cap_harness.providers.graspgenx.client import DEFAULT_GRASPGENX_URL, GraspGenXProvider
from cap_harness.providers.wevisone.client import DEFAULT_WEVISONE_URL, WeVisOneProvider
from cap_harness.run import run_program_with_api
from cap_harness.runtime import ProgramExecutionResult


def run_apple_preview(
    adapter: OpenRailAdapter,
    *,
    program_path: Path,
    recorder: RunRecorder,
    depth_filter: Mapping[str, object],
    gripper_sweep: Mapping[str, object],
    gripper_base_to_tcp: object,
    control_period_s: float,
    wevisone_url: str = DEFAULT_WEVISONE_URL,
    graspgenx_url: str = DEFAULT_GRASPGENX_URL,
    curobo_url: str = DEFAULT_CUROBO_URL,
) -> ProgramExecutionResult:
    """Run the CAP apple program against a caller-owned, read-only adapter.

    All geometry and filtering parameters are supplied by the caller; supplying
    them does not prove their physical calibration. The G1 planner's own model
    acknowledgement must still establish whether a plan is collision-aware.
    This preview refuses motion-enabled adapters, even if a modified program
    attempts to issue an action. The caller remains responsible for closing
    the adapter and finalizing the recorder.
    """
    if not isinstance(adapter, OpenRailAdapter):
        raise TypeError("adapter must be an OpenRailAdapter")
    if adapter.physical_motion_authorized or adapter.controlled_arms != ("right",):
        raise ValueError("apple preview requires a read-only, right-arm-only adapter")
    task = adapter.get_task_context()
    config = task.metadata.get("open_rail_pick_apple")
    if not isinstance(config, Mapping) or not isinstance(config.get("camera_name"), str):
        raise ValueError("task must specify the apple preview camera")
    if gripper_base_to_tcp is None:
        raise ValueError("apple preview requires an explicit gripper-base-to-TCP transform")

    vision = DepthRefinedSegmentationProvider(
        WeVisOneProvider(base_url=wevisone_url), **dict(depth_filter)
    )
    segmentation = TracedProvider(vision, recorder, "wevisone")
    grasping = TracedProvider(
        GraspGenXProvider(
            gripper_sweep,
            base_url=graspgenx_url,
            gripper_base_to_tcp=gripper_base_to_tcp,
        ),
        recorder,
        "graspgenx",
    )
    planner = TracedProvider(
        CuRoboProvider(
            base_url=curobo_url,
            control_period_s=control_period_s,
            strict_collision_ack=True,
        ),
        recorder,
        "curobo",
    )
    api = CapApi(
        adapter,
        segmentation_provider=segmentation,
        grasp_providers={"graspgenx": grasping},
        ik_providers={"curobo": planner},
        trajectory_planning_providers={"curobo": planner},
        integrated_pose_planning_providers={"curobo-integrated": planner},
        default_camera=config["camera_name"],
        recorder=recorder,
    )
    return run_program_with_api(api=api, program_path=program_path, recorder=recorder)


__all__ = ["run_apple_preview"]
