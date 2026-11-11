"""Compose the read-only apple preview with the existing CAP runner.

The caller owns the robot connection, calibration, model services and recorder.
This entry does not start an Open-RAIL policy or turn on physical control.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, TracedProvider
from cap_harness.contracts import GraspSet, PlanningScene, Pose
from cap_harness.open_rail.adapter import OpenRailAdapter
from cap_harness.open_rail.preview_chain import preview_pick_chain
from cap_harness.open_rail.scene_partition import (
    TargetScenePartition,
    verify_target_scene_partition,
)
from cap_harness.providers.curobo.attachment import AttachedObjectGeometry
from cap_harness.providers.curobo.client import DEFAULT_CUROBO_URL, CuRoboProvider
from cap_harness.providers.depth_refined import DepthRefinedSegmentationProvider
from cap_harness.providers.graspgenx.client import DEFAULT_GRASPGENX_URL, GraspGenXProvider
from cap_harness.providers.graspgenx.preview import write_grasp_preview
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
    chain_lift_offset_m: object | None = None,
    chain_attachment: AttachedObjectGeometry | Mapping[str, object] | None = None,
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
    full_chain = config.get("preview_full_chain", False)
    if type(full_chain) is not bool:
        raise ValueError("preview_full_chain must be a bool")
    if full_chain:
        offset = np.asarray(chain_lift_offset_m, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all() or not np.any(offset):
            raise ValueError("full chain preview needs an explicit finite nonzero lift offset")
        if isinstance(chain_attachment, Mapping):
            chain_attachment = AttachedObjectGeometry(**chain_attachment)
        if not isinstance(chain_attachment, AttachedObjectGeometry):
            raise ValueError("full chain preview needs explicit assumed attachment geometry")
    elif chain_lift_offset_m is not None or chain_attachment is not None:
        raise ValueError("chain preview inputs require preview_full_chain in task metadata")

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
            # This optional branch cannot yield executable trajectories. Allow
            # it to diagnose a G1 model that is explicitly unverified.
            strict_collision_ack=not full_chain,
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
    result = run_program_with_api(api=api, program_path=program_path, recorder=recorder)
    if result.ok and isinstance(result.result, Mapping):
        grasps = result.result.get("preview_grasps")
        scene = result.result.get("preview_scene")
        if isinstance(grasps, GraspSet) and grasps.ok and isinstance(scene, PlanningScene):
            # Both objects come from the same observation used by the program.
            # The PLY is evidence for inspection, never proof of collision clearance.
            write_grasp_preview(
                recorder.root / "media/grasp-candidates.ply",
                scene.point_cloud,
                grasps,
                snapshot_id=grasps.diagnostics.get("snapshot_id"),
                timestamp_s=grasps.diagnostics.get("camera_timestamp_s"),
            )
        if full_chain and result.result.get("stage") == "chain_preview_input":
            result = _run_chain_preview(
                result, planner, adapter, recorder, offset, chain_attachment,
            )
    return result


def _run_chain_preview(result, planner, adapter, recorder, offset, attachment):
    """Finish an optional static-scene preview without commanding the robot."""
    data = result.result
    partition, grasps, candidates = data.get("partition"), data.get("preview_grasps"), data.get("candidates")
    if not isinstance(partition, TargetScenePartition) or not isinstance(grasps, GraspSet) or not grasps.ok:
        raise ValueError("chain preview requires its original target partition and grasp set")
    verify_target_scene_partition(partition, partition.scene)
    mask = partition.source_localization.segmentation.mask
    mask_sha256 = hashlib.sha256(mask.tobytes()).hexdigest()
    if (partition.source_localization.diagnostics.get("target_mask_sha256") != mask_sha256
            or grasps.diagnostics.get("target_mask_sha256") != mask_sha256
            or grasps.diagnostics.get("snapshot_id") != partition.snapshot_id
            or grasps.diagnostics.get("snapshot_fingerprint") != partition.snapshot_fingerprint):
        raise ValueError("grasp candidates and scene partition do not share a target mask and snapshot")
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("chain preview requires candidates from the original grasp set")
    attempts = []
    for candidate, pregrasp in candidates:
        if not any(candidate is item for item in grasps.grasps):
            raise ValueError("chain preview candidate does not belong to the original grasp set")
        grasp_pose = candidate.pose
        lift = Pose(grasp_pose.position + offset, grasp_pose.quaternion_wxyz, grasp_pose.frame)
        with recorder.span("apple.chain_preview", category="planning", inputs={
            "snapshot_id": partition.snapshot_id, "target_mask_sha256": mask_sha256,
            "candidate": candidate, "pregrasp": pregrasp, "lift": lift,
            "assumed_attachment": attachment,
        }) as span:
            chain = preview_pick_chain(
                planner, partition, adapter.get_planning_context(), arm="right", pregrasp=pregrasp,
                approach=grasp_pose, lift=lift, attached_object=attachment,
            )
            span.output(chain, ok=chain.completed)
        attempts.append(chain)
        if chain.completed:
            break
    public = {key: value for key, value in data.items() if key not in ("partition", "candidates")}
    public.update({
        "ok": False, "stage": "chain_preview" if attempts[-1].completed else "chain_planning",
        "reason": "static preview only; contact, attachment, collision safety and grasp are unverified",
        "target_mask_sha256": mask_sha256, "chain_attempts": tuple(attempts),
    })
    return replace(result, result=public)


__all__ = ["run_apple_preview"]
