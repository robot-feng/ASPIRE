"""CAP program: locate an apple and plan a collision-checked pregrasp.

This stage uses the Open-RAIL task configuration supplied in TaskContext. It
does not execute a trajectory: G1 contact, lift and hold verification require
the validated execution and collision models described in the runbook.
"""

def pregrasp_pose(candidate, distance_m, base_frame):
    meta = candidate.metadata
    if (
        candidate.frame != base_frame
        or meta.get("grasp_frame") != "tcp"
        or meta.get("tcp_transform_applied") is not True
        or meta.get("approach_direction_frame") != base_frame
        or meta.get("approach_direction_semantics")
        != "graspgenx_base_positive_z_toward_object"
    ):
        raise ValueError("candidate lacks a framed GraspGen-X approach or TCP transform")
    direction = meta.get("approach_direction_in_frame")
    try:
        if len(direction) != 3:
            raise ValueError("candidate approach direction is unavailable")
        direction = tuple(float(x) for x in direction)
    except (TypeError, ValueError):
        raise ValueError("candidate approach direction is unavailable")
    norm = sum(x * x for x in direction) ** 0.5
    if not 0.99999 <= norm <= 1.00001:
        raise ValueError("candidate approach direction must be a finite unit vector")
    return Pose(
        position=tuple(
            float(candidate.pose.position[i]) - distance_m * direction[i] for i in range(3)
        ),
        quaternion_wxyz=candidate.pose.quaternion_wxyz,
        frame=base_frame,
    )


def plan_apple_pregrasp():
    context = get_task_context()
    config = context.metadata.get("open_rail_pick_apple")
    if config is None:
        return {"ok": False, "stage": "configuration", "reason": "missing task configuration"}
    try:
        config = dict(config)
        camera_name = str(config["camera_name"])
        arm = str(config["arm"])
        distance = float(config["pregrasp_distance_m"])
        maximum_skew = float(config["max_camera_state_skew_s"])
        minimum_depth_points = int(config["minimum_depth_points"])
        minimum_depth_fraction = float(config["minimum_depth_fraction"])
        max_candidates = int(config["max_candidates"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "stage": "configuration", "reason": "incomplete task configuration"}
    if (
        arm != "right"
        or not camera_name
        or not 0.0 < distance < float("inf")
        or not 0.0 <= maximum_skew < float("inf")
        or minimum_depth_points < 1
        or not 0.0 < minimum_depth_fraction <= 1.0
        or max_candidates < 1
    ):
        return {"ok": False, "stage": "configuration", "reason": "invalid task configuration"}

    observation = get_observation(camera_name)
    localization = localize_object_from_observation(
        "apple",
        observation,
        camera_name=camera_name,
        require_single_instance=True,
        min_valid_depth_points=minimum_depth_points,
        min_valid_depth_fraction=minimum_depth_fraction,
    )
    if not localization.ok:
        return {"ok": False, "stage": "localization", "error": localization.error}

    grasps = generate_grasps_for_localization(
        localization,
        observation,
        camera_name=camera_name,
        max_camera_state_skew_s=maximum_skew,
        backend="graspgenx",
        max_candidates=max_candidates,
    )
    if not grasps.ok:
        return {"ok": False, "stage": "grasp_generation", "error": grasps.error}
    if grasps.diagnostics.get("tcp_transform_applied") is not True:
        return {"ok": False, "stage": "grasp_generation", "reason": "TCP transform unavailable"}

    # The runner can optionally request a non-executable three-stage preview.
    # Keep the exact observation and localization together for its scene proof.
    if config.get("preview_full_chain") is True:
        try:
            voxel_size = float(config["preview_scene_voxel_size_m"])
            if not 0.0 < voxel_size < float("inf"):
                raise ValueError("invalid preview scene voxel size")
            partition = partition_target_scene(
                observation, localization, camera_name=camera_name,
                max_camera_state_skew_s=maximum_skew, voxel_size_m=voxel_size,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "stage": "scene_partition", "reason": str(exc)}
        candidates = []
        failures = []
        for candidate in grasps.grasps:
            try:
                pose = pregrasp_pose(candidate, distance, observation.robot_state.base_frame)
            except (TypeError, ValueError) as exc:
                failures.append(str(exc))
                continue
            candidates.append((candidate, pose))
        if not candidates:
            return {"ok": False, "stage": "pregrasp_planning", "failures": failures}
        return {
            "ok": False, "stage": "chain_preview_input", "snapshot_id": partition.snapshot_id,
            "partition": partition, "candidates": candidates, "preview_grasps": grasps,
            "preview_scene": get_planning_scene(observation),
        }

    # The full observed scene is kept for pregrasp planning, including apple
    # and table. Contact and held-object planning are separate future stages.
    scene = get_planning_scene(observation)
    strategy = MotionStrategy(
        ik_solver="curobo", trajectory_planner="curobo", pose_planner="curobo-integrated"
    )
    failures = []
    for candidate in grasps.grasps:
        try:
            pose = pregrasp_pose(candidate, distance, observation.robot_state.base_frame)
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
            continue
        plan = plan_motion(
            pose,
            arm=arm,
            strategy=strategy,
            planning_scene=scene,
            robot_state=observation.robot_state,
        )
        if plan.ok and plan.trajectory.collision_aware:
            return {
                "ok": False,
                "stage": "pregrasp_preview",
                "reason": "grasp approach, lift and verification are not yet available",
                "snapshot_id": localization.diagnostics.get("snapshot_id"),
                "candidate": candidate,
                "pregrasp_pose": pose,
                "trajectory": plan.trajectory,
                "preview_grasps": grasps,
                "preview_scene": scene,
            }
        failures.append("collision-checked pregrasp planning failed")
    return {"ok": False, "stage": "pregrasp_planning", "failures": failures,
            "preview_grasps": grasps, "preview_scene": scene}


result = plan_apple_pregrasp()
