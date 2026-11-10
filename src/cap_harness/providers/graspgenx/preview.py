"""Export a frame-labelled point-cloud preview of GraspGenX candidates.

The preview deliberately shows pose axes and approach directions, not a
guessed gripper body or opening. It never implies reachability or collision
clearance. PLY can be opened by common point-cloud viewers and needs no viewer
or plotting dependency in the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cap_harness.contracts import GraspSet, PointCloud


def write_grasp_preview(
    path: str | Path,
    scene: PointCloud,
    candidates: GraspSet,
    *,
    snapshot_id: str,
    timestamp_s: float,
    axis_length_m: float = 0.04,
    max_scene_points: int = 100_000,
) -> None:
    """Write an exclusive ASCII PLY for a scene and its candidate poses.

    ``snapshot_id`` is caller-supplied and must agree with the result's ID when
    available. Scene points and candidates must already share one frame. An
    approach line is drawn only for the explicitly framed GraspGenX direction;
    otherwise it is omitted rather than inferred from a possibly rotated TCP.
    Existing files are never overwritten.
    """
    if not isinstance(scene, PointCloud) or not isinstance(candidates, GraspSet):
        raise ValueError("scene and candidates must be CAP PointCloud and GraspSet")
    if not candidates.ok:
        raise ValueError("a successful grasp result is required")
    if any(candidate.frame != scene.frame for candidate in candidates.grasps):
        raise ValueError("scene and candidate frames do not match")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id must be a non-empty string")
    result_id = candidates.diagnostics.get("snapshot_id")
    if result_id is not None and result_id != snapshot_id:
        raise ValueError("candidate snapshot does not match preview snapshot")
    if isinstance(timestamp_s, bool):
        raise ValueError("timestamp_s must be finite")
    if isinstance(axis_length_m, bool):
        raise ValueError("axis_length_m must be positive and finite")
    try:
        timestamp = float(timestamp_s)
        axis_length = float(axis_length_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp and axis length must be numeric") from exc
    if not np.isfinite(timestamp):
        raise ValueError("timestamp_s must be finite")
    if not np.isfinite(axis_length) or axis_length <= 0:
        raise ValueError("axis_length_m must be positive and finite")
    if (
        isinstance(max_scene_points, bool)
        or not isinstance(max_scene_points, int)
        or max_scene_points < 1
    ):
        raise ValueError("max_scene_points must be a positive integer")

    stride = max(1, (len(scene.points) + max_scene_points - 1) // max_scene_points)
    points = [scene.points[::stride]]
    if scene.colors is None:
        scene_colors = np.full((len(points[0]), 3), 160, dtype=np.uint8)
    else:
        scene_colors = scene.colors[::stride]
        if np.issubdtype(scene_colors.dtype, np.floating):
            scene_colors = np.rint(scene_colors * 255.0)
        scene_colors = scene_colors.astype(np.uint8)
    colors = [scene_colors]
    candidate_ids = [np.full(len(points[0]), -1, dtype=np.int32)]
    components = [np.zeros(len(points[0]), dtype=np.uint8)]
    axis_colors = ((255, 64, 64), (64, 255, 64), (64, 128, 255))
    steps = np.linspace(0.0, axis_length, 25)[:, None]
    records = []
    for index, candidate in enumerate(candidates.grasps):
        transform = candidate.pose.as_matrix()
        origin, rotation = transform[:3, 3], transform[:3, :3]
        for axis, color in enumerate(axis_colors):
            points.append(origin + steps * rotation[:, axis])
            colors.append(np.tile(color, (len(steps), 1)))
            candidate_ids.append(np.full(len(steps), index, dtype=np.int32))
            components.append(np.full(len(steps), axis + 1, dtype=np.uint8))

        metadata = candidate.metadata
        direction = metadata.get("approach_direction_in_frame")
        direction_frame = metadata.get("approach_direction_frame")
        semantics = metadata.get("approach_direction_semantics")
        approach_shown = False
        if (
            direction is not None
            and direction_frame == scene.frame
            and semantics == "graspgenx_base_positive_z_toward_object"
        ):
            direction = np.asarray(direction, dtype=np.float64)
            if (
                direction.shape != (3,)
                or not np.all(np.isfinite(direction))
                or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-5)
            ):
                raise ValueError("candidate approach direction must be a finite unit vector")
            points.append(origin - steps * direction)
            colors.append(np.tile((255, 220, 0), (len(steps), 1)))
            candidate_ids.append(np.full(len(steps), index, dtype=np.int32))
            components.append(np.full(len(steps), 4, dtype=np.uint8))
            approach_shown = True
        records.append({"index": index, "score": candidate.score,
                        "pose_frame": str(metadata.get("grasp_frame", "unspecified")),
                        "approach_shown": approach_shown})

    all_points, all_colors = np.concatenate(points), np.concatenate(colors)
    vertices = np.column_stack((all_points, all_colors, np.concatenate(candidate_ids),
                                np.concatenate(components)))
    summary = {"frame": scene.frame, "snapshot_id": snapshot_id,
               "timestamp_s": timestamp, "snapshot_association": "caller_supplied",
               "scene_points_original": len(scene.points), "scene_points_shown": len(points[0]),
               "axis_length_m": axis_length, "candidates": records,
               "collision_checked": False, "gripper_geometry_shown": False}
    header = ["ply", "format ascii 1.0",
              "comment Candidate pose preview only; collision clearance and gripper geometry not shown.",
              "comment components: 0 scene; 1 pose X red; 2 pose Y green; 3 pose Z blue; 4 approach yellow",
              "comment yellow approach runs from pregrasp toward the pose origin",
              "comment metadata " + json.dumps(summary, separators=(",", ":"), ensure_ascii=True),
              f"element vertex {len(vertices)}", "property float x", "property float y",
              "property float z", "property uchar red", "property uchar green", "property uchar blue",
              "property int candidate_id", "property uchar component", "end_header"]
    output = Path(path)
    with output.open("x", encoding="ascii", newline="\n") as stream:
        try:
            stream.write("\n".join(header) + "\n")
            np.savetxt(stream, vertices, fmt=("%.9g", "%.9g", "%.9g", "%d", "%d", "%d", "%d", "%d"))
        except BaseException:
            stream.close()
            output.unlink(missing_ok=True)
            raise


__all__ = ["write_grasp_preview"]
