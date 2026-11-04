"""Explicit held-object geometry helpers for cuRobo configuration.

A caller must provide the measured parent-to-object transform and collision
spheres; these helpers serialize them into the pinned cuRobo v2 schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _validate_pose7(pose: Any, *, name: str) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be seven finite values [x, y, z, qw, qx, qy, qz]")
    norm = float(np.linalg.norm(value[3:]))
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(f"{name} quaternion must be normalized")
    return value


def pose7_to_matrix(pose: Any, *, name: str = "pose") -> np.ndarray:
    """Convert xyz + quaternion-wxyz to a validated homogeneous transform."""
    x, y, z, qw, qx, qy, qz = _validate_pose7(pose, name=name)
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [x, y, z]
    return matrix


def _validate_spheres(spheres: Any, *, name: str = "collision_spheres") -> np.ndarray:
    value = np.asarray(spheres, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or len(value) == 0:
        raise ValueError(f"{name} must have shape (N, 4) with at least one [x, y, z, radius]")
    if not np.all(np.isfinite(value)) or np.any(value[:, 3] <= 0.0):
        raise ValueError(f"{name} must contain finite centers and positive radii")
    return value


@dataclass(frozen=True)
class AttachedObjectGeometry:
    """Caller-verified rigid object geometry attached to a robot link.

    ``parent_to_object`` uses xyz + quaternion-wxyz. Sphere centers are in the
    object frame and each row is ``[x, y, z, radius]`` in meters. The data type
    carries no robot- or gripper-specific defaults.
    """

    object_link_name: str
    attachment_joint_name: str
    parent_link_name: str
    parent_to_object: tuple[float, float, float, float, float, float, float]
    collision_spheres: tuple[tuple[float, float, float, float], ...]
    contact_link_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("object_link_name", "attachment_joint_name", "parent_link_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty name")
        if self.object_link_name == self.parent_link_name:
            raise ValueError("object link and parent link must be different")
        parent_to_object = _validate_pose7(self.parent_to_object, name="parent_to_object")
        collision_spheres = _validate_spheres(self.collision_spheres)
        if isinstance(self.contact_link_names, str):
            raise ValueError("contact_link_names must be a sequence of link names, not one string")
        contacts = tuple(self.contact_link_names)
        if not contacts or any(not isinstance(name, str) or not name.strip() for name in contacts):
            raise ValueError("contact_link_names must explicitly list at least one link")
        if len(set(contacts)) != len(contacts):
            raise ValueError("contact_link_names must not contain duplicates")
        if self.object_link_name in contacts:
            raise ValueError("object link cannot be its own contact link")

        # A frozen dataclass only protects its attributes from reassignment;
        # callers may still pass mutable lists or arrays. Snapshot all sequence
        # fields as immutable Python tuples so later caller mutation cannot
        # change the collision geometry or its serialized payload.
        object.__setattr__(
            self, "parent_to_object", tuple(float(value) for value in parent_to_object)
        )
        object.__setattr__(
            self,
            "collision_spheres",
            tuple(tuple(float(value) for value in row) for row in collision_spheres),
        )
        object.__setattr__(self, "contact_link_names", contacts)

    def pose_in_base(self, parent_pose_in_base: Any) -> np.ndarray:
        """Compose a measured parent pose with the supplied fixed attachment transform."""
        parent = np.asarray(parent_pose_in_base, dtype=np.float64)
        if parent.shape != (4, 4) or not np.all(np.isfinite(parent)):
            raise ValueError("parent_pose_in_base must be a finite 4x4 transform")
        if not np.allclose(parent[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("parent_pose_in_base must have homogeneous last row [0, 0, 0, 1]")
        rotation = parent[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError("parent_pose_in_base rotation must be proper and orthonormal")
        return parent @ pose7_to_matrix(self.parent_to_object, name="parent_to_object")

    def curobo_kinematics_patch(self) -> dict[str, Any]:
        """Return v2 kinematics fields to merge into an existing robot config.

        The attached link is added to ``tool_frames`` for FK access, to
        ``collision_link_names`` for environment collision, and its explicitly
        supplied contact links are ignored for self collision. The caller still
        owns checking those names against the actual robot description.
        """
        return {
            "tool_frames": [self.object_link_name],
            "extra_links": {
                self.object_link_name: {
                    "fixed_transform": list(self.parent_to_object),
                    "joint_name": self.attachment_joint_name,
                    "joint_type": "FIXED",
                    "link_name": self.object_link_name,
                    "parent_link_name": self.parent_link_name,
                }
            },
            "collision_link_names": [self.object_link_name],
            "collision_spheres": {
                self.object_link_name: [
                    {"center": row[:3].tolist(), "radius": float(row[3])}
                    for row in _validate_spheres(self.collision_spheres)
                ]
            },
            "self_collision_ignore": {
                self.object_link_name: list(self.contact_link_names),
            },
        }

    def to_payload(self) -> dict[str, Any]:
        """Return the explicit JSON fields accepted by the cuRobo service."""
        return {
            "object_link_name": self.object_link_name,
            "attachment_joint_name": self.attachment_joint_name,
            "parent_link_name": self.parent_link_name,
            "parent_to_object": list(self.parent_to_object),
            "collision_spheres": [list(sphere) for sphere in self.collision_spheres],
            "contact_link_names": list(self.contact_link_names),
        }


def merge_attachment_into_kinematics(
    kinematics: Mapping[str, Any], attachment: AttachedObjectGeometry
) -> dict[str, Any]:
    """Copy and extend an inline-sphere cuRobo v2 kinematics mapping.

    This helper refuses file-backed collision sphere tables because it cannot
    safely infer or rewrite their contents. It also refuses a pre-existing
    link with the same name rather than silently replacing robot geometry.
    """
    result = deepcopy(dict(kinematics))
    patch = attachment.curobo_kinematics_patch()
    collision_spheres = result.get("collision_spheres")
    if not isinstance(collision_spheres, dict):
        raise ValueError("kinematics collision_spheres must be an inline mapping")
    if attachment.object_link_name in collision_spheres:
        raise ValueError(f"collision sphere link already exists: {attachment.object_link_name}")

    if result.get("extra_links") is None:
        result["extra_links"] = {}
    extra_links = result["extra_links"]
    if not isinstance(extra_links, dict) or attachment.object_link_name in extra_links:
        raise ValueError(
            f"extra link already exists or extra_links is invalid: {attachment.object_link_name}"
        )
    extra_links.update(patch["extra_links"])
    collision_spheres.update(patch["collision_spheres"])

    if result.get("collision_link_names") is None:
        result["collision_link_names"] = []
    collision_link_names = result["collision_link_names"]
    if not isinstance(collision_link_names, list):
        raise ValueError("collision_link_names must be a list")
    if attachment.object_link_name not in collision_link_names:
        collision_link_names.append(attachment.object_link_name)

    tool_frames = result.setdefault("tool_frames", [])
    if not isinstance(tool_frames, list):
        raise ValueError("tool_frames must be a list")
    if attachment.object_link_name not in tool_frames:
        tool_frames.append(attachment.object_link_name)

    if result.get("self_collision_ignore") is None:
        result["self_collision_ignore"] = {}
    self_collision_ignore = result["self_collision_ignore"]
    if not isinstance(self_collision_ignore, dict):
        raise ValueError("self_collision_ignore must be a mapping")
    existing_ignored = self_collision_ignore.get(attachment.object_link_name, [])
    if not isinstance(existing_ignored, list):
        raise ValueError("self_collision_ignore values must be lists")
    self_collision_ignore[attachment.object_link_name] = list(
        dict.fromkeys([*existing_ignored, *attachment.contact_link_names])
    )
    return result


def transform_spheres_to_base(object_pose_in_base: Any, spheres_object: Any) -> np.ndarray:
    """Transform object-local sphere centers into base coordinates."""
    pose = np.asarray(object_pose_in_base, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("object_pose_in_base must be a finite 4x4 transform")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("object_pose_in_base must have homogeneous last row [0, 0, 0, 1]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("object_pose_in_base rotation must be proper and orthonormal")
    spheres = _validate_spheres(spheres_object)
    transformed = spheres.copy()
    transformed[:, :3] = spheres[:, :3] @ rotation.T + pose[:3, 3]
    return transformed


def carve_attached_object_points(
    scene_points_base: Any,
    object_pose_in_base: Any,
    spheres_object: Any,
    *,
    padding_m: float,
) -> np.ndarray:
    """Remove only points covered by caller-supplied attached-object spheres.

    This avoids removing an axis-aligned box around the object, which can erase
    adjacent table or plate returns. Sphere fitting still determines the actual
    carve and may cover nearby support geometry if the supplied spheres are too
    large. ``padding_m`` must be explicit because it trades depth-noise tolerance
    against retained neighboring scene geometry.
    """
    points = np.asarray(scene_points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("scene_points_base must have shape (N, 3) and finite values")
    if not np.isfinite(padding_m) or padding_m < 0.0:
        raise ValueError("padding_m must be finite and non-negative")
    spheres = transform_spheres_to_base(object_pose_in_base, spheres_object)
    keep = np.ones(len(points), dtype=bool)
    # Chunking caps the temporary pairwise distance array for large RGB-D clouds.
    max_pairs = 1_000_000
    points_per_chunk = max(1, max_pairs // len(spheres))
    radii_sq = (spheres[:, 3] + float(padding_m)) ** 2
    for start in range(0, len(points), points_per_chunk):
        chunk = points[start : start + points_per_chunk]
        distance_sq = np.sum((chunk[:, None, :] - spheres[None, :, :3]) ** 2, axis=2)
        keep[start : start + len(chunk)] = np.all(distance_sq > radii_sq[None, :], axis=1)
    return points[keep]
