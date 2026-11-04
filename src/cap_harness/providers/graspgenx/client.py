"""Typed ASPIRE client for the GraspGenX inference service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import requests

from cap_harness.contracts import (
    GraspCandidate,
    GraspSet,
    ObjectGeometry,
    PointCloud,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.providers.http import HttpProviderClient, ProviderHttpError
from cap_harness.providers.wire import decode_numpy, encode_numpy

from .frames import gripper_base_pose_to_tcp
from .wire import GENERATE_GRASPS_PATH

DEFAULT_GRASPGENX_URL = "http://127.0.0.1:8120"
_SWEEP_FIELDS = ("extents_open", "offset_open", "extents_mid", "offset_mid")
_PLANNER_KNOBS = frozenset(
    {
        "moe_num_yaws",
        "moe_z_offsets_cm",
        "moe_outlier_threshold",
        "moe_outlier_k",
        "moe_obb_mode",
        "moe_skip_obb_rule",
        "moe_obb_density",
        "moe_obb_position_spacing_cm",
    }
)


def _sweep_volume(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and JSON-normalize GraspGenX sweep-volume-v2 parameters."""
    if not isinstance(value, Mapping):
        raise ValueError("sweep_volume_params must be a mapping")
    missing = set(_SWEEP_FIELDS) - set(value)
    allowed = set(_SWEEP_FIELDS) | {"gripper_type", "fingertip_depth"}
    unknown = set(value) - allowed
    if missing or unknown:
        raise ValueError(
            f"sweep_volume_params missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    result: dict[str, object] = {}
    for name in _SWEEP_FIELDS:
        array = np.asarray(value[name], dtype=np.float32)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"sweep_volume_params[{name!r}] must be a finite 3-vector")
        if name.startswith("extents") and np.any(array <= 0.0):
            raise ValueError(f"sweep_volume_params[{name!r}] must be positive")
        result[name] = [float(item) for item in array]
    gripper_type = value.get("gripper_type", 0)
    if (
        isinstance(gripper_type, bool)
        or not isinstance(gripper_type, int)
        or gripper_type not in (0, 1, 2)
    ):
        raise ValueError("gripper_type must be 0, 1, or 2")
    result["gripper_type"] = gripper_type
    fingertip_depth = value.get("fingertip_depth")
    if fingertip_depth is not None:
        if isinstance(fingertip_depth, bool):
            raise ValueError("fingertip_depth must be a positive finite value")
        depth = float(fingertip_depth)
        if not np.isfinite(depth) or depth <= 0.0:
            raise ValueError("fingertip_depth must be a positive finite value")
        result["fingertip_depth"] = depth
    return result


class GraspGenXProvider:
    """Generate gripper-base grasp candidates from a segmented point cloud.

    ``sweep_volume_params`` must describe the physical gripper being planned
    for. The returned pose is the GraspGenX gripper-base pose in the input
    point-cloud frame by default. A measured ``gripper_base_to_tcp`` transform
    can map candidates to a calibrated TCP; no default is assumed. The provider
    does not collision-check candidates against the rest of the scene.
    """

    def __init__(
        self,
        sweep_volume_params: Mapping[str, object],
        base_url: str = DEFAULT_GRASPGENX_URL,
        *,
        planner: str = "graspmoe",
        num_grasps: int = 200,
        planner_kwargs: Mapping[str, object] | None = None,
        gripper_base_to_tcp: object | None = None,
        timeout_s: float | tuple[float, float] = 180.0,
        max_retries: int = 1,
        retry_backoff_s: float = 0.1,
        session: requests.Session | None = None,
    ) -> None:
        if planner not in {"graspmoe", "diffusion"}:
            raise ValueError("planner must be 'graspmoe' or 'diffusion'")
        if isinstance(num_grasps, bool) or not isinstance(num_grasps, int) or num_grasps <= 0:
            raise ValueError("num_grasps must be a positive integer")
        knobs = dict(planner_kwargs or {})
        unknown_knobs = set(knobs) - _PLANNER_KNOBS
        if unknown_knobs:
            raise ValueError(f"unknown GraspGenX planner knobs: {sorted(unknown_knobs)}")
        self.sweep_volume_params = _sweep_volume(sweep_volume_params)
        self.planner = planner
        self.num_grasps = num_grasps
        self.planner_kwargs = knobs
        if gripper_base_to_tcp is None:
            self.gripper_base_to_tcp = None
        else:
            transform = np.asarray(gripper_base_to_tcp, dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("gripper_base_to_tcp must have shape (4, 4)")
            # matrix_to_pose validates finiteness, homogeneous form, and rotation.
            matrix_to_pose(transform, frame="gripper_base")
            transform = np.array(transform, copy=True)
            transform.setflags(write=False)
            self.gripper_base_to_tcp = transform
        self._http = HttpProviderClient(
            base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            session=session,
        )

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def health(self) -> bool:
        return self._http.health()

    def generate_grasps(
        self,
        point_cloud: PointCloud,
        *,
        scene_point_cloud: PointCloud | None = None,
        geometry: ObjectGeometry | None = None,
        max_candidates: int | None = None,
    ) -> GraspSet:
        if not isinstance(point_cloud, PointCloud):
            return self._failure(ErrorCode.INVALID_REQUEST, "point_cloud must be PointCloud")
        if geometry is not None:
            if not isinstance(geometry, ObjectGeometry):
                return self._failure(ErrorCode.INVALID_REQUEST, "geometry must be ObjectGeometry")
            if geometry.frame != point_cloud.frame:
                return self._failure(
                    ErrorCode.INVALID_REQUEST,
                    "geometry and point_cloud frames do not match",
                    details={
                        "geometry_frame": geometry.frame,
                        "point_cloud_frame": point_cloud.frame,
                    },
                )
        if scene_point_cloud is not None:
            if not isinstance(scene_point_cloud, PointCloud):
                return self._failure(
                    ErrorCode.INVALID_REQUEST,
                    "scene_point_cloud must be PointCloud or None",
                )
            if scene_point_cloud.frame != point_cloud.frame:
                return self._failure(
                    ErrorCode.INVALID_REQUEST,
                    "scene and object point-cloud frames do not match",
                    details={
                        "scene_frame": scene_point_cloud.frame,
                        "object_frame": point_cloud.frame,
                    },
                )
        if max_candidates is not None and (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            return self._failure(
                ErrorCode.INVALID_REQUEST,
                "max_candidates must be a positive integer or None",
            )

        payload = {
            "point_cloud_base64": encode_numpy(point_cloud.points.astype(np.float32)),
            "sweep_volume_params": self.sweep_volume_params,
            "planner": self.planner,
            "num_grasps": self.num_grasps,
            "planner_kwargs": self.planner_kwargs,
        }
        try:
            response = self._http.post_json(GENERATE_GRASPS_PATH, payload)
            transforms = decode_numpy(response.get("grasps_base64"))
            scores = decode_numpy(response.get("scores_base64"))
            branches = response.get("branch_tags", [])
            timing = response.get("timing", {})
            if not isinstance(timing, Mapping):
                raise ValueError("timing must be a mapping")
            candidates = self._decode_candidates(
                transforms,
                scores,
                branches,
                frame=point_cloud.frame,
                max_candidates=max_candidates,
                gripper_base_to_tcp=self.gripper_base_to_tcp,
            )
        except ProviderHttpError as exc:
            return GraspSet(ok=False, error=exc.error)
        except (TypeError, ValueError, KeyError) as exc:
            return self._failure(
                ErrorCode.GRASP_FAILED,
                "GraspGenX returned an invalid response",
                details={"provider": "graspgenx", "error": str(exc)},
            )

        if not candidates:
            return self._failure(
                ErrorCode.NO_GRASP,
                "GraspGenX returned no candidates",
                details={"provider": "graspgenx", "frame": point_cloud.frame},
            )
        diagnostics: dict[str, object] = {
            "provider": "graspgenx",
            "frame": point_cloud.frame,
            "pose_semantics": (
                "tcp_in_input_point_cloud_frame"
                if self.gripper_base_to_tcp is not None
                else "graspgenx_gripper_base_in_input_point_cloud_frame"
            ),
            "tcp_transform_applied": self.gripper_base_to_tcp is not None,
            "scene_collision_checked": False,
            "scene_point_count": 0 if scene_point_cloud is None else len(scene_point_cloud.points),
        }
        if geometry is not None:
            diagnostics["geometry_supplied"] = True
        if timing:
            diagnostics["timing"] = dict(timing)
        return GraspSet(ok=True, grasps=candidates, diagnostics=diagnostics)

    def _decode_candidates(
        self,
        transforms: np.ndarray,
        scores: np.ndarray,
        branches: object,
        *,
        frame: str,
        max_candidates: int | None,
        gripper_base_to_tcp: np.ndarray | None,
    ) -> list[GraspCandidate]:
        transforms = np.asarray(transforms, dtype=np.float64)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"grasps must have shape (N, 4, 4), got {transforms.shape}")
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(transforms):
            raise ValueError("grasp and score counts do not match")
        if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("grasp confidences must be finite values in [0, 1]")
        if isinstance(branches, str | bytes) or not isinstance(branches, list | tuple):
            raise ValueError("branch_tags must be a sequence")
        if branches and len(branches) != len(transforms):
            raise ValueError("branch tag and grasp counts do not match")
        if transforms.shape[0] == 0:
            return []

        order = np.argsort(-scores, kind="stable")
        if max_candidates is not None:
            order = order[:max_candidates]
        width_m = float(self.sweep_volume_params["extents_open"][0])  # type: ignore[index]
        candidates: list[GraspCandidate] = []
        for index in order:
            transform = transforms[index]
            grasp_frame = "gripper_base"
            pose = matrix_to_pose(transform, frame=frame)
            if gripper_base_to_tcp is not None:
                pose = gripper_base_pose_to_tcp(pose, gripper_base_to_tcp)
                grasp_frame = "tcp"
            candidates.append(
                GraspCandidate(
                    pose=pose,
                    score=float(scores[index]),
                    width_m=width_m,
                    metadata={
                        "provider": "graspgenx",
                        "grasp_frame": grasp_frame,
                        "graspgenx_pose_frame": "gripper_base",
                        "tcp_transform_applied": gripper_base_to_tcp is not None,
                        "branch": str(branches[index]) if branches else "unspecified",
                    },
                )
            )
        return candidates

    @staticmethod
    def _failure(
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> GraspSet:
        return GraspSet(
            ok=False,
            error=ApiError(code=code, message=message, details=details or {}),
        )


GraspGenXClient = GraspGenXProvider

__all__ = ["DEFAULT_GRASPGENX_URL", "GraspGenXClient", "GraspGenXProvider"]
