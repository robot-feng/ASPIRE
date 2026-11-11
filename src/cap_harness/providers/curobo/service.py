"""FastAPI service for cuRobo V2 IK and collision-aware motion planning."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import functools
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import tempfile
from typing import Any, Literal
import xml.etree.ElementTree as ET

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, PrivateAttr, model_validator
import uvicorn

from cap_harness.contracts import JOINT_DIMENSIONS
from cap_harness.providers.service_runtime import install_service_runtime

from ..wire import decode_numpy, encode_numpy
from .attachment import (
    AttachedObjectGeometry,
    carve_attached_object_points,
    merge_attachment_into_kinematics,
    pose7_to_matrix,
)
from .dual import build_dual_panda_config
from .sparse_scene import SparseSceneLimitError, sparse_voxel_mesh
from .wire import IK_PATH, PLAN_PATH, PLAN_SYNCHRONIZED_PATH

app = FastAPI(title="cap-harness cuRobo V2 provider")
# Bound concurrent GPU/inference requests so one instance safely serves
# multiple simulator clients (limit from CAP_HARNESS_SERVICE_CONCURRENCY).
install_service_runtime(app)
_PLANNERS: dict[tuple[str, float | None, float | None], Any] = {}
_IK_SOLVERS: dict[str, Any] = {}
_SEGMENTATION_KINEMATICS: dict[str, Any] = {}
_ATTACHED_CONFIG_PATHS: dict[str, str] = {}
_SEGMENTATION_TOLERANCE_M = 0.02
_LOGGER = logging.getLogger(__name__)


def _collision_disabled(model: str) -> bool:
    """Whether to drop every collision term from cuRobo.

    Collision checking is on for every model. ``CAP_HARNESS_CUROBO_DISABLE_COLLISION``
    disables it everywhere (any value but ``0``). It is read per request rather than
    at import, so a service restart is all it takes to change, and logged on every
    use so a blind plan cannot be mistaken for a checked one in a recorded run.
    """
    override = os.environ.get("CAP_HARNESS_CUROBO_DISABLE_COLLISION")
    return override is not None and override != "0"


_ROBOT_SEGMENTATION_DISTANCE_M = 0.05

#: Peak float64 elements per distance-matrix chunk in the robot carve. The dense
#: set is ~1250 spheres against ~46k points; chunking on the product rather than
#: a fixed point count keeps that under a few hundred MB whichever set is used.
_CARVE_CHUNK_ELEMENTS = 16_000_000


class Target(BaseModel):
    type: Literal["pose", "joints"]
    position: list[float] | None = None
    quaternion_wxyz: list[float] | None = None
    frame: str | None = None
    joint_positions: list[float] | None = None


class AttachedObjectRequest(BaseModel):
    object_link_name: str
    attachment_joint_name: str
    parent_link_name: str
    parent_to_object: list[float]
    collision_spheres: list[list[float]]
    contact_link_names: list[str]

    def to_geometry(self) -> AttachedObjectGeometry:
        return AttachedObjectGeometry(
            object_link_name=self.object_link_name,
            attachment_joint_name=self.attachment_joint_name,
            parent_link_name=self.parent_link_name,
            parent_to_object=tuple(self.parent_to_object),
            collision_spheres=tuple(tuple(sphere) for sphere in self.collision_spheres),
            contact_link_names=tuple(self.contact_link_names),
        )

    @model_validator(mode="after")
    def validate_geometry(self) -> AttachedObjectRequest:
        self.to_geometry()
        return self


class TargetScenePartitionRequest(BaseModel):
    mode: Literal["pixel_partition_v1", "pixel_partition_v2"]
    snapshot_id: str
    snapshot_fingerprint: str
    scene_fingerprint: str
    scene_points_sha256: str

    @model_validator(mode="after")
    def validate_fingerprints(self) -> TargetScenePartitionRequest:
        if not self.snapshot_id.strip():
            raise ValueError("partition snapshot_id must be nonempty")
        for value in (self.snapshot_fingerprint, self.scene_fingerprint, self.scene_points_sha256):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("partition fingerprints must be lowercase SHA-256")
        return self


class BaseRequest(BaseModel):
    base_frame: str
    model: Literal["panda", "dual_panda", "yam_real", "agibot_g1"]
    joint_positions: dict[str, list[float]]
    joint_names: dict[str, list[str]]
    base_transforms: dict[str, list[list[float]]]
    end_effector_links: dict[str, str]
    scene_points_base64: str | None = None
    voxel_size_m: float = 0.01
    interpolation_dt_s: float | None = None
    maximum_trajectory_dt_s: float | None = None
    attached_object: AttachedObjectRequest | None = None
    target_scene_partition: TargetScenePartitionRequest | None = None
    _g1_joint_map: dict[str, str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_target_scene_partition(self) -> BaseRequest:
        if self.target_scene_partition is None:
            return self
        if self.scene_points_base64 is None or (
            self.target_scene_partition.mode == "pixel_partition_v1" and self.attached_object is None
        ):
            raise ValueError("target scene partition requires a scene and v1 requires an attached object and scene")
        try:
            points = np.asarray(decode_numpy(self.scene_points_base64))
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or points.dtype != np.float32:
                raise ValueError("partition scene must have float32 shape (N, 3)")
            if not np.all(np.isfinite(points)):
                raise ValueError("partition scene must contain finite points")
            digest = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        except Exception as exc:
            raise ValueError(f"invalid partition scene payload: {exc}") from exc
        if digest != self.target_scene_partition.scene_points_sha256:
            raise ValueError("partition scene points do not match their SHA-256")
        return self


class IkRequest(BaseRequest):
    arm: str
    target: Target


class PlanRequest(BaseRequest):
    arm: str
    target: Target


class SynchronizedPlanRequest(BaseRequest):
    targets: dict[str, Target]


class PlanResponse(BaseModel):
    joint_positions_base64: str
    dt_s: float
    attachment_diagnostics: dict[str, Any] | None = None
    scene_partition_diagnostics: dict[str, Any] | None = None
    collision_diagnostics: dict[str, bool]


class SynchronizedPlanResponse(BaseModel):
    joint_positions_base64: dict[str, str]
    dt_s: float
    attachment_diagnostics: dict[str, Any] | None = None
    collision_diagnostics: dict[str, bool]


@functools.lru_cache(maxsize=1)
def _yam_real_robot_config() -> str:
    """CuRobo description of the physical bimanual station.

    Cached, and it must be. This writes a temp file whose name is random, and
    ``_planner`` keys its planner cache on that name. Uncached, every plan request
    produced a new name, so the key never repeated, so the cache never hit, so
    each request built a fresh ``MotionPlanner`` and called ``warmup`` -- which
    records CUDA graphs into private pools that are never released.

    The service then dies of an out-of-memory that looks like nothing to do with
    it: every plan is refused, and the arm reads as unable to reach anywhere.
    Measured on hardware before this decorator existed: 159 plan requests,
    158 solver initializations, 158 temp configs, and a bimanual planner leaked
    per request until 20 MiB could not be allocated. ``_yam_sim_robot_config``
    has always been cached, which is why only the real station showed it.

    Distinct from the YAM Sim config, which describes a **single** arm
    (``joint1..6`` with the fingers locked). The real station carries both arms
    in one model 0.62 m apart, and they can reach each other -- arm-to-arm
    self-collision is the thing a bimanual planner is for, and a single-arm
    description cannot express it at any setting.

    The config and the URDF are a matched pair and must stay one: this URDF
    declares the four finger joints ``fixed``, so they are absent from the
    kinematic tree, and its config correspondingly locks nothing. Pairing a
    config that locks the fingers with a URDF that articulates them (or the
    reverse) is rejected at load time. Its collision spheres are inline, because
    v2 will not load them from a path.

    The shipped config is in cuRobo's v1 schema, so it is converted here. Each
    step below is load-bearing; without them the v2 loader either refuses the
    config or accepts a subtly broken one.
    """
    configured = os.environ.get("CAP_HARNESS_CUROBO_YAM_REAL_CONFIG")
    station = Path(__file__).resolve().parents[2] / "yam_real" / "description" / "station"
    source = (
        Path(configured)
        if configured
        else station / "curobo" / "yam_dual_isaacsim_physics_fixed_fingers.yml"
    )
    if not source.is_file():
        raise FileNotFoundError(f"YAM real cuRobo config not found: {source}")

    import yaml

    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]

    urdf = (station / kinematics["urdf_path"]).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"YAM real station URDF not found: {urdf}")
    # v1 keys the v2 loader rejects outright. The USD group describes an Isaac
    # asset v2 neither reads nor tolerates; the URDF is the real source.
    for legacy in (
        "use_usd_kinematics",
        "usd_path",
        "usd_robot_root",
        "isaac_usd_path",
        "usd_flip_joints",
        "usd_flip_joint_limits",
        "asset_root_path",
    ):
        kinematics.pop(legacy, None)

    # v1 named one ee_link plus a link_names list; v2 takes tool_frames. BOTH
    # grasp frames have to survive -- keeping only ee_link leaves the right arm
    # with no tool frame, and every right-arm goal becomes unsolvable.
    end_effector = kinematics.pop("ee_link", "left_grasp")
    link_names = kinematics.pop("link_names", None) or []
    kinematics["tool_frames"] = list(dict.fromkeys([end_effector, *link_names]))

    # The self-collision ignore map is left exactly as shipped. An earlier version
    # of this function stripped a pair from it, on the inherited claim that the
    # config "ignores every populated pair" and that cuRobo divides by zero on an
    # empty table. That claim is false for this config: 15 collision links give
    # 105 pairs and the ignore map removes about 6, so ~99 pairs are checked. The
    # mutation was also a no-op here, since neither link ignored the other.

    cspace = kinematics.get("cspace") or {}
    if "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")

    kinematics["format_version"] = 2.0
    kinematics["urdf_path"] = str(urdf)

    output = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="cap_yam_real_curobo_", delete=False
    )
    with output:
        yaml.safe_dump(config, output, sort_keys=False)
    return output.name


def _yam_real_segmentation_config() -> str:
    """The station model again, but wearing segmentation geometry.

    Same URDF, same kinematics, same joint order as the planning config -- only
    the sphere set differs. Building it as a second config rather than editing
    the first is the whole point of the fix: the planner keeps its sparse,
    fast self-collision skeleton, and the carve gets geometry that actually
    follows the arm.

    Self-collision is stripped rather than translated. This model never plans;
    it exists to answer "where is the robot right now" so those points can be
    subtracted from the depth cloud, and carrying a 20-link pair table for
    that would only be a second thing to keep consistent.
    """
    import yaml

    source = Path(_yam_real_robot_config())
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]

    spheres_path = (
        Path(__file__).resolve().parents[2]
        / "yam_real/description/station/curobo/yam_dual_segmentation_spheres.yml"
    )
    if not spheres_path.is_file():
        raise FileNotFoundError(
            f"YAM segmentation spheres not found: {spheres_path}; "
            "reinstall the packaged YAM descriptions"
        )
    document = yaml.safe_load(spheres_path.read_text(encoding="utf-8"))
    spheres = document["collision_spheres"]

    kinematics["collision_spheres"] = spheres
    kinematics["collision_link_names"] = sorted(spheres)
    kinematics["mesh_link_names"] = sorted(spheres)
    # The planning config adds 5 mm to every sphere. The segmentation set states
    # its own radii and its tolerance is applied at the carve, so no hidden
    # inflation here -- otherwise the effective carve is a number that appears
    # in neither file.
    kinematics["collision_sphere_buffer"] = 0.0
    kinematics["self_collision_ignore"] = {}
    kinematics["self_collision_buffer"] = dict.fromkeys(spheres, 0.0)

    output = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="cap_yam_real_segmentation_", delete=False
    )
    with output:
        yaml.safe_dump(config, output, sort_keys=False)
    return output.name


def _segmentation_kinematics(request: BaseRequest) -> Any | None:
    """Kinematics carrying the dense segmentation spheres, or None if unmodelled.

    Only real YAM has a segmentation set. Every other model falls back to its
    self-collision spheres, which is what shipped and is left alone here.
    """
    if request.model != "yam_real":
        return None
    cached = _SEGMENTATION_KINEMATICS.get(request.model)
    if cached is not None:
        return cached
    from curobo.kinematics import Kinematics, KinematicsCfg

    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(_yam_real_segmentation_config()))
    _SEGMENTATION_KINEMATICS[request.model] = kinematics
    return kinematics


def _robot_config(request: BaseRequest) -> str:
    if request.model == "panda":
        robot_config = os.environ.get("CAP_HARNESS_CUROBO_PANDA_CONFIG", "franka.yml")
    elif request.model == "yam_real":
        robot_config = _yam_real_robot_config()
    elif request.model == "agibot_g1":
        configured = os.environ.get("CAP_HARNESS_CUROBO_G1_CONFIG")
        if not configured:
            raise HTTPException(status_code=503, detail="G1 cuRobo robot config is not configured")
        path = Path(configured)
        if not path.is_absolute() or not path.is_file():
            raise HTTPException(
                status_code=503,
                detail="G1 cuRobo robot config must be an existing absolute path",
            )
        if _collision_disabled(request.model):
            raise HTTPException(status_code=400, detail="G1 planning requires collision checking")
        robot_config = str(path.resolve())
    else:
        value = os.environ.get("CAP_HARNESS_CUROBO_DUAL_CONFIG")
        if value and Path(value).is_file():
            robot_config = value
        else:
            try:
                import curobo

                curobo_root = Path(curobo.__file__).resolve().parents[1]
                secondary = np.asarray(request.base_transforms["secondary"], dtype=np.float64)
                robot_config = str(
                    build_dual_panda_config(
                        curobo_root,
                        secondary,
                        Path(
                            os.environ.get(
                                "CAP_HARNESS_CUROBO_MODEL_CACHE", "/tmp/curobo-models"
                            )
                        ),
                    )
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"could not build dual-Panda model: {exc}"
                ) from exc

    if request.attached_object is None:
        return robot_config
    if _collision_disabled(request.model):
        raise HTTPException(
            status_code=400,
            detail="attached-object planning requires collision checking to be enabled",
        )
    if request.scene_points_base64 is None:
        raise HTTPException(
            status_code=400,
            detail="attached-object planning requires an observed scene point cloud",
        )
    try:
        return _attached_robot_config(robot_config, request.attached_object.to_geometry())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid attached-object robot config: {exc}"
        ) from exc


def _attachment_diagnostics(request: BaseRequest) -> dict[str, Any] | None:
    if request.attached_object is None:
        return None
    result = {
        "object_link": request.attached_object.object_link_name,
        "collision_model": "caller_supplied_spheres",
        "geometry_verified": False,
        "sphere_coverage_verified": False,
        "scene_carve_may_remove_support_points": True,
        "contact_links_ignored_for_self_collision": list(
            request.attached_object.contact_link_names
        ),
    }
    if request.target_scene_partition is not None and request.target_scene_partition.mode == "pixel_partition_v1":
        result["scene_partition"] = request.target_scene_partition.model_dump()
        # Robot geometry carving is still active. Even without the attachment
        # carve, a nearby surface may be removed by robot collision spheres.
        result["robot_scene_carve_may_remove_support_points"] = True
    return result


def _scene_partition_diagnostics(request: BaseRequest) -> dict[str, Any] | None:
    partition = request.target_scene_partition
    if partition is None or partition.mode != "pixel_partition_v2":
        return None
    return {
        "scene_partition": partition.model_dump(),
        "all_background_points_retained": True,
        "robot_scene_carve_applied": False,
        "target_interaction_verified": False,
        "occluded_space_verified": False,
    }


def _collision_diagnostics(request: BaseRequest) -> dict[str, bool]:
    """Acknowledge collision checks actually enabled for this successful request.

    Callers add this to a plan response only after ``_planner`` succeeds. When
    collision checking is enabled, a supplied scene has then passed decoding,
    finite/non-empty checks, workspace filtering, and scene-mesh construction.
    A request without points may still be planned against robot self-collision,
    but it has no world-collision guarantee.
    """
    disabled = _collision_disabled(request.model)
    scene_supplied = request.scene_points_base64 is not None
    result = {
        "collision_disabled": disabled,
        "scene_supplied": scene_supplied,
        "world_collision_check_enabled": not disabled and scene_supplied,
        "self_collision_check_enabled": not disabled,
    }
    if request.model == "agibot_g1":
        # Loading a YAML cannot establish that its collision geometry, lock
        # joints or tool frame correspond to the physical robot. Expose offline
        # plans without presenting them as verified execution trajectories.
        result["robot_model_verified"] = False
    if request.target_scene_partition is not None:
        result["target_partition_safety_verified"] = False
        result["robot_scene_carve_verified"] = False
    return result


def _attached_robot_config(
    robot_config: str, attachment: AttachedObjectGeometry
) -> str:
    """Create a stable temporary cuRobo config carrying an explicit fixed object link."""
    import yaml

    source = Path(robot_config)
    if not source.is_file():
        from curobo.content import get_robot_configs_path

        source = Path(get_robot_configs_path()) / robot_config
    if not source.is_file():
        raise FileNotFoundError(f"cuRobo robot config not found: {source}")

    source_bytes = source.read_bytes()
    payload = json.dumps(attachment.to_payload(), sort_keys=True, separators=(",", ":"))
    document = yaml.safe_load(source_bytes)
    if not isinstance(document, dict):
        raise ValueError("cuRobo config root must be a mapping")
    robot_cfg = document.get("robot_cfg", document)
    if not isinstance(robot_cfg, dict) or not isinstance(robot_cfg.get("kinematics"), dict):
        raise ValueError("cuRobo config must contain a kinematics mapping")
    urdf_path = _validate_attachment_links(robot_cfg["kinematics"], attachment)
    digest = hashlib.sha256(
        source_bytes + urdf_path.read_bytes() + payload.encode("utf-8")
    ).hexdigest()
    cached = _ATTACHED_CONFIG_PATHS.get(digest)
    if cached is not None and Path(cached).is_file():
        return cached

    robot_cfg["kinematics"] = merge_attachment_into_kinematics(
        robot_cfg["kinematics"], attachment
    )

    output_dir = Path(tempfile.gettempdir()) / "cap_harness_curobo_attached"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{digest}.yml"
    if not output.is_file():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", prefix=f"{digest}.", dir=output_dir,
            encoding="utf-8", delete=False,
        ) as handle:
            yaml.safe_dump(document, handle, sort_keys=False)
            temporary = Path(handle.name)
        os.replace(temporary, output)
    _ATTACHED_CONFIG_PATHS[digest] = str(output)
    return str(output)


def _validate_attachment_links(
    kinematics: Mapping[str, Any], attachment: AttachedObjectGeometry
) -> Path:
    """Check requested parent/contact links exist and reject URDF closure loops."""
    urdf_path = kinematics.get("urdf_path")
    if not isinstance(urdf_path, str) or not urdf_path:
        raise ValueError("attachment requires a robot URDF path")
    urdf = Path(urdf_path)
    if not urdf.is_absolute():
        from curobo.content import get_assets_path

        urdf = Path(get_assets_path()) / urdf
    if not urdf.is_file():
        raise FileNotFoundError(f"robot URDF not found: {urdf}")
    root = ET.parse(urdf).getroot()
    loop_joints = root.findall("loop_joint")
    if loop_joints:
        names = [joint.attrib.get("name", "<unnamed>") for joint in loop_joints]
        raise ValueError(
            "cuRobo request attachments require a tree URDF; unsupported loop_joint tags: "
            f"{names}"
        )
    links = {element.attrib.get("name") for element in root.findall("link")}
    required = {attachment.parent_link_name, *attachment.contact_link_names}
    missing = sorted(required - links)
    if missing:
        raise ValueError(f"attachment links are absent from robot URDF: {missing}")
    parent_by_child: dict[str, str] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        if child is not None and parent is not None:
            child_link = child.attrib.get("link")
            parent_link = parent.attrib.get("link")
            if child_link is not None and parent_link is not None:
                parent_by_child[child_link] = parent_link
    for contact in attachment.contact_link_names:
        ancestor = contact
        while ancestor != attachment.parent_link_name and ancestor in parent_by_child:
            ancestor = parent_by_child[ancestor]
        if ancestor != attachment.parent_link_name:
            raise ValueError(
                f"contact link {contact!r} is outside the attached gripper subtree "
                f"rooted at {attachment.parent_link_name!r}"
            )
    if attachment.object_link_name in links:
        raise ValueError(
            f"attachment object link already exists in robot URDF: {attachment.object_link_name}"
        )
    return urdf


def _filter_robot_points(
    points: np.ndarray,
    spheres: np.ndarray,
    *,
    padding_m: float,
) -> np.ndarray:
    """Remove depth points belonging to the current collision-sphere model."""
    keep = np.ones(len(points), dtype=bool)
    stride = max(1024, _CARVE_CHUNK_ELEMENTS // max(len(spheres), 1))
    for start in range(0, len(points), stride):
        chunk = points[start : start + stride]
        distances = np.linalg.norm(chunk[:, None, :] - spheres[None, :, :3], axis=-1)
        keep[start : start + len(chunk)] = np.all(
            distances > spheres[None, :, 3] + padding_m,
            axis=1,
        )
    return points[keep]


def _mesh_from_pointcloud(
    pointcloud: np.ndarray,
    pitch: float,
    name: str,
    pose: list[float],
    mesh_type: Any,
) -> Any:
    """Create a mesh from a pointcloud via voxelized surface extraction.

    Verbatim copy of the pinned cuRobo implementation
    (third_party/curobo curobo/_src/geom/types.py::Mesh.from_pointcloud,
    commit a35a708) except for the single PATCH block below. Upstream winds
    the boundary triangles inward (a one-voxel mesh has volume -pitch**3),
    which inverts warp's signed-distance queries: free space near surfaces
    reads as inside the mesh and the planner refuses reachable goals.
    Delete this copy and call Mesh.from_pointcloud once upstream fixes the
    winding.
    """
    if len(pointcloud) == 0:
        return mesh_type(name, pose=pose, vertices=[[0, 0, 0]], faces=[0, 0, 0])

    pts = np.asarray(pointcloud, dtype=np.float64)
    origin = pts.min(axis=0) - pitch
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)

    grid_shape = ijk.max(axis=0) + 3  # +2 pad so boundary is always empty
    occupied = np.zeros(grid_shape, dtype=bool)
    occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    # For each axis direction, find faces between occupied and empty voxels
    # Face normals: +x, -x, +y, -y, +z, -z
    face_templates = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # -x face
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],  # +x face
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # -y face
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # +y face
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # -z face
            [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],  # +z face
        ],
        dtype=np.float64,
    )

    verts_list = []
    faces_list = []
    vi = 0  # running vertex index

    axis_shifts = [
        (0, 1, 0, 1),
        (0, -1, 0, 1),
        (1, 1, 2, 3),
        (1, -1, 2, 3),
        (2, 1, 4, 5),
        (2, -1, 4, 5),
    ]

    for axis, direction, neg_tpl, pos_tpl in axis_shifts:
        shifted = np.roll(occupied, -direction, axis=axis)
        if direction > 0:
            boundary = occupied & ~shifted
            tpl = face_templates[pos_tpl]
        else:
            boundary = occupied & ~shifted
            tpl = face_templates[neg_tpl]

        coords = np.argwhere(boundary)
        if len(coords) == 0:
            continue

        quad_verts = coords[:, None, :] + tpl[None, :, :]
        quad_verts = quad_verts.reshape(-1, 3).astype(np.float64) * pitch + origin
        n = len(coords)
        idx = np.arange(n) * 4 + vi
        # PATCH(cap-harness): upstream is `idx, idx + 1, idx + 2 / idx,
        # idx + 2, idx + 3`, winding the triangles inward. Swap two indices
        # per triangle so face normals point out of the occupied voxels.
        tri_faces = np.column_stack(
            [
                idx,
                idx + 2,
                idx + 1,
                idx,
                idx + 3,
                idx + 2,
            ]
        ).reshape(-1, 3)

        verts_list.append(quad_verts)
        faces_list.append(tri_faces)
        vi += n * 4

    if not verts_list:
        return mesh_type(name, pose=pose, vertices=[[0, 0, 0]], faces=[0, 0, 0])

    all_verts = np.concatenate(verts_list, axis=0)
    all_faces = np.concatenate(faces_list, axis=0)

    return mesh_type(
        name, pose=pose, vertices=all_verts.tolist(), faces=all_faces.flatten().tolist()
    )


def _voxelized_scene_mesh(
    points: np.ndarray,
    *,
    voxel_size_m: float,
    mesh_type: Any | None = None,
) -> Any:
    """Build a cuRobo mesh without changing the declared scene resolution."""
    if mesh_type is None:
        from curobo._src.geom.types import Mesh

        mesh_type = Mesh
    return _mesh_from_pointcloud(
        points,
        pitch=float(voxel_size_m),
        name="public_rgbd_scene",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        mesh_type=mesh_type,
    )


def _sparse_voxelized_scene_mesh(points: np.ndarray, *, voxel_size_m: float,
                                 mesh_type: Any | None = None) -> Any:
    vertices, faces = sparse_voxel_mesh(points, float(voxel_size_m))
    if mesh_type is None:
        from curobo._src.geom.types import Mesh

        mesh_type = Mesh
    return mesh_type("public_rgbd_scene", pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     vertices=vertices.tolist(), faces=faces.flatten().tolist())


_SCENE_MAX_RANGE_M = 5.0


def _drop_distant_points(
    points: np.ndarray, *, max_range_m: float = _SCENE_MAX_RANGE_M
) -> np.ndarray:
    """Drop depth outliers far outside the workspace.

    The voxel surface extraction allocates a dense occupancy grid over the
    cloud's bounding box; a handful of degenerate depth pixels hundreds of
    meters out explode it (observed: a 646 TiB allocation). Planning scenes
    are expressed in the robot base frame, so anything beyond a room radius
    is sensor noise, not reachable geometry.
    """
    return points[np.linalg.norm(points, axis=1) <= max_range_m]


def _planner_attachment_sphere_indices(
    planner: Any, object_link_name: str, sphere_count: int
) -> np.ndarray:
    """Find attached-object sphere slots in the planner model or refuse the carve."""
    kinematics = getattr(planner, "kinematics", planner)
    params = getattr(getattr(kinematics, "config", None), "kinematics_config", None)
    get_indices = getattr(params, "get_sphere_index_from_link_name", None)
    if not callable(get_indices):
        raise HTTPException(
            status_code=500,
            detail="planner model cannot identify attached-object collision spheres",
        )
    try:
        indices = np.asarray(
            get_indices(object_link_name).detach().cpu(), dtype=np.int64
        ).reshape(-1)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"planner model has no attached-object sphere mapping: {exc}",
        ) from exc
    if len(indices) == 0:
        raise HTTPException(
            status_code=500,
            detail="planner model has no attached-object collision spheres",
        )
    if np.any(indices < 0) or np.any(indices >= sphere_count):
        raise HTTPException(
            status_code=500,
            detail="planner attached-object sphere indices are outside the FK sphere array",
        )
    return indices


def _carve_robot_points(request: BaseRequest, robot: Any, points: np.ndarray) -> np.ndarray:
    """Remove robot pixels using the embodiment's segmentation geometry."""
    segmentation = _segmentation_kinematics(request)
    source = segmentation if segmentation is not None else robot
    state = source.compute_kinematics(_state(request, source))
    spheres = np.asarray(state.robot_spheres.detach().cpu(), dtype=np.float64).reshape(-1, 4)
    if request.attached_object is not None and segmentation is None:
        object_indices = _planner_attachment_sphere_indices(
            robot,
            request.attached_object.object_link_name,
            len(spheres),
        )
        spheres = np.delete(spheres, object_indices, axis=0)
    margin = (
        _SEGMENTATION_TOLERANCE_M
        if segmentation is not None
        else max(float(request.voxel_size_m), _ROBOT_SEGMENTATION_DISTANCE_M)
    )
    filtered = _filter_robot_points(points, spheres, padding_m=margin)
    if request.attached_object is not None and request.target_scene_partition is None:
        try:
            current = _state(request, robot)
            poses = robot.compute_kinematics(current).tool_poses.to_dict()
            pose = poses[request.attached_object.object_link_name]
            position = np.asarray(pose.position.detach().cpu(), dtype=np.float64).reshape(-1, 3)[0]
            quaternion = np.asarray(
                pose.quaternion.detach().cpu(), dtype=np.float64
            ).reshape(-1, 4)[0]
            object_pose = pose7_to_matrix([*position, *quaternion], name="object FK pose")
            attachment = request.attached_object.to_geometry()
            filtered = carve_attached_object_points(
                filtered,
                object_pose,
                attachment.collision_spheres,
                padding_m=float(request.voxel_size_m),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"attached-object scene carve failed: {exc}",
            ) from exc
    return filtered


def _scene(request: BaseRequest, robot: Any | None = None) -> Any | None:
    if _collision_disabled(request.model):
        if request.target_scene_partition is not None and request.target_scene_partition.mode == "pixel_partition_v2":
            raise HTTPException(status_code=400, detail="preserved scene preview requires collision checking")
        _LOGGER.warning(
            "COLLISION DISABLED (CAP_HARNESS_CUROBO_DISABLE_COLLISION is set): the "
            "observed scene is being discarded. Any plan returned is unsafe to execute."
        )
        return None
    if request.scene_points_base64 is None:
        return None
    points = np.asarray(decode_numpy(request.scene_points_base64), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise HTTPException(status_code=400, detail="scene points must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise HTTPException(status_code=400, detail="scene points must be finite")
    preserve = request.target_scene_partition is not None and request.target_scene_partition.mode == "pixel_partition_v2"
    if preserve:
        pitch = float(request.voxel_size_m)
        if not np.isfinite(pitch) or pitch <= 0:
            raise HTTPException(status_code=400, detail="preserved scene voxel size must be positive and finite")
    else:
        points = _drop_distant_points(points)
    if len(points) == 0:
        raise HTTPException(status_code=422, detail="scene contains no points in range")
    if robot is not None and not preserve:
        try:
            before = len(points)
            points = _carve_robot_points(request, robot, points)
            _LOGGER.info("robot carve (self-collision): %d -> %d points", before, len(points))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"robot point removal failed: {exc}"
            ) from exc
        if len(points) == 0:
            raise HTTPException(status_code=422, detail="scene contains only robot points")
    try:
        from curobo.scene import Scene

        # PointCloud.get_mesh() silently uses cuRobo's 2 cm default pitch.
        # Preserve the public scene resolution so filtered 1 cm voxels do not
        # expand back into the robot and invalidate otherwise feasible goals.
        if preserve:
            # Build a bounded sparse surface without discarding distant points,
            # robot points, or geometry near a target or its support surface.
            obstacle = _sparse_voxelized_scene_mesh(points, voxel_size_m=pitch)
        else:
            obstacle = _voxelized_scene_mesh(points, voxel_size_m=request.voxel_size_m)
        return Scene(mesh=[obstacle])
    except SparseSceneLimitError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        if preserve:
            raise HTTPException(status_code=422, detail=f"preserved scene invalid: {exc}") from exc
        raise HTTPException(status_code=500, detail=f"scene conversion failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scene conversion failed: {exc}") from exc


def _purge_scene_mesh_cache(planner: Any) -> None:
    """Drop cached Warp BVHs so update_world loads the new scene geometry.

    cuRobo's mesh cache is keyed by name only (data_mesh.py
    _load_mesh_into_cache): loading a mesh whose name is already cached
    silently reuses the first geometry ever loaded under that name. Every
    request names its scene mesh "public_rgbd_scene", so without this purge
    a cached planner keeps checking collisions against the world from the
    first request it served.
    """
    cache = getattr(
        getattr(getattr(planner, "scene_collision_checker", None), "data", None),
        "meshes",
        None,
    )
    cache = getattr(cache, "wp_cache", None)
    if cache:
        cache.clear()


def _planner(request: BaseRequest) -> Any:
    robot = _robot_config(request)
    interpolation_dt = request.interpolation_dt_s
    maximum_dt = request.maximum_trajectory_dt_s
    for name, value in (
        ("interpolation_dt_s", interpolation_dt),
        ("maximum_trajectory_dt_s", maximum_dt),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise HTTPException(status_code=400, detail=f"{name} must be positive and finite")
    if interpolation_dt is not None and maximum_dt is not None and interpolation_dt > maximum_dt:
        raise HTTPException(
            status_code=400,
            detail="interpolation_dt_s must not exceed maximum_trajectory_dt_s",
        )
    key = (robot, interpolation_dt, maximum_dt)
    planner = _PLANNERS.get(key)
    try:
        if planner is None:
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

            config = MotionPlannerCfg.create(
                robot=robot,
                scene_model=None,
                collision_cache={"mesh": 1},
                # Kept on. If a specific pair ever turns out to be a false
                # positive, name it in ``self_collision_ignore`` in the robot
                # config rather than disabling the whole term.
                self_collision_check=not _collision_disabled(request.model),
            )
            if interpolation_dt is not None:
                config.trajopt_solver_config.interpolation_dt = float(interpolation_dt)
            if maximum_dt is not None:
                config.trajopt_solver_config.maximum_trajectory_dt = float(maximum_dt)
            planner = MotionPlanner(config)
            planner.warmup(enable_graph=True, num_warmup_iterations=2)
            _PLANNERS[key] = planner
        scene = _scene(request, planner)
        if scene is not None:
            _purge_scene_mesh_cache(planner)
            planner.update_world(scene)
        return planner
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"cuRobo initialization failed: {exc}") from exc


def _ik_solver(request: BaseRequest) -> Any:
    robot = _robot_config(request)
    solver = _IK_SOLVERS.get(robot)
    try:
        if solver is None:
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

            config = InverseKinematicsCfg.create(
                robot=robot,
                scene_model=None,
                collision_cache={"mesh": 1},
                num_seeds=32,
                # Kept in step with the planner above. If IK screened
                # self-collision while planning did not, IK would reject seeds
                # the planner is willing to use and the two would disagree about
                # the same pose.
                self_collision_check=not _collision_disabled(request.model),
            )
            solver = InverseKinematics(config)
            _IK_SOLVERS[robot] = solver
        scene = _scene(request, solver)
        if scene is not None:
            # Same name-keyed mesh cache as the planner above, and _IK_SOLVERS
            # keeps one solver per robot for the life of the service: without
            # this purge a cached solver screens every later request against the
            # first scene it ever loaded -- another step of the episode, or
            # another task entirely -- and rejects poses the planner accepts.
            _purge_scene_mesh_cache(solver)
            solver.update_world(scene)
        return solver
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"cuRobo IK initialization failed: {exc}"
        ) from exc


def _validate(request: BaseRequest) -> tuple[str, ...]:
    arms = tuple(request.joint_positions)
    joint_dimension = JOINT_DIMENSIONS["yam_real" if request.model == "yam_real" else "robosuite"]
    if not arms or set(request.joint_names) != set(arms):
        raise HTTPException(status_code=400, detail="joint mappings must share non-empty arms")
    if set(request.end_effector_links) != set(arms) or set(request.base_transforms) != set(arms):
        raise HTTPException(status_code=400, detail="robot context arm mappings do not match")
    if request.model == "agibot_g1":
        if len(arms) != 2:
            raise HTTPException(
                status_code=400, detail="G1 planning requires both observed arm states"
            )
        names = [name for arm in arms for name in request.joint_names[arm]]
        if len(set(names)) != len(names):
            raise HTTPException(status_code=400, detail="G1 joint names must be unique across arms")
    for arm in arms:
        joints = np.asarray(request.joint_positions[arm], dtype=np.float64)
        transform = np.asarray(request.base_transforms[arm], dtype=np.float64)
        joints_finite = bool(np.all(np.isfinite(joints)))
        if joints.shape != (joint_dimension,) or not joints_finite:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"invalid joints for {arm}",
                    "arm": arm,
                    "actual_shape": list(joints.shape),
                    "expected_shape": [joint_dimension],
                    "all_finite": joints_finite,
                },
            )
        if len(request.joint_names[arm]) != joint_dimension or transform.shape != (4, 4):
            raise HTTPException(status_code=400, detail=f"invalid robot context for {arm}")
    return arms


def _pose(target: Target) -> Any:
    if target.type != "pose" or target.position is None or target.quaternion_wxyz is None:
        raise HTTPException(status_code=400, detail="cuRobo V2 service requires a pose target")
    position = np.asarray(target.position, dtype=np.float32)
    quaternion = np.asarray(target.quaternion_wxyz, dtype=np.float32)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise HTTPException(status_code=400, detail="pose target has invalid shape")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
        raise HTTPException(status_code=400, detail="pose target must be finite")
    from curobo.types import Pose
    import torch

    return Pose(
        position=torch.as_tensor(position, device="cuda").reshape(1, 3),
        quaternion=torch.as_tensor(quaternion, device="cuda").reshape(1, 4),
    )


#: Pull encoder readings slightly inside modeled hard limits before planning.
_JOINT_LIMIT_MARGIN_RAD = 1e-3


def _clip_joint_positions(positions: np.ndarray, planner: Any) -> np.ndarray:
    """Clamp a planner-ordered joint vector inside its limits by a small margin."""
    clipped = np.asarray(positions, dtype=np.float64).reshape(-1).copy()
    kinematics = getattr(planner, "kinematics", None)
    if kinematics is None or not hasattr(kinematics, "get_joint_limits"):
        return clipped
    try:
        limits = kinematics.get_joint_limits()
        lower = np.asarray(limits.position_lower_limits.detach().cpu(), dtype=np.float64).reshape(
            -1
        )
        upper = np.asarray(limits.position_upper_limits.detach().cpu(), dtype=np.float64).reshape(
            -1
        )
    except Exception as exc:
        _LOGGER.warning("joint-limit clip skipped: %s", exc)
        return clipped
    if lower.shape != clipped.shape or upper.shape != clipped.shape:
        return clipped
    half = np.maximum((upper - lower) * 0.5 - 1e-9, 0.0)
    margin = np.minimum(_JOINT_LIMIT_MARGIN_RAD, half)
    before = clipped.copy()
    clipped = np.minimum(np.maximum(clipped, lower + margin), upper - margin)
    delta = clipped - before
    if np.any(np.abs(delta) > 0.0):
        _LOGGER.info(
            "clipped joint_positions into limits (margin=%.1e rad); max|delta|=%.3e",
            _JOINT_LIMIT_MARGIN_RAD,
            float(np.max(np.abs(delta))),
        )
    return clipped


def _model_joint_names(request: BaseRequest, arm: str) -> list[str]:
    """Map G1 SDK names only through an explicitly configured one-to-one map."""
    names = request.joint_names[arm]
    if request.model != "agibot_g1":
        return list(names)
    if request._g1_joint_map is None:
        request_names = [name for key in request.joint_names for name in request.joint_names[key]]
        path_value = os.environ.get("CAP_HARNESS_CUROBO_G1_JOINT_MAP")
        if path_value is None:
            mapping = {name: name for name in request_names}
        else:
            path = Path(path_value)
            if not path.is_absolute() or not path.is_file():
                raise HTTPException(
                    status_code=503, detail="G1 joint map must be an existing absolute path"
                )
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
                mapping = content["joint_name_map"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise HTTPException(status_code=503, detail=f"invalid G1 joint map: {exc}") from exc
            if not isinstance(mapping, dict):
                raise HTTPException(status_code=503, detail="G1 joint_name_map must be an object")
        if (
            set(mapping) != set(request_names)
            or any(
                not isinstance(v, str) or not v.strip() or v != v.strip()
                for v in mapping.values()
            )
            or len(set(mapping.values())) != len(request_names)
        ):
            raise HTTPException(status_code=400, detail="G1 joint map must uniquely cover both arm states")
        request._g1_joint_map = dict(mapping)
    return [request._g1_joint_map[name] for name in names]


def _joint_positions_in_model_order(request: BaseRequest, expected: list[str]) -> np.ndarray:
    """Match observed joint values to a model without guessing G1 name mappings."""
    ordered_arms = tuple(request.joint_positions)
    positions = np.concatenate([request.joint_positions[arm] for arm in ordered_arms])
    names = [name for arm in ordered_arms for name in _model_joint_names(request, arm)]
    if set(names) != set(expected):
        if request.model != "agibot_g1" and len(ordered_arms) == 1 and len(expected) == 7:
            names = expected
        else:
            raise HTTPException(
                status_code=400,
                detail=f"request joint names do not match cuRobo model: {names} != {expected}",
            )
    reorder = [names.index(name) for name in expected]
    return positions[reorder]


def _state(request: BaseRequest, planner: Any) -> Any:
    from curobo.types import JointState
    import torch

    expected = list(planner.joint_names)
    positions = _joint_positions_in_model_order(request, expected)
    clipped = _clip_joint_positions(positions, planner)
    if request.model == "agibot_g1" and not np.allclose(positions, clipped, rtol=0, atol=1e-7):
        raise HTTPException(status_code=400, detail="G1 observed joints require model-limit clipping")
    positions = clipped
    return JointState.from_position(
        torch.as_tensor(positions, device="cuda", dtype=torch.float32).reshape(1, -1),
        joint_names=expected,
    )


def _requested_indices(
    request: BaseRequest,
    arm: str,
    available_names: list[str],
) -> list[int]:
    requested = _model_joint_names(request, arm)
    if set(requested) <= set(available_names):
        return [available_names.index(name) for name in requested]
    canonical_panda = [f"panda_joint{index}" for index in range(1, 8)]
    if (
        request.model == "panda"
        and len(request.joint_positions) == 1
        and len(requested) == 7
        and set(canonical_panda) <= set(available_names)
    ):
        return [available_names.index(name) for name in canonical_panda]
    raise HTTPException(
        status_code=400,
        detail=f"joint names for {arm} do not match the cuRobo model",
    )


def _joint_goal(request: PlanRequest, planner: Any) -> Any:
    if request.target.type != "joints" or request.target.joint_positions is None:
        raise HTTPException(status_code=400, detail="joint target is required")
    target = np.asarray(request.target.joint_positions, dtype=np.float64)
    joint_dimension = JOINT_DIMENSIONS["yam_real" if request.model == "yam_real" else "robosuite"]
    if target.shape != (joint_dimension,) or not np.all(np.isfinite(target)):
        raise HTTPException(
            status_code=400,
            detail=f"joint target must contain {joint_dimension} finite values",
        )

    from curobo.types import JointState
    import torch

    current = _state(request, planner)
    positions = current.position.clone()
    expected = list(planner.joint_names)
    indices = torch.as_tensor(
        _requested_indices(request, request.arm, expected),
        device=positions.device,
        dtype=torch.long,
    )
    positions[:, indices] = torch.as_tensor(target, device=positions.device, dtype=positions.dtype)
    clipped = _clip_joint_positions(
        positions.detach().cpu().numpy().reshape(-1),
        planner,
    )
    return JointState.from_position(
        torch.as_tensor(clipped, device=positions.device, dtype=positions.dtype).reshape(1, -1),
        joint_names=expected,
    )


def _configuration_feasibility(
    planner: Any,
    current: Any,
    goal: Any,
) -> tuple[bool, bool] | None:
    """Check c-space endpoints with the planner's configured constraints."""
    graph_planner = getattr(planner, "graph_planner", None)
    if graph_planner is None:
        return None
    import torch

    samples = torch.cat(
        (
            current.position.reshape(1, -1),
            goal.position.reshape(1, -1),
        ),
        dim=0,
    )
    mask = graph_planner.check_samples_feasibility(samples)
    values = np.asarray(mask.detach().cpu(), dtype=bool).reshape(-1)
    if values.shape != (2,):
        return None
    return bool(values[0]), bool(values[1])


def _configuration_constraint_detail(planner: Any, state: Any) -> dict[str, object]:
    """Expose which feasibility term rejects one configuration.

    The planner reports only that a state is infeasible, which collapses joint
    bounds, self-collision and scene collision into one 422 and makes a
    real-arm rejection impossible to diagnose. This evaluates the graph
    planner's own feasibility rollout and names each constraint.

    Ported from the reference branch, which added it for the same reason.
    """
    graph_planner = getattr(planner, "graph_planner", None)
    if graph_planner is None:
        return {}
    try:
        samples = state.position.reshape(1, -1)
        graph_planner.check_samples_feasibility(samples)
        buffer = graph_planner._max_act_buffer
        buffer[:1, :, :] = samples.unsqueeze(1)
        metrics = graph_planner.feasibility_rollout.compute_metrics_from_action(buffer)
        values: dict[str, object] = {}
        costs = metrics.costs_and_constraints
        for kind, collection in (
            ("constraint", costs.constraints),
            ("hybrid", costs.hybrid_costs_constraints),
        ):
            for name, value in zip(collection.names, collection.values):
                array = np.asarray(value[0].detach().cpu(), dtype=np.float64)
                values[f"{kind}:{name}"] = {
                    "maximum": float(np.max(array)),
                    "sum": float(np.sum(array)),
                }
        total = costs.get_sum_constraint(sum_horizon=True, include_all_hybrid=False)
        if total is not None:
            values["sum_constraint"] = float(total.reshape(-1)[0].detach().cpu())
        return values
    except Exception as exc:
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def _goal_tool_frames(request: BaseRequest, planner: Any) -> list[str]:
    """Return goal frames, omitting a held object's FK-only frame."""
    ordered = list(getattr(planner, "tool_frames", ()))
    if request.attached_object is not None:
        # The attached object moves with its parent link; it is an FK/collision
        # frame, not a second stationary goal constraint.
        ordered = [
            frame
            for frame in ordered
            if frame != request.attached_object.object_link_name
        ]
    return ordered


def _ordered_goal_poses(
    request: BaseRequest,
    targets: Mapping[str, Target],
    planner: Any,
    current: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Complete partial bimanual pose goals with the inactive live tool pose."""
    ordered = _goal_tool_frames(request, planner)
    if request.model == "agibot_g1":
        tools = list(request.end_effector_links.values())
        if len(set(tools)) != len(tools) or not set(tools).issubset(ordered):
            raise HTTPException(status_code=400, detail="G1 tool frames do not match robot model")
    poses = {request.end_effector_links[arm]: _pose(target) for arm, target in targets.items()}
    missing = [name for name in ordered if name not in poses]
    if missing:
        current_poses = planner.compute_kinematics(current).tool_poses.to_dict()
        unavailable = [name for name in missing if name not in current_poses]
        if unavailable:
            raise HTTPException(
                status_code=500,
                detail=f"cuRobo kinematics omitted tool frames: {unavailable}",
            )
        poses.update({name: current_poses[name] for name in missing})
    if not ordered:
        ordered = list(poses)
    return poses, ordered


def _goal(
    request: BaseRequest,
    targets: Mapping[str, Target],
    planner: Any,
    current: Any,
) -> Any:
    from curobo.types import GoalToolPose

    poses, ordered = _ordered_goal_poses(request, targets, planner, current)
    return GoalToolPose.from_poses(poses, ordered_tool_frames=ordered, num_goalset=1)


def _tensor_summary(value: Any) -> dict[str, object] | None:
    """Return small JSON-safe diagnostics for a cuRobo result tensor."""
    if value is None:
        return None
    try:
        array = np.asarray(value.detach().cpu())
    except (AttributeError, TypeError, ValueError):
        return None
    if array.size == 0:
        return {"shape": list(array.shape), "count": 0}
    finite = array[np.isfinite(array)]
    summary: dict[str, object] = {
        "shape": list(array.shape),
        "count": int(array.size),
    }
    if finite.size:
        summary.update(
            {
                "minimum": float(np.min(finite)),
                "maximum": float(np.max(finite)),
            }
        )
    if array.dtype == np.bool_:
        summary["true_count"] = int(np.count_nonzero(array))
    return summary


def _constraint_scores(constraints: Mapping[str, object]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name in ("cspace", "self_collision", "scene_collision"):
        entry = constraints.get(f"constraint:{name}")
        value = 0.0
        if isinstance(entry, Mapping):
            raw = entry.get("maximum", entry.get("sum", 0.0))
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 0.0
        scores[name] = value
    return scores


def _dominant_constraint(constraints: Mapping[str, object]) -> str | None:
    scores = _constraint_scores(constraints)
    dominant = max(scores, key=scores.get)
    return dominant if scores[dominant] > 0.0 else None


def _sample_feasible(planner: Any, state: Any) -> bool | None:
    graph_planner = getattr(planner, "graph_planner", None)
    if graph_planner is None:
        return None
    mask = graph_planner.check_samples_feasibility(state.position.reshape(1, -1))
    values = np.asarray(mask.detach().cpu(), dtype=bool).reshape(-1)
    return bool(values[0]) if values.size else None


def _planning_failure_detail(
    result: Any,
    *,
    planner: Any | None = None,
    current: Any | None = None,
    goal_state: Any | None = None,
) -> dict[str, object]:
    """Build a 422 detail that identifies infeasible endpoints when possible."""
    if result is None:
        detail: dict[str, object] = {"message": "cuRobo returned no planning result"}
    else:
        detail = {"message": "cuRobo could not find a feasible plan"}
        for name in (
            "success",
            "feasible",
            "cspace_error",
            "position_error",
            "rotation_error",
        ):
            summary = _tensor_summary(getattr(result, name, None))
            if summary is not None:
                detail[name] = summary

    constraints_start: dict[str, object] = {}
    constraints_goal: dict[str, object] = {}
    start_feasible: bool | None = None
    goal_feasible: bool | None = None
    if planner is not None and current is not None:
        constraints_start = _configuration_constraint_detail(planner, current)
        start_feasible = _sample_feasible(planner, current)
        detail["constraints_start"] = constraints_start
        detail["start_state_feasible"] = start_feasible
    if planner is not None and goal_state is not None:
        constraints_goal = _configuration_constraint_detail(planner, goal_state)
        goal_feasible = _sample_feasible(planner, goal_state)
        detail["constraints_goal"] = constraints_goal
        detail["goal_state_feasible"] = goal_feasible

    if start_feasible is False:
        kind = _dominant_constraint(constraints_start) or "unknown"
        detail["failure_kind"] = f"start_{kind}"
        detail["message"] = f"cuRobo start state infeasible ({kind})"
    elif goal_feasible is False:
        kind = _dominant_constraint(constraints_goal) or "unknown"
        detail["failure_kind"] = f"goal_{kind}"
        detail["message"] = f"cuRobo goal state infeasible ({kind})"
    elif result is not None:
        kind = _dominant_constraint(constraints_start)
        detail["failure_kind"] = kind or "plan_failed"
        if kind is not None:
            detail["message"] = f"cuRobo could not find a feasible plan ({kind})"
    else:
        kind = _dominant_constraint(constraints_start) or _dominant_constraint(constraints_goal)
        detail["failure_kind"] = kind or "no_result"
    return detail


def _trajectory(
    result: Any,
    *,
    planner: Any | None = None,
    current: Any | None = None,
    goal_state: Any | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    if result is None or result.success is None or not bool(result.success.any()):
        raise HTTPException(
            status_code=422,
            detail=_planning_failure_detail(
                result, planner=planner, current=current, goal_state=goal_state
            ),
        )
    plan = result.get_interpolated_plan()
    positions = np.asarray(plan.position.detach().cpu(), dtype=np.float64).reshape(
        -1, plan.position.shape[-1]
    )
    names = list(plan.joint_names)
    plan_dt = getattr(plan, "dt", None)
    dt = float(plan_dt.reshape(-1)[0].item()) if plan_dt is not None else 0.025
    return positions, dt, names


def _check_g1_inactive_arm(
    request: IkRequest | PlanRequest, names: list[str], positions: np.ndarray
) -> None:
    """Ensure every uncommanded arm stays at its observed joint positions."""
    if request.model != "agibot_g1":
        return
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(names) or not np.all(np.isfinite(values)):
        raise HTTPException(status_code=500, detail="G1 planner returned malformed joint positions")
    if len(set(names)) != len(names):
        raise HTTPException(status_code=500, detail="G1 planner returned duplicate joint names")
    for arm in request.joint_positions:
        if arm == request.arm:
            continue
        model_joint_names = _model_joint_names(request, arm)
        missing = set(model_joint_names) - set(names)
        if missing:
            raise HTTPException(status_code=500, detail="G1 planner omitted inactive-arm joints")
        indices = [names.index(name) for name in model_joint_names]
        observed = np.asarray(request.joint_positions[arm], dtype=np.float64)
        if not np.allclose(values[:, indices], observed[None, :], rtol=0, atol=1e-6):
            raise HTTPException(status_code=422, detail="G1 plan would move the inactive arm")


def _solve_ik(request: IkRequest) -> dict[str, object]:
    if request.target_scene_partition is not None:
        raise HTTPException(status_code=400, detail="scene partitions are not supported for IK")
    _validate(request)
    if request.arm not in request.joint_positions:
        raise HTTPException(status_code=400, detail="unknown arm")
    if request.model == "agibot_g1" and request.target.frame != request.base_frame:
        raise HTTPException(status_code=400, detail="G1 target frame must match the robot base")
    solver = _ik_solver(request)
    current = _state(request, solver)
    # A single-arm goal on a model that declares two tool frames is refused by
    # cuRobo ("Ordered link names [...] not a subset of [...]"): every declared
    # frame must appear in the goal. The planning path already completes partial
    # goals with the inactive arm's live pose; IK has to do the same, or one-arm
    # IK is unsolvable on the bimanual station. Single-tool-frame models are
    # unaffected, which is why this only surfaces on a bimanual model.
    goal = _goal(request, {request.arm: request.target}, solver, current)
    result = solver.solve_pose(goal, current_state=current)
    if result.success is None or not bool(result.success.any()) or result.js_solution is None:
        raise HTTPException(status_code=422, detail="cuRobo IK target is unreachable")
    solution = np.asarray(result.js_solution.position.detach().cpu()).reshape(-1)
    solution_names = list(result.js_solution.joint_names)
    _check_g1_inactive_arm(request, solution_names, solution.reshape(1, -1))
    joints = solution[_requested_indices(request, request.arm, solution_names)]
    return {"joint_positions": joints.tolist()}


def _plan(request: PlanRequest) -> PlanResponse:
    _validate(request)
    if request.arm not in request.joint_positions:
        raise HTTPException(status_code=400, detail="unknown arm")
    if (
        request.model == "agibot_g1"
        and request.target.type == "pose"
        and request.target.frame != request.base_frame
    ):
        raise HTTPException(status_code=400, detail="G1 target frame must match the robot base")
    planner = _planner(request)
    current = _state(request, planner)
    goal_state: Any | None = None
    if request.target.type == "pose":
        result = planner.plan_pose(
            _goal(request, {request.arm: request.target}, planner, current),
            current,
        )
    else:
        joint_goal = _joint_goal(request, planner)
        goal_state = joint_goal
        feasibility = _configuration_feasibility(planner, current, joint_goal)
        if feasibility is not None and not all(feasibility):
            detail = _planning_failure_detail(
                None, planner=planner, current=current, goal_state=joint_goal
            )
            detail["start_state_feasible"] = feasibility[0]
            detail["goal_state_feasible"] = feasibility[1]
            raise HTTPException(status_code=422, detail=detail)
        result = planner.plan_cspace(joint_goal, current)
    positions, dt, names = _trajectory(
        result, planner=planner, current=current, goal_state=goal_state
    )
    _check_g1_inactive_arm(request, names, positions)
    indices = _requested_indices(request, request.arm, names)
    return PlanResponse(
        joint_positions_base64=encode_numpy(positions[:, indices]),
        dt_s=dt,
        attachment_diagnostics=_attachment_diagnostics(request),
        scene_partition_diagnostics=_scene_partition_diagnostics(request),
        collision_diagnostics=_collision_diagnostics(request),
    )


def _plan_synchronized(request: SynchronizedPlanRequest) -> SynchronizedPlanResponse:
    if request.target_scene_partition is not None:
        raise HTTPException(status_code=400, detail="scene partitions require single-arm planning")
    arms = _validate(request)
    if request.model == "agibot_g1":
        raise HTTPException(status_code=400, detail="G1 currently permits only single-arm planning")
    if set(request.targets) != set(arms):
        raise HTTPException(status_code=400, detail="targets must name every arm")
    planner = _planner(request)
    current = _state(request, planner)
    result = planner.plan_pose(
        _goal(request, request.targets, planner, current),
        current,
    )
    positions, dt, names = _trajectory(result, planner=planner, current=current)
    output: dict[str, str] = {}
    for arm in arms:
        requested = request.joint_names[arm]
        if not set(requested) <= set(names):
            raise HTTPException(status_code=500, detail=f"planned trajectory omitted {arm}")
        output[arm] = encode_numpy(positions[:, [names.index(name) for name in requested]])
    return SynchronizedPlanResponse(
        joint_positions_base64=output,
        dt_s=dt,
        attachment_diagnostics=_attachment_diagnostics(request),
        collision_diagnostics=_collision_diagnostics(request),
    )


async def _worker(function: Any, request: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(function, request))


async def _run_request(function: Any, request: Any) -> Any:
    try:
        return await _worker(function, request)
    except HTTPException as exc:
        _LOGGER.warning("cuRobo request rejected (%s): %s", exc.status_code, exc.detail)
        raise


@app.post(IK_PATH)
async def solve_ik(request: IkRequest) -> dict[str, object]:
    return await _run_request(_solve_ik, request)


@app.post(PLAN_PATH, response_model=PlanResponse, response_model_exclude_none=True)
async def plan(request: PlanRequest) -> PlanResponse:
    return await _run_request(_plan, request)


@app.post(
    PLAN_SYNCHRONIZED_PATH,
    response_model=SynchronizedPlanResponse,
    response_model_exclude_none=True,
)
async def plan_synchronized(request: SynchronizedPlanRequest) -> SynchronizedPlanResponse:
    return await _run_request(_plan_synchronized, request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cuRobo V2 provider")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8118)
    args = parser.parse_args()
    # uvicorn configures its own loggers and leaves the root at WARNING, which
    # silences this module. The carve counts are how you tell a scene that still
    # contains the robot from one that does not, so they default to visible.
    logging.basicConfig(
        level=os.environ.get("CAP_HARNESS_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
