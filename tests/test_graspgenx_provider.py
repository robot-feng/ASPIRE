"""CPU-only contract tests for the GraspGenX provider client."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from cap_harness.contracts import PointCloud
from cap_harness.errors import ErrorCode
from cap_harness.geometry import matrix_to_pose
from cap_harness.providers.graspgenx import (
    GraspGenXProvider,
    gripper_base_pose_to_tcp,
    service,
    tcp_pregrasp_pose,
)
from cap_harness.providers.graspgenx.wire import GENERATE_GRASPS_PATH
from cap_harness.providers.wire import decode_numpy, encode_numpy

SWEEP_VOLUME = {
    "extents_open": [0.09, 0.02, 0.05],
    "offset_open": [0.0, 0.0, 0.10],
    "extents_mid": [0.04, 0.02, 0.05],
    "offset_mid": [0.0, 0.0, 0.10],
    "gripper_type": 0,
    "fingertip_depth": 0.125,
}


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(self, method: str, url: str, *, json=None, timeout=None):
        self.calls.append((method, url, json))
        return _Response(self.payload)


def _cloud(frame: str = "head_optical") -> PointCloud:
    return PointCloud(
        points=np.array([[0.1, 0.2, 0.3], [0.11, 0.2, 0.3]], dtype=np.float32),
        frame=frame,
    )


def _response(grasps: np.ndarray, scores: np.ndarray, branches: list[str]) -> dict[str, Any]:
    return {
        "grasps_base64": encode_numpy(grasps),
        "scores_base64": encode_numpy(scores),
        "branch_tags": branches,
        "timing": {"infer_ms": 12.5},
    }


def test_provider_maps_ranked_gripper_base_poses_in_cloud_frame() -> None:
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    transforms[1, 0, 3] = 0.2
    session = _Session(_response(transforms, np.array([0.4, 0.9]), ["diff", "obb"]))
    provider = GraspGenXProvider(SWEEP_VOLUME, session=session)  # type: ignore[arg-type]

    result = provider.generate_grasps(_cloud(), max_candidates=1)

    assert result.ok
    assert len(result.grasps) == 1
    grasp = result.grasps[0]
    assert grasp.frame == "head_optical"
    assert np.allclose(grasp.pose.position, [0.2, 0.0, 0.0])
    assert np.isclose(grasp.score, 0.9)
    assert grasp.width_m is None
    assert np.isclose(grasp.metadata["sweep_box_open_extent_x_m"], 0.09)
    assert grasp.metadata["opening_width_semantics"] == "not_predicted"
    assert grasp.metadata["grasp_frame"] == "gripper_base"
    assert grasp.metadata["tcp_transform_applied"] is False
    assert grasp.metadata["branch"] == "obb"
    assert result.diagnostics["scene_collision_checked"] is False
    assert result.diagnostics["pose_semantics"] == (
        "graspgenx_gripper_base_in_input_point_cloud_frame"
    )
    assert result.diagnostics["timing"] == {"infer_ms": 12.5}
    assert session.calls[0][1].endswith(GENERATE_GRASPS_PATH)
    assert session.calls[0][2]["planner"] == "graspmoe"


def test_provider_composes_measured_gripper_base_to_tcp_transform() -> None:
    grasp = np.eye(4, dtype=np.float32)
    grasp[:3, 3] = [0.2, 0.3, 0.4]
    base_to_tcp = np.eye(4)
    base_to_tcp[:3, 3] = [0.0, 0.0, 0.12]
    session = _Session(_response(grasp[None], np.array([0.8]), ["diff"]))
    provider = GraspGenXProvider(
        SWEEP_VOLUME,
        session=session,  # type: ignore[arg-type]
        gripper_base_to_tcp=base_to_tcp,
    )

    result = provider.generate_grasps(_cloud("robot_base"))

    assert result.ok
    assert np.allclose(result.grasps[0].pose.position, [0.2, 0.3, 0.52])
    assert result.grasps[0].metadata["grasp_frame"] == "tcp"
    assert result.grasps[0].metadata["graspgenx_pose_frame"] == "gripper_base"
    assert result.grasps[0].metadata["tcp_transform_applied"] is True
    assert result.diagnostics["pose_semantics"] == "tcp_in_input_point_cloud_frame"
    assert result.diagnostics["tcp_transform_applied"] is True


def test_provider_rejects_invalid_gripper_base_to_tcp_transform() -> None:
    invalid = np.eye(4)
    invalid[0, 0] = 2.0

    try:
        GraspGenXProvider(SWEEP_VOLUME, gripper_base_to_tcp=invalid)
    except ValueError as exc:
        assert "orthonormal" in str(exc)
    else:
        raise AssertionError("invalid gripper-base-to-TCP rotation was accepted")


def test_frame_helpers_back_out_along_graspgenx_approach_axis() -> None:
    # GraspGenX local +Z points into the object. Rotate it onto world +X, and
    # rotate the TCP frame separately to ensure retreat follows the gripper
    # base approach axis rather than assuming the TCP has the same orientation.
    base_rotation_y = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    tcp_rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    grasp_base = np.eye(4)
    grasp_base[:3, :3] = base_rotation_y
    grasp_base[:3, 3] = [0.2, 0.3, 0.4]
    base_to_tcp = np.eye(4)
    base_to_tcp[:3, :3] = tcp_rotation_x
    base_to_tcp[:3, 3] = [0.0, 0.0, 0.12]

    grasp_tcp = gripper_base_pose_to_tcp(
        matrix_to_pose(grasp_base, frame="robot_base"),
        base_to_tcp,
    )
    pregrasp_tcp = tcp_pregrasp_pose(
        grasp_tcp,
        base_to_tcp,
        distance_m=0.1,
    )

    assert np.allclose(grasp_tcp.position, [0.32, 0.3, 0.4])
    assert np.allclose(pregrasp_tcp.position, [0.22, 0.3, 0.4])
    assert np.allclose(
        pregrasp_tcp.as_matrix()[:3, :3],
        grasp_tcp.as_matrix()[:3, :3],
    )
    assert pregrasp_tcp.frame == grasp_tcp.frame == "robot_base"


def test_provider_preserves_base_approach_direction_with_rotated_tcp() -> None:
    grasp_base = np.eye(4)
    grasp_base[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    grasp_base[:3, 3] = [0.2, 0.3, 0.4]
    base_to_tcp = np.eye(4)
    base_to_tcp[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    base_to_tcp[:3, 3] = [0.0, 0.0, 0.12]
    session = _Session(_response(grasp_base[None], np.array([0.8]), ["diff"]))
    provider = GraspGenXProvider(
        SWEEP_VOLUME, session=session, gripper_base_to_tcp=base_to_tcp  # type: ignore[arg-type]
    )

    result = provider.generate_grasps(_cloud("robot_base"))

    assert result.ok
    candidate = result.grasps[0]
    direction = np.asarray(candidate.metadata["approach_direction_in_frame"])
    assert np.allclose(direction, [1.0, 0.0, 0.0])
    assert candidate.metadata["approach_direction_frame"] == candidate.frame
    assert candidate.metadata["approach_direction_semantics"] == (
        "graspgenx_base_positive_z_toward_object"
    )
    # The TCP's +Z is not the GraspGen-X approach direction here.
    assert not np.allclose(candidate.pose.as_matrix()[:3, 2], direction)
    expected = tcp_pregrasp_pose(candidate.pose, base_to_tcp, distance_m=0.1)
    assert np.allclose(candidate.pose.position - 0.1 * direction, expected.position)


def test_provider_rejects_mismatched_scene_frame_without_inference() -> None:
    session = _Session(_response(np.eye(4)[None], np.array([0.5]), ["diff"]))
    provider = GraspGenXProvider(SWEEP_VOLUME, session=session)  # type: ignore[arg-type]

    result = provider.generate_grasps(_cloud(), scene_point_cloud=_cloud("robot_base"))

    assert not result.ok
    assert result.error.code is ErrorCode.INVALID_REQUEST
    assert session.calls == []


def test_provider_reports_no_candidates() -> None:
    session = _Session(_response(np.empty((0, 4, 4)), np.empty((0,)), []))
    provider = GraspGenXProvider(SWEEP_VOLUME, session=session)  # type: ignore[arg-type]

    result = provider.generate_grasps(_cloud())

    assert not result.ok
    assert result.error.code is ErrorCode.NO_GRASP


def test_provider_rejects_malformed_sweep_volume() -> None:
    malformed = {**SWEEP_VOLUME, "extents_open": [0.0, 0.02, 0.05]}

    try:
        GraspGenXProvider(malformed)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("non-positive physical gripper extent was accepted")


def test_service_forwards_params_only_request_to_official_engine() -> None:
    class _Backend:
        def _dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload["action"] == "infer_object"
            assert payload["planner"] == "graspmoe"
            assert payload["sweep_volume_params"]["fingertip_depth"] == 0.125
            return {
                "grasps": np.eye(4, dtype=np.float32)[None],
                "confidences": np.array([0.8], dtype=np.float32),
                "branch_tags": ["diff"],
                "timing": {"infer_ms": 4.0},
            }

    service.app.state.graspgenx_backend = _Backend()
    request = service.GenerateGraspsRequest(
        point_cloud_base64=encode_numpy(_cloud().points),
        sweep_volume_params=SWEEP_VOLUME,
    )
    response = asyncio.run(service.generate_grasps(request))

    assert decode_numpy(response.grasps_base64).shape == (1, 4, 4)
    assert np.allclose(decode_numpy(response.scores_base64), [0.8])
    assert response.branch_tags == ["diff"]
    assert response.timing == {"infer_ms": 4.0}
