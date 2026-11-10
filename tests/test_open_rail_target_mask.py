from __future__ import annotations

import base64
from dataclasses import replace

import numpy as np
import pytest

from cap_harness.api import CapApi
from cap_harness.contracts import CameraObservation, GraspSet, Observation, Pose, RobotState
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import mask_to_point_cloud
from cap_harness.open_rail.verification import HoldVerifier
from cap_harness.open_rail.hold_replay import HoldReplaySample, replay_hold
from cap_harness.providers.depth_refined import AmbiguousTargetDepth, DepthRefinedSegmentationProvider, refine_target_mask
from cap_harness.providers.wevisone import WeVisOneProvider


class _Response:
    status_code = 200
    text = ""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def json(self):
        return {
            "results": [
                {
                    "label": "apple",
                    "box_xyxy": [1, 1, 9, 7],
                    "shape": list(self.mask.shape),
                    "probabilities_base64": base64.b64encode(
                        self.mask.astype("<f4").tobytes()
                    ).decode("ascii"),
                }
            ],
            "total_instances": 1,
            "truncated": False,
        }


class _Session:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask

    def request(self, *args, **kwargs):
        return _Response(self.mask)


def _camera() -> CameraObservation:
    depth = np.full((8, 10), 1.4)
    depth[1:7, 1:9] = 0.6
    depth[2, 3] = 0.0
    depth[3, 3] = 0.9
    return CameraObservation(
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        depth_m=depth,
        intrinsics=np.array([[100, 0, 4], [0, 100, 3], [0, 0, 1]]),
        frame="head_rgb_optical",
        camera_pose=Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), "base"),
        timestamp_s=12.5,
    )


def _segment(camera: CameraObservation, mask: np.ndarray):
    return WeVisOneProvider(session=_Session(mask)).segment_text(camera, "apple")


def _refine(camera: CameraObservation, result, **overrides):
    options = {
        "max_depth_deviation_m": 0.03,
        "max_neighbor_depth_step_m": 0.01,
        "min_valid_pixels": 20,
        "min_valid_fraction": 0.8,
        "min_depth_inlier_fraction": 0.8,
        "min_component_fraction": 0.8,
    }
    return refine_target_mask(camera, result, **(options | overrides))


def test_wevisone_refinement_retains_scene_and_auditable_snapshot() -> None:
    camera = _camera()
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    result = _segment(camera, mask)

    refined, details = _refine(camera, result)

    assert result.ok
    assert result.diagnostics["camera_timestamp_s"] == 12.5
    assert result.diagnostics["snapshot_id"] == "head_rgb_optical@12.500000000"
    assert details["rgb_sha256"] == result.diagnostics["rgb_sha256"]
    assert details["depth_sha256"] == result.diagnostics["depth_sha256"]
    assert details["mask_pixels"] == 48
    assert details["valid_depth_pixels"] == 47
    assert details["depth_inlier_pixels"] == 46
    assert details["selected_component_pixels"] == 46
    assert not refined.mask[2, 3] and not refined.mask[3, 3]
    assert np.array_equal(result.segmentations[0].mask, mask)
    assert len(mask_to_point_cloud(camera, refined).points) == 46
    scene = replace(refined, mask=np.ones(mask.shape, dtype=bool))
    assert len(mask_to_point_cloud(camera, scene).points) == 79


def test_stale_or_different_rgb_snapshot_is_rejected() -> None:
    camera = _camera()
    result = _segment(camera, np.ones((8, 10), dtype=bool))
    with pytest.raises(ValueError, match="does not belong"):
        _refine(replace(camera, timestamp_s=12.6), result)
    different_rgb = camera.rgb.copy()
    different_rgb[0, 0, 0] = 1
    with pytest.raises(ValueError, match="does not belong"):
        _refine(replace(camera, rgb=different_rgb), result)
    different_depth = camera.depth_m.copy()
    different_depth[0, 0] += 0.01
    with pytest.raises(ValueError, match="does not belong"):
        _refine(replace(camera, depth_m=different_depth), result)
    different_intrinsics = camera.intrinsics.copy()
    different_intrinsics[0, 0] += 1
    with pytest.raises(ValueError, match="does not belong"):
        _refine(replace(camera, intrinsics=different_intrinsics), result)
    with pytest.raises(ValueError, match="timestamp is required"):
        _refine(replace(camera, timestamp_s=None), result)


def test_rejects_multiple_similarly_sized_depth_components() -> None:
    camera = _camera()
    depth = np.full((8, 10), 0.6)
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:4] = True
    mask[1:7, 6:9] = True
    camera = replace(camera, depth_m=depth)
    with pytest.raises(ValueError, match="ambiguous disconnected"):
        _refine(camera, _segment(camera, mask), min_valid_pixels=10)


def test_touching_objects_separated_by_depth_discontinuity_are_ambiguous() -> None:
    camera = _camera()
    depth = np.full((8, 10), 0.6)
    depth[:, 5:] = 0.62
    camera = replace(camera, depth_m=depth)
    result = _segment(camera, np.ones((8, 10), dtype=bool))
    with pytest.raises(ValueError, match="ambiguous disconnected"):
        _refine(camera, result, min_valid_pixels=10)


def _background_options():
    return {
        "background_ring_width_px": 1,
        "min_background_valid_pixels": 10,
        "min_background_valid_fraction": 0.5,
        "min_foreground_gap_m": 0.05,
    }


def test_background_depth_check_uses_original_mask_and_records_evidence() -> None:
    camera = _camera()
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    segmentation = _segment(camera, mask)
    refined, details = _refine(camera, segmentation, **_background_options())
    check = details["background_depth_check"]
    assert check["status"] == "passed"
    assert check["ring_pixels"] == 32
    assert check["valid_ring_pixels"] == 32
    assert check["target_median_depth_m"] == pytest.approx(0.6)
    assert check["background_median_depth_m"] == pytest.approx(1.4)
    assert check["foreground_gap_m"] == pytest.approx(0.8)
    assert not refined.mask[2, 3] and not refined.mask[3, 3]
    # Excluded target-mask holes are not incorrectly reused as background.
    assert check["valid_ring_pixels"] == 32


def test_foreground_background_ambiguity_is_not_reported_as_no_target() -> None:
    camera = replace(_camera(), depth_m=np.full((8, 10), 0.6))
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    options = {
        "max_depth_deviation_m": 0.05, "max_neighbor_depth_step_m": 0.01,
        "min_valid_pixels": 20, "min_valid_fraction": 0.8,
        "min_depth_inlier_fraction": 0.8, "min_component_fraction": 0.8,
        **_background_options(),
    }
    provider = DepthRefinedSegmentationProvider(WeVisOneProvider(session=_Session(mask)), **options)
    result = provider.segment_text(camera, "apple")
    assert not result.ok
    assert result.error.code is ErrorCode.POINT_CLOUD_FAILED
    assert result.error.details["depth_status"] == "uncertain"
    assert "uncertain" in result.error.message
    assert result.diagnostics["background_depth_check"]["foreground_gap_m"] == pytest.approx(0)
    np.testing.assert_array_equal(result.diagnostics["original_masks"][0], mask)


def test_background_check_rejects_missing_depth_and_clipped_mask_as_uncertain() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    camera = _camera()
    depth = np.array(camera.depth_m)
    depth[~mask] = 0
    camera = replace(camera, depth_m=depth)
    with pytest.raises(AmbiguousTargetDepth, match="insufficient valid surrounding depth"):
        _refine(camera, _segment(camera, mask), **_background_options())

    clipped = mask.copy()
    clipped[0, 3] = True
    camera = replace(_camera(), depth_m=np.full((8, 10), 0.6))
    with pytest.raises(AmbiguousTargetDepth, match="image boundary"):
        _refine(camera, _segment(camera, clipped), **_background_options())

    # A wider requested ring can be clipped even when the mask is not.
    with pytest.raises(AmbiguousTargetDepth, match="image boundary"):
        _refine(_camera(), _segment(_camera(), mask),
                **{**_background_options(), "background_ring_width_px": 2})


def test_background_depth_check_requires_complete_explicit_configuration() -> None:
    camera = _camera()
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    result = _segment(camera, mask)
    with pytest.raises(ValueError, match="all four explicit settings"):
        _refine(camera, result, background_ring_width_px=1)
    with pytest.raises(ValueError, match="foreground gap positive"):
        _refine(camera, result, **{**_background_options(), "min_foreground_gap_m": np.bool_(True)})


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_depth_deviation_m", 0),
        ("max_neighbor_depth_step_m", float("inf")),
        ("min_valid_pixels", True),
        ("min_valid_fraction", float("nan")),
        ("min_depth_inlier_fraction", 1.1),
        ("min_component_fraction", 0),
    ],
)
def test_invalid_quality_thresholds_are_rejected(field, value) -> None:
    camera = _camera()
    result = _segment(camera, np.ones((8, 10), dtype=bool))
    with pytest.raises(ValueError):
        _refine(camera, result, **{field: value})


def test_provider_filter_flows_through_cap_localization_and_hold_verifier(tmp_path) -> None:
    pytest.importorskip("imageio")
    from cap_harness.artifacts import RunRecorder, TracedProvider

    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    provider = DepthRefinedSegmentationProvider(
        WeVisOneProvider(session=_Session(mask)),
        max_depth_deviation_m=0.05,
        max_neighbor_depth_step_m=0.01,
        min_valid_pixels=20,
        min_valid_fraction=0.8,
        min_depth_inlier_fraction=0.8,
        min_component_fraction=0.8,
    )
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="test",
        suite="test",
        task_id=0,
        task_name="apple",
        seed=1,
        max_steps=10,
        capture_videos=False,
    )

    class CaptureGraspProvider:
        def __init__(self):
            self.clouds = None

        def generate_grasps(self, point_cloud, *, scene_point_cloud=None, **_kwargs):
            self.clouds = point_cloud, scene_point_cloud
            return GraspSet(ok=False, error=ApiError(ErrorCode.NO_GRASP, "fake provider"))

    grasp_provider = CaptureGraspProvider()
    api = CapApi(
        object(),
        home_joint_positions=np.zeros(7),
        segmentation_provider=TracedProvider(provider, recorder, "wevisone"),
        grasp_providers={"graspgenx": grasp_provider},
        recorder=recorder,
    )

    def sample(t: float, lift: float):
        yy, xx = np.indices((8, 10))
        depth = 0.60 + lift + 0.0005 * ((xx - 4.5) ** 2 + (yy - 3.5) ** 2)
        camera = replace(_camera(), depth_m=depth, timestamp_s=t)
        state = RobotState(
            joint_positions={"right": np.zeros(7)},
            joint_velocities={"right": np.zeros(7)},
            end_effector_poses={
                "right": Pose(
                    np.array([0.0, 0.0, 0.1 + lift]), np.array([1.0, 0.0, 0.0, 0.0]), "base"
                )
            },
            gripper_positions={"right": 0.5},
            base_frame="base",
            timestamp_s=t,
            embodiment="open_rail_7dof",
        )
        observation = Observation(cameras={"head": camera}, robot_state=state, timestamp_s=t)
        localization = api.localize_object_from_observation(
            "apple",
            observation,
            camera_name="head",
            require_single_instance=True,
            min_valid_depth_points=20,
            min_valid_depth_fraction=0.8,
        )
        assert localization.ok, localization.error
        assert localization.diagnostics["segmentation_source"] == "provider"
        assert localization.diagnostics["instance_count"] == 1
        return observation, localization

    baseline = sample(1.0, 0.0)
    grasps = api.generate_grasps_for_localization(
        baseline[1], baseline[0], camera_name="head", max_camera_state_skew_s=0.01
    )
    assert not grasps.ok and grasps.error.code is ErrorCode.NO_GRASP
    assert grasp_provider.clouds is not None
    assert len(grasp_provider.clouds[0].points) == 48
    assert len(grasp_provider.clouds[1].points) == 80
    verifier = HoldVerifier(
        *baseline,
        camera_name="head",
        arm="right",
        up_in_base=(0, 0, 1),
        minimum_lift_m=0.05,
        max_relative_error_m=0.02,
        max_sample_gap_s=1.1,
        max_state_skew_s=0.01,
    )
    captured = [sample(t, 0.1) for t in (2.0, 3.0, 4.0, 5.0)]
    results = [verifier.observe(*item) for item in captured]

    assert results[-1].status == "verified"
    replayed = replay_hold(
        *baseline,
        [HoldReplaySample(index, *item) for index, item in enumerate(captured)],
        expected_samples=4,
        capture_clock_verified=True,
        camera_name="head",
        arm="right",
        up_in_base=(0, 0, 1),
        minimum_lift_m=0.05,
        max_relative_error_m=0.02,
        max_sample_gap_s=1.1,
        max_state_skew_s=0.01,
    )
    assert replayed.status == "verified_history"
    assert replayed.current_hold_verified is False
    import hashlib
    import json

    provider_outputs = list(recorder.root.glob("trace/calls/*wevisone-segment-text/output.json"))
    assert len(provider_outputs) == 5
    payload = json.loads(provider_outputs[0].read_text())
    diagnostics = payload["diagnostics"]
    assert diagnostics["provider"] == "wevisone"
    assert diagnostics["depth_filter"]["mask_pixels"] == 48
    for saved in (diagnostics["original_masks"][0], diagnostics["filtered_mask"]):
        path = recorder.root / saved["$array"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["sha256"]
        np.testing.assert_array_equal(np.load(path), mask.astype(np.uint8))
    recorder.finalize(
        program_ok=True,
        task_success=True,
        termination_reason="program_completed",
        program_result=None,
    )


def test_provider_keeps_original_mask_when_depth_filter_fails(tmp_path) -> None:
    pytest.importorskip("imageio")
    import json

    from cap_harness.artifacts import RunRecorder, TracedProvider

    camera = _camera()
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    thresholds = dict(
        max_depth_deviation_m=0.05,
        max_neighbor_depth_step_m=0.01,
        min_valid_pixels=20,
        min_valid_fraction=0.8,
        min_depth_inlier_fraction=0.8,
        min_component_fraction=0.8,
    )
    thresholds["min_valid_pixels"] = 100
    provider = DepthRefinedSegmentationProvider(
        WeVisOneProvider(session=_Session(mask)), **thresholds
    )
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="test",
        suite="test",
        task_id=0,
        task_name="apple",
        seed=1,
        max_steps=10,
        capture_videos=False,
    )
    result = TracedProvider(provider, recorder, "wevisone").segment_text(camera, "apple")
    assert not result.ok and "insufficient valid depth" in result.error.message
    np.testing.assert_array_equal(result.diagnostics["original_masks"][0], mask)
    output = next(recorder.root.glob("trace/calls/*wevisone-segment-text/output.json"))
    evidence = json.loads(output.read_text())["diagnostics"]["original_masks"][0]
    np.testing.assert_array_equal(np.load(recorder.root / evidence["$array"]), mask)
    recorder.finalize(
        program_ok=False,
        task_success=False,
        termination_reason="program_error",
        program_result=None,
    )


def test_wevisone_overlay_uses_exact_provider_input_and_archives_rgbd(tmp_path) -> None:
    pytest.importorskip("imageio")
    import hashlib
    import json
    from types import SimpleNamespace

    from cap_harness.artifacts import RunRecorder

    camera = _camera()
    rgb = np.full(camera.rgb.shape, (17, 31, 47), dtype=np.uint8)
    camera = replace(camera, rgb=rgb)
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:7, 1:9] = True
    result = _segment(camera, mask)
    recorder = RunRecorder(
        output_root=tmp_path,
        benchmark="test",
        suite="test",
        task_id=0,
        task_name="apple",
        seed=1,
        max_steps=1,
        capture_videos=False,
    )
    # A later cached observation exists: never pair this mask with that frame.
    recorder.last_observation = SimpleNamespace(  # type: ignore[assignment]
        cameras={"head": replace(camera, rgb=np.zeros_like(camera.rgb), timestamp_s=13.0)}
    )
    with recorder.span(
        "wevisone.segment_text", category="provider", inputs={"args": (camera, "apple")}
    ) as span:
        span.output(result)

    metadata_path = next((recorder.root / "media/overlays").glob("wevisone-*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["snapshot_id"] == "head_rgb_optical@12.500000000"
    assert metadata["score_semantics"] == "neutral; provider emits no confidence score"
    for saved, expected in (
        (metadata["camera"]["rgb"], camera.rgb),
        (metadata["camera"]["depth_m"], camera.depth_m),
        (metadata["candidates"][0]["mask"], mask.astype(np.uint8)),
    ):
        path = recorder.root / saved["$array"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == saved["sha256"]
        np.testing.assert_array_equal(np.load(path), expected)

    # Reusing the same timestamps with altered RGB or depth cannot produce an overlay.
    changed_rgb = camera.rgb.copy()
    changed_rgb[0, 0, 0] += 1
    changed_depth = camera.depth_m.copy()
    changed_depth[0, 0] += 0.1
    for changed in (replace(camera, rgb=changed_rgb), replace(camera, depth_m=changed_depth)):
        with recorder.span(
            "wevisone.segment_text", category="provider", inputs={"args": (changed, "apple")}
        ) as span:
            span.output(result)
    assert len(list((recorder.root / "media/overlays").glob("wevisone-*.json"))) == 1
    recorder.finalize(
        program_ok=True,
        task_success=None,
        termination_reason="program_completed",
        program_result=None,
    )
