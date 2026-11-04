"""Synthetic image tests for saved-image ChArUco intrinsic calibration."""

from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

from cap_harness.open_rail.offline_camera_calibration import (
    BoardSpec,
    _detect_views,
    _read_fisheye_initial_intrinsics,
    calibrate_directory,
)


def _synthetic_images(directory, model: str) -> np.ndarray:
    board_spec = BoardSpec()
    board = board_spec.create()
    rendered = board.generateImage((700, 1000), marginSize=40, borderBits=1)
    detector = cv2.aruco.CharucoDetector(board)
    source_corners, source_ids, _, _ = detector.detectBoard(rendered)
    object_points = board.getChessboardCorners()[source_ids.ravel()].astype(np.float64)
    source_h, _ = cv2.findHomography(object_points[:, :2], source_corners.reshape(-1, 2))

    width, height = 640, 480
    if model == "fisheye":
        camera = np.array([[260.0, 0, 320], [0, 265.0, 240], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([-0.05, 0.001, 0.0, 0.0], dtype=np.float64)
        distort_points = cv2.fisheye.undistortPoints
        count = 40
    else:
        camera = np.array([[350.0, 0, 320], [0, 355.0, 240], [0, 0, 1]], dtype=np.float64)
        distortion = np.array([-0.08, 0.015, -0.001, 0.0002, 0.002], dtype=np.float64)
        distort_points = cv2.undistortPoints
        count = 22

    px, py = np.meshgrid(np.arange(width), np.arange(height))
    distorted_pixels = np.stack([px, py], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    ideal_normalized = distort_points(
        distorted_pixels, camera, distortion.reshape(-1, 1)
    ).reshape(height, width, 2)
    map_x = (ideal_normalized[:, :, 0] * camera[0, 0] + camera[0, 2]).astype(np.float32)
    map_y = (ideal_normalized[:, :, 1] * camera[1, 1] + camera[1, 2]).astype(np.float32)

    rng = np.random.default_rng(991)
    for index in range(count):
        if model == "fisheye":
            tx = rng.uniform(-0.19, 0.19)
            ty = rng.uniform(-0.12, 0.12)
            depth = rng.uniform(0.25, 0.44)
        else:
            tx = rng.uniform(-0.10, 0.10)
            ty = rng.uniform(-0.07, 0.07)
            depth = rng.uniform(0.34, 0.58)
        rvec = np.array(
            [rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3)],
            dtype=np.float64,
        ).reshape(3, 1)
        tvec = np.array([tx, ty, depth], dtype=np.float64).reshape(3, 1)
        ideal_corners, _ = cv2.projectPoints(object_points, rvec, tvec, camera, None)
        target_h, _ = cv2.findHomography(object_points[:, :2], ideal_corners.reshape(-1, 2))
        ideal = cv2.warpPerspective(
            rendered,
            target_h @ np.linalg.inv(source_h),
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        image = cv2.remap(
            ideal,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        assert cv2.imwrite(str(directory / f"view-{index:03d}.png"), image)
    return camera


@pytest.mark.parametrize("model", ["pinhole", "fisheye"])
def test_saved_synthetic_views_calibrate_selected_model(tmp_path, model):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    expected_camera = _synthetic_images(image_dir, model)
    output = tmp_path / f"{model}-result"

    result = calibrate_directory(
        image_dir,
        output,
        camera_name="synthetic",
        model=model,
        min_views=8,
    )

    assert result["camera_model"] == model
    assert result["board"]["printed_scale_verified"] is False
    intrinsics = result["intrinsics"]
    assert intrinsics["distortion_model"] == (
        "opencv_fisheye_equidistant" if model == "fisheye" else "opencv_pinhole_brown_conrady"
    )
    assert len(intrinsics["distortion_coefficients"]) == (4 if model == "fisheye" else 5)
    matrix = np.asarray(intrinsics["camera_matrix"])
    np.testing.assert_allclose(matrix[0, 0], expected_camera[0, 0], rtol=0.04)
    np.testing.assert_allclose(matrix[1, 1], expected_camera[1, 1], rtol=0.04)
    assert intrinsics["recomputed_rms_reprojection_error_px"] < 1.0
    assert len(intrinsics["per_frame"]) >= 8
    assert all(np.isfinite(v["reprojection_rms_px"]) for v in intrinsics["per_frame"])

    saved_samples = json.loads((output / "samples.json").read_text())
    assert len(saved_samples["accepted"]) == result["accepted_view_count"]
    assert all((output / sample["saved_image"]).is_file() for sample in saved_samples["accepted"])


def test_one_view_is_not_accepted_as_an_intrinsic_calibration(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _synthetic_images(image_dir, "pinhole")
    with pytest.raises(ValueError, match="need at least 30"):
        calibrate_directory(
            image_dir,
            tmp_path / "result",
            camera_name="synthetic",
            model="pinhole",
            min_views=30,
        )


def test_fisheye_initial_intrinsics_are_used_only_as_a_recorded_seed(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    expected_camera = _synthetic_images(image_dir, "fisheye")
    seed_file = tmp_path / "existing-fisheye.json"
    seed = {
        "image_size_px": [640, 480],
        "intrinsic": {
            "fu": float(expected_camera[0, 0]),
            "fv": float(expected_camera[1, 1]),
            "pu": float(expected_camera[0, 2]),
            "pv": float(expected_camera[1, 2]),
            "distortion_model": "equidistant",
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
        },
    }
    seed_file.write_text(json.dumps(seed))
    output = tmp_path / "fisheye-seeded-result"

    result = calibrate_directory(
        image_dir,
        output,
        camera_name="synthetic",
        model="fisheye",
        initial_intrinsics_path=seed_file,
    )

    initial = result["intrinsics"]["initial_guess"]
    assert initial["source_file"] == str(seed_file.resolve())
    assert initial["source_sha256"] == hashlib.sha256(seed_file.read_bytes()).hexdigest()
    assert initial["role"] == "optimizer_initial_guess_only"
    np.testing.assert_allclose(initial["camera_matrix"], expected_camera)
    assert initial["distortion_coefficients"] == [0.0, 0.0, 0.0, 0.0]
    saved = json.loads((output / "calibration.json").read_text())
    assert saved["intrinsics"]["initial_guess"] == initial


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"camera_matrix": [[1, 0], [0, 1]], "distortion_coefficients": [0, 0, 0, 0]}, "3x3"),
        ({"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "distortion_coefficients": [0, 0, 0]}, "four finite"),
        ({"image_size_px": [320, 240], "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "distortion_coefficients": [0, 0, 0, 0]}, "does not match"),
    ],
)
def test_fisheye_initial_intrinsics_reject_invalid_shape_or_image_size(
    tmp_path, payload, message
):
    path = tmp_path / "initial.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        _read_fisheye_initial_intrinsics(path, [640, 480])


def test_fisheye_initial_intrinsics_reject_nonfinite_values(tmp_path):
    path = tmp_path / "initial.json"
    path.write_text(
        json.dumps(
            {
                "camera_matrix": [[float("nan"), 0, 320], [0, 260, 240], [0, 0, 1]],
                "distortion_coefficients": [0, 0, 0, 0],
            }
        )
    )

    with pytest.raises(ValueError, match="finite"):
        _read_fisheye_initial_intrinsics(path, [640, 480])


def test_upscaled_detection_maps_corners_back_without_changing_image_size(tmp_path):
    image_dir = tmp_path / "large"
    image_dir.mkdir()
    _synthetic_images(image_dir, "pinhole")
    small_dir = tmp_path / "small"
    small_dir.mkdir()
    for path in image_dir.glob("*.png"):
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        small = cv2.resize(original, (320, 240), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(small_dir / path.name), small)

    views, _, image_size = _detect_views(
        small_dir, BoardSpec(), min_corners=6, detect_upscale=2.0
    )

    assert image_size == [320, 240]
    assert views
    assert all(view["detection_scale"] == 2.0 for view in views)
    for view in views:
        assert np.all(view["pixel_points"][:, 0] >= 0)
        assert np.all(view["pixel_points"][:, 0] < 320)
        assert np.all(view["pixel_points"][:, 1] >= 0)
        assert np.all(view["pixel_points"][:, 1] < 240)

