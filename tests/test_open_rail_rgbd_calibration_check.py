from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cap_harness.open_rail.offline_camera_calibration import BoardSpec
from cap_harness.open_rail.rgbd_calibration_check import _scale_summary, evaluate


def _fixture(tmp_path: Path):
    sample = tmp_path / "sample_0001"
    sample.mkdir()
    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    calibration = tmp_path.parent / (tmp_path.name + "-calibration.json")
    calibration.write_text(
        json.dumps(
            {
                "camera_model": "pinhole",
                "image_size_px": [640, 800],
                "intrinsics": {"camera_matrix": K.tolist(), "distortion_coefficients": [0] * 5},
            }
        )
    )
    board = BoardSpec().create()
    pattern = board.generateImage((560, 800), marginSize=0, borderBits=1)
    # A nominal 168 x 240 mm board, parallel to an undistorted camera at 600 mm.
    corners = np.array([[180, 200], [460, 200], [460, 600], [180, 600]], np.float32)
    H = cv2.getPerspectiveTransform(
        np.array([[0, 0], [559, 0], [559, 799], [0, 799]], np.float32), corners
    )
    image = cv2.warpPerspective(pattern, H, (640, 800), borderValue=200)
    cv2.imwrite(str(sample / "head_rgb.png"), image)
    np.save(sample / "head_depth_u16.npy", np.full((800, 640), 600, dtype=np.uint16))
    (sample / "snapshot.json").write_text(
        json.dumps(
            {
                "rgb_timestamp_ns": 1000000000,
                "depth_timestamp_ns": 1010000000,
                "is_synchronized": False,
            }
        )
    )
    return sample, calibration


def test_multiview_diagnostic_preserves_hypotheses_and_raw_time(tmp_path):
    sample, calibration = _fixture(tmp_path)
    report = evaluate(
        tmp_path,
        calibration,
        square_mm=24.0,
        candidate_scale_m=0.001,
        invalid_depth_values=(0, 65535),
    )
    assert len(report["frames"]) == 1
    frame = report["frames"][0]
    assert frame["corners"] >= 40
    assert frame["valid_corner_depth_count"] >= 40
    assert frame["depth_minus_rgb_ns"] == 10000000
    assert frame["source_reports_synchronized"] is False
    assert frame["median_raw_depth"] == 600
    assert frame["median_board_axial_z_m_nominal"] == pytest.approx(0.6, abs=0.004)
    assert frame["median_inferred_scale_m_per_raw_unit"] == pytest.approx(0.001, abs=0.00001)
    assert report["depth_scale_verified"] is False
    assert report["rgb_depth_registration_verified"] is False
    assert report["board_scale_physically_verified"] is False
    assert len(frame["depth_sha256"]) == 64


def test_board_size_rescales_inference_and_invalid_depth_is_not_silently_used(tmp_path):
    sample, calibration = _fixture(tmp_path)
    regular = evaluate(
        tmp_path,
        calibration,
        square_mm=24.0,
        candidate_scale_m=0.001,
        invalid_depth_values=(0, 65535),
    )["frames"][0]
    smaller = evaluate(
        tmp_path,
        calibration,
        square_mm=21.0,
        candidate_scale_m=0.001,
        invalid_depth_values=(0, 65535),
    )["frames"][0]
    assert smaller["median_inferred_scale_m_per_raw_unit"] / regular[
        "median_inferred_scale_m_per_raw_unit"
    ] == pytest.approx(21 / 24, abs=1e-5)
    np.save(sample / "head_depth_u16.npy", np.zeros((800, 640), dtype=np.uint16))
    invalid = evaluate(
        tmp_path,
        calibration,
        square_mm=24.0,
        candidate_scale_m=0.001,
        invalid_depth_values=(0, 65535),
    )["frames"][0]
    assert "fewer than 8 corners" in invalid["failure"]
    assert "median_inferred_scale_m_per_raw_unit" not in invalid


def test_missing_depth_or_bad_grid_is_a_per_frame_failure(tmp_path):
    sample, calibration = _fixture(tmp_path)
    np.save(sample / "head_depth_u16.npy", np.zeros((20, 20), dtype=np.uint16))
    report = evaluate(
        tmp_path,
        calibration,
        square_mm=24.0,
        candidate_scale_m=0.001,
        invalid_depth_values=(0, 65535),
    )
    assert "grid does not match" in report["frames"][0]["failure"]
    assert report["scale_summary"] is None


def test_multidistance_fit_recovers_known_scale_and_bias_without_claiming_verification():
    rows = [
        {
            "median_raw_depth": raw,
            "median_board_axial_z_m_nominal": raw * 0.001 + 0.01,
            "median_inferred_scale_m_per_raw_unit": (raw * 0.001 + 0.01) / raw,
        }
        for raw in (500, 700, 900)
    ]
    summary = _scale_summary(rows)
    assert summary["linear_fit"]["metres_per_raw_unit"] == pytest.approx(0.001)
    assert summary["linear_fit"]["intercept_m"] == pytest.approx(0.01)
    assert "assumptions" in summary["linear_fit"]["warning"]
    assert _scale_summary(rows[:2])["linear_fit"] is None


def test_paper_annotations_are_explicit_and_only_an_assumption(tmp_path):
    sample, calibration = _fixture(tmp_path)
    # Deliberately use the board boundary as a nominal 168 x 240 mm rectangle.
    annotations = {
        "sample_0001": {
            "corners_px": [[180, 200], [460, 200], [460, 600], [180, 600]],
            "edge_lengths_mm": [168, 240, 168, 240],
        }
    }
    report = evaluate(
        tmp_path,
        calibration,
        square_mm=24.0,
        candidate_scale_m=0.001,
        invalid_depth_values=(0, 65535),
        annotations=annotations,
    )
    paper = report["frames"][0]["paper_size_check"]
    assert paper["implied_square_length_mm_median"] == pytest.approx(24.0, abs=0.15)
    assert "no physical length was verified" in paper["warning"]
    assert len(report["frames"][0]["local_edge_comparison"]) == 4
