# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Offline ChArUco intrinsic calibration from saved images only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class BoardSpec:
    """Nominal physical dimensions for the supplied 7x10 A4 ChArUco PDF."""

    squares_x: int = 7
    squares_y: int = 10
    square_length_m: float = 0.024
    marker_length_m: float = 0.018
    dictionary: str = "DICT_4X4_50"

    def create(self):
        import cv2

        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("board needs at least 3 squares per side")
        if not 0 < self.marker_length_m < self.square_length_m:
            raise ValueError("board lengths must satisfy 0 < marker < square")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary: {self.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.dictionary)
        )
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            dictionary,
        )


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"image directory does not exist: {directory}")
    return sorted(
        (p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).casefold(),
    )


def _detect_views(
    image_dir: Path,
    board_spec: BoardSpec,
    min_corners: int,
    detect_upscale: float,
):
    import cv2

    board = board_spec.create()
    detector = cv2.aruco.CharucoDetector(board)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    views: list[dict] = []
    rejected: list[dict] = []
    image_size = None

    for path in _image_paths(image_dir):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": str(path), "reason": "could not decode image"})
            continue
        height, width = image.shape[:2]
        size = [width, height]
        if image_size is None:
            image_size = size
        if size != image_size:
            rejected.append(
                {"image": str(path), "reason": f"resolution {size} differs from {image_size}"}
            )
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, marker_ids = detector.detectBoard(gray)
        detection_scale = 1.0
        marker_count = 0 if marker_ids is None else len(marker_ids)
        original_count = 0 if ids is None else len(ids)
        original_marker_count = marker_count
        # Small/soft boards often gain corners from interpolation. Keep the
        # original image for calibration and map detections back to its pixels.
        if detect_upscale > 1 and original_count < 32:
            enlarged = cv2.resize(
                gray,
                None,
                fx=detect_upscale,
                fy=detect_upscale,
                interpolation=cv2.INTER_CUBIC,
            )
            up_corners, up_ids, _, up_marker_ids = detector.detectBoard(enlarged)
            if up_ids is not None and (ids is None or len(up_ids) > len(ids)):
                corners = np.asarray(up_corners, dtype=np.float64) / detect_upscale
                ids = up_ids
                marker_count = 0 if up_marker_ids is None else len(up_marker_ids)
                detection_scale = detect_upscale
        if ids is None or corners is None or len(ids) < min_corners:
            rejected.append(
                {
                    "image": str(path),
                    "reason": "too few ChArUco corners",
                    "corners": 0 if ids is None else int(len(ids)),
                    "markers": int(marker_count),
                }
            )
            continue
        ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
        if np.any(ids_flat < 0) or np.any(ids_flat >= len(board_points)):
            rejected.append({"image": str(path), "reason": "invalid ChArUco corner ids"})
            continue
        pixel_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        object_points = board_points[ids_flat]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        views.append(
            {
                "path": path,
                "image": image,
                "ids": ids_flat,
                "pixel_points": pixel_points,
                "object_points": object_points,
                "sha256": digest,
                "detection_scale": detection_scale,
                "markers": int(marker_count),
                "original_charuco_corners": int(original_count),
                "original_markers": int(original_marker_count),
            }
        )
    if image_size is None:
        raise ValueError(f"no readable images found under {image_dir}")
    return views, rejected, image_size


def _read_fisheye_initial_intrinsics(path: Path, image_size: list[int]):
    """Read K/D from a calibration JSON for use as an optimizer seed only."""
    raw_bytes = path.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid initial-intrinsics JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"initial-intrinsics JSON must contain an object: {path}")

    declared_size = document.get("image_size_px", document.get("image_size"))
    if declared_size is not None and list(declared_size) != list(image_size):
        raise ValueError(
            f"initial intrinsics image_size {declared_size} does not match {image_size}"
        )

    intrinsics = document.get("intrinsic", document.get("intrinsics", document))
    if not isinstance(intrinsics, dict):
        raise ValueError(f"initial-intrinsics JSON must contain an object: {path}")
    model_name = document.get("camera_model", intrinsics.get("distortion_model"))
    if model_name is not None and not any(
        token in str(model_name).lower() for token in ("fisheye", "equidistant")
    ):
        raise ValueError(f"initial intrinsics are not marked as fisheye/equidistant: {model_name}")

    try:
        if "camera_matrix" in intrinsics:
            camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(
                intrinsics["distortion_coefficients"], dtype=np.float64
            ).reshape(-1)
        else:
            camera_matrix = np.array(
                [
                    [intrinsics["fu"], 0.0, intrinsics["pu"]],
                    [0.0, intrinsics["fv"], intrinsics["pv"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            distortion = np.array(
                [intrinsics[f"k{index}"] for index in range(1, 5)],
                dtype=np.float64,
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"initial intrinsics must provide a 3x3 K and four D values: {path}") from exc

    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise ValueError("initial camera_matrix must be finite and have shape 3x3")
    if distortion.shape != (4,) or not np.isfinite(distortion).all():
        raise ValueError("initial fisheye distortion must contain four finite values")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("initial focal lengths must be positive")
    if (
        abs(camera_matrix[0, 1]) > 1e-9
        or abs(camera_matrix[1, 0]) > 1e-9
        or not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1e-9)
    ):
        raise ValueError("initial camera_matrix must use zero skew and standard homogeneous row")

    metadata = {
        "source_file": str(path.resolve()),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "role": "optimizer_initial_guess_only",
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
    }
    return camera_matrix, distortion.reshape(4, 1), metadata


def _solve(
    views: list[dict],
    image_size: list[int],
    model: str,
    fisheye_initial: tuple[np.ndarray, np.ndarray, dict] | None = None,
):
    import cv2

    width, height = image_size
    point_dtype = np.float32 if model == "pinhole" else np.float64
    object_points = [
        v["object_points"].astype(point_dtype).reshape(-1, 1, 3) for v in views
    ]
    image_points = [
        v["pixel_points"].astype(point_dtype).reshape(-1, 1, 2) for v in views
    ]
    if model == "pinhole":
        rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            (width, height),
            None,
            None,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-8,
            ),
        )
        project = lambda obj, rv, tv: cv2.projectPoints(
            obj, rv, tv, camera_matrix, distortion
        )[0]
        distortion_model = "opencv_pinhole_brown_conrady"
    elif model == "fisheye":
        if fisheye_initial is None:
            camera_matrix = np.array(
                [[max(width, height), 0, width / 2], [0, max(width, height), height / 2], [0, 0, 1]],
                dtype=np.float64,
            )
            distortion = np.zeros((4, 1), dtype=np.float64)
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        else:
            camera_matrix, distortion, _ = fisheye_initial
            camera_matrix = camera_matrix.copy()
            distortion = distortion.copy()
            flags = (
                cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                | cv2.fisheye.CALIB_FIX_SKEW
            )
        rms, camera_matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            (width, height),
            camera_matrix,
            distortion,
            flags=flags,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                150,
                1e-8,
            ),
        )
        project = lambda obj, rv, tv: cv2.fisheye.projectPoints(
            obj.reshape(1, -1, 3), rv, tv, camera_matrix, distortion
        )[0]
        distortion_model = "opencv_fisheye_equidistant"
    else:
        raise ValueError("model must be pinhole or fisheye")

    per_frame = []
    sum_sq = 0.0
    total_points = 0
    for view, rvec, tvec in zip(views, rvecs, tvecs, strict=True):
        projected = np.asarray(
            project(view["object_points"].reshape(-1, 1, 3), rvec, tvec)
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - view["pixel_points"], axis=1)
        sum_sq += float(np.sum(errors**2))
        total_points += len(errors)
        view["rvec"] = np.asarray(rvec, dtype=float).reshape(-1).tolist()
        view["tvec_m_nominal"] = np.asarray(tvec, dtype=float).reshape(-1).tolist()
        per_frame.append(
            {
                "image": view["path"].name,
                "sha256": view["sha256"],
                "charuco_corners": int(len(view["ids"])),
                "detected_markers": view["markers"],
                "original_charuco_corners": view["original_charuco_corners"],
                "original_markers": view["original_markers"],
                "detection_scale": view["detection_scale"],
                "reprojection_rms_px": float(np.sqrt(np.mean(errors**2))),
                "reprojection_max_px": float(np.max(errors)),
                "board_rvec_rad": view["rvec"],
                "board_tvec_m_nominal": view["tvec_m_nominal"],
            }
        )
    result = {
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(distortion, dtype=float).reshape(-1).tolist(),
        "distortion_model": distortion_model,
        "opencv_rms_reprojection_error_px": float(rms),
        "recomputed_rms_reprojection_error_px": float(np.sqrt(sum_sq / total_points)),
        "per_frame": per_frame,
    }
    if fisheye_initial is not None:
        result["initial_guess"] = fisheye_initial[2]
    return result


def calibrate_directory(
    image_dir: Path,
    output_dir: Path,
    *,
    camera_name: str,
    model: str,
    board: BoardSpec = BoardSpec(),
    min_views: int = 8,
    min_corners: int = 8,
    detect_upscale: float = 2.0,
    initial_intrinsics_path: Path | None = None,
) -> dict:
    """Calibrate intrinsics from saved images and write a reviewable result bundle."""
    if model not in {"pinhole", "fisheye"}:
        raise ValueError("model must be pinhole or fisheye")
    if min_views < 4 or min_corners < 6:
        raise ValueError("min_views must be >= 4 and min_corners >= 6")
    if not np.isfinite(detect_upscale) or not 1 <= detect_upscale <= 3:
        raise ValueError("detect_upscale must be between 1 and 3")
    if not camera_name.strip():
        raise ValueError("camera_name must be non-empty")
    views, rejected, image_size = _detect_views(
        Path(image_dir), board, min_corners, detect_upscale
    )
    if len(views) < min_views:
        raise ValueError(
            f"only {len(views)} usable views; need at least {min_views}; rejected={len(rejected)}"
        )
    fisheye_initial = None
    if initial_intrinsics_path is not None:
        if model != "fisheye":
            raise ValueError("--initial-intrinsics is currently supported only for fisheye")
        fisheye_initial = _read_fisheye_initial_intrinsics(
            Path(initial_intrinsics_path), image_size
        )
    result = _solve(views, image_size, model, fisheye_initial)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    samples_dir = output / "samples"
    samples_dir.mkdir()

    accepted_samples = []
    for index, view in enumerate(views):
        saved_name = f"sample-{index:03d}{view['path'].suffix.lower()}"
        shutil.copyfile(view["path"], samples_dir / saved_name)
        accepted_samples.append(
            {
                "image": str(view["path"]),
                "saved_image": f"samples/{saved_name}",
                "sha256": view["sha256"],
                "charuco_ids": view["ids"].tolist(),
                "pixel_corners_xy": view["pixel_points"].tolist(),
                "detection_scale": view["detection_scale"],
                "detected_markers": view["markers"],
                "original_charuco_corners": view["original_charuco_corners"],
                "original_markers": view["original_markers"],
            }
        )

    all_corners = np.concatenate([v["pixel_points"] for v in views], axis=0)
    width, height = image_size
    coverage = [
        float((all_corners[:, 0].max() - all_corners[:, 0].min()) / width),
        float((all_corners[:, 1].max() - all_corners[:, 1].min()) / height),
    ]
    quality_warnings = []
    if len(views) < 15:
        quality_warnings.append(
            f"only {len(views)} usable views; 15-25 varied views are recommended"
        )
    if any(len(v["ids"]) < 12 for v in views):
        quality_warnings.append(
            "some views have fewer than 12 corners; move the board closer or improve focus"
        )
    if coverage[0] < 0.5 or coverage[1] < 0.5:
        quality_warnings.append(
            "detected corners cover less than half of an image dimension; vary board position"
        )

    calibration = {
        "schema": "aspire.camera_intrinsics.v1",
        "camera_name": camera_name,
        "camera_model": model,
        "image_size_px": image_size,
        "board": {
            **asdict(board),
            "source_pdf": "charuco_A4_7x10_24mm.pdf",
            "printed_scale_verified": False,
        },
        "intrinsics": result,
        "accepted_view_count": len(views),
        "rejected_image_count": len(rejected),
        "corner_coverage_fraction_xy": coverage,
        "quality_warnings": quality_warnings,
        "notes": [
            "Intrinsic calibration only; no camera-to-robot/base extrinsic is estimated.",
            "Board dimensions are nominal PDF dimensions; printed scale was not measured.",
            "Do not use per-frame board translations as metric until printed square "
            "size is measured.",
            "Select pinhole for the head RGB view and fisheye for a fisheye view explicitly.",
        ],
    }
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (output / "samples.json").write_text(
        json.dumps(
            {"accepted": accepted_samples, "rejected": rejected}, indent=2
        )
        + "\n"
    )
    return calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate", help="Calibrate saved camera images")
    calibrate.add_argument("--images", type=Path, required=True, help="Folder of saved images")
    calibrate.add_argument("--output-dir", type=Path, required=True, help="New results folder")
    calibrate.add_argument("--camera-name", required=True, help="Recorded camera role/name")
    calibrate.add_argument(
        "--model", choices=("pinhole", "fisheye"), required=True,
        help="pinhole for head RGB; fisheye for an actual fisheye stream",
    )
    calibrate.add_argument("--min-views", type=int, default=8)
    calibrate.add_argument("--min-corners", type=int, default=8)
    calibrate.add_argument(
        "--detect-upscale",
        type=float,
        default=2.0,
        help="try a larger detector image for views with <32 corners; 1 disables",
    )
    calibrate.add_argument("--square-mm", type=float, default=24.0)
    calibrate.add_argument("--marker-mm", type=float, default=18.0)
    calibrate.add_argument(
        "--initial-intrinsics",
        type=Path,
        help="optional fisheye K/D JSON used only to initialize OpenCV's optimizer",
    )
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        board = BoardSpec(
            square_length_m=args.square_mm / 1000.0,
            marker_length_m=args.marker_mm / 1000.0,
        )
        result = calibrate_directory(
            args.images,
            args.output_dir,
            camera_name=args.camera_name,
            model=args.model,
            board=board,
            min_views=args.min_views,
            min_corners=args.min_corners,
            detect_upscale=args.detect_upscale,
            initial_intrinsics_path=args.initial_intrinsics,
        )
        print(json.dumps(result["intrinsics"], indent=2))
        print(f"Saved calibration and samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
