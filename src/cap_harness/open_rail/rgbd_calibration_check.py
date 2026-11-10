"""Compare saved ChArUco RGB-D views without touching a camera or changing calibration.

All metric quantities depend on the supplied board dimensions. Pixelwise depth
comparisons assume RGB/depth registration and axial-Z depth; the report explicitly
records these assumptions and is not a calibration certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cap_harness.open_rail.offline_camera_calibration import BoardSpec, _detect_views


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _intrinsics(path: Path):
    data = json.loads(path.read_text())
    if data.get("camera_model") != "pinhole":
        raise ValueError("calibration must declare a pinhole camera model")
    width, height = data["image_size_px"]
    K = np.asarray(data["intrinsics"]["camera_matrix"], dtype=float)
    D = np.asarray(data["intrinsics"]["distortion_coefficients"], dtype=float).reshape(-1)
    if (
        K.shape != (3, 3)
        or D.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(K).all()
        or not np.isfinite(D).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or width <= 0
        or height <= 0
        or not np.allclose(K[2], [0, 0, 1])
    ):
        raise ValueError("invalid pinhole calibration")
    return K, D, [width, height]


def _timestamps(snapshot):
    rgb, depth = snapshot.get("rgb_timestamp_ns"), snapshot.get("depth_timestamp_ns")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (rgb, depth)):
        return None
    return depth - rgb


def _scale_summary(rows):
    usable = [r for r in rows if "median_inferred_scale_m_per_raw_unit" in r]
    if not usable:
        return None
    scales = np.array([r["median_inferred_scale_m_per_raw_unit"] for r in usable])
    raw = np.array([r["median_raw_depth"] for r in usable])
    depth = np.array([r["median_board_axial_z_m_nominal"] for r in usable])
    result = {
        "views": len(usable),
        "median": float(np.median(scales)),
        "range": [float(np.min(scales)), float(np.max(scales))],
        "nominal_board_depth_range_m": [float(np.min(depth)), float(np.max(depth))],
    }
    if len(usable) < 3 or np.ptp(depth) < 0.1 or np.ptp(raw) < 1:
        result["linear_fit"] = None
        result["linear_fit_reason"] = "need at least 3 views spanning 0.1 m of nominal board depth"
        return result
    slope, intercept = np.linalg.lstsq(
        np.column_stack((raw, np.ones(len(raw)))), depth, rcond=None
    )[0]
    residual = slope * raw + intercept - depth
    result["linear_fit"] = {
        "metres_per_raw_unit": float(slope),
        "intercept_m": float(intercept),
        "view_median_residual_rms_m": float(np.sqrt(np.mean(residual**2))),
        "warning": "Descriptive fit only; depends on board size, pixel registration and depth semantics assumptions.",
    }
    return result


def _paper_test(image_points, object_points, annotated, K, D, square_mm):
    import cv2

    corners = np.asarray(annotated["corners_px"], dtype=float)
    edges_mm = np.asarray(annotated["edge_lengths_mm"], dtype=float)
    if (
        corners.shape != (4, 2)
        or edges_mm.shape != (4,)
        or not np.isfinite(corners).all()
        or not np.isfinite(edges_mm).all()
        or np.any(edges_mm <= 0)
        or not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2))
    ):
        raise ValueError(
            "paper corners must be an ordered convex quadrilateral with four positive edge lengths"
        )
    und_board = cv2.undistortPoints(image_points.reshape(-1, 1, 2), K, D, P=K)
    homography, inliers = cv2.findHomography(und_board, object_points[:, :2], cv2.RANSAC, 0.001)
    if homography is None or np.count_nonzero(inliers) < 8:
        raise ValueError("paper measurement homography is not supported by enough corners")
    und_paper = cv2.undistortPoints(corners.reshape(-1, 1, 2), K, D, P=K)
    board_xy = cv2.perspectiveTransform(und_paper, homography).reshape(-1, 2)
    lengths_m = np.linalg.norm(np.roll(board_xy, -1, axis=0) - board_xy, axis=1)
    if np.any(lengths_m <= 1e-9):
        raise ValueError("paper edge projection is degenerate")
    estimates = square_mm * (edges_mm / 1000 / lengths_m)
    return {
        "paper_corners_px": corners.tolist(),
        "assumed_paper_edge_lengths_mm": edges_mm.tolist(),
        "implied_square_length_mm_by_edge": estimates.tolist(),
        "implied_square_length_mm_median": float(np.median(estimates)),
        "warning": "Assumes marked edges are full, flat paper of the stated physical dimensions; no physical length was verified.",
    }


def _edge_test(gray, depth, annotated, invalid):
    corners = np.asarray(annotated["corners_px"], dtype=float)
    h, w = depth.shape
    edge_reports = []
    for edge, (a, b) in enumerate(zip(corners, np.roll(corners, -1, axis=0), strict=True)):
        direction = b - a
        length = np.linalg.norm(direction)
        if length < 30:
            edge_reports.append({"edge": edge, "samples": 0, "reason": "short edge"})
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        offsets = []
        for fraction in np.linspace(0.1, 0.9, 21):
            center = a + fraction * direction
            rgb_grad, depth_grad, shifts = [], [], []
            for shift in range(-10, 11):
                pair = [np.rint(center + (shift + d) * normal).astype(int) for d in (-3, 3)]
                if any(x < 0 or x >= w or y < 0 or y >= h for x, y in pair):
                    continue
                (x1, y1), (x2, y2) = pair
                z1, z2 = int(depth[y1, x1]), int(depth[y2, x2])
                if z1 <= 0 or z2 <= 0 or z1 in invalid or z2 in invalid:
                    continue
                shifts.append(shift)
                rgb_grad.append(abs(float(gray[y2, x2]) - float(gray[y1, x1])))
                depth_grad.append(abs(z2 - z1))
            if not shifts:
                continue
            i, j = int(np.argmax(rgb_grad)), int(np.argmax(depth_grad))
            if rgb_grad[i] >= 20 and depth_grad[j] >= 40:
                offsets.append(shifts[j] - shifts[i])
        row = {"edge": edge, "samples": len(offsets), "depth_minus_rgb_offsets_px": offsets}
        if offsets:
            row["median_offset_px"] = float(np.median(offsets))
            row["range_px"] = [int(min(offsets)), int(max(offsets))]
        else:
            row["reason"] = (
                "no matching RGB/depth discontinuities above fixed exploratory thresholds"
            )
        edge_reports.append(row)
    return edge_reports


def evaluate(
    directory: Path,
    calibration: Path,
    *,
    square_mm: float,
    candidate_scale_m: float,
    invalid_depth_values: tuple[int, ...],
    annotations: dict | None = None,
):
    import cv2

    if (
        not np.isfinite(square_mm)
        or square_mm <= 0
        or not np.isfinite(candidate_scale_m)
        or candidate_scale_m <= 0
    ):
        raise ValueError("square length and candidate depth scale must be positive finite values")
    if not invalid_depth_values or any(
        isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 65535
        for x in invalid_depth_values
    ):
        raise ValueError("explicit invalid depth values must be uint16 integers")
    K, D, image_size = _intrinsics(calibration)
    board = BoardSpec(square_length_m=square_mm / 1000, marker_length_m=0.75 * square_mm / 1000)
    views, rejected, actual_size = _detect_views(directory, board, min_corners=12, detect_upscale=2)
    if actual_size != image_size:
        raise ValueError("image resolution does not match supplied intrinsic calibration")
    by_frame = []
    for view in views:
        sample = view["path"].parent
        dep_path, meta_path = sample / "head_depth_u16.npy", sample / "snapshot.json"
        row = {
            "sample": sample.name,
            "source_directory": str(sample.resolve()),
            "rgb_file": view["path"].name,
            "rgb_sha256": view["sha256"],
            "corners": len(view["ids"]),
        }
        try:
            meta = json.loads(meta_path.read_text())
            depth = np.load(dep_path, allow_pickle=False)
            if depth.shape == (image_size[1], image_size[0], 1):
                depth = depth[..., 0]
            if depth.dtype != np.uint16 or depth.shape != (image_size[1], image_size[0]):
                raise ValueError(
                    "depth dtype or grid does not match RGB; alignment cannot be inferred"
                )
            row["depth_sha256"] = _hash(dep_path)
            row["snapshot_sha256"] = _hash(meta_path)
            row["depth_file"] = dep_path.name
            row["snapshot_file"] = meta_path.name
            row["rgb_timestamp_ns"] = meta.get("rgb_timestamp_ns")
            row["depth_timestamp_ns"] = meta.get("depth_timestamp_ns")
            row["depth_minus_rgb_ns"] = _timestamps(meta)
            row["source_reports_synchronized"] = meta.get("is_synchronized")
            object_points = view["object_points"].astype(np.float64)
            pixels = view["pixel_points"].astype(np.float64)
            ok, rv, tv = cv2.solvePnP(object_points, pixels, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise ValueError("board pose estimation failed")
            projected = cv2.projectPoints(object_points, rv, tv, K, D)[0].reshape(-1, 2)
            row["board_reprojection_rms_px"] = float(
                np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1)))
            )
            n = cv2.Rodrigues(rv)[0][:, 2]
            xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
            rays = np.column_stack((xy, np.ones(len(xy))))
            denom = rays @ n
            depths_m = float(n @ tv.reshape(-1)) / denom
            raw, predicted = [], []
            for (px, py), distance in zip(pixels, depths_m, strict=True):
                x, y = int(round(px)), int(round(py))
                if (
                    not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0])
                    or not np.isfinite(distance)
                    or distance <= 0
                ):
                    continue
                patch = depth[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3].reshape(-1)
                # Nonpositive raw depths cannot describe a point in front of
                # the camera, irrespective of the device's sentinel list.
                good = patch[(patch > 0) & ~np.isin(patch, invalid_depth_values)]
                if len(good) >= 5:
                    raw.append(float(np.median(good)))
                    predicted.append(float(distance))
            if len(raw) < 8:
                raise ValueError("fewer than 8 corners have valid local depth")
            raw, predicted = np.array(raw), np.array(predicted)
            row.update(
                {
                    "valid_corner_depth_count": len(raw),
                    "median_raw_depth": float(np.median(raw)),
                    "median_board_axial_z_m_nominal": float(np.median(predicted)),
                    "median_inferred_scale_m_per_raw_unit": float(np.median(predicted / raw)),
                    "candidate_scale_m_per_raw_unit": candidate_scale_m,
                    "candidate_depth_minus_board_z_m_median": float(
                        np.median(raw * candidate_scale_m - predicted)
                    ),
                    "candidate_depth_minus_board_z_m_abs_p95": float(
                        np.percentile(np.abs(raw * candidate_scale_m - predicted), 95)
                    ),
                    "pixels_assumed_rgb_aligned": True,
                    "depth_assumed_axial_z": True,
                }
            )
            if annotations and sample.name in annotations:
                try:
                    ann = annotations[sample.name]
                    paper_corners = np.asarray(ann["corners_px"], dtype=float)
                    if (
                        paper_corners.shape != (4, 2)
                        or not np.isfinite(paper_corners).all()
                        or np.any(paper_corners < 0)
                        or np.any(paper_corners[:, 0] >= image_size[0])
                        or np.any(paper_corners[:, 1] >= image_size[1])
                    ):
                        raise ValueError("paper corners must be visible within the source image")
                    row["paper_size_check"] = _paper_test(
                        pixels, object_points, ann, K, D, square_mm
                    )
                    row["local_edge_comparison"] = _edge_test(
                        cv2.cvtColor(view["image"], cv2.COLOR_BGR2GRAY),
                        depth,
                        ann,
                        set(invalid_depth_values),
                    )
                except (KeyError, TypeError, ValueError, cv2.error) as exc:
                    row["paper_annotation_failure"] = str(exc)
        except (KeyError, OSError, TypeError, ValueError, cv2.error) as exc:
            row["failure"] = str(exc)
        by_frame.append(row)
    return {
        "schema": "aspire.open_rail.rgbd_board_diagnostic.v1",
        "calibration_path": str(calibration),
        "calibration_sha256": _hash(calibration),
        "board_square_mm_input": square_mm,
        "board_scale_physically_verified": False,
        "candidate_depth_scale_m_per_unit": candidate_scale_m,
        "depth_scale_verified": False,
        "invalid_depth_values_assumed": list(invalid_depth_values),
        "rgb_depth_registration_verified": False,
        "notes": [
            "Offline diagnostic only. No result is written to robot configuration.",
            "Board metric pose scales with printed square size; a print setting or A4 inference is not a physical measurement.",
            "RGB/depth pixels are compared assuming registration and depth values are assumed axial Z; neither is proved by the comparison.",
            "Per-frame timestamps describe the source; no synchronization limit or shared clock is assumed.",
            "Adjacent frames may depict the same stationary board; view count and fit residuals do not establish independent observations or uncertainty.",
            "Edge checks use operator-supplied paper corners, a search range of 10 px, RGB gradient 20 and raw depth gradient 40; selected matching discontinuities do not prove full-frame registration.",
        ],
        "detector_rejections": rejected,
        "frames": by_frame,
        "scale_summary": _scale_summary(by_frame),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--square-mm", type=float, required=True)
    parser.add_argument("--candidate-scale-m", type=float, required=True)
    parser.add_argument("--invalid-depth-values", type=int, nargs="+", required=True)
    parser.add_argument("--paper-annotations", type=Path)
    args = parser.parse_args(argv)
    annotations = (
        None if args.paper_annotations is None else json.loads(args.paper_annotations.read_text())
    )
    report = evaluate(
        args.samples,
        args.intrinsics,
        square_mm=args.square_mm,
        candidate_scale_m=args.candidate_scale_m,
        invalid_depth_values=tuple(args.invalid_depth_values),
        annotations=annotations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(
        f"Wrote {len(report['frames'])} detected views to {args.output}; this does not verify calibration."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
