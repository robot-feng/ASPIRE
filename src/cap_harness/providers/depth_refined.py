"""Filter one segmented target using its synchronized metric depth image."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_dilation

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.protocols import SegmentationProvider


class AmbiguousTargetDepth(ValueError):
    """The mask exists, but its depth cannot distinguish a target from background."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def refine_target_mask(
    camera: CameraObservation,
    segmentation_set: SegmentationSet,
    *,
    max_depth_deviation_m: float,
    max_neighbor_depth_step_m: float,
    min_valid_pixels: int,
    min_valid_fraction: float,
    min_depth_inlier_fraction: float,
    min_component_fraction: float,
    background_ring_width_px: int | None = None,
    min_background_valid_pixels: int | None = None,
    min_background_valid_fraction: float | None = None,
    min_foreground_gap_m: float | None = None,
) -> tuple[Segmentation, dict[str, Any]]:
    """Return a target mask for this exact RGB-D snapshot and quality counts.

    Depth is filtered around the median of valid mask pixels, then the largest
    8-connected component with bounded adjacent depth changes is retained. All
    thresholds are explicit: they must be chosen for the calibrated sensor,
    object and task. This rejects
    ambiguous/disconnected target results; it does not prove that the mask is
    correct, remove background at the same depth, or validate calibration.
    ``camera`` and the original segmentation set are left unchanged, so the
    complete RGB-D observation remains available for scene collision geometry.
    """
    if not isinstance(camera, CameraObservation) or not isinstance(
        segmentation_set, SegmentationSet
    ):
        raise ValueError("camera and segmentation_set must use ASPIRE contracts")
    if camera.timestamp_s is None:
        raise ValueError("camera timestamp is required to bind the target mask")
    if not segmentation_set.ok or len(segmentation_set.segmentations) != 1:
        raise ValueError("exactly one successful target segmentation is required")
    source = segmentation_set.diagnostics
    snapshot_id = f"{camera.frame}@{camera.timestamp_s:.9f}"
    rgb_sha256 = hashlib.sha256(camera.rgb.tobytes()).hexdigest()
    depth_sha256 = hashlib.sha256(camera.depth_m.tobytes()).hexdigest()
    intrinsics_sha256 = hashlib.sha256(camera.intrinsics.tobytes()).hexdigest()
    camera_pose_sha256 = hashlib.sha256(camera.camera_pose.as_matrix().tobytes()).hexdigest()
    if (
        source.get("snapshot_id") != snapshot_id
        or source.get("camera_frame") != camera.frame
        or source.get("camera_timestamp_s") != camera.timestamp_s
        or source.get("rgb_sha256") != rgb_sha256
        or source.get("depth_sha256") != depth_sha256
        or source.get("intrinsics_sha256") != intrinsics_sha256
        or source.get("camera_pose_sha256") != camera_pose_sha256
        or source.get("camera_parent_frame") != camera.camera_pose.frame
    ):
        raise ValueError("segmentation does not belong to this RGB-D snapshot")
    total = source.get("total_instances")
    if source.get("truncated") is not False or type(total) is not int or total != 1:
        raise ValueError("target instance set is truncated or ambiguous")
    segmentation = segmentation_set.segmentations[0]
    if segmentation.frame != camera.frame or segmentation.mask.shape != camera.depth_m.shape:
        raise ValueError("segmentation frame or shape does not match camera depth")

    if isinstance(max_depth_deviation_m, bool):
        raise ValueError("max_depth_deviation_m must be finite and positive")
    try:
        depth_limit = float(max_depth_deviation_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_depth_deviation_m must be finite and positive") from exc
    if not np.isfinite(depth_limit) or depth_limit <= 0:
        raise ValueError("max_depth_deviation_m must be finite and positive")
    if isinstance(max_neighbor_depth_step_m, bool):
        raise ValueError("max_neighbor_depth_step_m must be finite and positive")
    try:
        neighbor_limit = float(max_neighbor_depth_step_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_neighbor_depth_step_m must be finite and positive") from exc
    if not np.isfinite(neighbor_limit) or neighbor_limit <= 0:
        raise ValueError("max_neighbor_depth_step_m must be finite and positive")
    if (
        isinstance(min_valid_pixels, bool)
        or not isinstance(min_valid_pixels, int)
        or min_valid_pixels < 1
    ):
        raise ValueError("min_valid_pixels must be a positive integer")
    fractions = {}
    for name, value in (
        ("min_valid_fraction", min_valid_fraction),
        ("min_depth_inlier_fraction", min_depth_inlier_fraction),
        ("min_component_fraction", min_component_fraction),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be in (0, 1]")
        try:
            fraction = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be in (0, 1]") from exc
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
        fractions[name] = fraction

    mask_count = int(np.count_nonzero(segmentation.mask))
    valid = segmentation.mask & np.isfinite(camera.depth_m) & (camera.depth_m > 0)
    valid_count = int(np.count_nonzero(valid))
    if (
        mask_count == 0
        or valid_count < min_valid_pixels
        or valid_count / mask_count < fractions["min_valid_fraction"]
    ):
        raise ValueError(
            f"target mask has insufficient valid depth: {valid_count} of {mask_count} pixels"
        )
    median_depth = float(np.median(camera.depth_m[valid]))
    inliers = valid & (np.abs(camera.depth_m - median_depth) <= depth_limit)
    inlier_count = int(np.count_nonzero(inliers))
    if (
        inlier_count < min_valid_pixels
        or inlier_count / valid_count < fractions["min_depth_inlier_fraction"]
    ):
        raise ValueError(
            "target depth is inconsistent with the configured tolerance: "
            f"{inlier_count} of {valid_count} valid pixels are inliers"
        )
    ys, xs = np.nonzero(inliers)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    depth_roi = camera.depth_m[y0:y1, x0:x1]
    indices = np.full(depth_roi.shape, -1, dtype=np.int32)
    indices[ys - y0, xs - x0] = np.arange(inlier_count, dtype=np.int32)
    rows, cols = [], []
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        source_indices = indices[
            : indices.shape[0] - dy, max(0, -dx) : indices.shape[1] - max(0, dx)
        ]
        target_indices = indices[dy:, max(0, dx) : indices.shape[1] - max(0, -dx)]
        source_depth = depth_roi[
            : indices.shape[0] - dy, max(0, -dx) : indices.shape[1] - max(0, dx)
        ]
        target_depth = depth_roi[dy:, max(0, dx) : indices.shape[1] - max(0, -dx)]
        adjacent = (
            (source_indices >= 0)
            & (target_indices >= 0)
            & (np.abs(source_depth - target_depth) <= neighbor_limit)
        )
        rows.append(source_indices[adjacent])
        cols.append(target_indices[adjacent])
    row_indices = np.concatenate(rows)
    col_indices = np.concatenate(cols)
    graph = coo_matrix(
        (np.ones(len(row_indices), dtype=np.uint8), (row_indices, col_indices)),
        shape=(inlier_count, inlier_count),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=component_count)
    selected = int(np.argmax(sizes))
    component_pixels = int(sizes[selected])
    if (
        component_pixels < min_valid_pixels
        or component_pixels / inlier_count < fractions["min_component_fraction"]
    ):
        raise ValueError(
            "target depth has ambiguous disconnected components: "
            f"largest contains {component_pixels} of {inlier_count} inliers "
            f"across {component_count} components"
        )
    refined_mask = np.zeros_like(inliers)
    refined_mask[ys[labels == selected], xs[labels == selected]] = True
    refined = replace(segmentation, mask=refined_mask)
    quality = {
        "snapshot_id": snapshot_id,
        "camera_frame": camera.frame,
        "camera_timestamp_s": camera.timestamp_s,
        "rgb_sha256": rgb_sha256,
        "depth_sha256": depth_sha256,
        "intrinsics_sha256": intrinsics_sha256,
        "camera_pose_sha256": camera_pose_sha256,
        "camera_parent_frame": camera.camera_pose.frame,
        "source_provider": source.get("provider"),
        "source_query": source.get("query"),
        "source_mask_threshold": source.get("mask_threshold"),
        "mask_pixels": mask_count,
        "valid_depth_pixels": valid_count,
        "depth_inlier_pixels": inlier_count,
        "depth_component_count": int(component_count),
        "selected_component_pixels": component_pixels,
        "median_depth_m": median_depth,
        "max_depth_deviation_m": depth_limit,
        "max_neighbor_depth_step_m": neighbor_limit,
        "min_valid_pixels": min_valid_pixels,
        **fractions,
    }
    background_options = (
        background_ring_width_px, min_background_valid_pixels,
        min_background_valid_fraction, min_foreground_gap_m,
    )
    if any(value is not None for value in background_options):
        if any(value is None for value in background_options):
            raise ValueError("background depth check requires all four explicit settings")
        if (
            isinstance(background_ring_width_px, bool)
            or not isinstance(background_ring_width_px, int)
            or not 1 <= background_ring_width_px <= min(segmentation.mask.shape)
        ):
            raise ValueError("background_ring_width_px must be a positive integer within the image")
        if (
            isinstance(min_background_valid_pixels, bool)
            or not isinstance(min_background_valid_pixels, int)
            or min_background_valid_pixels < 1
        ):
            raise ValueError("min_background_valid_pixels must be a positive integer")
        try:
            min_fraction = float(min_background_valid_fraction)
            minimum_gap = float(min_foreground_gap_m)
        except (TypeError, ValueError) as exc:
            raise ValueError("background depth thresholds must be finite numbers") from exc
        if (
            isinstance(min_background_valid_fraction, (bool, np.bool_))
            or not np.isfinite(min_fraction) or not 0.0 < min_fraction <= 1.0
            or isinstance(min_foreground_gap_m, (bool, np.bool_))
            or not np.isfinite(minimum_gap) or minimum_gap <= 0
        ):
            raise ValueError("background fraction must be in (0, 1] and foreground gap positive")
        # The ring excludes the entire source mask: pixels dropped by depth
        # filtering must not become evidence of an independent background.
        original = segmentation.mask
        # The requested ring is truncated even when the mask itself does not
        # touch the boundary but lies closer to it than the ring width.
        mask_y, mask_x = np.nonzero(original)
        height, width = original.shape
        boundary_clipped = bool(
            mask_y.min() < background_ring_width_px
            or mask_x.min() < background_ring_width_px
            or mask_y.max() + background_ring_width_px >= height
            or mask_x.max() + background_ring_width_px >= width
        )
        ring = binary_dilation(original, structure=np.ones((3, 3), dtype=bool),
                               iterations=background_ring_width_px) & ~original
        ring_count = int(np.count_nonzero(ring))
        ring_valid = ring & np.isfinite(camera.depth_m) & (camera.depth_m > 0)
        valid_ring_count = int(np.count_nonzero(ring_valid))
        valid_ring_fraction = valid_ring_count / ring_count if ring_count else 0.0
        background = {
            "status": "uncertain", "ring_width_px": background_ring_width_px,
            "ring_pixels": ring_count, "valid_ring_pixels": valid_ring_count,
            "valid_ring_fraction": valid_ring_fraction, "frame_clipped": boundary_clipped,
            "min_background_valid_pixels": min_background_valid_pixels,
            "min_background_valid_fraction": min_fraction,
            "min_foreground_gap_m": minimum_gap,
        }
        if boundary_clipped:
            raise AmbiguousTargetDepth("target depth is uncertain: mask touches the image boundary", background)
        if valid_ring_count < min_background_valid_pixels or valid_ring_fraction < min_fraction:
            raise AmbiguousTargetDepth("target depth is uncertain: insufficient valid surrounding depth", background)
        target_median = float(np.median(camera.depth_m[refined_mask]))
        background_median = float(np.median(camera.depth_m[ring_valid]))
        gap = background_median - target_median
        background.update(target_median_depth_m=target_median,
                          background_median_depth_m=background_median,
                          foreground_gap_m=gap)
        if gap < minimum_gap:
            raise AmbiguousTargetDepth("target depth is uncertain: insufficient foreground/background separation", background)
        background["status"] = "passed"
        quality["background_depth_check"] = background
    return refined, quality


class DepthRefinedSegmentationProvider:
    """Opt-in depth refinement that preserves the standard provider interface.

    Results carry both raw and refined 0/1 uint8 masks and quality metadata in
    diagnostics. Wrap this provider in ASPIRE's TracedProvider with a
    RunRecorder to persist those arrays, including their file hashes. The
    underlying provider and observation are unchanged. This wrapper is for
    text-segmented objects, not point prompts.
    """

    def __init__(
        self,
        provider: SegmentationProvider,
        *,
        max_depth_deviation_m: float,
        max_neighbor_depth_step_m: float,
        min_valid_pixels: int,
        min_valid_fraction: float,
        min_depth_inlier_fraction: float,
        min_component_fraction: float,
        background_ring_width_px: int | None = None,
        min_background_valid_pixels: int | None = None,
        min_background_valid_fraction: float | None = None,
        min_foreground_gap_m: float | None = None,
    ) -> None:
        if not callable(getattr(provider, "segment_text", None)):
            raise ValueError("provider must support text segmentation")
        self.provider = provider
        self.thresholds = {
            "max_depth_deviation_m": max_depth_deviation_m,
            "max_neighbor_depth_step_m": max_neighbor_depth_step_m,
            "min_valid_pixels": min_valid_pixels,
            "min_valid_fraction": min_valid_fraction,
            "min_depth_inlier_fraction": min_depth_inlier_fraction,
            "min_component_fraction": min_component_fraction,
            "background_ring_width_px": background_ring_width_px,
            "min_background_valid_pixels": min_background_valid_pixels,
            "min_background_valid_fraction": min_background_valid_fraction,
            "min_foreground_gap_m": min_foreground_gap_m,
        }

    def segment_text(self, observation: CameraObservation, query: str) -> SegmentationSet:
        raw = self.provider.segment_text(observation, query)
        if not isinstance(raw, SegmentationSet):
            return SegmentationSet(
                ok=False,
                error=ApiError(
                    ErrorCode.PERCEPTION_FAILED, "provider returned an invalid segmentation result"
                ),
            )
        if not raw.ok:
            return raw
        raw_masks = tuple(item.mask.astype(np.uint8) for item in raw.segmentations)
        for mask in raw_masks:
            mask.setflags(write=False)
        try:
            refined, quality = refine_target_mask(observation, raw, **self.thresholds)
        except ValueError as exc:
            ambiguity = exc.details if isinstance(exc, AmbiguousTargetDepth) else None
            return SegmentationSet(
                ok=False,
                error=ApiError(
                    ErrorCode.POINT_CLOUD_FAILED, f"target depth filtering failed: {exc}",
                    details={"depth_status": "uncertain"} if ambiguity is not None else {},
                ),
                diagnostics={
                    **raw.diagnostics,
                    "original_masks": raw_masks,
                    "depth_filter_error": str(exc),
                    "depth_filter_options": self.thresholds,
                    **({"background_depth_check": ambiguity} if ambiguity is not None else {}),
                },
            )
        refined_mask = refined.mask.astype(np.uint8)
        refined_mask.setflags(write=False)
        return SegmentationSet(
            ok=True,
            segmentations=(refined,),
            diagnostics={
                **raw.diagnostics,
                "depth_filter": quality,
                "original_masks": raw_masks,
                "filtered_mask": refined_mask,
            },
        )

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        del observation, points_px, point_labels
        return SegmentationSet(
            ok=False,
            error=ApiError(
                ErrorCode.UNSUPPORTED, "depth-filtered target provider supports text prompts only"
            ),
        )


__all__ = ["AmbiguousTargetDepth", "DepthRefinedSegmentationProvider", "refine_target_mask"]
