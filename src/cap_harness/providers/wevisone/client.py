"""Typed ASPIRE segmentation client for WeVisOne-VL."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
import hashlib
import struct
import zlib

import numpy as np
import requests

from cap_harness.contracts import CameraObservation, Segmentation, SegmentationSet
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.providers.http import HttpProviderClient, ProviderHttpError
from cap_harness.providers.wevisone.wire import SEGMENT_PATH

DEFAULT_WEVISONE_URL = "http://127.0.0.1:8117"
DEFAULT_MASK_THRESHOLD = 0.5
DEFAULT_MAX_INSTANCES = 8
MAX_SERVICE_INSTANCES = 16


def _encode_rgb_png(image: np.ndarray) -> str:
    rgb = np.asarray(image)
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
    height, width, _ = rgb.shape
    scanlines = b"".join(b"\x00" + row.tobytes(order="C") for row in rgb)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = binascii.crc32(payload, binascii.crc32(kind)) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _decode_probabilities(value: object, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError("mask probability payload must be base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("mask probability payload is invalid base64") from exc
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if len(raw) != expected:
        raise ValueError(f"mask payload has {len(raw)} bytes; expected {expected}")
    probabilities = np.frombuffer(raw, dtype="<f4").reshape(shape)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("mask probabilities must be finite values in [0, 1]")
    return probabilities


class WeVisOneProvider:
    """Text-prompted instance segmentation backed by an isolated model service.

    WeVisOne-VL returns alpha probabilities, not calibrated object confidence.
    ASPIRE's current binary-mask contract therefore uses a configured threshold
    and a neutral score of 0.5; diagnostics state that the score is not a model
    confidence. Original probabilities remain available from the service output
    only for debugging and are never mislabeled as ASPIRE's binary mask.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_WEVISONE_URL,
        *,
        timeout_s: float | tuple[float, float] = (10.0, 180.0),
        max_retries: int = 1,
        retry_backoff_s: float = 0.1,
        mask_threshold: float = DEFAULT_MASK_THRESHOLD,
        max_instances: int = DEFAULT_MAX_INSTANCES,
        session: requests.Session | None = None,
    ) -> None:
        threshold = float(mask_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("mask_threshold must be finite and in (0, 1)")
        if (
            isinstance(max_instances, bool)
            or not isinstance(max_instances, int)
            or not 1 <= max_instances <= MAX_SERVICE_INSTANCES
        ):
            raise ValueError(f"max_instances must be an integer in [1, {MAX_SERVICE_INSTANCES}]")
        self._http = HttpProviderClient(
            base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            session=session,
        )
        self.mask_threshold = threshold
        self.max_instances = max_instances

    @property
    def base_url(self) -> str:
        return self._http.base_url

    def health(self) -> bool:
        return self._http.health("/healthz")

    def segment_text(self, observation: CameraObservation, query: str) -> SegmentationSet:
        if not isinstance(observation, CameraObservation):
            return self._failure(ErrorCode.INVALID_REQUEST, "observation must be CameraObservation")
        if not isinstance(query, str) or not query.strip():
            return self._failure(ErrorCode.INVALID_REQUEST, "query must be a non-empty string")
        query = query.strip()
        image_shape = observation.rgb.shape[:2]
        try:
            response = self._http.post_json(
                SEGMENT_PATH,
                {
                    "image_base64": _encode_rgb_png(observation.rgb),
                    "query": query,
                    "max_instances": self.max_instances,
                },
            )
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                raise ValueError("response field 'results' must be a list")
            if len(raw_results) > self.max_instances:
                raise ValueError("provider returned more instances than requested")
            total_instances = response.get("total_instances")
            truncated = response.get("truncated")
            if (
                isinstance(total_instances, bool)
                or not isinstance(total_instances, int)
                or total_instances < len(raw_results)
            ):
                raise ValueError(
                    "response field 'total_instances' must be an integer >= result count"
                )
            if not isinstance(truncated, bool):
                raise ValueError("response field 'truncated' must be a boolean")
            if truncated != (total_instances > len(raw_results)):
                raise ValueError("response truncation metadata disagrees with result count")
            if truncated:
                return self._failure(
                    ErrorCode.TRUNCATED,
                    "WeVisOne-VL returned only part of the matching instance set",
                    details={
                        "provider": "wevisone",
                        "query": query,
                        "total_instances": total_instances,
                        "returned_instances": len(raw_results),
                        "max_instances": self.max_instances,
                    },
                )

            segmentations: list[Segmentation] = []
            for index, item in enumerate(raw_results):
                if not isinstance(item, Mapping):
                    raise ValueError("each result must be an object")
                shape = self._shape(item.get("shape"))
                if shape != image_shape:
                    raise ValueError(
                        f"mask shape {shape} does not match image shape {image_shape}"
                    )
                probabilities = _decode_probabilities(item.get("probabilities_base64"), shape)
                mask = probabilities >= self.mask_threshold
                if not bool(np.any(mask)):
                    # Ignore empty matte outputs while retaining other instances.
                    continue
                label = item.get("label", query)
                if not isinstance(label, str) or not label.strip():
                    label = query
                box = item.get("box_xyxy")
                box_xyxy = None
                if box is not None:
                    box_xyxy = np.asarray(box, dtype=np.float64)
                    if box_xyxy.shape != (4,) or not np.isfinite(box_xyxy).all():
                        raise ValueError(f"result {index} box_xyxy must be finite length 4")
                segmentations.append(
                    Segmentation(
                        mask=mask,
                        label=label,
                        score=0.5,
                        camera_name=observation.frame,
                        frame=observation.frame,
                        box_xyxy=box_xyxy,
                    )
                )
        except ProviderHttpError as exc:
            return SegmentationSet(ok=False, error=exc.error)
        except (TypeError, ValueError, KeyError) as exc:
            return self._failure(
                ErrorCode.PERCEPTION_FAILED,
                "WeVisOne-VL returned an invalid segmentation response",
                details={"provider": "wevisone", "query": query, "error": str(exc)},
            )

        if not segmentations:
            return self._failure(
                ErrorCode.NO_SEGMENTATION,
                f"WeVisOne-VL found no non-empty mask for {query!r}",
                details={"provider": "wevisone", "query": query},
            )
        return SegmentationSet(
            ok=True,
            segmentations=segmentations,
            diagnostics={
                "provider": "wevisone",
                "query": query,
                "camera_frame": observation.frame,
                "camera_timestamp_s": observation.timestamp_s,
                "rgb_sha256": hashlib.sha256(observation.rgb.tobytes()).hexdigest(),
                "depth_sha256": hashlib.sha256(observation.depth_m.tobytes()).hexdigest(),
                "intrinsics_sha256": hashlib.sha256(observation.intrinsics.tobytes()).hexdigest(),
                "camera_pose_sha256": hashlib.sha256(
                    observation.camera_pose.as_matrix().tobytes()
                ).hexdigest(),
                "camera_parent_frame": observation.camera_pose.frame,
                "snapshot_id": (
                    None
                    if observation.timestamp_s is None
                    else f"{observation.frame}@{observation.timestamp_s:.9f}"
                ),
                "mask_threshold": self.mask_threshold,
                "score_semantics": "neutral; provider emits no confidence score",
                "total_instances": total_instances,
                "returned_instances": len(raw_results),
                "truncated": False,
            },
        )

    def segment_points(
        self,
        observation: CameraObservation,
        points_px: np.ndarray,
        *,
        point_labels: np.ndarray | None = None,
    ) -> SegmentationSet:
        if not isinstance(observation, CameraObservation):
            return self._failure(ErrorCode.INVALID_REQUEST, "observation must be CameraObservation")
        del points_px, point_labels
        return self._failure(
            ErrorCode.UNSUPPORTED,
            "WeVisOne-VL ASPIRE provider currently supports text prompts only",
        )

    @staticmethod
    def _shape(value: object) -> tuple[int, int]:
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
        ):
            raise ValueError("mask shape must contain two positive integer dimensions")
        return int(value[0]), int(value[1])

    @staticmethod
    def _failure(
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> SegmentationSet:
        return SegmentationSet(
            ok=False,
            error=ApiError(code=code, message=message, details=details or {}),
        )


WeVisOneVLProvider = WeVisOneProvider

__all__ = ["DEFAULT_WEVISONE_URL", "WeVisOneProvider", "WeVisOneVLProvider"]
