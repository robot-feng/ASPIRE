"""Isolated WeVisOne-VL inference service for ASPIRE."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import uvicorn

from cap_harness.providers.service_runtime import install_service_runtime

from .compat_457 import Transformers457SnapshotView
from .model_path import resolve_model_source, verify_snapshot
from .wire import SEGMENT_PATH

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="cap-harness WeVisOne-VL provider")
install_service_runtime(app)
_ENGINE: Any | None = None
_COMPAT_SNAPSHOT_VIEW: Transformers457SnapshotView | None = None
_INFERENCE_LOCK = asyncio.Semaphore(1)
_MAX_INSTANCES = 8


class SegmentRequest(BaseModel):
    image_base64: str
    query: str
    max_instances: int = Field(default=8, ge=1, le=16)


class MaskResult(BaseModel):
    label: str
    box_xyxy: list[float]
    probabilities_base64: str
    shape: list[int]


class SegmentResponse(BaseModel):
    results: list[MaskResult]
    total_instances: int
    truncated: bool
    response: str
    latency_s: float


def _decode_image(value: str) -> Image.Image:
    try:
        payload = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid RGB image: {exc}") from exc


def _encode_probabilities(mask: object, expected_shape: tuple[int, int]) -> str:
    probabilities = np.asarray(mask, dtype=np.float32)
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"mask shape {probabilities.shape} does not match image {expected_shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("mask probabilities contain non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("mask probabilities must lie in [0, 1]")
    raw = np.ascontiguousarray(probabilities, dtype="<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _format_result(result: Any, query: str, *, width: int, height: int, limit: int) -> SegmentResponse:
    boxes = list(getattr(result, "boxes", ()))
    masks = list(getattr(result, "masks", ()))
    if len(boxes) != len(masks):
        raise ValueError(f"inference returned {len(boxes)} boxes but {len(masks)} masks")
    instances: list[MaskResult] = []
    for box, mask in zip(boxes[:limit], masks[:limit], strict=True):
        normalized = np.asarray(box, dtype=np.float64)
        if normalized.shape != (4,) or not np.isfinite(normalized).all():
            raise ValueError("inference box must be finite normalized xyxy")
        x1, y1, x2, y2 = normalized
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"inference box is outside normalized image bounds: {normalized}")
        instances.append(
            MaskResult(
                label=query,
                box_xyxy=[float(x1 * width), float(y1 * height), float(x2 * width), float(y2 * height)],
                probabilities_base64=_encode_probabilities(mask, (height, width)),
                shape=[height, width],
            )
        )
    return SegmentResponse(
        results=instances,
        total_instances=len(masks),
        truncated=len(masks) > limit,
        response=str(getattr(result, "response", "")),
        latency_s=float(getattr(result, "latency_s", 0.0)),
    )


@app.post(SEGMENT_PATH, response_model=SegmentResponse)
async def segment(request: SegmentRequest) -> SegmentResponse:
    if _ENGINE is None:
        raise HTTPException(status_code=503, detail="WeVisOne-VL engine is not initialized")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty")
    image = _decode_image(request.image_base64)
    async with _INFERENCE_LOCK:
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _ENGINE.predict(
                    image,
                    task="segmentation",
                    query=query,
                    decode_matting=True,
                ),
            )
            return _format_result(
                result,
                query,
                width=image.width,
                height=image.height,
                limit=min(request.max_instances, _MAX_INSTANCES),
            )
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("WeVisOne-VL segmentation failed")
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASPIRE's WeVisOne-VL service")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument(
        "--compat-transformers-457",
        action="store_true",
        help="adapt the supported local Qwen3-VL snapshot schema in a disposable view",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8117)
    args = parser.parse_args()

    global _COMPAT_SNAPSHOT_VIEW, _ENGINE
    model_source = resolve_model_source(args.model)
    source_path = os.path.expanduser(model_source)
    if os.path.isabs(source_path):
        if not os.path.isdir(source_path):
            raise FileNotFoundError(f"local WeVisOne-VL snapshot does not exist: {source_path}")
        verify_snapshot(source_path)
        # Prevent an accidental Hub fallback when a tokenizer/sidecar file is
        # missing from the selected local directory.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.compat_transformers_457:
        if args.backend != "transformers":
            raise ValueError("--compat-transformers-457 requires --backend transformers")
        if not os.path.isabs(source_path) or not os.path.isdir(source_path):
            raise ValueError("--compat-transformers-457 requires a local model snapshot directory")
        _COMPAT_SNAPSHOT_VIEW = Transformers457SnapshotView(source_path)
        model_source = str(_COMPAT_SNAPSHOT_VIEW.path)
    # Hugging Face libraries may cache offline settings at import time.
    from wevisone_vl.engine import WeVisOneVL

    LOGGER.info("Loading WeVisOne-VL from %s on %s", model_source, args.device)
    _ENGINE = WeVisOneVL(model_source, device=args.device, backend=args.backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
