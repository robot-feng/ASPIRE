"""ASPIRE FastAPI service that hosts the official GraspGenX inference engine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

from cap_harness.providers.service_runtime import install_service_runtime
from cap_harness.providers.wire import decode_numpy, encode_numpy

from .wire import GENERATE_GRASPS_PATH

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="cap-harness GraspGenX provider")
install_service_runtime(app)
_GPU_LOCK = asyncio.Semaphore(1)


class GenerateGraspsRequest(BaseModel):
    point_cloud_base64: str
    sweep_volume_params: dict[str, Any]
    planner: Literal["graspmoe", "diffusion"] = "graspmoe"
    num_grasps: int = Field(default=200, ge=1)
    planner_kwargs: dict[str, Any] = Field(default_factory=dict)


class GenerateGraspsResponse(BaseModel):
    grasps_base64: str
    scores_base64: str
    branch_tags: list[str]
    timing: dict[str, float] = Field(default_factory=dict)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_backend() -> Any:
    repo = _repo_root()
    vendor = Path(os.environ.get("CAP_HARNESS_VENDOR_ROOT", repo / "third_party")).expanduser()
    vendor = vendor.resolve()
    checkpoint_root = Path(
        os.environ.get(
            "CAP_HARNESS_GRASPGENX_CHECKPOINTS",
            vendor / "models" / "graspgenx-checkpoints" / "release",
        )
    ).expanduser().resolve()
    source_root = vendor / "GraspGenX"
    if not source_root.is_dir():
        raise FileNotFoundError(f"GraspGenX source submodule not found: {source_root}")
    required = (
        checkpoint_root / "gen" / "config.yaml",
        checkpoint_root / "gen" / "epoch_736.pth",
        checkpoint_root / "dis" / "config.yaml",
        checkpoint_root / "dis" / "epoch_1056.pth",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "GraspGenX checkpoints are incomplete; missing: " + ", ".join(missing)
        )
    assets = Path(os.environ.get("CAP_HARNESS_GRASPGENX_ASSETS", source_root / "assets"))
    os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(checkpoint_root.parent)
    os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(assets.resolve())
    sys.path.insert(0, str(source_root))
    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    LOGGER.info("Loading GraspGenX from %s with checkpoints %s", source_root, checkpoint_root)
    return GraspGenXZMQServer(
        config_path=str(checkpoint_root),
        assets_dir=str(assets.resolve()),
        host="127.0.0.1",
        port=0,
    )


@app.on_event("startup")
async def _startup() -> None:
    app.state.graspgenx_backend = await asyncio.to_thread(_load_backend)


@app.post(GENERATE_GRASPS_PATH, response_model=GenerateGraspsResponse)
async def generate_grasps(request: GenerateGraspsRequest) -> GenerateGraspsResponse:
    backend = getattr(app.state, "graspgenx_backend", None)
    if backend is None:
        raise HTTPException(status_code=503, detail="GraspGenX model is not initialized")
    try:
        points = np.asarray(decode_numpy(request.point_cloud_base64), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid point cloud: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise HTTPException(status_code=400, detail="point cloud must have non-empty shape (N, 3)")
    payload = {
        "action": "infer_object",
        "point_cloud": points,
        "sweep_volume_params": request.sweep_volume_params,
        "planner": request.planner,
        "num_grasps": request.num_grasps,
        **request.planner_kwargs,
    }
    try:
        async with _GPU_LOCK:
            result = await asyncio.to_thread(backend._dispatch, payload)
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid GraspGenX request: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("GraspGenX inference failed")
        raise HTTPException(status_code=503, detail=f"GraspGenX inference failed: {exc}") from exc
    return GenerateGraspsResponse(
        grasps_base64=encode_numpy(np.asarray(result["grasps"], dtype=np.float32)),
        scores_base64=encode_numpy(np.asarray(result["confidences"], dtype=np.float32)),
        branch_tags=list(result.get("branch_tags", [])),
        timing={str(key): float(value) for key, value in result.get("timing", {}).items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASPIRE GraspGenX provider")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["app", "generate_grasps", "main"]
