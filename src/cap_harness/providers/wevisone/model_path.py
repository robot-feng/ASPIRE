"""Offline-first model path selection and checkpoint integrity checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from typing import Any

DEFAULT_MODEL_ID = "tencent/WeVisOne-VL-2B"
MODEL_PATH_ENV = "WEVISONE_MODEL_PATH"
DEFAULT_LOCAL_RELATIVE_PATH = Path("third_party/models/wevisone-vl-2b")
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "mask_decoder.pt",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _find_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    working_directory = Path.cwd().resolve()
    candidates = [working_directory, *working_directory.parents, *module_path.parents]
    for candidate in candidates:
        has_project_file = (candidate / "pyproject.toml").is_file()
        has_source_tree = (candidate / "src" / "cap_harness").is_dir()
        if has_project_file and has_source_tree:
            return candidate
    return module_path.parents[4]


def resolve_model_source(
    requested: str | None = None,
    *,
    env: dict[str, str] | None = None,
    repo_root: Path | str | None = None,
) -> str:
    """Choose explicit CLI, environment, local snapshot, then HF ID."""
    environment = os.environ if env is None else env
    root = Path(repo_root) if repo_root is not None else _find_repo_root()
    configured = requested or environment.get(MODEL_PATH_ENV)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        if configured.startswith("~/"):
            return str(candidate)
        if configured.startswith("."):
            return str(candidate.resolve())
        if configured.startswith("third_party/"):
            return str((root / candidate).resolve())
        if candidate.exists():
            return str(candidate.resolve())
        return configured

    local = root / DEFAULT_LOCAL_RELATIVE_PATH
    # A present but incomplete checkout is an operator-visible error. Do not
    # silently switch to a Hub ID and start another download in that case.
    if local.exists():
        return str(local)
    return DEFAULT_MODEL_ID


def verify_safetensors(path: Path | str) -> dict[str, Any]:
    """Validate the Safetensors header and all tensor byte ranges without loading weights."""
    model_path = Path(path)
    file_size = model_path.stat().st_size
    if file_size < 10:
        raise ValueError("safetensors file is too small to contain a header")
    with model_path.open("rb") as stream:
        header_length_raw = stream.read(8)
        header_length = struct.unpack("<Q", header_length_raw)[0]
        if header_length <= 0 or header_length > file_size - 8:
            raise ValueError("safetensors header length is outside the file bounds")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("safetensors header is not valid JSON") from exc

    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    data_size = file_size - 8 - header_length
    tensor_count = 0
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"safetensors entry {name!r} must be an object")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
        ):
            raise ValueError(f"safetensors entry {name!r} has invalid dtype, shape, or offsets")
        if len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
            raise ValueError(f"safetensors entry {name!r} has invalid data offsets")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(f"safetensors entry {name!r} has an invalid shape")
        start, end = offsets
        expected_bytes = _DTYPE_BYTES[dtype]
        for dimension in shape:
            expected_bytes *= dimension
        if start < 0 or end < start or end > data_size or end - start != expected_bytes:
            raise ValueError(f"safetensors entry {name!r} points outside tensor data")
        tensor_count += 1

    if tensor_count == 0:
        raise ValueError("safetensors file contains no tensors")
    return {
        "file": str(model_path),
        "bytes": file_size,
        "tensor_count": tensor_count,
        "data_bytes": data_size,
    }


def verify_snapshot(path: Path | str) -> dict[str, Any]:
    """Verify local release files and the Safetensors weight index, no GPU/model load."""
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ValueError(f"model directory does not exist: {root}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"model snapshot is missing files: {', '.join(missing)}")
    cache = root / ".cache" / "huggingface" / "download"
    incomplete = sorted(item.name for item in cache.glob("*.incomplete")) if cache.is_dir() else []
    if incomplete:
        raise ValueError(f"model snapshot has incomplete downloads: {', '.join(incomplete)}")

    with (root / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"unexpected model_type: {config.get('model_type')!r}")
    weights = verify_safetensors(root / "model.safetensors")
    sidecar = root / "mask_decoder.pt"
    if sidecar.stat().st_size == 0:
        raise ValueError("mask decoder sidecar is empty")
    return {
        "snapshot": str(root.resolve()),
        "model_type": config["model_type"],
        "architecture": (config.get("architectures") or [None])[0],
        "weights": weights,
        "mask_decoder_bytes": sidecar.stat().st_size,
        "required_files": list(REQUIRED_MODEL_FILES),
    }


__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_PATH_ENV",
    "resolve_model_source",
    "verify_safetensors",
    "verify_snapshot",
]
