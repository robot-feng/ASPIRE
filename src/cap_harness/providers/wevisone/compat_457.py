"""Temporary local snapshot view for the supported Transformers 4.57 schema."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any

_SPECIAL_TOKEN_ATTRIBUTES = {
    "<|im_start|>": "im_start_token",
    "<|im_end|>": "im_end_token",
    "<|vision_start|>": "vision_start_token",
    "<|vision_end|>": "vision_end_token",
    "<|vision_pad|>": "vision_pad_token",
    "<|image_pad|>": "image_token",
    "<|video_pad|>": "video_token",
}
_ROPE_FIELDS = {"rope_type", "rope_theta", "mrope_section", "mrope_interleaved"}


class Transformers457SnapshotView:
    """Own a disposable snapshot directory with only old-schema JSON adapted."""

    def __init__(self, source: Path | str) -> None:
        self.source = Path(source).expanduser().resolve()
        if not self.source.is_dir():
            raise ValueError(f"Transformers 4.57 compatibility requires a local snapshot: {source}")

        self._temporary = tempfile.TemporaryDirectory(prefix="wevisone-transformers-457-")
        self.path = Path(self._temporary.name) / "snapshot"
        try:
            self._build()
        except BaseException:
            self._temporary.cleanup()
            raise

    def cleanup(self) -> None:
        self._temporary.cleanup()

    def _build(self) -> None:
        tokenizer_path = self.source / "tokenizer_config.json"
        config_path = self.source / "config.json"
        if not tokenizer_path.is_file() or not config_path.is_file():
            raise ValueError("snapshot must include config.json and tokenizer_config.json")

        tokenizer = _read_json(tokenizer_path)
        raw_tokens = tokenizer.get("extra_special_tokens")
        if raw_tokens != list(_SPECIAL_TOKEN_ATTRIBUTES):
            raise ValueError(
                "unsupported WeVisOne tokenizer extra_special_tokens; expected the released Qwen3-VL list"
            )
        tokenizer["extra_special_tokens"] = {
            attribute: token for token, attribute in _SPECIAL_TOKEN_ATTRIBUTES.items()
        }

        config = _read_json(config_path)
        text_config = config.get("text_config")
        if (
            config.get("model_type") != "qwen3_vl"
            or not isinstance(text_config, dict)
            or text_config.get("model_type") != "qwen3_vl_text"
        ):
            raise ValueError("unsupported model config; expected a Qwen3-VL text_config")
        rope = text_config.get("rope_parameters")
        if not isinstance(rope, dict) or set(rope) != _ROPE_FIELDS:
            raise ValueError("unsupported rope_parameters schema for Transformers 4.57")
        if rope.get("rope_type") != "default":
            raise ValueError("only default Qwen3-VL RoPE is supported by this compatibility view")
        theta = rope.get("rope_theta")
        if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
            raise ValueError("rope_theta must be a positive finite number")
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list)
            or len(sections) != 3
            or any(type(section) is not int or section <= 0 for section in sections)
        ):
            raise ValueError("mrope_section must contain three positive integers")
        if rope.get("mrope_interleaved") is not True:
            raise ValueError("only interleaved Qwen3-VL RoPE is supported")

        text_config.pop("rope_parameters")
        text_config["rope_theta"] = theta
        text_config["rope_scaling"] = {
            "rope_type": "default",
            "mrope_section": sections,
            "mrope_interleaved": True,
        }

        self.path.mkdir()
        for item in self.source.iterdir():
            destination = self.path / item.name
            if item.name == "tokenizer_config.json":
                _write_json(destination, tokenizer)
            elif item.name == "config.json":
                _write_json(destination, config)
            else:
                destination.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid snapshot JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"snapshot JSON must contain an object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["Transformers457SnapshotView"]
