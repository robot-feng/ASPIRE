from __future__ import annotations

import base64
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.contracts import CameraObservation, Pose
from cap_harness.errors import ErrorCode
from cap_harness.providers.wevisone import WeVisOneProvider
from cap_harness.providers.wevisone.compat_457 import Transformers457SnapshotView
from cap_harness.providers.wevisone.model_path import (
    REQUIRED_MODEL_FILES,
    resolve_model_source,
    verify_snapshot,
)
from cap_harness.providers.wevisone.service import _format_result


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def request(self, method: str, url: str, *, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json, "timeout": timeout})
        return FakeResponse(200, self.payload)


def _observation(rgb: np.ndarray | None = None) -> CameraObservation:
    image = np.zeros((2, 3, 3), dtype=np.uint8) if rgb is None else rgb
    return CameraObservation(
        rgb=image,
        depth_m=np.ones(image.shape[:2], dtype=np.float32),
        intrinsics=np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]),
        frame="head_optical",
        camera_pose=Pose(
            position=np.zeros(3),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            frame="robot_base",
        ),
        timestamp_s=10.0,
    )


def _mask(probabilities: np.ndarray) -> dict:
    return {
        "label": "apple",
        "box_xyxy": [0.0, 0.0, 3.0, 2.0],
        "shape": list(probabilities.shape),
        "probabilities_base64": base64.b64encode(
            np.asarray(probabilities, dtype="<f4").tobytes()
        ).decode("ascii"),
    }


def _response(
    results: list[dict], *, total_instances: int | None = None, truncated: bool = False
) -> dict:
    return {
        "results": results,
        "total_instances": len(results) if total_instances is None else total_instances,
        "truncated": truncated,
    }


def test_text_segmentation_preserves_all_instances_and_thresholds_probabilities() -> None:
    first = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]], dtype=np.float32)
    second = np.array([[0.1, 0.9, 0.2], [0.7, 0.1, 0.8]], dtype=np.float32)
    session = FakeSession(_response([_mask(first), _mask(second)]))
    provider = WeVisOneProvider(session=session)  # type: ignore[arg-type]

    result = provider.segment_text(_observation(), "apple")

    assert result.ok
    assert len(result.segmentations) == 2
    np.testing.assert_array_equal(result.segmentations[0].mask, first >= 0.5)
    np.testing.assert_array_equal(result.segmentations[1].mask, second >= 0.5)
    assert all(item.label == "apple" for item in result.segmentations)
    assert all(item.frame == "head_optical" for item in result.segmentations)
    assert result.diagnostics["score_semantics"] == "neutral; provider emits no confidence score"
    assert result.diagnostics["total_instances"] == 2
    assert result.diagnostics["truncated"] is False
    assert session.calls[0]["json"]["query"] == "apple"


def test_no_masks_returns_explicit_no_segmentation() -> None:
    provider = WeVisOneProvider(session=FakeSession(_response([])))  # type: ignore[arg-type]

    result = provider.segment_text(_observation(), "apple")

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.NO_SEGMENTATION


def test_empty_probability_masks_are_not_reported_as_objects() -> None:
    provider = WeVisOneProvider(
        session=FakeSession(
            _response([_mask(np.zeros((2, 3), dtype=np.float32))])
        )  # type: ignore[arg-type]
    )

    result = provider.segment_text(_observation(), "apple")

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.NO_SEGMENTATION


@pytest.mark.parametrize(
    "item",
    [
        {"shape": [2, 2], "probabilities_base64": "AA=="},
        {"shape": [2, 3], "probabilities_base64": "not base64"},
        {
            "shape": [2, 3],
            "probabilities_base64": base64.b64encode(np.full((2, 3), 2, dtype="<f4").tobytes()).decode(),
        },
    ],
)
def test_invalid_masks_return_perception_error(item: dict) -> None:
    provider = WeVisOneProvider(session=FakeSession(_response([item])))  # type: ignore[arg-type]

    result = provider.segment_text(_observation(), "apple")

    assert not result.ok
    assert result.error is not None
    assert result.error.code is ErrorCode.PERCEPTION_FAILED


def test_invalid_observation_and_point_prompts_return_typed_failures() -> None:
    provider = WeVisOneProvider(session=FakeSession(_response([])))  # type: ignore[arg-type]

    invalid = provider.segment_text(object(), "apple")  # type: ignore[arg-type]
    points = provider.segment_points(_observation(), np.array([[1.0, 1.0]]))

    assert not invalid.ok and invalid.error.code is ErrorCode.INVALID_REQUEST
    assert not points.ok and points.error.code is ErrorCode.UNSUPPORTED


def test_truncated_single_instance_request_fails_closed() -> None:
    provider = WeVisOneProvider(
        max_instances=1,
        session=FakeSession(
            _response([_mask(np.ones((2, 3), dtype=np.float32))], total_instances=3, truncated=True)
        ),
    )  # type: ignore[arg-type]

    result = provider.segment_text(_observation(), "apple")

    assert not result.ok
    assert result.segmentations == ()
    assert result.error is not None
    assert result.error.code is ErrorCode.TRUNCATED
    assert result.error.details["total_instances"] == 3
    assert result.error.details["returned_instances"] == 1


def test_service_reports_total_before_truncating_results() -> None:
    result = SimpleNamespace(
        boxes=[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]],
        masks=[np.ones((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)],
        response="two objects",
        latency_s=0.1,
    )

    response = _format_result(result, "apple", width=3, height=2, limit=1)

    assert len(response.results) == 1
    assert response.total_instances == 2
    assert response.truncated is True


def test_model_source_uses_complete_local_snapshot_and_env_override(tmp_path, monkeypatch) -> None:
    local = tmp_path / "third_party" / "models" / "wevisone-vl-2b"
    local.mkdir(parents=True)
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "src" / "cap_harness").mkdir(parents=True)
    for name in REQUIRED_MODEL_FILES:
        (local / name).write_bytes(b"snapshot-file")

    assert resolve_model_source(repo_root=tmp_path) == str(local)
    monkeypatch.chdir(tmp_path)
    assert (
        resolve_model_source(
            env={"WEVISONE_MODEL_PATH": "third_party/models/wevisone-vl-2b"},
        )
        == str(local)
    )
    assert (
        resolve_model_source(env={"WEVISONE_MODEL_PATH": "/models/alternate"}, repo_root=tmp_path)
        == "/models/alternate"
    )
    (local / ".cache/huggingface/download").mkdir(parents=True)
    (local / ".cache/huggingface/download/model.incomplete").touch()
    assert resolve_model_source(repo_root=tmp_path) == str(local)


def test_incomplete_default_snapshot_does_not_fall_back_to_hub(tmp_path) -> None:
    local = tmp_path / "third_party" / "models" / "wevisone-vl-2b"
    local.mkdir(parents=True)

    assert resolve_model_source(repo_root=tmp_path) == str(local)


def test_offline_snapshot_verifier_checks_safetensors_tensor_ranges(tmp_path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (root / name).write_bytes(b"snapshot-file")
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "architectures": ["Qwen3VLForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )

    header = json.dumps(
        {"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    result = verify_snapshot(root)

    assert result["model_type"] == "qwen3_vl"
    assert result["weights"]["tensor_count"] == 1
    assert result["weights"]["data_bytes"] == 4

    cache = root / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model.incomplete").touch()
    with pytest.raises(ValueError, match="incomplete downloads"):
        verify_snapshot(root)


def _write_transformers_457_snapshot(root) -> None:
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"weights stay untouched")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "extra_special_tokens": [
                    "<|im_start|>",
                    "<|im_end|>",
                    "<|vision_start|>",
                    "<|vision_end|>",
                    "<|vision_pad|>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "model_type": "qwen3_vl_text",
                    "head_dim": 128,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 5_000_000,
                        "mrope_section": [24, 20, 20],
                        "mrope_interleaved": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_transformers_457_view_adapts_only_supported_configs_and_preserves_source(tmp_path) -> None:
    source = tmp_path / "snapshot"
    _write_transformers_457_snapshot(source)
    original_config = (source / "config.json").read_bytes()
    original_tokenizer = (source / "tokenizer_config.json").read_bytes()
    original_weights = (source / "model.safetensors").read_bytes()

    view = Transformers457SnapshotView(source)
    try:
        adapted_config = json.loads((view.path / "config.json").read_text(encoding="utf-8"))
        adapted_tokenizer = json.loads(
            (view.path / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        assert adapted_config["text_config"]["rope_theta"] == 5_000_000
        assert adapted_config["text_config"]["rope_scaling"] == {
            "rope_type": "default",
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        }
        assert "rope_parameters" not in adapted_config["text_config"]
        assert adapted_tokenizer["extra_special_tokens"] == {
            "im_start_token": "<|im_start|>",
            "im_end_token": "<|im_end|>",
            "vision_start_token": "<|vision_start|>",
            "vision_end_token": "<|vision_end|>",
            "vision_pad_token": "<|vision_pad|>",
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
        }
        assert (view.path / "model.safetensors").is_symlink()
        assert (source / "config.json").read_bytes() == original_config
        assert (source / "tokenizer_config.json").read_bytes() == original_tokenizer
        assert (source / "model.safetensors").read_bytes() == original_weights
    finally:
        view.cleanup()


@pytest.mark.parametrize(
    "config_mutation,token_mutation,error",
    [
        (lambda config: config["text_config"]["rope_parameters"].update(rope_type="linear"), None, "only default"),
        (lambda config: config["text_config"]["rope_parameters"].update(mrope_interleaved=False), None, "interleaved"),
        (None, lambda tokens: tokens["extra_special_tokens"].reverse(), "unsupported WeVisOne tokenizer"),
    ],
)
def test_transformers_457_view_rejects_unknown_tokenizer_or_rope_variants(
    tmp_path, config_mutation, token_mutation, error
) -> None:
    source = tmp_path / "snapshot"
    _write_transformers_457_snapshot(source)
    if config_mutation is not None:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config_mutation(config)
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if token_mutation is not None:
        tokenizer = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
        token_mutation(tokenizer)
        (source / "tokenizer_config.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        Transformers457SnapshotView(source)
