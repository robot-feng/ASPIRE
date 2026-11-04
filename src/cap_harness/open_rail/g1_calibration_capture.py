"""Read and archive raw packets through an existing G1 camera receiver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from cap_harness.open_rail.g1_runtime import OpenRailG1ObservationError


def _enum_metadata(value: Any) -> dict[str, Any]:
    name = getattr(value, "name", None)
    raw_value = getattr(value, "value", None)
    if not isinstance(raw_value, (str, int, float, bool)) and raw_value is not None:
        raw_value = str(raw_value)
    return {
        "display": str(value),
        "name": name if isinstance(name, str) else None,
        "value": raw_value,
    }


def _packet_payload(packet: Any) -> bytes:
    payload = packet.get_image_data()
    if payload is None:
        raise OpenRailG1ObservationError("camera packet has no payload")
    if hasattr(payload, "tobytes"):
        raw = payload.tobytes(order="C")
    else:
        try:
            raw = bytes(payload)
        except (TypeError, ValueError) as exc:
            raise OpenRailG1ObservationError(
                "camera packet payload is not a bytes-like value"
            ) from exc
    if not raw:
        raise OpenRailG1ObservationError("camera packet payload is empty")
    return raw


def capture_raw_camera_packets(
    camera_receiver: Any,
    camera_names: Mapping[str, str],
    camera_keys: Sequence[str],
    reference_timestamp_ns: int,
    *,
    image_packet_type: Any,
) -> list[dict[str, Any]]:
    """Read selected channels using the already-open CameraReceiverCosine.

    Payload bytes are preserved exactly as returned by the SDK. No image
    decoding, color conversion, or calibration claim is made here.
    """
    if not isinstance(camera_names, Mapping):
        raise OpenRailG1ObservationError("configured camera names must be a mapping")
    if isinstance(camera_keys, (str, bytes)):
        raise OpenRailG1ObservationError("camera_keys must be a sequence of logical keys")
    if isinstance(reference_timestamp_ns, bool) or not isinstance(
        reference_timestamp_ns, int
    ) or reference_timestamp_ns < 0:
        raise OpenRailG1ObservationError("reference_timestamp_ns must be a non-negative integer")
    keys = tuple(camera_keys)
    if not keys or len(set(keys)) != len(keys):
        raise OpenRailG1ObservationError("camera_keys must be non-empty and unique")

    records: list[dict[str, Any]] = []
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise OpenRailG1ObservationError("camera keys must be non-empty strings")
        logical_key = key.strip()
        channel = camera_names.get(logical_key)
        if not isinstance(channel, str) or not channel.strip():
            raise OpenRailG1ObservationError(
                f"camera key {logical_key!r} is not subscribed by this receiver"
            )
        packet = camera_receiver.get_packet_nearest(channel, reference_timestamp_ns)
        if packet is None:
            raise OpenRailG1ObservationError(
                f"no packet for camera {logical_key!r} near reference timestamp"
            )
        if packet.type != image_packet_type:
            raise OpenRailG1ObservationError(
                f"camera {logical_key!r} returned a non-image packet"
            )
        packet_timestamp_ns = int(packet.send_timestamp)
        width = int(packet.image_width)
        height = int(packet.image_height)
        if packet_timestamp_ns < 0 or width <= 0 or height <= 0:
            raise OpenRailG1ObservationError(
                f"camera {logical_key!r} packet has invalid timestamp or dimensions"
            )
        raw = _packet_payload(packet)
        get_topic = getattr(camera_receiver, "get_topic", None)
        dds_topic = get_topic(channel) if callable(get_topic) else channel
        records.append(
            {
                "metadata": {
                    "camera_key": logical_key,
                    "sdk_channel": channel,
                    "dds_topic": str(dds_topic),
                    "packet_type": _enum_metadata(packet.type),
                    "encoding_format": _enum_metadata(packet.encoding_format),
                    "color_format": _enum_metadata(packet.color_format),
                    "width": width,
                    "height": height,
                    "reference_timestamp_ns": reference_timestamp_ns,
                    "packet_timestamp_ns": packet_timestamp_ns,
                    "timestamp_delta_ns": packet_timestamp_ns - reference_timestamp_ns,
                    "payload_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "payload": raw,
            }
        )
    return records


def save_calibration_sample(
    output_dir: str | Path,
    sample_index: int,
    packets: Sequence[Mapping[str, Any]],
    *,
    target: str = "A4 ChArUco",
) -> Path:
    """Write raw packet payloads and an identity/timestamp manifest atomically."""
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 1:
        raise ValueError("sample_index must be a positive integer")
    if not packets:
        raise ValueError("packets must contain at least one camera packet")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a non-empty label")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"sample_{sample_index:04d}"
    if destination.exists():
        raise FileExistsError(f"capture sample already exists: {destination}")

    temp_dir = Path(tempfile.mkdtemp(prefix=".capture-", dir=root))
    try:
        camera_entries: list[dict[str, Any]] = []
        used_keys: set[str] = set()
        for packet_record in packets:
            metadata = packet_record.get("metadata")
            payload = packet_record.get("payload")
            if not isinstance(metadata, Mapping) or not isinstance(payload, bytes):
                raise ValueError("each packet must contain metadata and raw bytes payload")
            camera_key = metadata.get("camera_key")
            if not isinstance(camera_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", camera_key):
                raise ValueError("camera_key contains unsupported filename characters")
            if camera_key in used_keys:
                raise ValueError(f"duplicate camera packet key: {camera_key}")
            used_keys.add(camera_key)
            filename = f"{camera_key}.packet"
            (temp_dir / filename).write_bytes(payload)
            entry = dict(metadata)
            entry["payload_file"] = filename
            camera_entries.append(entry)

        manifest = {
            "schema": "cap-harness/open-rail-calibration-sample/1",
            "calibration_target": target.strip(),
            "board_detection_performed": False,
            "camera_packets": camera_entries,
        }
        (temp_dir / "sample.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, destination)
        return destination
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


__all__ = ["capture_raw_camera_packets", "save_calibration_sample"]
