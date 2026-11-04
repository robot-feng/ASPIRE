from __future__ import annotations

from enum import Enum
import json

import numpy as np
import pytest

from cap_harness.open_rail.g1_calibration_capture import (
    capture_raw_camera_packets,
    save_calibration_sample,
)
from cap_harness.open_rail.g1_runtime import (
    OpenRailG1ObservationError,
    OpenRailG1ObservationRuntime,
)
from cap_harness.open_rail.profile import (
    default_g1_observation_profile_path,
    load_g1_observation_profile,
)


class PacketType(Enum):
    IMAGE = 1
    OTHER = 2


class Encoding(Enum):
    JPEG = 1


class ColorFormat(Enum):
    RGB = 1


class FakePacket:
    type = PacketType.IMAGE
    encoding_format = Encoding.JPEG
    color_format = ColorFormat.RGB
    image_width = 2
    image_height = 1

    def __init__(self, timestamp: int, payload: bytes):
        self.send_timestamp = timestamp
        self.payload = payload

    def get_image_data(self) -> bytes:
        return self.payload


class FakeCameraReceiver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.packets = {
            "head_color": FakePacket(1000, b"jpeg-head"),
            "head_left_fisheye": FakePacket(1003, b"jpeg-fisheye"),
        }

    def get_topic(self, channel: str) -> str:
        return f"/camera/{channel}"

    def get_packet_nearest(self, channel: str, timestamp_ns: int) -> FakePacket:
        self.calls.append((channel, timestamp_ns))
        return self.packets[channel]


def test_capture_preserves_raw_packets_identity_formats_and_timestamps(tmp_path) -> None:
    camera = FakeCameraReceiver()
    packets = capture_raw_camera_packets(
        camera,
        {"head": "head_color", "head_left_fisheye": "head_left_fisheye"},
        ("head", "head_left_fisheye"),
        1000,
        image_packet_type=PacketType.IMAGE,
    )

    assert camera.calls == [("head_color", 1000), ("head_left_fisheye", 1000)]
    assert packets[0]["payload"] == b"jpeg-head"
    assert packets[1]["payload"] == b"jpeg-fisheye"
    assert packets[0]["metadata"]["encoding_format"]["name"] == "JPEG"
    assert packets[1]["metadata"]["color_format"]["name"] == "RGB"
    assert packets[1]["metadata"]["dds_topic"] == "/camera/head_left_fisheye"
    assert packets[1]["metadata"]["timestamp_delta_ns"] == 3

    sample = save_calibration_sample(tmp_path, 1, packets)
    assert (sample / "head.packet").read_bytes() == b"jpeg-head"
    assert (sample / "head_left_fisheye.packet").read_bytes() == b"jpeg-fisheye"
    metadata = json.loads((sample / "sample.json").read_text())
    assert metadata["calibration_target"] == "A4 ChArUco"
    assert metadata["board_detection_performed"] is False
    assert metadata["camera_packets"][0]["sha256"] == packets[0]["metadata"]["sha256"]
    with pytest.raises(FileExistsError):
        save_calibration_sample(tmp_path, 1, packets)


def test_capture_rejects_unsubscribed_or_missing_packets() -> None:
    camera = FakeCameraReceiver()
    with pytest.raises(OpenRailG1ObservationError, match="not subscribed"):
        capture_raw_camera_packets(
            camera, {"head": "head_color"}, ("unknown",), 1000,
            image_packet_type=PacketType.IMAGE,
        )

    camera.packets["head_color"] = None
    with pytest.raises(OpenRailG1ObservationError, match="no packet"):
        capture_raw_camera_packets(
            camera, {"head": "head_color"}, ("head",), 1000,
            image_packet_type=PacketType.IMAGE,
        )


class SnapshotProxy:
    def __init__(self) -> None:
        self.capture_call = None
        self.observations = [
            {
                "cam.head": np.full((2, 3, 3), 10, dtype=np.uint8),
                "obs.state": np.zeros(20, dtype=np.float32),
                "rgb_timestamp_ns": 1000,
                "rgb_packet_timestamp_ns": 1000,
                "rgb_order": "RGB",
                "rgb_encoding_format": "JPEG",
                "rgb_color_format": "RGB",
                "rgb_decoder": "opencv-imdecode-bgr-to-rgb",
                "state_timestamps_ns": {"head": 1000, "waist": 1000},
            }
        ]

    def retrieve_observation(self):
        return self.observations.pop(0)

    def capture_calibration_packets(self, camera_keys, reference_timestamp_ns):
        self.capture_call = (camera_keys, reference_timestamp_ns)
        return [{"metadata": {"camera_key": "head"}, "payload": b"raw"}]

    def close(self) -> None:
        pass


def test_runtime_uses_one_read_only_proxy_for_raw_calibration_packets() -> None:
    profile = load_g1_observation_profile(default_g1_observation_profile_path())
    proxy = SnapshotProxy()
    runtime = OpenRailG1ObservationRuntime(
        open_rail_root="/unused",
        profile=profile,
        calibration_camera_keys=("head_left_fisheye",),
        allow_physical_motion=False,
        proxy=proxy,
    )

    captured = runtime.capture_calibration_packets()

    assert proxy.capture_call == (["head", "head_left_fisheye"], 1000)
    assert captured[0]["payload"] == b"raw"
    runtime.close()


def test_calibration_channels_must_be_enabled_by_profile() -> None:
    profile = load_g1_observation_profile(default_g1_observation_profile_path())
    with pytest.raises(ValueError, match="not enabled by this G1 profile"):
        OpenRailG1ObservationRuntime(
            open_rail_root="/unused",
            profile=profile,
            calibration_camera_keys=("head_center_fisheye",),
            proxy=SnapshotProxy(),
        )

