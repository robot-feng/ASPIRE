"""Small profile boundary for Open-RAIL robot variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class G1ObservationProfile:
    """ASPIRE-side description of one Open-RAIL body and its observation keys.

    Profiles contain only mappings that have evidence. Calibration fields stay
    nullable until verified on the target hardware.
    """

    name: str
    robot_type: str
    robot_module: str
    config_factory: str
    public_camera_key: str
    rgb_channel: str
    depth_channel: str
    state_dimension: int
    state_layout: Mapping[str, object]
    calibration: Mapping[str, object]


def load_g1_observation_profile(path: str | Path) -> G1ObservationProfile:
    """Load the AgiBot G1 observation profile without filling unknowns."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("PyYAML is required to load an Open-RAIL profile") from exc

    profile_path = Path(path)
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "cap-harness/open-rail-observation/1":
        raise ValueError(f"unsupported Open-RAIL profile schema in {profile_path}")
    runtime = raw.get("runtime") or {}
    camera = raw.get("camera") or {}
    state = raw.get("state") or {}
    required = {
        "name": raw.get("name"),
        "robot_type": raw.get("robot_type"),
        "robot_module": runtime.get("robot_module"),
        "config_factory": runtime.get("config_factory"),
        "public_camera_key": camera.get("public_camera_key"),
        "rgb_channel": camera.get("rgb_channel"),
        "depth_channel": camera.get("depth_channel"),
    }
    for key, value in required.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Open-RAIL profile requires non-empty {key}")
    if required["robot_type"] != "agibot_g1":
        raise ValueError("G1 observation profile robot_type must be 'agibot_g1'")
    dimension = state.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Open-RAIL profile state.dimension must be a positive integer")
    layout = state.get("layout") or {}
    if not isinstance(layout, dict):
        raise ValueError("Open-RAIL profile state.layout must be a mapping")
    for key, bounds in layout.items():
        if (
            not isinstance(key, str)
            or not isinstance(bounds, list)
            or len(bounds) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) for x in bounds)
            or not 0 <= bounds[0] < bounds[1] <= dimension
        ):
            raise ValueError(f"invalid state layout entry {key!r}: {bounds!r}")
    calibration = raw.get("calibration") or {}
    if not isinstance(calibration, dict):
        raise ValueError("Open-RAIL profile calibration must be a mapping")
    return G1ObservationProfile(
        name=required["name"],
        robot_type=required["robot_type"],
        robot_module=required["robot_module"],
        config_factory=required["config_factory"],
        public_camera_key=required["public_camera_key"],
        rgb_channel=required["rgb_channel"],
        depth_channel=required["depth_channel"],
        state_dimension=dimension,
        state_layout=MappingProxyType(dict(layout)),
        calibration=MappingProxyType(dict(calibration)),
    )


def default_g1_observation_profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "open_rail"
        / "agibot_g1_observation.yaml"
    )


__all__ = [
    "G1ObservationProfile",
    "default_g1_observation_profile_path",
    "load_g1_observation_profile",
]
