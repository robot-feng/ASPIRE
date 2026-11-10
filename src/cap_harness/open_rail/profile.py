"""Small profile boundary for Open-RAIL robot variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


def _freeze(value: object) -> object:
    """Recursively freeze profile data before exposing it as runtime health."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_mapping(value: dict[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze(item) for key, item in value.items()})


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
    selected_arm: str
    state_dimension: int
    state_layout: Mapping[str, object]
    state_topics: Mapping[str, object]
    arm_message_names: tuple[str, ...]
    calibration: Mapping[str, object]
    camera: Mapping[str, object]
    arms: Mapping[str, object]
    gripper: Mapping[str, object]
    control: Mapping[str, object]


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
    control = raw.get("control") or {}
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
    topics = state.get("topics") or {}
    arm_names = state.get("arm_message_names") or []
    if not isinstance(arm_names, list) or any(not isinstance(name, str) or not name.strip() for name in arm_names):
        raise ValueError("state.arm_message_names must be a list of non-empty names")
    if arm_names and (len(arm_names) != 14 or len(set(arm_names)) != 14):
        raise ValueError("state.arm_message_names must contain 14 unique names")
    if not isinstance(topics, dict) or any(
        not isinstance(name, str) or not isinstance(topic, str) or not topic.strip()
        for name, topic in topics.items()
    ):
        raise ValueError("Open-RAIL profile state.topics must map names to non-empty topics")
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
    arms = raw.get("arms") or {}
    gripper = raw.get("gripper") or {}
    if not isinstance(arms, dict):
        raise ValueError("Open-RAIL profile arms must be a mapping")
    if not isinstance(gripper, dict):
        raise ValueError("Open-RAIL profile gripper must be a mapping")
    if not isinstance(control, dict):
        raise ValueError("Open-RAIL profile control must be a mapping")
    selected_arm = control.get("selected_arm")
    if not isinstance(selected_arm, str) or selected_arm not in arms:
        raise ValueError("Open-RAIL profile control.selected_arm must name a configured arm")
    allowed_commands = control.get("allowed_commands")
    if allowed_commands != ["right_arm_joint_position", "right_gripper_position"]:
        raise ValueError("G1 profile may allow only right-arm and right-gripper commands")
    read_only_parts = control.get("read_only_parts")
    if read_only_parts != ["left_arm", "head", "waist", "torso", "base"]:
        raise ValueError("G1 profile must keep the left arm, head, waist, torso, and base read-only")
    return G1ObservationProfile(
        name=required["name"],
        robot_type=required["robot_type"],
        robot_module=required["robot_module"],
        config_factory=required["config_factory"],
        public_camera_key=required["public_camera_key"],
        rgb_channel=required["rgb_channel"],
        depth_channel=required["depth_channel"],
        selected_arm=selected_arm,
        state_dimension=dimension,
        state_layout=_freeze_mapping(layout),
        state_topics=_freeze_mapping(topics),
        arm_message_names=tuple(arm_names),
        calibration=_freeze_mapping(calibration),
        camera=_freeze_mapping(camera),
        arms=_freeze_mapping(arms),
        gripper=_freeze_mapping(gripper),
        control=_freeze_mapping(control),
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
