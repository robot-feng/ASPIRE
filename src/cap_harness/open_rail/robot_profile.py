"""Robot-specific metadata for the Open-RAIL embodiment boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from cap_harness.contracts import RobotPlanningContext


@dataclass(frozen=True, slots=True)
class OpenRailProfile:
    """Verified kinematic identity for one robot supported by Open-RAIL.

    The profile is data for one robot; perception and grasp providers do not
    depend on it. Joint names, frames, and transforms must come from that robot's
    description and calibration rather than from a guessed G1 convention.
    """

    name: str
    robot_type: str
    model: str
    base_frame: str
    arms: Sequence[str]
    joint_names: Mapping[str, Sequence[str]]
    end_effector_links: Mapping[str, str]
    base_transforms: Mapping[str, np.ndarray]
    arm_aliases: Mapping[str, str]

    def __post_init__(self) -> None:
        for field_name in ("name", "robot_type", "model", "base_frame"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

        arms = tuple(str(arm).strip() for arm in self.arms)
        if not arms or any(not arm for arm in arms) or len(set(arms)) != len(arms):
            raise ValueError("arms must contain unique non-empty names")
        dimension = None
        joint_names: dict[str, tuple[str, ...]] = {}
        end_effector_links: dict[str, str] = {}
        base_transforms: dict[str, np.ndarray] = {}
        for arm in arms:
            names = tuple(str(name).strip() for name in self.joint_names.get(arm, ()))
            if len(names) not in (6, 7) or any(not name for name in names):
                raise ValueError(f"joint_names[{arm!r}] must contain 6 or 7 names")
            if dimension is None:
                dimension = len(names)
            elif len(names) != dimension:
                raise ValueError("all Open-RAIL arms must have the same joint dimension")
            joint_names[arm] = names

            link = str(self.end_effector_links.get(arm, "")).strip()
            if not link:
                raise ValueError(f"end_effector_links[{arm!r}] must be non-empty")
            end_effector_links[arm] = link

            transform = np.asarray(self.base_transforms.get(arm), dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise ValueError(f"base_transforms[{arm!r}] must be a finite 4x4 matrix")
            if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
                raise ValueError(f"base_transforms[{arm!r}] must be homogeneous")
            rotation = transform[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6) or not np.isclose(
                np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-6
            ):
                raise ValueError(f"base_transforms[{arm!r}] must have a proper rotation")
            transform = np.array(transform, copy=True)
            transform.setflags(write=False)
            base_transforms[arm] = transform

        aliases = {str(alias): str(arm) for alias, arm in self.arm_aliases.items()}
        if any(arm not in arms for arm in aliases.values()):
            raise ValueError("arm_aliases must refer to a configured arm")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "joint_names", MappingProxyType(joint_names))
        object.__setattr__(self, "end_effector_links", MappingProxyType(end_effector_links))
        object.__setattr__(self, "base_transforms", MappingProxyType(base_transforms))
        object.__setattr__(self, "arm_aliases", MappingProxyType(aliases))

    @property
    def joint_dimension(self) -> int:
        return len(next(iter(self.joint_names.values())))

    @property
    def embodiment(self) -> str:
        return f"open_rail_{self.joint_dimension}dof"

    def resolve_arm(self, arm: str) -> str:
        """Resolve shared API aliases while preserving native arm names."""
        name = str(arm)
        if name in self.arms:
            return name
        return self.arm_aliases.get(name, name)

    def planning_context(self) -> RobotPlanningContext:
        """Return the verified robot model metadata used by motion planners."""
        return RobotPlanningContext(
            embodiment=self.embodiment,
            model=self.model,
            joint_names=self.joint_names,
            base_transforms=self.base_transforms,
            end_effector_links=self.end_effector_links,
        )


__all__ = ["OpenRailProfile"]
