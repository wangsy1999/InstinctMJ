from __future__ import annotations

from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg


@dataclass(kw_only=True)
class InstinctActuatorCfg(BuiltinPositionActuatorCfg):
    """Builtin position actuator config with joint velocity limit metadata."""

    velocity_limit: float


@dataclass(kw_only=True)
class DelayedInstinctActuatorCfg(InstinctActuatorCfg):
    """Position actuator cfg with mjlab-native integrated command delay fields."""
