from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrototypeState:
    """Observed routing state of one prototype before component divergence."""

    family: str
    base: int
    scale_count: int
    direction_count: int
    active_pose_count: int
    scale_mode: str
    direction_mode: str
    null_score: float


@dataclass(frozen=True, slots=True)
class ComponentAssignment:
    """A future state-driven conversion without mutating the source bank."""

    source_family: str
    source_base: int
    target_component: str
    reason: str = ""
