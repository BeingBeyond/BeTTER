from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationThresholds:
    near_distance: float = 0.35
    on_overlap_ratio: float = 0.3
    on_z_offset_min: float = 0.0
    on_z_offset_max: float = 0.1
    above_overlap_ratio: float = 0.3
    in_containment_ratio: float = 0.5
    touching_distance: float = 0.01
    vertical_align_distance: float = 0.1
    axis_align_angle: float = 0.1745
    left_right_tolerance: float = 0.02
    height_epsilon: float = 0.02
    contain_xy_ratio: float = 0.5
    directional_epsilon: float = 1e-3
