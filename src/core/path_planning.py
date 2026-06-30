"""
Safe path generation helpers.

This is a pure math module with no Isaac Sim or USD dependency.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List

from .interpolation import interpolate_transforms


def generate_adaptive_safe_path(
    T_start: np.ndarray,
    T_end: np.ndarray,
    min_z: float = 0.25,
    lin_speed: float = 0.6,
    ang_speed: float = 1.5,
    min_steps: int = 20,
    dt: float = 1 / 60.0,
) -> List[np.ndarray]:
    """
    Generate an adaptive safe path between two SE(3) transforms.

    The end point is clamped above ``min_z`` to avoid direct table
    intersections. The number of interpolation steps is chosen from linear and
    angular travel distance, with ``min_steps`` as a lower bound.

    Args:
        T_start: Start 4x4 transform matrix.
        T_end: End 4x4 transform matrix.
        min_z: Minimum safe end-point height in meters.
        lin_speed: Target linear speed in meters per second.
        ang_speed: Target angular speed in radians per second.
        min_steps: Lower bound for generated steps.
        dt: Simulation or control time step in seconds.

    Returns:
        List of transform matrices with length determined by travel distance.
    """
    pos_start = T_start[:3, 3]
    pos_end = T_end[:3, 3].copy()

    # Clamp the end height above the safety floor.
    pos_end[2] = max(pos_end[2], min_z)

    T_end_adjusted = T_end.copy()
    T_end_adjusted[:3, 3] = pos_end

    # Linear travel distance.
    dist = np.linalg.norm(pos_end - pos_start)

    # Angular travel distance.
    rot_diff = T_end_adjusted[:3, :3] @ T_start[:3, :3].T
    rot_angle = np.linalg.norm(R.from_matrix(rot_diff).as_rotvec())

    # Adaptive step count.
    steps_lin = dist / (lin_speed * dt)
    steps_rot = rot_angle / (ang_speed * dt)
    total_steps = int(max(steps_lin, steps_rot, min_steps))

    return interpolate_transforms(T_start, T_end_adjusted, total_steps)


def generate_constrained_path(
    T_start: np.ndarray,
    T_end: np.ndarray,
    steps: int = 60,
    min_z: float = 0.05,
) -> List[np.ndarray]:
    """
    Generate a fixed-step interpolated path with a lower Z bound.

    This helper is retained as a simple fixed-step variant; new code should
    prefer ``generate_adaptive_safe_path`` when speed consistency matters.

    Args:
        T_start: Start transform matrix.
        T_end: End transform matrix.
        steps: Fixed number of interpolation steps.
        min_z: Minimum Z height for every path point.

    Returns:
        List of transform matrices.
    """
    T_list = interpolate_transforms(T_start, T_end, steps)
    result = []
    for T in T_list:
        T = T.copy()
        T[2, 3] = max(T[2, 3], min_z)
        result.append(T)
    return result
