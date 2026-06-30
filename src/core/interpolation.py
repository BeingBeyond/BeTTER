"""
SE(3) trajectory interpolation helpers.

This is a pure math module with no Isaac Sim or USD dependency.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from typing import List

from .transforms import decompose_transform_matrix, create_transform_matrix
from .quaternion import wxyz_to_xyzw


def interpolate_transforms(
    start_mat: np.ndarray,
    end_mat: np.ndarray,
    num_steps: int,
) -> List[np.ndarray]:
    """
    Interpolate between two SE(3) transform matrices.

    Positions are linearly interpolated and rotations are interpolated with
    spherical linear interpolation.

    Args:
        start_mat: Start 4x4 transform matrix.
        end_mat: End 4x4 transform matrix.
        num_steps: Number of interpolation steps, including both endpoints.

    Returns:
        List of ``num_steps`` transform matrices.
    """
    start_pos, start_quat = decompose_transform_matrix(start_mat)
    end_pos, end_quat = decompose_transform_matrix(end_mat)

    ratios = np.linspace(0.0, 1.0, num_steps)

    # Linear position interpolation.
    interp_positions = np.outer(1 - ratios, start_pos) + np.outer(ratios, end_pos)

    # Rotational SLERP.
    key_rots = R.from_quat([start_quat, end_quat])
    slerp_obj = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp_obj(ratios)

    result = []
    for i in range(num_steps):
        mat = np.eye(4)
        mat[:3, 3] = interp_positions[i]
        mat[:3, :3] = interp_rots[i].as_matrix()
        result.append(mat)

    return result


def slerp_poses(
    pos1: np.ndarray, quat1_wxyz: np.ndarray,
    pos2: np.ndarray, quat2_wxyz: np.ndarray,
    alpha: float,
) -> tuple:
    """
    Interpolate between two poses with linear position blending and SLERP.

    ``alpha=0`` returns the first pose and ``alpha=1`` returns the second pose.
    Input quaternions use the Isaac Sim ``[w, x, y, z]`` convention.

    Returns:
        (interp_pos, interp_quat_wxyz)
    """
    interp_pos = pos1 * (1 - alpha) + pos2 * alpha

    q1_xyzw = wxyz_to_xyzw(quat1_wxyz)
    q2_xyzw = wxyz_to_xyzw(quat2_wxyz)
    key_rots = R.from_quat([q1_xyzw, q2_xyzw])
    slerp_obj = Slerp([0.0, 1.0], key_rots)
    interp_quat_xyzw = slerp_obj([alpha])[0].as_quat()

    from .quaternion import xyzw_to_wxyz
    return interp_pos, xyzw_to_wxyz(interp_quat_xyzw)
