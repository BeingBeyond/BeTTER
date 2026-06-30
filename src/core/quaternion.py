"""
Quaternion utility functions.

Conventions:
- Isaac Sim and USD use the ``[w, x, y, z]`` convention.
- SciPy uses the ``[x, y, z, w]`` convention.
- Function names use ``wxyz`` and ``xyzw`` to make the expected layout explicit.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union

ArrayLike = Union[np.ndarray, list, tuple]


def wxyz_to_xyzw(quat: ArrayLike) -> np.ndarray:
    """
    Convert an Isaac Sim ``[w, x, y, z]`` quaternion to SciPy ``[x, y, z, w]``.

    Example:
        >>> wxyz_to_xyzw([1, 0, 0, 0])
        array([0., 0., 0., 1.])
    """
    q = np.asarray(quat, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(quat: ArrayLike) -> np.ndarray:
    """
    Convert a SciPy ``[x, y, z, w]`` quaternion to Isaac Sim ``[w, x, y, z]``.

    Example:
        >>> xyzw_to_wxyz([0, 0, 0, 1])
        array([1., 0., 0., 0.])
    """
    q = np.asarray(quat, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def normalize_quat(quat: ArrayLike) -> np.ndarray:
    """Normalize a quaternion in any component order."""
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Cannot normalize near-zero quaternion")
    return q / norm


def quat_multiply_wxyz(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    """
    Multiply two quaternions in ``[w, x, y, z]`` format.

    Prefer ``scipy.spatial.transform.Rotation`` for new code; this helper is
    kept for compatibility with older internal call sites.
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    # Convert to xyzw, use SciPy for composition, then convert back.
    r1 = R.from_quat(wxyz_to_xyzw(q1))
    r2 = R.from_quat(wxyz_to_xyzw(q2))
    return xyzw_to_wxyz((r1 * r2).as_quat())


def rotate_vector_by_quat_wxyz(quat_wxyz: ArrayLike, vec: ArrayLike) -> np.ndarray:
    """Rotate a vector with a ``[w, x, y, z]`` quaternion."""
    r = R.from_quat(wxyz_to_xyzw(quat_wxyz))
    return r.apply(np.asarray(vec, dtype=float))


def adjust_orientation_wxyz(quat_wxyz: ArrayLike) -> np.ndarray:
    """
    Adjust an orientation so the local positive X axis points forward.

    If the transformed X-axis has a negative world X component, rotate the
    orientation by 180 degrees around the world Z axis.

    Args:
        quat_wxyz: Quaternion in ``[w, x, y, z]`` order.

    Returns:
        Adjusted quaternion in ``[w, x, y, z]`` order.
    """
    rot = R.from_quat(wxyz_to_xyzw(quat_wxyz))
    if rot.apply(np.array([1.0, 0.0, 0.0]))[0] < 0:
        rot = R.from_euler("z", 180, degrees=True) * rot
    return xyzw_to_wxyz(rot.as_quat())


def rotate_orientation_by_z_wxyz(quat_wxyz: ArrayLike, angle_deg: float) -> np.ndarray:
    """
    Rotate a quaternion by a given angle around the world Z axis.

    Args:
        quat_wxyz: Quaternion in ``[w, x, y, z]`` order.
        angle_deg: Rotation angle in degrees.

    Returns:
        Rotated quaternion in ``[w, x, y, z]`` order.
    """
    rot = R.from_quat(wxyz_to_xyzw(quat_wxyz))
    rot = R.from_euler("z", angle_deg, degrees=True) * rot
    return xyzw_to_wxyz(rot.as_quat())
