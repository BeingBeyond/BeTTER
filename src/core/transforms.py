"""
SE(3) transform matrix utilities.

This is a pure math module with no Isaac Sim or USD dependency.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def create_transform_matrix(pos: np.ndarray, rot_quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Construct a 4x4 SE(3) transform matrix from position and orientation.

    Args:
        pos: Position vector ``[x, y, z]``.
        rot_quat_xyzw: Rotation quaternion in SciPy ``[x, y, z, w]`` order.

    Returns:
        A 4x4 transform matrix.
    """
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(rot_quat_xyzw).as_matrix()
    return mat


def decompose_transform_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a 4x4 transform matrix into position and orientation.

    Args:
        mat: A 4x4 transform matrix.

    Returns:
        ``(pos, quat_xyzw)`` with position ``[x, y, z]`` and SciPy quaternion
        ``[x, y, z, w]``.
    """
    pos = mat[:3, 3].copy()
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat_xyzw


def compute_relative_transform(source_mat: np.ndarray, ref_mat: np.ndarray) -> np.ndarray:
    """
    Compute the transform from ``ref_mat`` to ``source_mat``.

    T_rel = T_ref^{-1} * T_src

    Args:
        source_mat: Source frame transform in world coordinates.
        ref_mat: Reference frame transform in world coordinates.

    Returns:
        Relative transform matrix ``T_rel``.
    """
    return np.linalg.inv(ref_mat) @ source_mat


def apply_relative_transform(ref_mat: np.ndarray, rel_mat: np.ndarray) -> np.ndarray:
    """
    Apply a relative transform to a reference frame.

    T_src_new = T_ref_new * T_rel

    Args:
        ref_mat: New reference frame transform in world coordinates.
        rel_mat: Relative transform matrix.

    Returns:
        New world transform matrix.
    """
    return ref_mat @ rel_mat


def pose_to_transform(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Construct a transform matrix from position and an Isaac Sim quaternion.

    Args:
        pos: Position ``[x, y, z]``.
        quat_wxyz: Quaternion in Isaac Sim ``[w, x, y, z]`` order.

    Returns:
        A 4x4 transform matrix.
    """
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return create_transform_matrix(pos, quat_xyzw)


def transform_to_pose(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a transform matrix into position and an Isaac Sim quaternion.

    Args:
        mat: A 4x4 transform matrix.

    Returns:
        ``(pos, quat_wxyz)`` with the quaternion in ``[w, x, y, z]`` order.
    """
    pos, quat_xyzw = decompose_transform_matrix(mat)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def compute_delta_pose(pose1_pos: np.ndarray, pose1_quat_wxyz: np.ndarray,
                       pose2_pos: np.ndarray, pose2_quat_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the relative pose from ``pose2`` to ``pose1``.

    Args:
        pose1_pos, pose1_quat_wxyz: First pose, with quaternion in wxyz order.
        pose2_pos, pose2_quat_wxyz: Second pose, with quaternion in wxyz order.

    Returns:
        (delta_pos, delta_quat_wxyz)
    """
    T1 = pose_to_transform(pose1_pos, pose1_quat_wxyz)
    T2 = pose_to_transform(pose2_pos, pose2_quat_wxyz)
    T_delta = T1 @ np.linalg.inv(T2)
    return transform_to_pose(T_delta)


def propagate_pose(
    current_pos1: np.ndarray, current_quat1_wxyz: np.ndarray,
    prev_pos1: np.ndarray, prev_quat1_wxyz: np.ndarray,
    prev_pos2: np.ndarray, prev_quat2_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate one object's pose through another object's rigid motion.

    Returns:
        (new_pos2, new_quat2_wxyz)
    """
    T_curr1 = pose_to_transform(current_pos1, current_quat1_wxyz)
    T_prev1 = pose_to_transform(prev_pos1, prev_quat1_wxyz)
    T_prev2 = pose_to_transform(prev_pos2, prev_quat2_wxyz)
    T_new2 = T_prev2 @ np.linalg.inv(T_prev1) @ T_curr1
    return transform_to_pose(T_new2)
