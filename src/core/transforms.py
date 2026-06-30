"""
SE(3) 变换矩阵操作。

纯数学模块，无 Isaac Sim / USD 依赖。
统一了 LoHoBench 中分散在以下位置的重复实现：
- src/lohobench/mimicgen/mimicgen_utils.py
- src/lohobench/utils/transform_utils.py
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


def create_transform_matrix(pos: np.ndarray, rot_quat_xyzw: np.ndarray) -> np.ndarray:
    """
    从位置和旋转四元数构造 4x4 SE(3) 变换矩阵。

    Args:
        pos: 位置向量 [x, y, z]
        rot_quat_xyzw: 旋转四元数 [x, y, z, w]（SciPy 格式）

    Returns:
        4x4 变换矩阵
    """
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(rot_quat_xyzw).as_matrix()
    return mat


def decompose_transform_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 4x4 变换矩阵分解为位置和旋转四元数。

    Args:
        mat: 4x4 变换矩阵

    Returns:
        (pos, quat_xyzw): 位置 [x,y,z] 和四元数 [x,y,z,w]
    """
    pos = mat[:3, 3].copy()
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, quat_xyzw


def compute_relative_transform(source_mat: np.ndarray, ref_mat: np.ndarray) -> np.ndarray:
    """
    计算 source 相对于 ref 的变换。
    T_rel = T_ref^{-1} * T_src

    Args:
        source_mat: 源坐标系的世界变换矩阵
        ref_mat: 参考坐标系的世界变换矩阵

    Returns:
        相对变换矩阵 T_rel
    """
    return np.linalg.inv(ref_mat) @ source_mat


def apply_relative_transform(ref_mat: np.ndarray, rel_mat: np.ndarray) -> np.ndarray:
    """
    将相对变换应用到新的参考坐标系，得到新的世界坐标。
    T_src_new = T_ref_new * T_rel

    Args:
        ref_mat: 新参考坐标系的世界变换矩阵
        rel_mat: 相对变换矩阵

    Returns:
        新的世界变换矩阵
    """
    return ref_mat @ rel_mat


def pose_to_transform(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """
    从 (pos, quat_wxyz) 构造变换矩阵。
    兼容 LoHoBench transform_utils.py 中的 wxyz 格式。

    Args:
        pos: 位置 [x, y, z]
        quat_wxyz: 四元数 [w, x, y, z]（Isaac Sim 格式）

    Returns:
        4x4 变换矩阵
    """
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return create_transform_matrix(pos, quat_xyzw)


def transform_to_pose(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将变换矩阵分解为 (pos, quat_wxyz)。
    返回 Isaac Sim 格式的四元数 [w, x, y, z]。

    Args:
        mat: 4x4 变换矩阵

    Returns:
        (pos, quat_wxyz): 位置和 wxyz 格式四元数
    """
    pos, quat_xyzw = decompose_transform_matrix(mat)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat_wxyz


def compute_delta_pose(pose1_pos: np.ndarray, pose1_quat_wxyz: np.ndarray,
                       pose2_pos: np.ndarray, pose2_quat_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个位姿之间的差分：pose1 相对于 pose2 的变换。

    Args:
        pose1_pos, pose1_quat_wxyz: 位姿 1（wxyz 格式）
        pose2_pos, pose2_quat_wxyz: 位姿 2（wxyz 格式）

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
    根据物体 1 的运动，推算物体 2 的新位姿（刚体运动传播）。
    对应 LoHoBench transform_utils.py 中的 compute_pose2。

    Returns:
        (new_pos2, new_quat2_wxyz)
    """
    T_curr1 = pose_to_transform(current_pos1, current_quat1_wxyz)
    T_prev1 = pose_to_transform(prev_pos1, prev_quat1_wxyz)
    T_prev2 = pose_to_transform(prev_pos2, prev_quat2_wxyz)
    T_new2 = T_prev2 @ np.linalg.inv(T_prev1) @ T_curr1
    return transform_to_pose(T_new2)
