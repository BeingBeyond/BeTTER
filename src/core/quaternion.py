"""
四元数工具函数。

统一了 LoHoBench 中分散在以下位置的重复实现：
- src/lohobench/mimicgen/mimicgen_utils.py (wxyz_to_xyzw, xyzw_to_wxyz)
- src/lohobench/env/utils/quaternion_utils.py (同上)
- src/lohobench/utils/transform_utils.py (quaternion_multiply, quaternion_rotate_vector)

约定：
- Isaac Sim / USD 使用 [w, x, y, z] 格式
- SciPy 使用 [x, y, z, w] 格式
- 本模块函数名中 wxyz = Isaac Sim 格式，xyzw = SciPy 格式
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union

ArrayLike = Union[np.ndarray, list, tuple]


def wxyz_to_xyzw(quat: ArrayLike) -> np.ndarray:
    """
    Isaac Sim 格式 [w, x, y, z] → SciPy 格式 [x, y, z, w]

    Example:
        >>> wxyz_to_xyzw([1, 0, 0, 0])
        array([0., 0., 0., 1.])
    """
    q = np.asarray(quat, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw_to_wxyz(quat: ArrayLike) -> np.ndarray:
    """
    SciPy 格式 [x, y, z, w] → Isaac Sim 格式 [w, x, y, z]

    Example:
        >>> xyzw_to_wxyz([0, 0, 0, 1])
        array([1., 0., 0., 0.])
    """
    q = np.asarray(quat, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]])


def normalize_quat(quat: ArrayLike) -> np.ndarray:
    """归一化四元数（任意格式）。"""
    q = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Cannot normalize near-zero quaternion")
    return q / norm


def quat_multiply_wxyz(q1: ArrayLike, q2: ArrayLike) -> np.ndarray:
    """
    四元数乘法 q1 * q2，输入输出均为 [w, x, y, z] 格式。

    注意：推荐使用 scipy.spatial.transform.Rotation 代替此函数，
    此实现仅为兼容 LoHoBench 旧代码。
    """
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    # 转为 xyzw，用 scipy 计算，再转回
    r1 = R.from_quat(wxyz_to_xyzw(q1))
    r2 = R.from_quat(wxyz_to_xyzw(q2))
    return xyzw_to_wxyz((r1 * r2).as_quat())


def rotate_vector_by_quat_wxyz(quat_wxyz: ArrayLike, vec: ArrayLike) -> np.ndarray:
    """
    用四元数旋转向量：v' = q * v * q^{-1}
    输入四元数为 [w, x, y, z] 格式。
    """
    r = R.from_quat(wxyz_to_xyzw(quat_wxyz))
    return r.apply(np.asarray(vec, dtype=float))


def adjust_orientation_wxyz(quat_wxyz: ArrayLike) -> np.ndarray:
    """
    调整朝向，使 X 轴正方向朝前（若 X 分量为负则绕 Z 轴旋转 180°）。
    对应 LoHoBench transform_utils.py 中的 adjust_orientation。

    Args:
        quat_wxyz: 四元数 [w, x, y, z]

    Returns:
        调整后的四元数 [w, x, y, z]
    """
    rot = R.from_quat(wxyz_to_xyzw(quat_wxyz))
    if rot.apply(np.array([1.0, 0.0, 0.0]))[0] < 0:
        rot = R.from_euler("z", 180, degrees=True) * rot
    return xyzw_to_wxyz(rot.as_quat())


def rotate_orientation_by_z_wxyz(quat_wxyz: ArrayLike, angle_deg: float) -> np.ndarray:
    """
    绕 Z 轴旋转四元数指定角度。
    对应 LoHoBench transform_utils.py 中的 rot_orientation_by_z_axis。

    Args:
        quat_wxyz: 四元数 [w, x, y, z]
        angle_deg: 旋转角度（度）

    Returns:
        旋转后的四元数 [w, x, y, z]
    """
    rot = R.from_quat(wxyz_to_xyzw(quat_wxyz))
    rot = R.from_euler("z", angle_deg, degrees=True) * rot
    return xyzw_to_wxyz(rot.as_quat())
