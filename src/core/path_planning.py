"""
安全路径生成。

纯数学模块，无 Isaac Sim / USD 依赖。
来源：src/lohobench/mimicgen/mimicgen_utils.py 中的 generate_adaptive_safe_path。
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
    自适应安全路径生成器。

    逻辑：
    1. 强制将终点 Z 坐标提升到 min_z 以上，防止直接插入桌面。
    2. 根据线速度和角速度动态计算所需步数。
    3. 生成线性插值路径（位置线性 + 旋转 SLERP）。

    Args:
        T_start: 起始 4x4 变换矩阵
        T_end: 终止 4x4 变换矩阵
        min_z: 终点最低安全高度（米），默认 0.25m
        lin_speed: 目标线速度（m/s），用于计算步数
        ang_speed: 目标角速度（rad/s），用于计算步数
        min_steps: 最小步数下限
        dt: 仿真时间步长（秒）

    Returns:
        变换矩阵列表，长度由距离和速度动态决定
    """
    pos_start = T_start[:3, 3]
    pos_end = T_end[:3, 3].copy()

    # 强制终点高度不低于 min_z
    pos_end[2] = max(pos_end[2], min_z)

    T_end_adjusted = T_end.copy()
    T_end_adjusted[:3, 3] = pos_end

    # 线性距离
    dist = np.linalg.norm(pos_end - pos_start)

    # 角距离
    rot_diff = T_end_adjusted[:3, :3] @ T_start[:3, :3].T
    rot_angle = np.linalg.norm(R.from_matrix(rot_diff).as_rotvec())

    # 动态步数
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
    带 Z 轴下限约束的标准插值路径。

    注意：此函数在 LoHoBench mimicgen_replay.py 中定义但从未被调用，
    已被 generate_adaptive_safe_path 取代。保留此实现供参考。

    Args:
        T_start: 起始变换矩阵
        T_end: 终止变换矩阵
        steps: 固定步数
        min_z: 路径上每个点的最低 Z 高度

    Returns:
        变换矩阵列表
    """
    T_list = interpolate_transforms(T_start, T_end, steps)
    result = []
    for T in T_list:
        T = T.copy()
        T[2, 3] = max(T[2, 3], min_z)
        result.append(T)
    return result
