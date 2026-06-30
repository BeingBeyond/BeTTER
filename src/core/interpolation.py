"""
轨迹插值工具。

纯数学模块，无 Isaac Sim / USD 依赖。
来源：src/lohobench/mimicgen/mimicgen_utils.py 中的 interpolate_transforms。
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
    在两个 SE(3) 变换矩阵之间插值。
    - 位置：线性插值
    - 旋转：SLERP（球面线性插值）

    Args:
        start_mat: 起始 4x4 变换矩阵
        end_mat: 终止 4x4 变换矩阵
        num_steps: 插值步数（含首尾）

    Returns:
        长度为 num_steps 的变换矩阵列表
    """
    start_pos, start_quat = decompose_transform_matrix(start_mat)
    end_pos, end_quat = decompose_transform_matrix(end_mat)

    ratios = np.linspace(0.0, 1.0, num_steps)

    # 位置线性插值
    interp_positions = np.outer(1 - ratios, start_pos) + np.outer(ratios, end_pos)

    # 旋转 SLERP
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
    在两个位姿之间做 SLERP 插值（alpha=0 返回 pose1，alpha=1 返回 pose2）。
    输入四元数为 Isaac Sim [w, x, y, z] 格式。

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
