"""
PD 速度控制器。

纯数学模块，无 Isaac Sim / USD 依赖。
来源：scripts/mimicgen_replay.py 中的 MimicGenPDVelocityAgent。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional


class PDVelocityController:
    """
    SE(3) PD 速度控制器。

    给定当前位姿和目标位姿，计算 6D twist（线速度 + 角速度）。
    带微分项（D 项）用于抑制高频抖动。

    Args:
        kp_pos: 位置比例增益
        kd_pos: 位置微分增益
        kp_rot: 旋转比例增益
        kd_rot: 旋转微分增益
        dt: 控制时间步长（秒）
        max_lin_vel: 线速度限幅（m/s）
        max_ang_vel: 角速度限幅（rad/s）
    """

    def __init__(
        self,
        kp_pos: float = 4.0,
        kd_pos: float = 0.4,
        kp_rot: float = 1.5,
        kd_rot: float = 0.2,
        dt: float = 1 / 30.0,
        max_lin_vel: float = 1.0,
        max_ang_vel: float = 3.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_rot = kp_rot
        self.kd_rot = kd_rot
        self.dt = dt
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel

        self._prev_pos_err: Optional[np.ndarray] = None
        self._prev_rot_err: Optional[np.ndarray] = None

    def reset(self):
        """重置微分项历史（每个 episode 开始时调用）。"""
        self._prev_pos_err = None
        self._prev_rot_err = None

    def compute_twist(
        self,
        current_tf: np.ndarray,
        target_tf: np.ndarray,
    ) -> np.ndarray:
        """
        计算从当前位姿到目标位姿的 6D twist。

        Args:
            current_tf: 当前 4x4 变换矩阵
            target_tf: 目标 4x4 变换矩阵

        Returns:
            6D twist [vx, vy, vz, wx, wy, wz]，已限幅
        """
        # 位置误差
        pos_err = target_tf[:3, 3] - current_tf[:3, 3]
        if self._prev_pos_err is None:
            pos_d_err = np.zeros(3)
        else:
            pos_d_err = (pos_err - self._prev_pos_err) / self.dt
        self._prev_pos_err = pos_err

        # 旋转误差（旋转向量）
        rot_diff = target_tf[:3, :3] @ current_tf[:3, :3].T
        rot_err = R.from_matrix(rot_diff).as_rotvec()
        if self._prev_rot_err is None:
            rot_d_err = np.zeros(3)
        else:
            rot_d_err = (rot_err - self._prev_rot_err) / self.dt
        self._prev_rot_err = rot_err

        # PD 输出
        lin_vel = self.kp_pos * pos_err + self.kd_pos * pos_d_err
        ang_vel = self.kp_rot * rot_err + self.kd_rot * rot_d_err

        return np.concatenate([
            np.clip(lin_vel, -self.max_lin_vel, self.max_lin_vel),
            np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel),
        ])


def apply_twist_to_pose(
    current_pos: np.ndarray,
    current_quat_xyzw: np.ndarray,
    twist: np.ndarray,
    local_rotation: bool = False,
) -> np.ndarray:
    """
    将 6D twist 应用到当前位姿，得到新位姿。

    Args:
        current_pos: 当前位置 [x, y, z]
        current_quat_xyzw: 当前四元数 [x, y, z, w]（SciPy 格式）
        twist: 6D twist [dx, dy, dz, drx, dry, drz]
        local_rotation: True 表示 twist 在末端坐标系下，False 表示在世界坐标系下

    Returns:
        新位姿 [x, y, z, qx, qy, qz, qw]（7D，xyzw 格式）
    """
    delta_pos = twist[:3].copy()

    if local_rotation:
        r_curr = R.from_quat(current_quat_xyzw)
        delta_pos = r_curr.apply(delta_pos)

    target_pos = current_pos + delta_pos

    delta_rot_vec = twist[3:]
    if np.linalg.norm(delta_rot_vec) < 1e-6:
        target_quat = current_quat_xyzw
    else:
        delta_rot = R.from_rotvec(delta_rot_vec)
        current_rot = R.from_quat(current_quat_xyzw)
        if local_rotation:
            target_rot = current_rot * delta_rot
        else:
            target_rot = delta_rot * current_rot
        target_quat = target_rot.as_quat()

    return np.concatenate([target_pos, target_quat])
