"""
PD velocity controller utilities.

This is a pure math module with no Isaac Sim or USD dependency.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional


class PDVelocityController:
    """
    SE(3) PD velocity controller.

    Given current and target poses, compute a 6D twist containing linear and
    angular velocity. Derivative terms damp high-frequency jitter.

    Args:
        kp_pos: Position proportional gain.
        kd_pos: Position derivative gain.
        kp_rot: Rotation proportional gain.
        kd_rot: Rotation derivative gain.
        dt: Control time step in seconds.
        max_lin_vel: Linear velocity limit in meters per second.
        max_ang_vel: Angular velocity limit in radians per second.
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
        """Reset derivative history at the start of an episode."""
        self._prev_pos_err = None
        self._prev_rot_err = None

    def compute_twist(
        self,
        current_tf: np.ndarray,
        target_tf: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a 6D twist from the current pose to the target pose.

        Args:
            current_tf: Current 4x4 transform matrix.
            target_tf: Target 4x4 transform matrix.

        Returns:
            Clipped 6D twist ``[vx, vy, vz, wx, wy, wz]``.
        """
        # Position error.
        pos_err = target_tf[:3, 3] - current_tf[:3, 3]
        if self._prev_pos_err is None:
            pos_d_err = np.zeros(3)
        else:
            pos_d_err = (pos_err - self._prev_pos_err) / self.dt
        self._prev_pos_err = pos_err

        # Rotation error as a rotation vector.
        rot_diff = target_tf[:3, :3] @ current_tf[:3, :3].T
        rot_err = R.from_matrix(rot_diff).as_rotvec()
        if self._prev_rot_err is None:
            rot_d_err = np.zeros(3)
        else:
            rot_d_err = (rot_err - self._prev_rot_err) / self.dt
        self._prev_rot_err = rot_err

        # PD output.
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
    Apply a 6D twist to a pose and return the new pose.

    Args:
        current_pos: Current position ``[x, y, z]``.
        current_quat_xyzw: Current quaternion in SciPy ``[x, y, z, w]`` order.
        twist: 6D twist [dx, dy, dz, drx, dry, drz]
        local_rotation: If true, interpret the twist in the end-effector frame;
            otherwise interpret it in the world frame.

    Returns:
        New 7D pose ``[x, y, z, qx, qy, qz, qw]`` in xyzw quaternion order.
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
