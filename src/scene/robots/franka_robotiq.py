from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import importlib.util

import numpy as np

from .base import RobotBase
from .spec import EmbodimentSpec


class FrankaRobotiqRobot(RobotBase):
    """Franka+Robotiq embodiment with normalized legacy action interface."""

    def __init__(self, instance_id: str, prim_path: str, spec: EmbodimentSpec) -> None:
        super().__init__(instance_id=instance_id, prim_path=prim_path, spec=spec)
        self._world_position = np.zeros(3, dtype=np.float32)
        self._world_orientation_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._eef_pose = np.zeros(6, dtype=np.float32)
        self._gripper_closeness = np.zeros(1, dtype=np.float32)
        self._articulation = None

    def initialize(self) -> None:
        from omni.isaac.core.articulations import Articulation  # type: ignore[import]

        self._articulation = Articulation(self._prim_path)
        self._articulation.initialize()
        self._initialized = True

    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._world_position.copy(), self._world_orientation_wxyz.copy()

    def set_world_pose(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        if position is not None:
            pos = np.asarray(position, dtype=np.float32)
            if pos.shape != (3,):
                raise ValueError(f"position must have shape (3,), got {pos.shape}")
            self._world_position = pos.copy()

        if orientation is not None:
            quat = np.asarray(orientation, dtype=np.float32)
            if quat.shape != (4,):
                raise ValueError(f"orientation must have shape (4,), got {quat.shape}")
            self._world_orientation_wxyz = quat.copy()

    def get_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        position = self._eef_pose[:3].copy()
        orientation = self._eef_pose[3:].copy()
        return position, orientation

    def set_eef_target(self, eef_pose: np.ndarray) -> None:
        pose = np.asarray(eef_pose, dtype=np.float32)
        if pose.shape != (6,):
            raise ValueError(f"eef_pose must have shape (6,), got {pose.shape}")
        self._eef_pose = pose.copy()

    def set_gripper_closeness(self, closeness: np.ndarray) -> None:
        command = np.asarray(closeness, dtype=np.float32)
        if command.shape != (1,):
            raise ValueError(
                f"gripper_closeness must have shape (1,), got {command.shape}"
            )
        self._gripper_closeness = np.clip(command, 0.0, 1.0)
        if self._articulation is not None:
            self.set_gripper_target_from_closeness(float(self._gripper_closeness[0]))

    def get_joint_positions(self) -> Dict[str, float]:
        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        if not dof_names:
            return {}

        positions = articulation.get_joint_positions()
        if positions is None:
            return {}

        positions_array = np.asarray(positions, dtype=float).reshape(-1)
        if positions_array.shape[0] != len(dof_names):
            raise RuntimeError(
                f"Joint position size mismatch for '{self._prim_path}': "
                f"{positions_array.shape[0]} values for {len(dof_names)} joints."
            )

        return {
            joint_name: float(position)
            for joint_name, position in zip(dof_names, positions_array)
        }

    def set_joint_positions(self, positions: Dict[str, float]) -> None:
        articulation = self._require_articulation()
        if not positions:
            return

        dof_names = self._get_dof_names(articulation)
        unknown_joints = sorted(set(positions.keys()) - set(dof_names))
        if unknown_joints:
            raise KeyError(
                f"Unknown joints for articulation '{self._prim_path}': {unknown_joints}. "
                f"Available joints: {dof_names}"
            )

        joint_indices = np.asarray(
            [articulation.get_dof_index(joint_name) for joint_name in positions.keys()],
            dtype=np.int64,
        )
        joint_targets = np.asarray(
            [float(positions[joint_name]) for joint_name in positions.keys()],
            dtype=np.float32,
        )
        articulation.set_joint_positions(
            positions=joint_targets,
            joint_indices=joint_indices,
        )

    def get_joint_velocities(self) -> Dict[str, float]:
        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        if not dof_names:
            return {}

        velocities = articulation.get_joint_velocities()
        if velocities is None:
            return {}

        velocities_array = np.asarray(velocities, dtype=float).reshape(-1)
        if velocities_array.shape[0] != len(dof_names):
            raise RuntimeError(
                f"Joint velocity size mismatch for '{self._prim_path}': "
                f"{velocities_array.shape[0]} values for {len(dof_names)} joints."
            )

        return {
            joint_name: float(velocity)
            for joint_name, velocity in zip(dof_names, velocities_array)
        }

    def set_joint_velocities(self, velocities: Dict[str, float]) -> None:
        articulation = self._require_articulation()
        if not velocities:
            return

        dof_names = self._get_dof_names(articulation)
        unknown_joints = sorted(set(velocities.keys()) - set(dof_names))
        if unknown_joints:
            raise KeyError(
                f"Unknown joints for articulation '{self._prim_path}': {unknown_joints}. "
                f"Available joints: {dof_names}"
            )

        joint_indices = np.asarray(
            [articulation.get_dof_index(joint_name) for joint_name in velocities.keys()],
            dtype=np.int64,
        )
        joint_velocities = np.asarray(
            [float(velocities[joint_name]) for joint_name in velocities.keys()],
            dtype=np.float32,
        )
        articulation.set_joint_velocities(
            velocities=joint_velocities,
            joint_indices=joint_indices,
        )

    def set_joint_positions_by_order(
        self,
        positions: Sequence[float],
        *,
        joint_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        names = list(dof_names if joint_names is None else [str(name) for name in joint_names])
        values = np.asarray(positions, dtype=np.float32).reshape(-1)

        if len(values) != len(names):
            raise ValueError(
                f"Joint position sequence length mismatch for '{self._prim_path}': "
                f"{len(values)} values for {len(names)} joint names."
            )
        if joint_names is None and len(values) != len(dof_names):
            raise ValueError(
                f"Full joint position state for '{self._prim_path}' must have "
                f"{len(dof_names)} values, got {len(values)}."
            )

        mapping = {name: float(value) for name, value in zip(names, values)}
        self.set_joint_positions(mapping)
        return mapping

    def set_joint_position_targets(self, positions: Dict[str, float]) -> Dict[str, float]:
        articulation = self._require_articulation()
        if not positions:
            return {}
        if not hasattr(articulation, "get_articulation_controller"):
            return {}

        controller = articulation.get_articulation_controller()
        if not hasattr(controller, "apply_action"):
            return {}

        dof_names = self._get_dof_names(articulation)
        unknown_joints = sorted(set(positions.keys()) - set(dof_names))
        if unknown_joints:
            raise KeyError(
                f"Unknown joints for articulation '{self._prim_path}': {unknown_joints}. "
                f"Available joints: {dof_names}"
            )

        joint_indices = np.asarray(
            [articulation.get_dof_index(joint_name) for joint_name in positions.keys()],
            dtype=np.int64,
        )
        joint_targets = np.asarray(
            [float(positions[joint_name]) for joint_name in positions.keys()],
            dtype=np.float32,
        )
        controller.apply_action(
            _articulation_action(
                joint_positions=joint_targets,
                joint_indices=joint_indices,
            )
        )
        return {
            joint_name: float(value)
            for joint_name, value in zip(positions.keys(), joint_targets)
        }

    def set_joint_velocities_by_order(
        self,
        velocities: Sequence[float],
        *,
        joint_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        names = list(dof_names if joint_names is None else [str(name) for name in joint_names])
        values = np.asarray(velocities, dtype=np.float32).reshape(-1)

        if len(values) != len(names):
            raise ValueError(
                f"Joint velocity sequence length mismatch for '{self._prim_path}': "
                f"{len(values)} values for {len(names)} joint names."
            )
        if joint_names is None and len(values) != len(dof_names):
            raise ValueError(
                f"Full joint velocity state for '{self._prim_path}' must have "
                f"{len(dof_names)} values, got {len(values)}."
            )

        mapping = {name: float(value) for name, value in zip(names, values)}
        self.set_joint_velocities(mapping)
        return mapping

    def get_gripper_joint_positions(self) -> Dict[str, float]:
        joint_positions = self.get_joint_positions()
        return {
            joint_name: joint_positions[joint_name]
            for joint_name in self._spec.joint_names.get("gripper", [])
            if joint_name in joint_positions
        }

    def set_gripper_joint_positions(self, positions: Sequence[float]) -> Dict[str, float]:
        gripper_joint_names = list(self._spec.joint_names.get("gripper", []))
        values = np.asarray(positions, dtype=np.float32).reshape(-1)
        if len(values) != len(gripper_joint_names):
            if len(gripper_joint_names) == 1 and len(values) > 1:
                # Legacy LoHoBench Robotiq states stored both active knuckle targets.
                # The canonical PhysX-mimic model exposes only finger_joint as active.
                values = values[:1]
            else:
                raise ValueError(
                    f"Expected {len(gripper_joint_names)} gripper joint values for "
                    f"'{self._prim_path}', got {len(values)}."
                )

        mapping = {
            joint_name: float(value)
            for joint_name, value in zip(gripper_joint_names, values)
        }
        self.set_joint_positions(mapping)
        return mapping

    def set_gripper_target_from_closeness(self, closeness: float) -> Dict[str, float]:
        profile = self._get_gripper_action_profile()
        alpha = float(np.clip(closeness, 0.0, 1.0))
        opened = np.asarray(profile["opened_positions"], dtype=np.float32)
        closed = np.asarray(profile["closed_positions"], dtype=np.float32)
        target = opened + alpha * (closed - opened)
        return self._set_gripper_profile_joint_positions(profile, target)

    def set_gripper_target_from_legacy_action(
        self,
        gripper_action: Any,
    ) -> Dict[str, float]:
        profile = self._get_gripper_action_profile()
        action_values = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
        if len(action_values) != 1:
            raise ValueError(
                f"Expected one legacy gripper_action value for '{self._prim_path}', "
                f"got {len(action_values)}."
            )

        target = (
            profile["closed_positions"]
            if float(action_values[0]) > float(profile["close_threshold"])
            else profile["opened_positions"]
        )
        return self._set_gripper_profile_joint_positions(profile, target)

    def restore_legacy_mimicgen_action_target(
        self,
        action: Dict[str, Any],
        *,
        zero_velocities: bool = True,
    ) -> Dict[str, Any]:
        """Restore a LoHoBench/MimicGen action target onto this articulation.

        MimicGen steps store measured robot state in ``step["robot"]`` and a
        command target in ``step["action"]["execution"]``. The latter contains
        seven Franka arm joint targets plus a binary Robotiq ``gripper_action``
        using LoHoBench's convention: values greater than the configured
        threshold close the gripper; smaller values open it.
        """
        execution = action.get("execution", action)
        if "joint_positions" not in execution:
            raise KeyError("Legacy action target is missing 'joint_positions'.")

        arm_joint_names = list(self._spec.joint_names.get("arm", []))
        arm_joint_positions = np.asarray(
            execution["joint_positions"],
            dtype=np.float32,
        ).reshape(-1)
        restored_positions = self.set_joint_positions_by_order(
            arm_joint_positions,
            joint_names=arm_joint_names,
        )

        gripper_target_positions = {}
        if "gripper_action" in execution:
            gripper_target_positions = self.set_gripper_target_from_legacy_action(
                execution["gripper_action"]
            )
            restored_positions.update(gripper_target_positions)

        restored_joint_names = list(restored_positions.keys())
        restored_position_targets = self.set_joint_position_targets(restored_positions)
        restored_velocities = {}
        if zero_velocities:
            restored_velocities = {name: 0.0 for name in restored_joint_names}
            self.set_joint_velocities(restored_velocities)

        return {
            "input_arm_joint_count": int(len(arm_joint_positions)),
            "restored_joint_count": int(len(restored_joint_names)),
            "restored_joint_names": restored_joint_names,
            "restored_positions": restored_positions,
            "restored_position_targets": restored_position_targets,
            "restored_velocities": restored_velocities,
            "gripper_target_positions": gripper_target_positions,
            "used_gripper_action_target": bool(gripper_target_positions),
        }

    def restore_legacy_mimicgen_robot_state(
        self,
        robot_state: Dict[str, Any],
        *,
        zero_velocities: bool = True,
    ) -> Dict[str, Any]:
        """Restore LoHoBench/MimicGen robot state onto the current articulation.

        LoHoBench Robotiq recordings store a full 13-DoF joint vector in
        ``robot_state["joint_positions"]``. The first seven entries are Franka arm
        joints and the remaining entries are the runtime Robotiq gripper DoFs in
        the articulation's DoF order. Older or partial records may only contain
        seven arm joints plus a legacy ``gripper_state``; that path is treated as
        a compatibility fallback and may collapse two old gripper state values to
        the single active ``finger_joint`` command value.
        """
        if "joint_positions" not in robot_state:
            raise KeyError("Legacy robot state is missing 'joint_positions'.")

        articulation = self._require_articulation()
        dof_names = self._get_dof_names(articulation)
        full_joint_positions = np.asarray(
            robot_state["joint_positions"],
            dtype=np.float32,
        ).reshape(-1)

        used_gripper_state_fallback = False
        if len(full_joint_positions) == len(dof_names):
            restored_positions = self.set_joint_positions_by_order(full_joint_positions)
            restored_joint_names = list(restored_positions.keys())
        elif len(full_joint_positions) == len(self._spec.joint_names.get("arm", [])):
            arm_joint_names = list(self._spec.joint_names.get("arm", []))
            restored_positions = self.set_joint_positions_by_order(
                full_joint_positions,
                joint_names=arm_joint_names,
            )
            restored_joint_names = list(restored_positions.keys())

            if "gripper_state" in robot_state:
                restored_positions.update(
                    self.set_gripper_joint_positions(robot_state["gripper_state"])
                )
                restored_joint_names = list(restored_positions.keys())
                used_gripper_state_fallback = True
        else:
            raise ValueError(
                f"Unsupported legacy joint_positions length for '{self._prim_path}': "
                f"{len(full_joint_positions)}. Expected full runtime DoF count "
                f"{len(dof_names)} or arm count {len(self._spec.joint_names.get('arm', []))}."
            )

        restored_position_targets = self.set_joint_position_targets(restored_positions)
        restored_velocities = {}
        if zero_velocities:
            restored_velocities = {name: 0.0 for name in restored_joint_names}
            self.set_joint_velocities(restored_velocities)
        elif "joint_velocities" in robot_state:
            full_joint_velocities = np.asarray(
                robot_state["joint_velocities"],
                dtype=np.float32,
            ).reshape(-1)
            if len(full_joint_velocities) == len(dof_names):
                restored_velocities = self.set_joint_velocities_by_order(
                    full_joint_velocities
                )
            elif len(full_joint_velocities) == len(restored_joint_names):
                restored_velocities = self.set_joint_velocities_by_order(
                    full_joint_velocities,
                    joint_names=restored_joint_names,
                )
            else:
                raise ValueError(
                    f"Unsupported legacy joint_velocities length for '{self._prim_path}': "
                    f"{len(full_joint_velocities)}."
                )

        return {
            "input_joint_count": int(len(full_joint_positions)),
            "runtime_dof_count": int(len(dof_names)),
            "restored_joint_count": int(len(restored_joint_names)),
            "restored_joint_names": restored_joint_names,
            "restored_positions": restored_positions,
            "restored_position_targets": restored_position_targets,
            "restored_velocities": restored_velocities,
            "used_gripper_state_fallback": used_gripper_state_fallback,
        }

    def _require_articulation(self):
        if self._articulation is None:
            raise RuntimeError(
                f"FrankaRobotiqRobot '{self._prim_path}' is not initialized."
            )
        if (
            hasattr(self._articulation, "handles_initialized")
            and not self._articulation.handles_initialized
        ):
            self._articulation.initialize()
        return self._articulation

    def _get_gripper_action_profile(self) -> Dict[str, Any]:
        profile = self._spec.runtime_control.get("gripper_action_profile")
        if not isinstance(profile, dict):
            raise RuntimeError(
                f"Embodiment '{self.embodiment_id}' does not define "
                "runtime_control.gripper_action_profile."
            )

        joint_names = [
            str(name)
            for name in profile.get(
                "joint_names",
                self._spec.joint_names.get("gripper", []),
            )
        ]
        opened_positions = np.asarray(
            profile.get("opened_positions"),
            dtype=np.float32,
        ).reshape(-1)
        closed_positions = np.asarray(
            profile.get("closed_positions"),
            dtype=np.float32,
        ).reshape(-1)
        if not joint_names:
            raise ValueError(
                f"Embodiment '{self.embodiment_id}' gripper_action_profile "
                "must define at least one joint name."
            )
        if len(opened_positions) != len(joint_names):
            raise ValueError(
                f"gripper_action_profile.opened_positions length mismatch for "
                f"'{self.embodiment_id}': {len(opened_positions)} values for "
                f"{len(joint_names)} joints."
            )
        if len(closed_positions) != len(joint_names):
            raise ValueError(
                f"gripper_action_profile.closed_positions length mismatch for "
                f"'{self.embodiment_id}': {len(closed_positions)} values for "
                f"{len(joint_names)} joints."
            )

        return {
            "joint_names": joint_names,
            "opened_positions": [float(value) for value in opened_positions],
            "closed_positions": [float(value) for value in closed_positions],
            "close_threshold": float(profile.get("close_threshold", 0.5)),
        }

    def _set_gripper_profile_joint_positions(
        self,
        profile: Dict[str, Any],
        positions: Sequence[float],
    ) -> Dict[str, float]:
        values = np.asarray(positions, dtype=np.float32).reshape(-1)
        joint_names = list(profile["joint_names"])
        if len(values) != len(joint_names):
            raise ValueError(
                f"Expected {len(joint_names)} gripper target values for "
                f"'{self._prim_path}', got {len(values)}."
            )

        mapping = {
            joint_name: float(value)
            for joint_name, value in zip(joint_names, values)
        }
        return self.set_joint_position_targets(mapping)

    @staticmethod
    def _get_dof_names(articulation) -> list[str]:
        dof_names = articulation.dof_names
        if dof_names is None:
            return []
        return [str(name) for name in dof_names]


def _articulation_action(*, joint_positions: np.ndarray, joint_indices: np.ndarray):
    if importlib.util.find_spec("isaacsim") is None:
        return SimpleNamespace(
            joint_positions=joint_positions,
            joint_indices=joint_indices,
        )

    from isaacsim.core.utils.types import ArticulationAction  # type: ignore[import]

    return ArticulationAction(
        joint_positions=joint_positions,
        joint_indices=joint_indices,
    )
