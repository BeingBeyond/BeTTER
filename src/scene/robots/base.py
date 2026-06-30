from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .spec import EmbodimentSpec


ROBOT_RUNTIME_STATE_SCHEMA_VERSION = "better.robot_runtime_state.v1"


class RobotBase(ABC):
    """Base abstraction for embodiment-specific robot integration.

    The base class is runtime-agnostic and keeps simulator bindings optional,
    enabling pure unit tests for registry/config/action compatibility.
    """

    def __init__(self, instance_id: str, prim_path: str, spec: EmbodimentSpec) -> None:
        self._instance_id = instance_id
        self._prim_path = prim_path
        self._spec = spec
        self._initialized = False

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def prim_path(self) -> str:
        return self._prim_path

    @property
    def embodiment_id(self) -> str:
        return self._spec.embodiment_id

    @property
    def spec(self) -> EmbodimentSpec:
        return self._spec

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Default no-op init for pure-python unit usage."""
        self._initialized = True

    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("RobotBase.get_world_pose() is not implemented")

    def set_world_pose(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError("RobotBase.set_world_pose() is not implemented")

    def get_eef_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("RobotBase.get_eef_pose() is not implemented")

    def set_eef_target(self, eef_pose: np.ndarray) -> None:
        """Set end-effector target pose from legacy 6D action vector."""
        raise NotImplementedError("RobotBase.set_eef_target() is not implemented")

    def set_gripper_closeness(self, closeness: np.ndarray) -> None:
        """Set normalized gripper closure command from legacy 1D vector.

        This is an abstract command scalar, not a calibrated jaw width or force.
        """
        raise NotImplementedError("RobotBase.set_gripper_closeness() is not implemented")

    def get_joint_positions(self) -> Dict[str, float]:
        """Return current joint positions keyed by joint name."""
        raise NotImplementedError("RobotBase.get_joint_positions() is not implemented")

    def set_joint_positions(self, positions: Dict[str, float]) -> None:
        """Set joint positions by joint-name mapping."""
        raise NotImplementedError("RobotBase.set_joint_positions() is not implemented")

    def get_joint_velocities(self) -> Dict[str, float]:
        """Return current joint velocities keyed by joint name."""
        raise NotImplementedError("RobotBase.get_joint_velocities() is not implemented")

    def set_joint_velocities(self, velocities: Dict[str, float]) -> None:
        """Set joint velocities by joint-name mapping."""
        raise NotImplementedError("RobotBase.set_joint_velocities() is not implemented")

    def capture_runtime_state(
        self,
        *,
        include_runtime_control: bool = True,
    ) -> Dict[str, Any]:
        """Capture a storage-agnostic robot state payload.

        This payload is intentionally independent of JSON, pickle, or HDF5.
        Storage layers should serialize these fields without changing their
        semantics; replay code should restore from this schema rather than from
        legacy action or dataset-specific wrappers.
        """
        joint_positions = _float_mapping(self.get_joint_positions())
        joint_velocities = _float_mapping(self.get_joint_velocities())
        joint_groups = {
            group_name: {
                "joint_names": [
                    joint_name
                    for joint_name in [str(name) for name in joint_names]
                    if joint_name in joint_positions
                ],
                "joint_positions": _select_mapping(
                    joint_positions,
                    [str(name) for name in joint_names],
                ),
                "joint_velocities": _select_mapping(
                    joint_velocities,
                    [str(name) for name in joint_names],
                ),
            }
            for group_name, joint_names in self._spec.joint_names.items()
        }

        state: Dict[str, Any] = {
            "schema_version": ROBOT_RUNTIME_STATE_SCHEMA_VERSION,
            "embodiment_id": self.embodiment_id,
            "instance_id": self.instance_id,
            "prim_path": self.prim_path,
            "runtime_dof_names": list(joint_positions),
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_groups": joint_groups,
        }
        if include_runtime_control:
            state["runtime_control"] = dict(self._spec.runtime_control)
        return state

    def restore_runtime_state(
        self,
        state: Dict[str, Any],
        *,
        require_all_joints: bool = True,
        restore_velocities: bool = True,
        zero_missing_velocities: bool = True,
        mirror_position_targets: bool = True,
    ) -> Dict[str, Any]:
        """Restore a state captured by :meth:`capture_runtime_state`.

        The input may store positions/velocities either as name-keyed mappings
        or as ordered arrays with ``runtime_dof_names``. This keeps the robot
        interface compatible with compact HDF5 array storage while preserving a
        single semantic restore path.
        """
        if not isinstance(state, dict):
            raise TypeError(f"Runtime state must be a dict, got {type(state).__name__}.")
        if "joint_positions" not in state:
            raise KeyError("Runtime state is missing 'joint_positions'.")

        runtime_joint_names = list(self.get_joint_positions())
        logged_joint_names = _state_joint_names(state)
        logged_positions = _state_float_mapping(
            state["joint_positions"],
            field_name="joint_positions",
            joint_names=logged_joint_names,
        )

        unknown_joints = sorted(set(logged_positions) - set(runtime_joint_names))
        missing_joints = sorted(set(runtime_joint_names) - set(logged_positions))
        if unknown_joints:
            raise KeyError(
                f"Runtime state contains joints not present in articulation "
                f"'{self.prim_path}': {unknown_joints}. Runtime joints: {runtime_joint_names}"
            )
        if missing_joints and require_all_joints:
            raise KeyError(
                f"Runtime state is missing articulation joints for '{self.prim_path}': "
                f"{missing_joints}. Use require_all_joints=False only for intentional "
                "subset restore."
            )

        restored_positions = {
            joint_name: logged_positions[joint_name]
            for joint_name in runtime_joint_names
            if joint_name in logged_positions
        }
        self.set_joint_positions(restored_positions)

        restored_position_targets: Dict[str, float] = {}
        if mirror_position_targets and hasattr(self, "set_joint_position_targets"):
            restored_position_targets = self.set_joint_position_targets(restored_positions)

        restored_velocities: Dict[str, float] = {}
        logged_velocity_names: set[str] = set()
        if restore_velocities:
            logged_velocities: Dict[str, float] = {}
            if "joint_velocities" in state:
                logged_velocities = _state_float_mapping(
                    state["joint_velocities"],
                    field_name="joint_velocities",
                    joint_names=logged_joint_names,
                )
                logged_velocity_names = set(logged_velocities)
            missing_velocity_joints = sorted(set(restored_positions) - set(logged_velocities))
            if missing_velocity_joints and not zero_missing_velocities:
                raise KeyError(
                    f"Runtime state is missing velocities for restored joints: "
                    f"{missing_velocity_joints}."
                )
            restored_velocities = {
                joint_name: logged_velocities.get(joint_name, 0.0)
                for joint_name in restored_positions
            }
            self.set_joint_velocities(restored_velocities)

        return {
            "schema_version": state.get("schema_version"),
            "input_joint_count": len(logged_positions),
            "runtime_dof_count": len(runtime_joint_names),
            "restored_joint_count": len(restored_positions),
            "restored_joint_names": list(restored_positions),
            "missing_joint_names": missing_joints,
            "restored_positions": restored_positions,
            "restored_position_targets": restored_position_targets,
            "restored_velocities": restored_velocities,
            "restored_recorded_velocities": bool(
                restore_velocities
                and logged_velocity_names
                and set(restored_positions).issubset(logged_velocity_names)
            ),
        }

    def apply_action(self, action: Dict[str, Any]) -> None:
        """Apply legacy action schema with compatibility validation.

        Required keys:
        - eef_pose: shape (6,)
        - gripper_closeness: shape (1,)
        """
        if "eef_pose" not in action:
            raise KeyError("Missing action key: 'eef_pose'")
        if "gripper_closeness" not in action:
            raise KeyError("Missing action key: 'gripper_closeness'")

        eef_pose = np.asarray(action["eef_pose"], dtype=np.float32)
        gripper = np.asarray(action["gripper_closeness"], dtype=np.float32)

        if eef_pose.shape != (6,):
            raise ValueError(
                f"'eef_pose' must have shape (6,), got {eef_pose.shape}"
            )
        if gripper.shape != (1,):
            raise ValueError(
                f"'gripper_closeness' must have shape (1,), got {gripper.shape}"
            )

        self.set_eef_target(eef_pose)
        self.set_gripper_closeness(gripper)


def _float_mapping(values: Dict[str, Any]) -> Dict[str, float]:
    return {str(name): float(value) for name, value in values.items()}


def _select_mapping(values: Dict[str, float], names: list[str]) -> Dict[str, float]:
    return {name: float(values[name]) for name in names if name in values}


def _state_joint_names(state: Dict[str, Any]) -> list[str] | None:
    joint_names = state.get("runtime_dof_names")
    if joint_names is None:
        return None
    return [str(name) for name in joint_names]


def _state_float_mapping(
    values: Any,
    *,
    field_name: str,
    joint_names: list[str] | None,
) -> Dict[str, float]:
    if isinstance(values, dict):
        return {str(name): float(value) for name, value in values.items()}

    if joint_names is None:
        raise KeyError(
            f"Runtime state field '{field_name}' is ordered but missing "
            "'runtime_dof_names'."
        )

    values_array = np.asarray(values, dtype=float).reshape(-1)
    if len(values_array) != len(joint_names):
        raise ValueError(
            f"Runtime state field '{field_name}' has {len(values_array)} values for "
            f"{len(joint_names)} runtime_dof_names."
        )
    return {
        joint_name: float(value)
        for joint_name, value in zip(joint_names, values_array)
    }
