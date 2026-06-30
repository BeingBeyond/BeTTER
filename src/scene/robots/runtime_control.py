from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .spec import EmbodimentSpec


def load_controller_profile(profiles_path: str | Path, profile_id: str) -> Dict[str, Any]:
    path = Path(profiles_path)
    if not path.exists():
        raise FileNotFoundError(f"Controller profiles file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", [])
    for profile in profiles:
        if profile.get("profile_id") == profile_id:
            return dict(profile)

    available = sorted(str(profile.get("profile_id")) for profile in profiles)
    raise KeyError(
        f"Controller profile '{profile_id}' not found in {path}. "
        f"Available profiles: {available}"
    )


def with_gain_overrides(
    profile: Dict[str, Any],
    *,
    arm_kp: Optional[float] = None,
    arm_kd: Optional[float] = None,
) -> Dict[str, Any]:
    if arm_kp is None and arm_kd is None:
        return profile

    overridden = dict(profile)
    groups = []
    for group in profile.get("joint_groups", []):
        updated = dict(group)
        joint_names = [str(name) for name in group.get("joint_names", [])]
        is_arm_group = bool(joint_names) and all(
            name.startswith("panda_joint") for name in joint_names
        )
        if is_arm_group:
            if arm_kp is not None:
                updated["stiffness"] = float(arm_kp)
            if arm_kd is not None:
                updated["damping"] = float(arm_kd)
        groups.append(updated)

    overridden["joint_groups"] = groups
    overridden["overrides"] = {
        "arm_kp": None if arm_kp is None else float(arm_kp),
        "arm_kd": None if arm_kd is None else float(arm_kd),
    }
    return overridden


def resolve_runtime_control(
    embodiment: EmbodimentSpec,
    *,
    controller_profile_id: Optional[str] = None,
    disable_robot_gravity: Optional[bool] = None,
) -> Dict[str, Any]:
    runtime_control = dict(embodiment.runtime_control)
    if controller_profile_id is not None:
        runtime_control["controller_profile_id"] = str(controller_profile_id)
    if disable_robot_gravity is not None:
        runtime_control["disable_robot_gravity"] = bool(disable_robot_gravity)
    return runtime_control


def _to_numpy_1d(articulation_view, values, name: str, expected_len: int) -> np.ndarray:
    array = np.asarray(
        articulation_view._backend_utils.to_numpy(values),
        dtype=float,
    ).reshape(-1)
    if array.shape != (expected_len,):
        raise RuntimeError(
            f"Unexpected {name} readback shape. Expected ({expected_len},), got {array.shape}."
        )
    return array


def _require_articulation(robot):
    if hasattr(robot, "_require_articulation"):
        return robot._require_articulation()

    if (
        hasattr(robot, "handles_initialized")
        and not robot.handles_initialized
        and hasattr(robot, "initialize")
    ):
        robot.initialize()
    return robot


def _get_dof_names(robot, articulation) -> list[str]:
    if hasattr(robot, "_get_dof_names"):
        return robot._get_dof_names(articulation)

    dof_names = articulation.dof_names
    if dof_names is None:
        return []
    return [str(name) for name in dof_names]


def _prim_path(robot, articulation) -> str:
    return str(
        getattr(robot, "prim_path", getattr(articulation, "prim_path", "<unknown>"))
    )


def _read_joint_controller_profile(robot, applied_groups: list[dict]) -> tuple[list[dict], dict]:
    articulation = _require_articulation(robot)
    controller = articulation.get_articulation_controller()
    articulation_view = articulation._articulation_view
    dof_names = _get_dof_names(robot, articulation)
    dof_count = len(dof_names)

    readback_kps, readback_kds = controller.get_gains()
    readback_kps = _to_numpy_1d(articulation_view, readback_kps, "stiffness", dof_count)
    readback_kds = _to_numpy_1d(articulation_view, readback_kds, "damping", dof_count)
    readback_efforts = _to_numpy_1d(
        articulation_view,
        controller.get_max_efforts(),
        "max effort",
        dof_count,
    )
    readback_velocities = _to_numpy_1d(
        articulation_view,
        articulation_view.get_joint_max_velocities()[0],
        "max joint velocity",
        dof_count,
    )

    summary_errors = {
        "stiffness": 0.0,
        "damping": 0.0,
        "velocity_limit": 0.0,
        "effort_limit": 0.0,
    }
    groups_with_readback = []
    for group in applied_groups:
        joint_names = list(group["joint_names"])
        joint_indices = np.asarray(
            [articulation.get_dof_index(name) for name in joint_names], dtype=np.int64
        )

        stiffness_by_joint = {
            name: float(readback_kps[int(index)])
            for name, index in zip(joint_names, joint_indices)
        }
        damping_by_joint = {
            name: float(readback_kds[int(index)])
            for name, index in zip(joint_names, joint_indices)
        }
        velocity_by_joint = {
            name: float(readback_velocities[int(index)])
            for name, index in zip(joint_names, joint_indices)
        }
        effort_by_joint = {
            name: float(readback_efforts[int(index)])
            for name, index in zip(joint_names, joint_indices)
        }

        stiffness_error = (
            0.0
            if group["stiffness"] is None
            else max(
                abs(value - float(group["stiffness"]))
                for value in stiffness_by_joint.values()
            )
        )
        damping_error = (
            0.0
            if group["damping"] is None
            else max(
                abs(value - float(group["damping"]))
                for value in damping_by_joint.values()
            )
        )
        velocity_error = (
            0.0
            if group["velocity_limit"] is None
            else max(
                abs(value - float(group["velocity_limit"]))
                for value in velocity_by_joint.values()
            )
        )
        effort_error = (
            0.0
            if group["effort_limit"] is None
            else max(
                abs(value - float(group["effort_limit"]))
                for value in effort_by_joint.values()
            )
        )

        summary_errors["stiffness"] = max(summary_errors["stiffness"], stiffness_error)
        summary_errors["damping"] = max(summary_errors["damping"], damping_error)
        summary_errors["velocity_limit"] = max(
            summary_errors["velocity_limit"], velocity_error
        )
        summary_errors["effort_limit"] = max(summary_errors["effort_limit"], effort_error)

        updated_group = dict(group)
        updated_group["readback"] = {
            "stiffness_by_joint": stiffness_by_joint,
            "damping_by_joint": damping_by_joint,
            "velocity_limit_by_joint": velocity_by_joint,
            "effort_limit_by_joint": effort_by_joint,
            "max_abs_error": {
                "stiffness": float(stiffness_error),
                "damping": float(damping_error),
                "velocity_limit": float(velocity_error),
                "effort_limit": float(effort_error),
            },
        }
        groups_with_readback.append(updated_group)

    return groups_with_readback, {key: float(value) for key, value in summary_errors.items()}


def apply_joint_controller_profile(robot, profile: Dict[str, Any]) -> Dict[str, Any]:
    articulation = _require_articulation(robot)
    controller = articulation.get_articulation_controller()
    articulation_view = articulation._articulation_view
    dof_names = _get_dof_names(robot, articulation)
    dof_name_set = set(dof_names)
    full_kps, full_kds = controller.get_gains()
    full_kps = np.asarray(full_kps, dtype=np.float32).copy()
    full_kds = np.asarray(full_kds, dtype=np.float32).copy()
    if full_kps.shape != (len(dof_names),) or full_kds.shape != (len(dof_names),):
        raise RuntimeError(
            "Unexpected articulation gain shape. "
            f"Expected ({len(dof_names)},), got kps={full_kps.shape}, kds={full_kds.shape}."
        )

    applied_groups = []
    gains_changed = False
    for group in profile.get("joint_groups", []):
        joint_names = [str(name) for name in group["joint_names"]]
        missing_joints = sorted(set(joint_names) - dof_name_set)
        if missing_joints:
            raise RuntimeError(
                f"Controller profile group '{group.get('name', '<unnamed>')}' "
                f"references joints not found in articulation '{_prim_path(robot, articulation)}': "
                f"{missing_joints}. Available joints: {dof_names}"
            )

        joint_indices = np.asarray(
            [articulation.get_dof_index(name) for name in joint_names], dtype=np.int64
        )
        stiffness = None if group.get("stiffness") is None else float(group["stiffness"])
        damping = None if group.get("damping") is None else float(group["damping"])
        if stiffness is not None:
            full_kps[joint_indices] = stiffness
            gains_changed = True
        if damping is not None:
            full_kds[joint_indices] = damping
            gains_changed = True

        velocity_limit = group.get("velocity_limit")
        if velocity_limit is not None:
            velocities = np.full(
                len(joint_indices), float(velocity_limit), dtype=np.float32
            )
            articulation_view.set_max_joint_velocities(
                np.expand_dims(velocities, axis=0), joint_indices=joint_indices
            )

        effort_limit = group.get("effort_limit")
        if effort_limit is not None:
            efforts = np.full(len(joint_indices), float(effort_limit), dtype=np.float32)
            controller.set_max_efforts(efforts, joint_indices=joint_indices)

        applied_groups.append(
            {
                "name": str(group.get("name", "")),
                "joint_names": joint_names,
                "stiffness": stiffness,
                "damping": damping,
                "velocity_limit": (
                    None if velocity_limit is None else float(velocity_limit)
                ),
                "effort_limit": None if effort_limit is None else float(effort_limit),
            }
        )

    if gains_changed:
        controller.set_gains(kps=full_kps, kds=full_kds)

    groups_with_readback, readback_max_abs_error = _read_joint_controller_profile(
        robot, applied_groups
    )

    return {
        "profile_id": str(profile["profile_id"]),
        "source_files": [str(path) for path in profile.get("source_files", [])],
        "joint_groups": groups_with_readback,
        "readback_max_abs_error": readback_max_abs_error,
        "overrides": profile.get("overrides", {}),
    }


def apply_embodiment_runtime_control(
    robot,
    runtime_control: Dict[str, Any],
    *,
    controller_profiles_path: Optional[str | Path] = None,
    arm_kp: Optional[float] = None,
    arm_kd: Optional[float] = None,
) -> Dict[str, Any]:
    articulation = _require_articulation(robot)

    disable_robot_gravity = bool(runtime_control.get("disable_robot_gravity", False))
    if disable_robot_gravity:
        articulation.disable_gravity()
    else:
        articulation.enable_gravity()

    profile_report = None
    controller_profile_id = runtime_control.get("controller_profile_id")
    if controller_profile_id is not None:
        if controller_profiles_path is None:
            raise ValueError(
                "controller_profiles_path is required when runtime_control defines "
                "'controller_profile_id'."
            )
        profile = load_controller_profile(controller_profiles_path, str(controller_profile_id))
        profile = with_gain_overrides(profile, arm_kp=arm_kp, arm_kd=arm_kd)
        profile_report = apply_joint_controller_profile(robot, profile)

    return {
        "disable_robot_gravity": disable_robot_gravity,
        "controller_profile_id": (
            None if controller_profile_id is None else str(controller_profile_id)
        ),
        "controller_profile": profile_report,
    }
