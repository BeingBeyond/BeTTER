#!/usr/bin/env python3
"""Keyboard teleop feel test for FrankaRobotiq with BeTTER runtime PD control.

This script intentionally does not import Isaac Lab. Its keyboard bindings
mirror Isaac Lab's Se3Keyboard, and its Cartesian jog path uses a local
Jacobian damped-least-squares step to produce joint position targets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY_PATH = REPO_ROOT / "configs" / "embodiments" / "registry.v1.json"
CONTROLLER_PROFILES_PATH = (
    REPO_ROOT / "configs" / "embodiments" / "controller_profiles.v1.json"
)

FRANKA_ARM_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}

READY_ARM_POSE = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.0,
}

READY_GRIPPER_POSE = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "RevoluteJoint": 0.0,
    "RevoluteJoint_0": 0.0,
}


@dataclass
class JogCommandReport:
    raw_twist: list[float]
    smoothed_twist: list[float]
    commanded_joint_targets: dict[str, float]
    gripper_action: float
    gripper_target_positions: dict[str, float]
    gripper_actual_positions: dict[str, float]
    max_abs_joint_delta_rad: float


class RuntimeJacobianJogController:
    """Local Cartesian jog controller for interactive hand-feel tests."""

    def __init__(
        self,
        *,
        arm_joint_names: list[str],
        jacobian_link_name: str,
        damping: float,
        smoothing_alpha: float,
        use_virtual_target: bool,
        virtual_sync_threshold: float,
        max_joint_delta_per_control: float,
        command_frame: str,
    ) -> None:
        self.arm_joint_names = list(arm_joint_names)
        self.jacobian_link_name = str(jacobian_link_name)
        self.damping = float(damping)
        self.smoothing_alpha = float(smoothing_alpha)
        self.use_virtual_target = bool(use_virtual_target)
        self.virtual_sync_threshold = float(virtual_sync_threshold)
        self.max_joint_delta_per_control = float(max_joint_delta_per_control)
        self.command_frame = str(command_frame)
        self._virtual_joint_targets: np.ndarray | None = None
        self._last_twist = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        self._virtual_joint_targets = None
        self._last_twist = np.zeros(6, dtype=np.float64)

    def compute_and_apply(
        self,
        robot,
        *,
        raw_action: np.ndarray,
        control_dt: float,
    ) -> JogCommandReport:
        raw_action = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if raw_action.shape[0] != 7:
            raise ValueError(f"Expected 7D keyboard command, got shape {raw_action.shape}.")

        raw_twist = raw_action[:6]
        self._last_twist = raw_twist.copy()

        current_joint_positions = _joint_vector(robot, self.arm_joint_names)
        base_joint_positions = current_joint_positions
        if self.use_virtual_target:
            if self._virtual_joint_targets is None:
                self._virtual_joint_targets = current_joint_positions.copy()
            virtual_error = float(np.linalg.norm(self._virtual_joint_targets - current_joint_positions))
            if virtual_error > self.virtual_sync_threshold:
                self._virtual_joint_targets = current_joint_positions.copy()
            base_joint_positions = self._virtual_joint_targets

        twist_world = self._twist_in_world_frame(robot, raw_twist)
        jacobian = _jacobian_for_link(robot, self.jacobian_link_name, self.arm_joint_names)
        joint_velocity = _damped_least_squares(jacobian, twist_world, damping=self.damping)
        joint_delta = joint_velocity * float(control_dt)
        joint_delta = _clipped_joint_delta(
            joint_delta,
            max_norm=float(self.max_joint_delta_per_control),
        )
        target_joint_positions = _apply_joint_limits(
            self.arm_joint_names,
            base_joint_positions + joint_delta,
        )

        if self.use_virtual_target:
            self._virtual_joint_targets = target_joint_positions.copy()

        joint_targets = {
            joint_name: float(position)
            for joint_name, position in zip(self.arm_joint_names, target_joint_positions)
        }
        gripper_action = -1.0 if float(raw_action[6]) > 0.0 else 1.0
        gripper_targets = _gripper_targets_from_legacy_action(
            robot=robot,
            gripper_action=gripper_action,
        )
        joint_targets.update(gripper_targets)
        applied_targets = robot.set_joint_position_targets(joint_targets)
        gripper_actual_positions = _joint_positions_by_name(
            robot,
            list(gripper_targets.keys()),
        )

        return JogCommandReport(
            raw_twist=raw_twist.astype(float).tolist(),
            smoothed_twist=self._last_twist.astype(float).tolist(),
            commanded_joint_targets=applied_targets,
            gripper_action=float(gripper_action),
            gripper_target_positions=gripper_targets,
            gripper_actual_positions=gripper_actual_positions,
            max_abs_joint_delta_rad=float(np.max(np.abs(joint_delta))),
        )

    def _twist_in_world_frame(self, robot, twist: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=np.float64).reshape(6)
        if self.command_frame == "world":
            return twist
        if self.command_frame != "robot_base":
            raise ValueError(f"Unsupported command_frame: {self.command_frame!r}")

        _position, orientation = robot.get_world_pose()
        rotation = _quat_wxyz_to_matrix(np.asarray(orientation, dtype=np.float64))
        transformed = np.zeros(6, dtype=np.float64)
        transformed[:3] = rotation @ twist[:3]
        transformed[3:] = rotation @ twist[3:]
        return transformed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument(
        "--reference-prim-path",
        type=str,
        default="/World/BeTTERFrankaRobotiq",
        help="Prim path where robot_mimic.usd will be instanced.",
    )
    parser.add_argument(
        "--embodiment-id",
        type=str,
        default="franka_robotiq",
        help="Embodiment registry id to load. Use franka_robotiq_legacy_asset for legacy asset A/B tests.",
    )
    parser.add_argument(
        "--controller-profile-id",
        type=str,
        default=None,
        help="Override registry runtime_control.controller_profile_id for A/B controller tests.",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=1.0 / 120.0,
        help="Physics dt. Default keeps the BeTTER 120Hz physics rate.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=1.0 / 15.0,
        help="Teleop command period. Default is 15Hz, i.e. 8 physics steps at 120Hz.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Multiplies the default keyboard sensitivity.",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=None,
        help="Cartesian translational velocity per held key in m/s. Defaults to 0.1*sensitivity.",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=None,
        help="Cartesian rotational velocity per held key in rad/s. Defaults to 0.4*sensitivity.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="Deprecated compatibility option. Keyboard twist commands are not smoothed.",
    )
    parser.add_argument(
        "--dls-damping",
        type=float,
        default=0.05,
        help="Damped least-squares regularization for the runtime Jacobian jog.",
    )
    parser.add_argument(
        "--max-joint-delta-per-control",
        type=float,
        default=0.08,
        help="Clip joint delta norm per teleop command before applying targets.",
    )
    parser.add_argument(
        "--virtual-sync-threshold",
        type=float,
        default=0.5,
        help="Reset virtual target to measured joints when their L2 error exceeds this value.",
    )
    parser.add_argument(
        "--no-virtual-target",
        action="store_true",
        help="Compute each jog command from measured joints instead of integrated virtual joint target.",
    )
    parser.add_argument(
        "--command-frame",
        choices=("world", "robot_base"),
        default="world",
        help="Frame used for keyboard W/S/A/D/Q/E and rotation commands.",
    )
    parser.add_argument(
        "--jacobian-link-name",
        type=str,
        default="base_link",
        help="Runtime articulation body name whose Jacobian is used for Cartesian jogging.",
    )
    parser.add_argument(
        "--home-pose",
        choices=("ready", "usd"),
        default="ready",
        help="Start from a simple ready pose or preserve the USD default pose.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=24,
        help="Physics frames to step after loading and applying the ready pose.",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Optional auto-exit duration for smoke testing. Omit for interactive teleop.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="Print a compact status line every N teleop commands. Use 0 to disable.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Optional JSONL path for recording FrankaRobotiq state snapshots.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record one state snapshot every N teleop commands when --record-path is set.",
    )
    parser.add_argument(
        "--record-flush-every",
        type=int,
        default=1,
        help="Flush the JSONL recorder every N written snapshots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from isaacsim import SimulationApp

    launch_config: dict[str, Any] = {"headless": bool(args.headless)}
    if not bool(args.headless):
        launch_config.update(
            {
                "hide_ui": False,
                "renderer": "RaytracedLighting",
            }
        )
    simulation_app = SimulationApp(launch_config)

    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage

    from src.scene.robots import (
        FrankaRobotiqFrameResolver,
        apply_embodiment_runtime_control,
        create_robot,
        load_registry,
        resolve_embodiment_paths,
        resolve_runtime_control,
    )
    from src.scene.recording import JsonlRobotStateLogger, RobotStateSnapshotter
    from src.scene.teleop import LocalSe3Keyboard, LocalSe3KeyboardConfig

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=float(args.physics_dt),
        rendering_dt=float(args.control_dt),
    )
    world.scene.add_default_ground_plane()

    registry = load_registry(REGISTRY_PATH)
    embodiment = registry.get_embodiment(str(args.embodiment_id))
    runtime_control = resolve_runtime_control(
        embodiment,
        controller_profile_id=args.controller_profile_id,
    )
    runtime_embodiment = replace(embodiment, runtime_control=runtime_control)
    resolved_paths = resolve_embodiment_paths(registry, embodiment)
    robot_usd_path = resolved_paths["runtime_asset_path"]
    _require_file(robot_usd_path)

    add_reference_to_stage(str(robot_usd_path), prim_path=args.reference_prim_path)
    articulation_root_path = _find_articulation_root_path(args.reference_prim_path)
    robot = create_robot(
        instance_id="franka_robotiq_keyboard_pd_jog",
        prim_path=articulation_root_path,
        embodiment=runtime_embodiment,
    )

    world.reset()
    robot.initialize()
    runtime_control_report = apply_embodiment_runtime_control(
        robot,
        runtime_control,
        controller_profiles_path=CONTROLLER_PROFILES_PATH,
    )
    if args.home_pose == "ready":
        _apply_ready_pose(robot)

    for _ in range(max(0, int(args.warmup_frames))):
        world.step(render=not bool(args.headless))

    if args.home_pose == "ready":
        print(
            "[franka_robotiq_keyboard_pd_jog] home_pose "
            + json.dumps(_home_pose_report(robot), sort_keys=True),
            flush=True,
        )

    stop_requested = {"value": False}

    pos_sensitivity = (
        float(args.pos_sensitivity)
        if args.pos_sensitivity is not None
        else 0.1 * float(args.sensitivity)
    )
    rot_sensitivity = (
        float(args.rot_sensitivity)
        if args.rot_sensitivity is not None
        else 0.4 * float(args.sensitivity)
    )
    keyboard = LocalSe3Keyboard(
        LocalSe3KeyboardConfig(
            pos_sensitivity=pos_sensitivity,
            rot_sensitivity=rot_sensitivity,
            gripper_term=True,
        )
    )
    keyboard.add_callback("ESCAPE", lambda: stop_requested.__setitem__("value", True))

    jog_controller = RuntimeJacobianJogController(
        arm_joint_names=list(embodiment.joint_names["arm"]),
        jacobian_link_name=str(args.jacobian_link_name),
        damping=float(args.dls_damping),
        smoothing_alpha=float(args.smoothing_alpha),
        use_virtual_target=not bool(args.no_virtual_target),
        virtual_sync_threshold=float(args.virtual_sync_threshold),
        max_joint_delta_per_control=float(args.max_joint_delta_per_control),
        command_frame=str(args.command_frame),
    )

    physics_steps_per_control = _physics_steps_per_control(
        physics_dt=float(args.physics_dt),
        control_dt=float(args.control_dt),
    )
    print(
        "[franka_robotiq_keyboard_pd_jog] runtime_control "
        + json.dumps(_compact_runtime_control(runtime_control_report), sort_keys=True),
        flush=True,
    )
    print(
        "[franka_robotiq_keyboard_pd_jog] "
        f"asset={robot_usd_path} articulation={articulation_root_path} "
        f"physics_dt={args.physics_dt} control_dt={args.control_dt} "
        f"physics_steps_per_control={physics_steps_per_control}",
        flush=True,
    )
    print(str(keyboard), flush=True)

    state_logger = None
    record_every = max(1, int(args.record_every))
    if args.record_path is not None:
        frame_resolver = FrankaRobotiqFrameResolver.from_registry(
            registry,
            embodiment_id=str(args.embodiment_id),
        )
        runtime_metadata = {
            "embodiment_id": str(args.embodiment_id),
            "runtime_asset_path": str(robot_usd_path),
            "reference_prim_path": str(args.reference_prim_path),
            "articulation_root_path": str(articulation_root_path),
            "physics_dt": float(args.physics_dt),
            "control_dt": float(args.control_dt),
            "physics_steps_per_control": int(physics_steps_per_control),
            "runtime_control": _compact_runtime_control(runtime_control_report),
        }
        state_logger = JsonlRobotStateLogger(
            args.record_path,
            RobotStateSnapshotter(
                robot,
                frame_resolver=frame_resolver,
                runtime_metadata=runtime_metadata,
            ),
            metadata=runtime_metadata,
            flush_every=int(args.record_flush_every),
        )
        print(
            "[franka_robotiq_keyboard_pd_jog] recording "
            + json.dumps(
                {
                    "record_path": str(args.record_path),
                    "record_every": int(record_every),
                    "record_flush_every": int(args.record_flush_every),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    start_time = time.monotonic()
    command_index = 0
    try:
        while simulation_app.is_running() and not bool(stop_requested["value"]):
            raw_action = keyboard.advance()
            report = jog_controller.compute_and_apply(
                robot,
                raw_action=raw_action,
                control_dt=float(args.control_dt),
            )
            command_index += 1
            if int(args.print_every) > 0 and command_index % int(args.print_every) == 0:
                print(
                    "[franka_robotiq_keyboard_pd_jog] status "
                    + json.dumps(
                        {
                            "command_index": command_index,
                            "raw_twist_norm": float(np.linalg.norm(report.raw_twist)),
                            "smoothed_twist_norm": float(
                                np.linalg.norm(report.smoothed_twist)
                            ),
                            "max_abs_joint_delta_rad": report.max_abs_joint_delta_rad,
                            "gripper_action": report.gripper_action,
                            "gripper_targets": report.gripper_target_positions,
                            "gripper_positions": report.gripper_actual_positions,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            for _ in range(physics_steps_per_control):
                robot.set_joint_position_targets(report.commanded_joint_targets)
                world.step(render=not bool(args.headless))

            if state_logger is not None and command_index % record_every == 0:
                state_logger.log(
                    step_index=command_index,
                    timestamp=time.monotonic() - start_time,
                    command={
                        "raw_action": raw_action.astype(float).tolist(),
                        "keyboard_close_gripper": bool(keyboard.close_gripper),
                        "control_dt": float(args.control_dt),
                    },
                    command_report=_jog_report_mapping(report),
                )

            if args.run_seconds is not None:
                if time.monotonic() - start_time >= float(args.run_seconds):
                    break
    finally:
        if state_logger is not None:
            state_logger.close()
        keyboard.shutdown()
        simulation_app.close()

    return 0


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def _find_articulation_root_path(search_root: str) -> str:
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdPhysics

    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(search_root)
    if not root_prim.IsValid():
        raise RuntimeError(f"Reference prim does not exist after add_reference_to_stage: {search_root}")

    candidates = []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            candidates.append(str(prim.GetPath()))

    if not candidates:
        raise RuntimeError(f"No articulation root found under reference prim {search_root}.")

    candidates.sort(key=len)
    return candidates[0]


def _apply_ready_pose(robot) -> None:
    available_joints = set(robot.get_joint_positions())
    ready_positions = {
        name: value
        for name, value in READY_ARM_POSE.items()
        if name in available_joints
    }
    ready_positions.update(
        {
            name: value
            for name, value in READY_GRIPPER_POSE.items()
            if name in available_joints
        }
    )
    robot.set_joint_positions(ready_positions)
    robot.set_joint_velocities({joint_name: 0.0 for joint_name in ready_positions})
    targets = {
        name: value
        for name, value in READY_ARM_POSE.items()
        if name in available_joints
    }
    targets.update(_gripper_targets_from_legacy_action(robot=robot, gripper_action=-1.0))
    robot.set_joint_position_targets(targets)


def _home_pose_report(robot) -> dict[str, Any]:
    positions = robot.get_joint_positions()
    arm_errors = {
        name: abs(float(positions[name]) - float(target))
        for name, target in READY_ARM_POSE.items()
        if name in positions
    }
    gripper_errors = {
        name: abs(float(positions[name]) - float(target))
        for name, target in READY_GRIPPER_POSE.items()
        if name in positions
    }
    return {
        "arm_max_abs_error": max(arm_errors.values(), default=0.0),
        "gripper_max_abs_error": max(gripper_errors.values(), default=0.0),
        "arm_positions": {
            name: float(positions[name])
            for name in READY_ARM_POSE
            if name in positions
        },
        "gripper_positions": {
            name: float(positions[name])
            for name in READY_GRIPPER_POSE
            if name in positions
        },
    }


def _jog_report_mapping(report: JogCommandReport) -> dict[str, Any]:
    return {
        "raw_twist": report.raw_twist,
        "smoothed_twist": report.smoothed_twist,
        "commanded_joint_targets": report.commanded_joint_targets,
        "gripper_action": report.gripper_action,
        "gripper_target_positions": report.gripper_target_positions,
        "gripper_actual_positions": report.gripper_actual_positions,
        "max_abs_joint_delta_rad": report.max_abs_joint_delta_rad,
    }


def _physics_steps_per_control(*, physics_dt: float, control_dt: float) -> int:
    ratio = float(control_dt) / float(physics_dt)
    rounded = int(round(ratio))
    if rounded < 1:
        raise ValueError(
            f"control_dt={control_dt} must be at least one physics step of physics_dt={physics_dt}."
        )
    if abs(ratio - rounded) > 1.0e-6:
        raise ValueError(
            f"control_dt={control_dt} is not an integer multiple of physics_dt={physics_dt}."
        )
    return rounded


def _joint_vector(robot, joint_names: list[str]) -> np.ndarray:
    positions = robot.get_joint_positions()
    missing = [name for name in joint_names if name not in positions]
    if missing:
        raise RuntimeError(
            f"Robot state is missing joints {missing}; available joints are {list(positions)}."
        )
    return np.asarray([positions[name] for name in joint_names], dtype=np.float64)


def _joint_positions_by_name(robot, joint_names: list[str]) -> dict[str, float]:
    positions = robot.get_joint_positions()
    missing = [name for name in joint_names if name not in positions]
    if missing:
        raise RuntimeError(
            f"Robot state is missing joints {missing}; available joints are {list(positions)}."
        )
    return {name: float(positions[name]) for name in joint_names}


def _gripper_targets_from_legacy_action(*, robot, gripper_action: float) -> dict[str, float]:
    profile = robot.spec.runtime_control.get("gripper_action_profile")
    if not isinstance(profile, Mapping):
        raise RuntimeError(
            f"Embodiment '{robot.embodiment_id}' does not define "
            "runtime_control.gripper_action_profile."
        )

    joint_names = [
        str(name)
        for name in profile.get("joint_names", robot.spec.joint_names.get("gripper", []))
    ]
    opened = np.asarray(profile["opened_positions"], dtype=np.float32).reshape(-1)
    closed = np.asarray(profile["closed_positions"], dtype=np.float32).reshape(-1)
    if len(joint_names) != len(opened) or len(joint_names) != len(closed):
        raise ValueError(
            "gripper_action_profile length mismatch: "
            f"{len(joint_names)} joints, {len(opened)} opened values, "
            f"{len(closed)} closed values."
        )

    values = closed if float(gripper_action) > float(profile.get("close_threshold", 0.5)) else opened
    return {name: float(value) for name, value in zip(joint_names, values)}


def _jacobian_for_link(robot, link_name: str, arm_joint_names: list[str]) -> np.ndarray:
    articulation = robot._require_articulation()
    articulation_view = articulation._articulation_view
    body_names = _body_names(articulation_view)
    if link_name not in body_names:
        raise RuntimeError(
            f"Jacobian link '{link_name}' was not found. Available body names: {body_names}"
        )
    link_index = body_names.index(link_name)

    jacobians = articulation_view.get_jacobians()
    jacobians_np = np.asarray(
        articulation_view._backend_utils.to_numpy(jacobians),
        dtype=np.float64,
    )
    if jacobians_np.ndim != 4 or jacobians_np.shape[0] != 1 or jacobians_np.shape[2] != 6:
        raise RuntimeError(f"Unexpected Jacobian shape: {jacobians_np.shape}")

    dof_names = robot._get_dof_names(articulation)
    joint_indices = [dof_names.index(name) for name in arm_joint_names]
    link_jacobian = jacobians_np[0, link_index]
    arm_jacobian = link_jacobian[:, joint_indices]
    if arm_jacobian.shape != (6, len(arm_joint_names)):
        raise RuntimeError(
            f"Unexpected arm Jacobian shape: {arm_jacobian.shape}; "
            f"raw Jacobian shape is {jacobians_np.shape}."
        )
    return arm_jacobian


def _body_names(articulation_view) -> list[str]:
    body_names = getattr(articulation_view, "body_names", None)
    if body_names is None:
        body_names = getattr(articulation_view, "_body_names", None)
    if body_names is None:
        raise RuntimeError("Articulation view does not expose body_names.")
    return [str(name) for name in body_names]


def _damped_least_squares(jacobian: np.ndarray, twist: np.ndarray, *, damping: float) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    twist = np.asarray(twist, dtype=np.float64).reshape(6)
    lhs = jacobian @ jacobian.T + (float(damping) ** 2) * np.eye(6, dtype=np.float64)
    return jacobian.T @ np.linalg.solve(lhs, twist)


def _clipped_joint_delta(joint_delta: np.ndarray, *, max_norm: float) -> np.ndarray:
    joint_delta = np.asarray(joint_delta, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(joint_delta))
    if max_norm <= 0.0 or norm <= max_norm:
        return joint_delta
    return joint_delta * (float(max_norm) / norm)


def _apply_joint_limits(joint_names: list[str], joint_positions: np.ndarray) -> np.ndarray:
    values = np.asarray(joint_positions, dtype=np.float64).reshape(-1).copy()
    if len(values) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} joint positions, got {len(values)} values."
        )
    for index, joint_name in enumerate(joint_names):
        if joint_name in FRANKA_ARM_LIMITS:
            lower, upper = FRANKA_ARM_LIMITS[joint_name]
            values[index] = float(np.clip(values[index], lower, upper))
    return values


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("Cannot convert zero quaternion to rotation matrix.")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _compact_runtime_control(report: Mapping[str, Any]) -> dict[str, Any]:
    profile = report.get("controller_profile")
    compact = {
        "disable_robot_gravity": bool(report["disable_robot_gravity"]),
        "controller_profile_id": report.get("controller_profile_id"),
    }
    if isinstance(profile, Mapping):
        compact["controller_profile"] = {
            "profile_id": profile.get("profile_id"),
            "joint_groups": [
                {
                    "name": group.get("name"),
                    "joint_count": len(group.get("joint_names", [])),
                    "stiffness": group.get("stiffness"),
                    "damping": group.get("damping"),
                    "velocity_limit": group.get("velocity_limit"),
                    "effort_limit": group.get("effort_limit"),
                    "readback_max_abs_error": (
                        group.get("readback", {}).get("max_abs_error")
                        if isinstance(group.get("readback"), Mapping)
                        else None
                    ),
                }
                for group in profile.get("joint_groups", [])
            ],
            "readback_max_abs_error": profile.get("readback_max_abs_error"),
        }
    else:
        compact["controller_profile"] = None
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
