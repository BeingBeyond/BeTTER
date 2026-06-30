#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK_DIR = REPO_ROOT / "assets" / "tasks" / "loose_packing" / "Packing_a_Fruit_Lunch"
DEFAULT_ASSET_REGISTRY_ROOT = REPO_ROOT / "assets" / "objects" / "registry"
DEFAULT_BACKGROUND_REGISTRY = REPO_ROOT / "assets" / "scenes" / "backgrounds" / "registry.v2.json"
DEFAULT_RECORD_DIR = REPO_ROOT / "outputs" / "teleop"
DEFAULT_RECORD_PREFIX = "episode"
EMBODIMENT_REGISTRY_PATH = REPO_ROOT / "configs" / "embodiments" / "registry.v1.json"
CONTROLLER_PROFILES_PATH = REPO_ROOT / "configs" / "embodiments" / "controller_profiles.v1.json"
DEFAULT_ROBOT_POSITION = (-0.05, 0.0, -0.1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyboard FrankaRobotiq teleop inside a resolved BeTTER task episode."
    )
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--asset-registry-root", type=Path, default=DEFAULT_ASSET_REGISTRY_ROOT)
    parser.add_argument("--variation-id", default="BASE")
    parser.add_argument("--episode-seed", type=int, default=1234)
    parser.add_argument("--prim-root", default="/World/Objects")
    parser.add_argument("--load-background", dest="load_background", action="store_true", default=True)
    parser.add_argument("--no-background", dest="load_background", action="store_false")
    parser.add_argument("--background-registry", type=Path, default=DEFAULT_BACKGROUND_REGISTRY)
    parser.add_argument("--background-scene-id", default=None)
    parser.add_argument("--background-material-id", default=None)
    parser.add_argument("--background-layout-id", default=None)
    parser.add_argument("--background-prim-path", default="/World/Background")
    parser.add_argument("--reference-prim-path", default="/World/BeTTERFrankaRobotiq")
    parser.add_argument(
        "--robot-position",
        type=float,
        nargs=3,
        default=DEFAULT_ROBOT_POSITION,
        metavar=("X", "Y", "Z"),
        help="World translation for the robot reference root prim.",
    )
    parser.add_argument(
        "--embodiment-id",
        default="franka_robotiq",
        help="Embodiment registry id to load. Use franka_robotiq_legacy_asset for legacy asset A/B tests.",
    )
    parser.add_argument(
        "--controller-profile-id",
        default=None,
        help="Override registry runtime_control.controller_profile_id for A/B controller tests.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--force-keyboard-in-headless",
        action="store_true",
        help="Try to subscribe to keyboard events even when --headless is set.",
    )
    parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--control-dt", type=float, default=1.0 / 15.0)
    parser.add_argument("--sensitivity", type=float, default=2.0)
    parser.add_argument("--pos-sensitivity", type=float, default=None)
    parser.add_argument("--rot-sensitivity", type=float, default=None)
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="Deprecated compatibility option. Keyboard twist commands are not smoothed.",
    )
    parser.add_argument("--dls-damping", type=float, default=0.05)
    parser.add_argument("--max-joint-delta-per-control", type=float, default=0.08)
    parser.add_argument("--virtual-sync-threshold", type=float, default=0.5)
    parser.add_argument("--no-virtual-target", action="store_true")
    parser.add_argument("--command-frame", choices=("world", "robot_base"), default="world")
    parser.add_argument("--jacobian-link-name", default="base_link")
    parser.add_argument("--home-pose", choices=("ready", "usd"), default="ready")
    parser.add_argument("--warmup-frames", type=int, default=24)
    parser.add_argument("--run-seconds", type=float, default=None)
    parser.add_argument("--max-control-steps", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--single-episode", action="store_true")
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--record-prefix", default=DEFAULT_RECORD_PREFIX)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--record-flush-every", type=int, default=1)
    parser.add_argument("--skip-initial-record", action="store_true")
    parser.add_argument("--skip-ground-plane", action="store_true")
    parser.add_argument(
        "--add-default-lights",
        dest="add_default_lights",
        action="store_true",
        default=True,
        help="Add local fallback lights when running without a background scene.",
    )
    parser.add_argument("--no-default-lights", dest="add_default_lights", action="store_false")
    parser.add_argument("--evaluate-conditions", dest="evaluate_conditions", action="store_true", default=True)
    parser.add_argument("--no-evaluate-conditions", dest="evaluate_conditions", action="store_false")
    parser.add_argument("--goal-consecutive-steps", type=int, default=20)
    parser.add_argument("--fail-consecutive-steps", type=int, default=1)
    parser.add_argument("--auto-reset-on-success", dest="auto_reset_on_success", action="store_true", default=True)
    parser.add_argument("--no-auto-reset-on-success", dest="auto_reset_on_success", action="store_false")
    parser.add_argument(
        "--manual-reset-shortcut",
        "--manual-reset-key",
        dest="manual_reset_shortcut",
        default="SHIFT+R",
        help="Keyboard shortcut that finishes the current trajectory and starts the next one.",
    )
    parser.add_argument("--stop-on-condition", action="store_true")
    parser.add_argument("--condition-near-distance", type=float, default=0.35)
    parser.add_argument("--condition-touching-distance", type=float, default=0.01)
    parser.add_argument("--condition-height-epsilon", type=float, default=0.02)
    parser.add_argument("--condition-on-overlap-ratio", type=float, default=0.3)
    parser.add_argument("--condition-on-z-offset-min", type=float, default=0.0)
    parser.add_argument("--condition-on-z-offset-max", type=float, default=0.1)
    parser.add_argument("--condition-above-overlap-ratio", type=float, default=0.3)
    parser.add_argument("--condition-in-containment-ratio", type=float, default=0.5)
    parser.add_argument("--condition-vertical-align-distance", type=float, default=0.1)
    parser.add_argument("--condition-axis-align-angle", type=float, default=0.1745)
    parser.add_argument("--condition-left-right-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--condition-contain-xy-ratio",
        type=float,
        default=None,
        help="Deprecated compatibility alias for --condition-in-containment-ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.headless) and int(args.max_control_steps) <= 0 and args.run_seconds is None:
        raise ValueError(
            "Headless task teleop needs --max-control-steps > 0 or --run-seconds set."
        )
    if (
        bool(args.headless)
        and not bool(args.single_episode)
        and int(args.max_episodes) <= 0
        and args.run_seconds is None
    ):
        raise ValueError(
            "Headless continuous task teleop needs --max-episodes > 0 or --run-seconds set."
        )

    from isaacsim import SimulationApp

    launch_config: dict[str, Any] = {"headless": bool(args.headless)}
    if not bool(args.headless):
        launch_config.update({"hide_ui": False, "renderer": "RaytracedLighting"})
    simulation_app = SimulationApp(launch_config)

    keyboard = None
    state_logger = None
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage

        from src.scene.episode_conditions import build_episode_condition_evaluator
        from src.scene.episode_runtime import (
            EpisodeRuntimeLoadConfig,
            load_resolved_episode_into_stage,
            restore_episode_object_poses,
        )
        from src.scene.recording import (
            EpisodeStateSnapshotter,
            JsonlEpisodeStateLogger,
            PickleEpisodeStateLogger,
            RobotStateSnapshotter,
        )
        from src.scene.robots import (
            FrankaRobotiqFrameResolver,
            apply_embodiment_runtime_control,
            create_robot,
            load_registry,
            resolve_embodiment_paths,
            resolve_runtime_control,
        )
        from src.scene.teleop import LocalSe3Keyboard, LocalSe3KeyboardConfig
        from src.task.conditions.schema import RelationThresholds
        from src.task.evaluation import TaskEvaluatorConfig
        from src.task.specs import ResolvedBackgroundSpec, load_task_spec, resolve_episode
        from tools.teleop.franka_robotiq_keyboard_pd_jog import (
            READY_ARM_POSE,
            READY_GRIPPER_POSE,
            RuntimeJacobianJogController,
            _apply_ready_pose,
            _compact_runtime_control,
            _find_articulation_root_path,
            _jog_report_mapping,
            _physics_steps_per_control,
        )

        physics_steps_per_control = _physics_steps_per_control(
            physics_dt=float(args.physics_dt),
            control_dt=float(args.control_dt),
        )
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(args.physics_dt),
            rendering_dt=float(args.control_dt),
        )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable after SimulationApp startup.")
        if not bool(args.load_background):
            if not bool(args.skip_ground_plane):
                _add_local_ground_plane(stage)
            if bool(args.add_default_lights):
                _add_local_lighting(stage)

        background = None
        if bool(args.load_background):
            background = ResolvedBackgroundSpec(
                scene_id=_resolve_background_scene_id(args),
                material_id=args.background_material_id,
                layout_id=args.background_layout_id,
                registry_path=str(args.background_registry.expanduser().resolve()),
                prim_path=str(args.background_prim_path),
            )

        task_dir = args.task_dir.expanduser().resolve()
        asset_registry_root = args.asset_registry_root.expanduser().resolve()
        task_spec = load_task_spec(task_dir)
        episode = resolve_episode(
            task_spec,
            args.variation_id,
            asset_registry_root=asset_registry_root,
            episode_seed=int(args.episode_seed),
            background=background,
            prim_root=str(args.prim_root),
        )
        handle = load_resolved_episode_into_stage(
            stage,
            episode,
            config=EpisodeRuntimeLoadConfig(
                load_background=bool(args.load_background),
                load_objects=False,
                restore_object_poses=False,
                capture_light_baseline=False,
                require_background_registry=bool(args.load_background),
            ),
        )
        loaded_episode = None

        registry = load_registry(EMBODIMENT_REGISTRY_PATH)
        embodiment = registry.get_embodiment(str(args.embodiment_id))
        runtime_control = resolve_runtime_control(
            embodiment,
            controller_profile_id=args.controller_profile_id,
        )
        runtime_embodiment = replace(embodiment, runtime_control=runtime_control)
        resolved_paths = resolve_embodiment_paths(registry, embodiment)
        robot_usd_path = resolved_paths["runtime_asset_path"]
        _require_file(robot_usd_path)

        add_reference_to_stage(str(robot_usd_path), prim_path=str(args.reference_prim_path))
        _set_reference_prim_position(stage, str(args.reference_prim_path), args.robot_position)
        articulation_root_path = _find_articulation_root_path(str(args.reference_prim_path))
        robot = create_robot(
            instance_id="franka_robotiq_task_teleop",
            prim_path=articulation_root_path,
            embodiment=runtime_embodiment,
        )
        robot.set_world_pose(position=np.asarray(args.robot_position, dtype=np.float32))

        world.reset()
        robot.initialize()
        runtime_control_report = apply_embodiment_runtime_control(
            robot,
            runtime_control,
            controller_profiles_path=CONTROLLER_PROFILES_PATH,
        )

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
        use_keyboard = not bool(args.headless) or bool(args.force_keyboard_in_headless)
        stop_requested = {"value": False}
        manual_reset_requested = {"value": False}
        manual_reset_key, manual_reset_modifiers = _parse_keyboard_shortcut(args.manual_reset_shortcut)
        manual_reset_shortcut = _format_keyboard_shortcut(manual_reset_key, manual_reset_modifiers)
        if use_keyboard:
            keyboard = LocalSe3Keyboard(
                LocalSe3KeyboardConfig(
                    pos_sensitivity=pos_sensitivity,
                    rot_sensitivity=rot_sensitivity,
                    gripper_term=True,
                )
            )
            keyboard.add_callback("ESCAPE", lambda: stop_requested.__setitem__("value", True))
            if manual_reset_modifiers:
                keyboard.add_modified_callback(
                    manual_reset_key,
                    manual_reset_modifiers,
                    lambda: manual_reset_requested.__setitem__("value", True),
                )
            else:
                keyboard.add_callback(
                    manual_reset_key,
                    lambda: manual_reset_requested.__setitem__("value", True),
                )
            print(str(keyboard), flush=True)
            print(
                "[franka_robotiq_task_teleop] callbacks "
                + json.dumps(
                    {
                        "exit": "ESCAPE",
                        "finish_and_reset_episode": manual_reset_shortcut,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            print(
                "[franka_robotiq_task_teleop] headless_zero_command_source",
                flush=True,
            )

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

        timing = {
            "physics_dt": float(world.get_physics_dt()),
            "control_dt": float(world.get_rendering_dt()),
            "physics_steps_per_control": int(physics_steps_per_control),
        }
        compact_runtime_control = _compact_runtime_control(runtime_control_report)
        runtime_metadata = {
            "embodiment_id": str(args.embodiment_id),
            "task_dir": str(task_dir),
            "asset_registry_root": str(asset_registry_root),
            "runtime_asset_path": str(robot_usd_path),
            "reference_prim_path": str(args.reference_prim_path),
            "articulation_root_path": str(articulation_root_path),
            "robot_position": [float(value) for value in args.robot_position],
            "background_loaded": handle.background_scene is not None,
            "timing": timing,
            "runtime_control": compact_runtime_control,
            "controller": {
                "type": "runtime_jacobian_keyboard_pd_jog",
                "command_frame": str(args.command_frame),
                "jacobian_link_name": str(args.jacobian_link_name),
                "dls_damping": float(args.dls_damping),
                "input_smoothing": "none",
                "smoothing_alpha_arg": float(args.smoothing_alpha),
                "max_joint_delta_per_control": float(args.max_joint_delta_per_control),
                "use_virtual_target": not bool(args.no_virtual_target),
            },
        }

        record_enabled = not bool(args.no_record)
        evaluator = None
        thresholds = None
        record_task_name = _path_component(str(episode.task_id or args.task_dir.name))
        record_variation_name = _path_component(str(episode.variation_id or args.variation_id))
        active_record_path = None
        active_final_record_path = None
        active_runtime_metadata = None
        process_start_time = time.monotonic()
        episode_start_time = process_start_time
        episode_index = 0
        command_index = 0
        total_command_count = 0
        logged_count = 0
        total_logged_count = 0
        completed_episodes = 0
        last_logged_step_index = None
        final_evaluation = None

        def _resolve_collection_episode(collection_episode_index: int):
            return resolve_episode(
                task_spec,
                args.variation_id,
                asset_registry_root=asset_registry_root,
                episode_seed=int(args.episode_seed) + int(collection_episode_index) - 1,
                background=background,
                prim_root=str(args.prim_root),
            )

        def _build_condition_state():
            if not bool(args.evaluate_conditions):
                return None, None
            in_containment_ratio = (
                float(args.condition_contain_xy_ratio)
                if args.condition_contain_xy_ratio is not None
                else float(args.condition_in_containment_ratio)
            )
            return build_episode_condition_evaluator(
                episode,
                config=TaskEvaluatorConfig(
                    goal_consecutive_steps=max(int(args.goal_consecutive_steps), 1),
                    fail_consecutive_steps=max(int(args.fail_consecutive_steps), 1),
                ),
            ), RelationThresholds(
                near_distance=float(args.condition_near_distance),
                touching_distance=float(args.condition_touching_distance),
                height_epsilon=float(args.condition_height_epsilon),
                on_overlap_ratio=float(args.condition_on_overlap_ratio),
                on_z_offset_min=float(args.condition_on_z_offset_min),
                on_z_offset_max=float(args.condition_on_z_offset_max),
                above_overlap_ratio=float(args.condition_above_overlap_ratio),
                in_containment_ratio=in_containment_ratio,
                contain_xy_ratio=in_containment_ratio,
                vertical_align_distance=float(args.condition_vertical_align_distance),
                axis_align_angle=float(args.condition_axis_align_angle),
                left_right_tolerance=float(args.condition_left_right_tolerance),
            )

        def _close_state_logger() -> None:
            nonlocal state_logger
            if state_logger is not None:
                state_logger.close()
                state_logger = None

        def _open_state_logger(record_path: Path, metadata: Mapping[str, Any]):
            logger_format = _episode_state_logger_format(record_path)
            logger_cls = (
                PickleEpisodeStateLogger
                if logger_format == "pickle"
                else JsonlEpisodeStateLogger
            )
            return logger_cls(
                record_path,
                EpisodeStateSnapshotter(
                    stage=stage,
                    episode=episode,
                    robot_snapshotter=RobotStateSnapshotter(
                        robot,
                        frame_resolver=FrankaRobotiqFrameResolver.from_registry(
                            registry,
                            embodiment_id=str(args.embodiment_id),
                        ),
                        runtime_metadata=metadata,
                    ),
                    runtime_metadata=metadata,
                ),
                metadata=metadata,
                flush_every=int(args.record_flush_every),
            )

        def _log_state(
            *,
            phase: str,
            evaluation,
            raw_action=None,
            report=None,
            timestamp: float | None = None,
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal logged_count, total_logged_count, last_logged_step_index
            if state_logger is None:
                return
            payload_extra = {
                "phase": str(phase),
                "collection_episode_index": int(episode_index),
            }
            if extra is not None:
                payload_extra.update(dict(extra))
            command = None
            command_report = None
            if raw_action is not None:
                command = {
                    "raw_action": np.asarray(raw_action, dtype=float).tolist(),
                    "keyboard_close_gripper": (
                        None if keyboard is None else bool(keyboard.close_gripper)
                    ),
                    "control_dt": float(args.control_dt),
                }
            if report is not None:
                command_report = _jog_report_mapping(report)
            state_logger.log(
                step_index=command_index,
                timestamp=(
                    float(timestamp)
                    if timestamp is not None
                    else float(time.monotonic() - episode_start_time)
                ),
                command=command,
                command_report=command_report,
                evaluation=evaluation,
                extra=payload_extra,
            )
            logged_count += 1
            total_logged_count += 1
            last_logged_step_index = int(command_index)

        def _start_episode(*, reason: str) -> None:
            nonlocal active_record_path, active_final_record_path, active_runtime_metadata
            nonlocal command_index, episode_index, episode_start_time
            nonlocal evaluator, thresholds, final_evaluation
            nonlocal last_logged_step_index, logged_count, state_logger
            nonlocal episode, loaded_episode

            _close_state_logger()
            episode_index += 1
            previous_episode = loaded_episode
            episode = _resolve_collection_episode(episode_index)
            _remove_episode_object_prims(stage, previous_episode)
            load_resolved_episode_into_stage(
                stage,
                episode,
                config=EpisodeRuntimeLoadConfig(
                    load_background=False,
                    load_objects=True,
                    restore_object_poses=True,
                    capture_light_baseline=False,
                    require_background_registry=False,
                ),
            )
            loaded_episode = episode
            command_index = 0
            logged_count = 0
            last_logged_step_index = None
            final_evaluation = None
            active_final_record_path = None
            active_record_path = (
                None
                if not record_enabled
                else _available_path(
                    _episode_record_path(
                        args.record_dir,
                        task_name=record_task_name,
                        variation_name=record_variation_name,
                        status="in_progress",
                        record_prefix=str(args.record_prefix),
                        episode_index=episode_index,
                    )
                )
            )
            planned_success_record_path = (
                None
                if not record_enabled
                else _episode_record_path(
                    args.record_dir,
                    task_name=record_task_name,
                    variation_name=record_variation_name,
                    status="success",
                    record_prefix=str(args.record_prefix),
                    episode_index=episode_index,
                )
            )
            planned_failed_record_path = (
                None
                if not record_enabled
                else _episode_record_path(
                    args.record_dir,
                    task_name=record_task_name,
                    variation_name=record_variation_name,
                    status="failed",
                    record_prefix=str(args.record_prefix),
                    episode_index=episode_index,
                )
            )
            active_runtime_metadata = {
                **runtime_metadata,
                "collection": {
                    "episode_index": int(episode_index),
                    "staging_record_path": None if active_record_path is None else str(active_record_path),
                    "record_dir": None if not record_enabled else str(args.record_dir),
                    "record_prefix": None if not record_enabled else str(args.record_prefix),
                    "record_task_name": record_task_name,
                    "record_variation_name": record_variation_name,
                    "planned_success_record_path": (
                        None if planned_success_record_path is None else str(planned_success_record_path)
                    ),
                    "planned_failed_record_path": (
                        None if planned_failed_record_path is None else str(planned_failed_record_path)
                    ),
                    "start_reason": str(reason),
                    "manual_reset_shortcut": manual_reset_shortcut,
                    "continuous": not bool(args.single_episode),
                    "auto_reset_on_success": bool(args.auto_reset_on_success),
                    "evaluate_conditions": bool(args.evaluate_conditions),
                    "warmup_frames": int(args.warmup_frames),
                    "episode_seed": int(episode.episode_seed),
                    "selected_assets": _episode_asset_summary(episode),
                },
            }

            world.reset()
            robot.initialize()
            apply_embodiment_runtime_control(
                robot,
                runtime_control,
                controller_profiles_path=CONTROLLER_PROFILES_PATH,
            )
            restore_episode_object_poses(stage, episode)
            robot.set_world_pose(position=np.asarray(args.robot_position, dtype=np.float32))
            if args.home_pose == "ready":
                _apply_ready_pose(robot)

            for _ in range(max(0, int(args.warmup_frames))):
                world.step(render=not bool(args.headless))

            jog_controller.reset()
            if keyboard is not None:
                keyboard.reset()
            evaluator, thresholds = _build_condition_state()
            episode_start_time = time.monotonic()

            if active_record_path is not None:
                state_logger = _open_state_logger(active_record_path, active_runtime_metadata)

            home_report = _home_pose_report(
                robot,
                arm_targets=READY_ARM_POSE,
                gripper_targets=READY_GRIPPER_POSE,
            )
            print(
                "[franka_robotiq_task_teleop] episode_start "
                + json.dumps(
                    {
                        "episode_index": int(episode_index),
                        "staging_record_path": None if active_record_path is None else str(active_record_path),
                        "reason": str(reason),
                        "task_id": episode.task_id,
                        "variation_id": episode.variation_id,
                        "episode_seed": int(episode.episode_seed),
                        "object_count": len(episode.objects),
                        "selected_assets": _episode_asset_summary(episode),
                        "runtime_control": compact_runtime_control,
                        "home_pose": home_report,
                        "timing": timing,
                        "warmup_frames": int(args.warmup_frames),
                        "evaluate_conditions": bool(args.evaluate_conditions),
                        "auto_reset_on_success": bool(args.auto_reset_on_success),
                        "robot_position": [float(value) for value in args.robot_position],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            if state_logger is not None and not bool(args.skip_initial_record):
                final_evaluation = _evaluate_conditions(
                    stage=stage,
                    episode=episode,
                    evaluator=evaluator,
                    thresholds=thresholds,
                )
                _log_state(
                    phase="initial",
                    evaluation=final_evaluation,
                    timestamp=0.0,
                    extra={"start_reason": str(reason)},
                )

        def _finish_episode(*, reason: str, evaluation) -> None:
            nonlocal completed_episodes, active_final_record_path
            if state_logger is not None:
                _log_state(
                    phase=str(reason),
                    evaluation=evaluation,
                    extra={"finish_reason": str(reason)},
                )
            _close_state_logger()
            result_status = _episode_result_status(reason=reason, evaluation=evaluation)
            active_final_record_path = None
            if active_record_path is not None:
                active_final_record_path = _finalize_episode_record_path(
                    active_record_path,
                    record_dir=args.record_dir,
                    task_name=record_task_name,
                    variation_name=record_variation_name,
                    status=result_status,
                    record_prefix=str(args.record_prefix),
                    episode_index=episode_index,
                )
            completed_episodes += 1
            print(
                "[franka_robotiq_task_teleop] episode_done "
                + json.dumps(
                    {
                        "episode_index": int(episode_index),
                        "record_path": (
                            None if active_final_record_path is None else str(active_final_record_path)
                        ),
                        "staging_record_path": (
                            None if active_record_path is None else str(active_record_path)
                        ),
                        "result_status": result_status,
                        "reason": str(reason),
                        "logged_count": int(logged_count),
                        "command_count": int(command_index),
                        "evaluation": _evaluation_to_dict(evaluation),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def _max_episodes_reached() -> bool:
            return int(args.max_episodes) > 0 and completed_episodes >= int(args.max_episodes)

        print(
            "[franka_robotiq_task_teleop] start "
            + json.dumps(
                {
                    "record_dir": None if not record_enabled else str(args.record_dir),
                    "record_prefix": None if not record_enabled else str(args.record_prefix),
                    "record_mode": "single_episode" if bool(args.single_episode) else "continuous_sequence",
                    "task_id": episode.task_id,
                    "variation_id": episode.variation_id,
                    "base_episode_seed": int(args.episode_seed),
                    "episode_seed_policy": "base_episode_seed + collection_episode_index - 1",
                    "object_count": len(episode.objects),
                    "runtime_control": compact_runtime_control,
                    "timing": timing,
                    "robot_position": [float(value) for value in args.robot_position],
                    "sensitivity": float(args.sensitivity),
                    "pos_sensitivity": float(pos_sensitivity),
                    "rot_sensitivity": float(rot_sensitivity),
                    "evaluate_conditions": bool(args.evaluate_conditions),
                    "auto_reset_on_success": bool(args.auto_reset_on_success),
                    "manual_reset_shortcut": manual_reset_shortcut,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        _start_episode(reason="startup")

        while simulation_app.is_running() and not bool(stop_requested["value"]):
            if args.run_seconds is not None and time.monotonic() - process_start_time >= float(args.run_seconds):
                break

            if bool(manual_reset_requested["value"]):
                manual_reset_requested["value"] = False
                _finish_episode(reason="manual_reset", evaluation=final_evaluation)
                if bool(args.single_episode) or _max_episodes_reached():
                    break
                _start_episode(reason="manual_reset")
                continue

            if int(args.max_control_steps) > 0 and command_index >= int(args.max_control_steps):
                _finish_episode(reason="max_control_steps", evaluation=final_evaluation)
                if bool(args.single_episode) or _max_episodes_reached():
                    break
                _start_episode(reason="max_control_steps")
                continue

            raw_action = (
                keyboard.advance()
                if keyboard is not None
                else np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            )
            report = jog_controller.compute_and_apply(
                robot,
                raw_action=raw_action,
                control_dt=float(args.control_dt),
            )
            command_index += 1
            total_command_count += 1

            for _ in range(physics_steps_per_control):
                robot.set_joint_position_targets(report.commanded_joint_targets)
                world.step(render=not bool(args.headless))

            if report.gripper_target_positions:
                joint_positions = robot.get_joint_positions()
                report.gripper_actual_positions = {
                    name: float(joint_positions[name])
                    for name in report.gripper_target_positions
                    if name in joint_positions
                }

            final_evaluation = _evaluate_conditions(
                stage=stage,
                episode=episode,
                evaluator=evaluator,
                thresholds=thresholds,
            )
            if state_logger is not None and command_index % max(int(args.record_every), 1) == 0:
                _log_state(
                    phase="running",
                    raw_action=raw_action,
                    report=report,
                    evaluation=final_evaluation,
                )

            if int(args.print_every) > 0 and command_index % int(args.print_every) == 0:
                print(
                    "[franka_robotiq_task_teleop] status "
                    + json.dumps(
                        {
                            "episode_index": int(episode_index),
                            "command_index": int(command_index),
                            "total_command_count": int(total_command_count),
                            "logged_count": int(logged_count),
                            "total_logged_count": int(total_logged_count),
                            "staging_record_path": (
                                None if active_record_path is None else str(active_record_path)
                            ),
                            "raw_twist_norm": float(np.linalg.norm(report.raw_twist)),
                            "smoothed_twist_norm": float(np.linalg.norm(report.smoothed_twist)),
                            "max_abs_joint_delta_rad": report.max_abs_joint_delta_rad,
                            "gripper_action": report.gripper_action,
                            "gripper_targets": report.gripper_target_positions,
                            "gripper_positions": report.gripper_actual_positions,
                            "evaluation": _evaluation_to_dict(final_evaluation),
                            "condition_details": _evaluate_condition_details(
                                stage=stage,
                                episode=episode,
                                thresholds=thresholds,
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if final_evaluation is not None and bool(final_evaluation.terminated):
                reason = str(final_evaluation.termination_type)
                if bool(args.stop_on_condition):
                    _finish_episode(reason=reason, evaluation=final_evaluation)
                    break
                if reason == "success" and bool(args.auto_reset_on_success):
                    _finish_episode(reason="success", evaluation=final_evaluation)
                    if bool(args.single_episode) or _max_episodes_reached():
                        break
                    _start_episode(reason="success")
                    continue
                if reason == "failure":
                    _finish_episode(reason="failure", evaluation=final_evaluation)
                    if bool(args.single_episode) or _max_episodes_reached():
                        break
                    _start_episode(reason="failure")
                    continue

        if state_logger is not None:
            _finish_episode(reason="shutdown", evaluation=final_evaluation)

        final_report = {
            "success": True,
            "record_dir": None if not record_enabled else str(args.record_dir),
            "record_prefix": None if not record_enabled else str(args.record_prefix),
            "record_mode": "single_episode" if bool(args.single_episode) else "continuous_sequence",
            "active_staging_record_path": None if active_record_path is None else str(active_record_path),
            "active_final_record_path": (
                None if active_final_record_path is None else str(active_final_record_path)
            ),
            "completed_episodes": int(completed_episodes),
            "active_episode_index": int(episode_index),
            "active_episode_logged_count": int(logged_count),
            "total_logged_count": int(total_logged_count),
            "active_episode_command_count": int(command_index),
            "total_command_count": int(total_command_count),
            "runtime_asset_path": str(robot_usd_path),
            "reference_prim_path": str(args.reference_prim_path),
            "articulation_root_path": str(articulation_root_path),
            "robot_position": [float(value) for value in args.robot_position],
            "object_count": len(episode.objects),
            "timing": timing,
            "evaluate_conditions": bool(args.evaluate_conditions),
            "auto_reset_on_success": bool(args.auto_reset_on_success),
            "evaluation": _evaluation_to_dict(final_evaluation),
        }
        print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if state_logger is not None:
            finish_episode = locals().get("_finish_episode")
            if finish_episode is not None:
                finish_episode(reason="shutdown", evaluation=locals().get("final_evaluation"))
            else:
                state_logger.close()
        if keyboard is not None:
            keyboard.shutdown()
        simulation_app.close()


def _resolve_background_scene_id(args: argparse.Namespace) -> str:
    if args.background_scene_id is not None:
        return str(args.background_scene_id)
    payload = json.loads(args.background_registry.expanduser().resolve().read_text(encoding="utf-8"))
    return str(payload["default_scene_id"])


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def _remove_episode_object_prims(stage, episode) -> None:
    if episode is None:
        return
    for obj in sorted(episode.objects.values(), key=lambda item: item.prim_path, reverse=True):
        prim = stage.GetPrimAtPath(str(obj.prim_path))
        if prim is not None and (not hasattr(prim, "IsValid") or prim.IsValid()):
            stage.RemovePrim(str(obj.prim_path))


def _episode_asset_summary(episode) -> dict[str, dict[str, str]]:
    return {
        instance_id: {
            "semantic_name": str(obj.semantic_name),
            "asset_key": str(obj.asset.asset_key),
            "source_uid": str(obj.asset.source_uid),
            "asset_path": str(obj.asset.asset_path),
        }
        for instance_id, obj in sorted(episode.objects.items())
    }


def _episode_record_path(
    record_dir: Path,
    *,
    task_name: str,
    variation_name: str,
    status: str,
    record_prefix: str,
    episode_index: int,
) -> Path:
    directory = Path(record_dir)
    prefix = str(record_prefix).strip()
    if not prefix:
        raise ValueError("--record-prefix cannot be empty.")
    return (
        directory
        / _path_component(task_name)
        / _path_component(variation_name)
        / _path_component(status)
        / f"{_path_component(prefix)}_{int(episode_index):03d}.pkl"
    )


def _finalize_episode_record_path(
    staging_path: Path,
    *,
    record_dir: Path,
    task_name: str,
    variation_name: str,
    status: str,
    record_prefix: str,
    episode_index: int,
) -> Path:
    target_path = _episode_record_path(
        record_dir,
        task_name=task_name,
        variation_name=variation_name,
        status=status,
        record_prefix=record_prefix,
        episode_index=episode_index,
    )
    target_path = _available_path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    Path(staging_path).replace(target_path)
    return target_path


def _episode_result_status(*, reason: str, evaluation) -> str:
    if evaluation is not None and str(evaluation.termination_type) == "success":
        return "success"
    if str(reason) == "success":
        return "success"
    return "failed"


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output path for: {path}")


def _path_component(value: str) -> str:
    text = str(value).strip().replace(" ", "_")
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def _episode_state_logger_format(record_path: Path) -> str:
    suffix = Path(record_path).suffix.lower()
    if suffix in ("", ".pkl", ".pickle"):
        return "pickle"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError(
        f"Unsupported record file suffix {suffix!r}; expected .pkl or .jsonl."
    )


def _parse_keyboard_shortcut(shortcut: str) -> tuple[str, frozenset[str]]:
    parts = [part.strip().upper() for part in str(shortcut).split("+") if part.strip()]
    if not parts:
        raise ValueError("Manual reset shortcut cannot be empty.")
    aliases = {"CONTROL": "CTRL", "OPTION": "ALT", "CMD": "SUPER", "COMMAND": "SUPER"}
    normalized = [aliases.get(part, part) for part in parts]
    key = normalized[-1]
    modifiers = frozenset(normalized[:-1])
    allowed_modifiers = {"SHIFT", "CTRL", "ALT", "SUPER"}
    invalid = sorted(modifier for modifier in modifiers if modifier not in allowed_modifiers)
    if invalid:
        raise ValueError(
            f"Unsupported manual reset shortcut modifiers {invalid}; "
            f"allowed modifiers are {sorted(allowed_modifiers)}."
        )
    return key, modifiers


def _format_keyboard_shortcut(key: str, modifiers: frozenset[str]) -> str:
    ordered_modifiers = [name for name in ("SHIFT", "CTRL", "ALT", "SUPER") if name in modifiers]
    return "+".join([*ordered_modifiers, str(key).upper()])


def _add_local_ground_plane(stage, prim_path: str = "/World/GroundPlane") -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.42, 0.42)])

    xform = UsdGeom.Xformable(cube.GetPrim())
    _set_or_add_xform_op(
        xform,
        UsdGeom.XformOp.TypeTranslate,
        Gf.Vec3d(0.0, 0.0, -0.011),
    )
    _set_or_add_xform_op(
        xform,
        UsdGeom.XformOp.TypeScale,
        Gf.Vec3d(10.0, 10.0, 0.01),
    )
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _add_local_lighting(stage) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    UsdGeom.Xform.Define(stage, "/World/DefaultLights")

    dome = UsdLux.DomeLight.Define(stage, "/World/DefaultLights/Dome")
    dome.CreateIntensityAttr(350.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/DefaultLights/Key")
    key.CreateIntensityAttr(2200.0)
    key.CreateAngleAttr(0.55)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    _set_or_add_xform_op(
        UsdGeom.Xformable(key.GetPrim()),
        UsdGeom.XformOp.TypeRotateXYZ,
        Gf.Vec3f(-45.0, 0.0, -35.0),
    )

    fill = UsdLux.RectLight.Define(stage, "/World/DefaultLights/Fill")
    fill.CreateIntensityAttr(600.0)
    fill.CreateWidthAttr(4.0)
    fill.CreateHeightAttr(4.0)
    fill.CreateColorAttr(Gf.Vec3f(0.8, 0.9, 1.0))
    fill_xform = UsdGeom.Xformable(fill.GetPrim())
    _set_or_add_xform_op(fill_xform, UsdGeom.XformOp.TypeTranslate, Gf.Vec3d(0.0, -1.8, 2.4))
    _set_or_add_xform_op(fill_xform, UsdGeom.XformOp.TypeRotateXYZ, Gf.Vec3f(-60.0, 0.0, 0.0))


def _set_or_add_xform_op(xformable, op_type, value) -> None:
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            op.Set(value)
            return
    if op_type == UsdGeom.XformOp.TypeTranslate:
        xformable.AddTranslateOp().Set(value)
        return
    if op_type == UsdGeom.XformOp.TypeScale:
        xformable.AddScaleOp().Set(value)
        return
    if op_type == UsdGeom.XformOp.TypeRotateXYZ:
        xformable.AddRotateXYZOp().Set(value)
        return
    raise ValueError(f"Unsupported xform op type: {op_type}")


def _set_reference_prim_position(stage, prim_path: str, position) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        raise RuntimeError(f"Robot reference prim is missing after USD reference: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate = Gf.Vec3d(*(float(value) for value in position))
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(translate)
            return
    xformable.AddTranslateOp().Set(translate)


def _evaluate_conditions(*, stage, episode, evaluator, thresholds):
    if evaluator is None:
        return None
    from src.scene.episode_conditions import evaluate_episode_conditions

    return evaluate_episode_conditions(
        stage,
        episode,
        evaluator=evaluator,
        thresholds=thresholds,
    )


def _evaluate_condition_details(*, stage, episode, thresholds):
    if thresholds is None:
        return None
    from src.scene.episode_conditions import evaluate_episode_condition_details

    return evaluate_episode_condition_details(
        stage,
        episode,
        thresholds=thresholds,
    )


def _evaluation_to_dict(evaluation) -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "goal_raw": bool(evaluation.goal_raw),
        "goal_streak": int(evaluation.goal_streak),
        "goal_passed": bool(evaluation.goal_passed),
        "fail_raw": bool(evaluation.fail_raw),
        "fail_streak": int(evaluation.fail_streak),
        "fail_passed": bool(evaluation.fail_passed),
        "terminated": bool(evaluation.terminated),
        "termination_type": str(evaluation.termination_type),
        "reason": str(evaluation.reason),
    }


def _home_pose_report(
    robot,
    *,
    arm_targets: Mapping[str, float],
    gripper_targets: Mapping[str, float],
) -> dict[str, Any]:
    positions = robot.get_joint_positions()
    arm_errors = {
        name: abs(float(positions[name]) - float(target))
        for name, target in arm_targets.items()
        if name in positions
    }
    gripper_errors = {
        name: abs(float(positions[name]) - float(target))
        for name, target in gripper_targets.items()
        if name in positions
    }
    return {
        "arm_max_abs_error": max(arm_errors.values(), default=0.0),
        "gripper_max_abs_error": max(gripper_errors.values(), default=0.0),
        "arm_positions": {
            name: float(positions[name])
            for name in arm_targets
            if name in positions
        },
        "gripper_positions": {
            name: float(positions[name])
            for name in gripper_targets
            if name in positions
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
