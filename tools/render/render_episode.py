#!/usr/bin/env python3
"""Render a resolved BeTTER episode or a recorded task teleop state log.

Default behavior is state-snapshot rendering: each output frame restores the
recorded robot/object state, zeroes velocities, and advances one small physics
sync tick so Isaac articulation poses are pushed to the renderer. Output video
FPS is independent from this sync tick and defaults to the logged 15 Hz cadence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK_DIR = REPO_ROOT / "assets" / "tasks" / "loose_packing" / "Packing_a_Fruit_Lunch"
DEFAULT_ASSET_REGISTRY_ROOT = REPO_ROOT / "assets" / "objects" / "registry"
DEFAULT_BACKGROUND_REGISTRY = REPO_ROOT / "assets" / "scenes" / "backgrounds" / "registry.v2.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "renders" / "franka_robotiq_episode"
EMBODIMENT_REGISTRY_PATH = REPO_ROOT / "configs" / "embodiments" / "registry.v1.json"
CONTROLLER_PROFILES_PATH = REPO_ROOT / "configs" / "embodiments" / "controller_profiles.v1.json"
DEFAULT_ROBOT_POSITION = (-0.05, 0.0, -0.1)
DEFAULT_RENDER_SYNC_DT = 1.0 / 120.0
FIXED_CAMERA_PRESET_NAMES = ("front_camera", "left_shoulder_camera", "right_shoulder_camera")
CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "front_camera": {
        "prim_path": "/World/Cameras/front_camera",
        "resolution": (448, 448),
        "position": (1.6, 0.0, 0.9),
        "orientation_wxyz": (0.627211, 0.326506, 0.326506, 0.627211),
        "focal_length": 24.0,
        "horizontal_aperture": 20.955,
        "vertical_aperture": 20.955,
        "clipping_range": (0.01, 10000.0),
    },
    "left_shoulder_camera": {
        "prim_path": "/World/Cameras/left_shoulder_camera",
        "resolution": (448, 448),
        "position": (-0.3, 0.9, 0.9),
        "orientation_wxyz": (0.339444, 0.176704, -0.426600, -0.819491),
        "focal_length": 24.0,
        "horizontal_aperture": 20.955,
        "vertical_aperture": 20.955,
        "clipping_range": (0.01, 10000.0),
    },
    "right_shoulder_camera": {
        "prim_path": "/World/Cameras/right_shoulder_camera",
        "resolution": (448, 448),
        "position": (-0.3, -0.9, 0.9),
        "orientation_wxyz": (0.819491, 0.426600, -0.176704, -0.339444),
        "focal_length": 24.0,
        "horizontal_aperture": 20.955,
        "vertical_aperture": 20.955,
        "clipping_range": (0.01, 10000.0),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Recorded state log produced by tools/teleop/franka_robotiq_task_teleop.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--embodiment-id", default="franka_robotiq")
    parser.add_argument("--controller-profile-id", default=None)
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=DEFAULT_RENDER_SYNC_DT,
        help="Physics timestep used for the per-frame articulation/render sync step.",
    )
    parser.add_argument(
        "--rendering-dt",
        type=float,
        default=DEFAULT_RENDER_SYNC_DT,
        help=(
            "Isaac World rendering timestep used for sync. Keep this equal to "
            "--physics-dt for one sync step per saved frame; output video FPS is "
            "controlled by --render-fps."
        ),
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=15.0,
        help="Output video frame-rate metadata. This does not change restored state sampling.",
    )
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--skip-ground-plane", action="store_true")
    parser.add_argument(
        "--add-default-lights",
        dest="add_default_lights",
        action="store_true",
        default=True,
        help="Add local fallback lights when rendering without a background scene.",
    )
    parser.add_argument("--no-default-lights", dest="add_default_lights", action="store_false")
    parser.add_argument("--skip-runtime-control", action="store_true")
    parser.add_argument("--disable-gravity", dest="disable_gravity", action="store_true", default=True)
    parser.add_argument("--enable-gravity", dest="disable_gravity", action="store_false")
    parser.add_argument(
        "--step-physics",
        dest="step_physics",
        action="store_true",
        default=True,
        help="Run the per-frame physics sync step after each restored state. Enabled by default.",
    )
    parser.add_argument(
        "--no-step-physics",
        dest="step_physics",
        action="store_false",
        help=(
            "Experimental render-only path. This can leave Isaac articulation poses "
            "visually stale in camera output."
        ),
    )
    parser.add_argument(
        "--zero-velocities",
        dest="zero_velocities",
        action="store_true",
        default=True,
        help="Zero robot/object velocities after every state restore.",
    )
    parser.add_argument("--keep-logged-velocities", dest="zero_velocities", action="store_false")
    parser.add_argument("--allow-partial-joints", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum output frames. 0 means all log frames, or one static frame without a log.",
    )
    parser.add_argument(
        "--startup-frames",
        type=int,
        default=32,
        help=(
            "Unsaved warmup frames before saving output. If a state log is provided, "
            "each warmup frame restores a sampled state first and primes camera buffers."
        ),
    )
    parser.add_argument(
        "--startup-state-sampling",
        choices=("random", "uniform"),
        default="random",
        help="How to sample state rows during startup warmup.",
    )
    parser.add_argument(
        "--startup-seed",
        type=int,
        default=1234,
        help="Deterministic seed for random startup state sampling.",
    )
    parser.add_argument(
        "--first-frame-warmup-frames",
        type=int,
        default=8,
        help=(
            "Unsaved warmup frames before the first output frame. The first state is "
            "restored again before saving frame 0."
        ),
    )
    parser.add_argument("--flush-warmup-frames", type=int, default=3)
    parser.add_argument("--disable-no-ghosting", action="store_true")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["front_camera"],
        help=(
            "Camera presets to render. Use 'all' for the fixed benchmark "
            "front/left/right cameras, or 'custom' to use --camera-* arguments."
        ),
    )
    parser.add_argument("--camera-id", default="render_camera")
    parser.add_argument("--camera-prim-path", default="/World/Cameras/RenderCamera")
    parser.add_argument("--camera-resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--camera-position", type=float, nargs=3, default=(-0.4, 0.0, 0.8))
    parser.add_argument(
        "--camera-orientation-wxyz",
        type=float,
        nargs=4,
        default=(0.612372, 0.353553, -0.353553, -0.612372),
    )
    parser.add_argument("--camera-focal-length", type=float, default=18.14756)
    parser.add_argument("--camera-horizontal-aperture", type=float, default=15.2908)
    parser.add_argument("--camera-vertical-aperture", type=float, default=15.2908)
    parser.add_argument("--camera-clipping-range", type=float, nargs=2, default=(0.01, 10.0))
    parser.add_argument("--write-video", dest="write_video", action="store_true", default=True)
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.add_argument(
        "--save-images",
        dest="save_images",
        action="store_true",
        default=False,
        help="Also save per-frame RGB PNG files under OUTPUT_DIR/rgb. Off by default.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory for MP4 outputs. Defaults to OUTPUT_DIR/videos.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="MP4 frame rate. Defaults to --render-fps.",
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        default=8,
        help="imageio/ffmpeg quality, typically 0-10.",
    )
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _preflight_args(args)
    log_payload = _load_episode_state_log(args.record_path) if args.record_path else None

    from isaacsim import SimulationApp

    launch_config: dict[str, Any] = {
        "headless": bool(args.headless),
        "hide_ui": bool(args.headless),
        "renderer": str(args.renderer),
        "multi_gpu": False,
    }
    if not bool(args.disable_no_ghosting):
        launch_config["extra_args"] = _build_no_ghosting_extra_args()

    simulation_app = SimulationApp(launch_config)

    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage

        from src.scene.episode_runtime import (
            EpisodeRuntimeLoadConfig,
            restore_episode_object_poses,
            runtime_object_poses_from_dict,
            load_resolved_episode_into_stage,
        )
        from src.scene.robots import (
            apply_embodiment_runtime_control,
            create_robot,
            load_registry,
            resolve_embodiment_paths,
            resolve_runtime_control,
        )
        from src.task.specs.episode import resolve_episode, resolved_episode_from_dict
        from src.task.specs.loader import load_task_spec
        from src.task.specs.schema import ResolvedBackgroundSpec
        from tools.teleop.franka_robotiq_keyboard_pd_jog import (
            _apply_ready_pose,
            _find_articulation_root_path,
            _physics_steps_per_control,
        )

        _verify_renderer_settings()

        episode = (
            resolved_episode_from_dict(log_payload["metadata"]["resolved_episode"])
            if log_payload is not None
            else _resolve_static_episode(args)
        )
        if bool(args.load_background) and episode.background is None:
            episode = replace(episode, background=_resolve_background(args, ResolvedBackgroundSpec))
        elif not bool(args.load_background) and episode.background is not None:
            episode = replace(episode, background=None)
        state_rows = [] if log_payload is None else list(log_payload["states"])

        physics_steps_per_render = _physics_steps_per_control(
            physics_dt=float(args.physics_dt),
            control_dt=float(args.rendering_dt),
        )
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(args.physics_dt),
            rendering_dt=float(args.rendering_dt),
        )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable after SimulationApp startup.")
        if not bool(args.load_background):
            if not bool(args.skip_ground_plane):
                _add_local_ground_plane(stage)
            if bool(args.add_default_lights):
                _add_local_lighting(stage)

        _remove_prim_subtree(stage, str(args.prim_root))
        _ensure_xform(stage, str(args.prim_root))
        _remove_prim_subtree(stage, str(args.reference_prim_path))

        load_resolved_episode_into_stage(
            stage,
            episode,
            config=EpisodeRuntimeLoadConfig(
                load_background=bool(args.load_background),
                restore_object_poses=not bool(state_rows),
                capture_light_baseline=False,
                require_background_registry=bool(args.load_background),
            ),
        )

        registry = load_registry(EMBODIMENT_REGISTRY_PATH)
        embodiment = registry.get_embodiment(str(args.embodiment_id))
        runtime_control = resolve_runtime_control(
            embodiment,
            controller_profile_id=args.controller_profile_id,
        )
        runtime_embodiment = replace(embodiment, runtime_control=runtime_control)
        resolved_paths = resolve_embodiment_paths(registry, embodiment)
        robot_usd_path = resolved_paths["runtime_asset_path"]
        if not robot_usd_path.exists():
            raise FileNotFoundError(f"Robot USD does not exist: {robot_usd_path}")

        add_reference_to_stage(str(robot_usd_path), prim_path=str(args.reference_prim_path))
        _set_reference_prim_position(stage, str(args.reference_prim_path), args.robot_position)
        articulation_root_path = _find_articulation_root_path(str(args.reference_prim_path))
        robot = create_robot(
            instance_id="franka_robotiq_render",
            prim_path=articulation_root_path,
            embodiment=runtime_embodiment,
        )
        robot.set_world_pose(position=_np_array(args.robot_position))

        world.reset()
        if bool(args.disable_gravity):
            _disable_stage_gravity(stage)

        if not state_rows:
            restore_episode_object_poses(stage, episode)
        robot.initialize()
        if not bool(args.skip_runtime_control):
            apply_embodiment_runtime_control(
                robot,
                runtime_control,
                controller_profiles_path=CONTROLLER_PROFILES_PATH,
            )

        if state_rows:
            _restore_state_row(
                args=args,
                stage=stage,
                episode=episode,
                robot=robot,
                state_row=state_rows[0],
            )
        else:
            _apply_ready_pose(robot)
            _zero_robot_velocities(robot)
            if bool(args.zero_velocities):
                _zero_episode_object_velocities(stage, episode)

        cameras = _create_cameras(stage, args)
        startup_state_indices = _render_startup(
            args=args,
            world=world,
            stage=stage,
            episode=episode,
            robot=robot,
            cameras=cameras,
            states=state_rows,
        )
        if not bool(args.disable_no_ghosting):
            _force_renderer_flush(
                world,
                stage=stage,
                root_prim_path="/World",
                warmup_frames=int(args.flush_warmup_frames),
                step_physics=bool(args.step_physics),
                physics_steps_per_render=int(physics_steps_per_render),
                cameras=cameras,
            )

        out_dir = args.output_dir.expanduser().resolve()
        camera_specs_by_id = {
            str(spec["camera_id"]): spec for spec in _selected_camera_specs(args)
        }
        rgb_dirs = (
            {camera_id: out_dir / "rgb" / camera_id for camera_id in cameras}
            if bool(args.save_images)
            else {}
        )
        for rgb_dir in rgb_dirs.values():
            rgb_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        selected_indices = _selected_state_indices(
            state_count=len(state_rows),
            stride=int(args.frame_stride),
            max_frames=int(args.max_frames),
        )
        if not selected_indices:
            selected_indices = [None]

        print(
            "[render_episode] start "
            + json.dumps(
                {
                    "record_path": None if args.record_path is None else str(args.record_path),
                    "output_dir": str(out_dir),
                    "frame_count": len(selected_indices),
                    "step_physics": bool(args.step_physics),
                    "physics_dt": float(args.physics_dt),
                    "rendering_dt": float(args.rendering_dt),
                    "render_fps": float(args.render_fps),
                    "startup_frames": int(args.startup_frames),
                    "startup_state_sampling": str(args.startup_state_sampling),
                    "startup_seed": int(args.startup_seed),
                    "startup_state_indices": startup_state_indices,
                    "video_fps": _video_fps(args),
                    "video_backend": _preferred_video_backend() if bool(args.write_video) else None,
                    "write_video": bool(args.write_video),
                    "save_images": bool(args.save_images),
                    "load_background": bool(args.load_background),
                    "default_lights": bool(args.add_default_lights) and not bool(args.load_background),
                    "physics_steps_per_render_if_enabled": int(physics_steps_per_render),
                    "cameras": list(cameras.keys()),
                    "robot_position": [float(value) for value in args.robot_position],
                },
                sort_keys=True,
            ),
            flush=True,
        )

        video_paths = _video_paths(args, out_dir=out_dir, camera_ids=list(cameras)) if bool(args.write_video) else {}
        video_writers = (
            _VideoWriterSet.open(video_paths, fps=_video_fps(args), quality=int(args.video_quality))
            if video_paths
            else None
        )
        try:
            for output_index, state_index in enumerate(selected_indices):
                state_row = None if state_index is None else state_rows[state_index]
                if state_row is not None:
                    _restore_state_row(
                        args=args,
                        stage=stage,
                        episode=episode,
                        robot=robot,
                        state_row=state_row,
                    )
                if output_index == 0:
                    _render_warmup_frames(
                        world,
                        frame_count=int(args.first_frame_warmup_frames),
                        step_physics=bool(args.step_physics),
                        physics_steps_per_render=int(physics_steps_per_render),
                        cameras=cameras,
                    )
                    if state_row is not None:
                        _restore_state_row(
                            args=args,
                            stage=stage,
                            episode=episode,
                            robot=robot,
                            state_row=state_row,
                        )
                _render_frame(
                    world,
                    step_physics=bool(args.step_physics),
                    physics_steps_per_render=int(physics_steps_per_render),
                )
                rgb_paths: dict[str, str] = {}
                for camera_id, camera in cameras.items():
                    rgb = _camera_rgb(camera)
                    if bool(args.save_images):
                        rgb_path = rgb_dirs[camera_id] / f"{output_index:06d}.png"
                        _write_rgb_png(rgb_path, rgb)
                        rgb_paths[camera_id] = str(rgb_path.relative_to(out_dir))
                    if video_writers is not None:
                        video_writers.append(camera_id, rgb)
                manifest_row = {
                    "frame_index": int(output_index),
                    "state_row_index": None if state_index is None else int(state_index),
                    "logged_step_index": None if state_row is None else state_row.get("step_index"),
                    "timestamp": None if state_row is None else state_row.get("timestamp"),
                }
                if bool(args.save_images):
                    manifest_row["rgb"] = rgb_paths
                _append_manifest(manifest_path, manifest_row)
                if int(args.print_every) > 0 and (
                    output_index == 0
                    or (output_index + 1) % int(args.print_every) == 0
                    or output_index == len(selected_indices) - 1
                ):
                    print(
                        "[render_episode] progress "
                        + json.dumps(
                            {
                                "frame_index": int(output_index),
                                "state_row_index": None if state_index is None else int(state_index),
                                "rgb": {
                                    camera_id: str(out_dir / rel_path)
                                    for camera_id, rel_path in rgb_paths.items()
                                },
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            if video_writers is not None:
                video_writers.close()

        metadata = {
            "schema_version": "better.render_episode.v1",
            "record_path": None if args.record_path is None else str(args.record_path),
            "output_dir": str(out_dir),
            "frame_count": len(selected_indices),
            "cameras": {
                camera_id: {
                    "prim_path": str(camera_specs_by_id[camera_id]["prim_path"]),
                    "resolution": [
                        int(v) for v in camera_specs_by_id[camera_id]["resolution"]
                    ],
                    "position": [
                        float(v) for v in camera_specs_by_id[camera_id]["position"]
                    ],
                    "orientation_wxyz": [
                        float(v)
                        for v in camera_specs_by_id[camera_id]["orientation_wxyz"]
                    ],
                }
                for camera_id in cameras
            },
            "timing": {
                "physics_dt": float(args.physics_dt),
                "rendering_dt": float(args.rendering_dt),
                "render_fps": float(args.render_fps),
                "video_fps": _video_fps(args),
                "video_backend": _preferred_video_backend() if bool(args.write_video) else None,
                "write_video": bool(args.write_video),
                "save_images": bool(args.save_images),
                "step_physics": bool(args.step_physics),
                "load_background": bool(args.load_background),
                "default_lights": bool(args.add_default_lights) and not bool(args.load_background),
                "physics_steps_per_render_if_enabled": int(physics_steps_per_render),
                "startup_frames": int(args.startup_frames),
                "startup_state_sampling": str(args.startup_state_sampling),
                "startup_seed": int(args.startup_seed),
                "startup_state_indices": startup_state_indices,
                "first_frame_warmup_frames": int(args.first_frame_warmup_frames),
                "flush_warmup_frames": (
                    0
                    if bool(args.disable_no_ghosting)
                    else int(args.flush_warmup_frames)
                ),
            },
            "episode": {
                "task_id": episode.task_id,
                "variation_id": episode.variation_id,
                "episode_seed": int(episode.episode_seed),
                "object_count": len(episode.objects),
            },
            "robot": {
                "runtime_asset_path": str(robot_usd_path),
                "reference_prim_path": str(args.reference_prim_path),
                "articulation_root_path": str(articulation_root_path),
                "robot_position": [float(value) for value in args.robot_position],
            },
            "videos": {
                camera_id: _manifest_path(path, root=out_dir)
                for camera_id, path in video_paths.items()
            },
        }
        if bool(args.save_images):
            metadata["images"] = {
                camera_id: _manifest_path(path, root=out_dir)
                for camera_id, path in rgb_dirs.items()
            }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print(
            "[render_episode] done "
            + json.dumps(
                {
                    "output_dir": str(out_dir),
                    "frame_count": len(selected_indices),
                    "save_images": bool(args.save_images),
                    "manifest": str(manifest_path),
                    "videos": {camera_id: str(path) for camera_id, path in video_paths.items()},
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


def _preflight_args(args: argparse.Namespace) -> None:
    if args.record_path is not None:
        record_path = args.record_path.expanduser().resolve()
        if not record_path.exists():
            raise SystemExit(f"[render_episode] record file does not exist: {record_path}")
        if not record_path.is_file():
            raise SystemExit(f"[render_episode] record path is not a file: {record_path}")
    else:
        task_dir = args.task_dir.expanduser().resolve()
        if not task_dir.exists():
            raise SystemExit(f"[render_episode] task directory does not exist: {task_dir}")

    if bool(args.load_background):
        background_registry = args.background_registry.expanduser().resolve()
        if not background_registry.exists():
            raise SystemExit(
                f"[render_episode] background registry does not exist: {background_registry}"
            )
        if not background_registry.is_file():
            raise SystemExit(
                f"[render_episode] background registry path is not a file: {background_registry}"
            )

    camera_names = _requested_camera_names(args)
    unknown = [name for name in camera_names if name not in CAMERA_PRESETS and name != "custom"]
    if unknown:
        available = ", ".join((*FIXED_CAMERA_PRESET_NAMES, "all", "custom"))
        raise SystemExit(
            f"[render_episode] unknown camera preset(s): {unknown}. Available: {available}"
        )

    if int(args.startup_frames) < 0:
        raise SystemExit("[render_episode] --startup-frames must be >= 0")
    if int(args.first_frame_warmup_frames) < 0:
        raise SystemExit("[render_episode] --first-frame-warmup-frames must be >= 0")
    if int(args.flush_warmup_frames) < 0:
        raise SystemExit("[render_episode] --flush-warmup-frames must be >= 0")
    if not bool(args.write_video) and not bool(args.save_images):
        raise SystemExit("[render_episode] nothing to write: keep --write-video enabled or pass --save-images.")
    if bool(args.write_video):
        if _preferred_video_backend() is None:
            raise SystemExit(
                "[render_episode] MP4 writing requires either imageio with an "
                "ffmpeg/pyav backend or OpenCV. Install one of those or pass --no-video."
            )
        _video_fps(args)
        if int(args.video_quality) < 0:
            raise SystemExit("[render_episode] --video-quality must be >= 0")


def _requested_camera_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    for item in args.cameras:
        names.extend(part for part in str(item).split(",") if part)
    if not names:
        return ["front_camera"]
    if "all" in names:
        return list(FIXED_CAMERA_PRESET_NAMES)
    return names


def _build_no_ghosting_extra_args() -> list[str]:
    return [
        "--/rtx/hydra/LOD/enabled=false",
        "--/rtx/hydra/LOD/maxRefinementLevel=3",
        "--/rtx/texturestreaming/enabled=false",
        "--/rtx/texturestreaming/minMipLevel=0",
        "--/rtx/texturestreaming/memoryBudget=24000",
        "--/rtx/hydra/enableDynamicResolution=false",
        "--/rtx/hydra/syncLoading=true",
        "--/rtx/hydra/material/syncLoading=true",
        "--/rtx/materialDb/syncLoads=true",
        "--/omni.kit.plugin/syncUsdLoads=true",
        "--/rtx/post/motionblur/enabled=false",
        "--/rtx/post/aa/op=2",
        "--/rtx/hydra/culling/encoding/enabled=false",
        "--/rtx/hydra/culling/sensor/enabled=false",
        "--/rtx/realtime/resetAccumulationOnCameraMove=true",
        "--/rtx/realtime/dlss/enabled=false",
        "--/rtx/hydra/asyncShaderCompile=false",
        "--/rtx/hydra/waitOnShaderCompilation=true",
        "--/rtx/realtime/neuray/resetAccumulation=true",
        "--/rtx/realtime/optix/denoiser/blendFactor=0.0",
        "--/rtx/hydra/forceUpdate=true",
    ]


def _verify_renderer_settings() -> None:
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/rtx/hydra/LOD/enabled", False)
    settings.set("/rtx/texturestreaming/enabled", False)
    settings.set("/rtx/hydra/enableDynamicResolution", False)
    settings.set("/rtx/texturestreaming/minMipLevel", 0)
    settings.set("/rtx/texturestreaming/memoryBudget", 24000)


def _load_episode_state_log(path: Path) -> dict[str, Any]:
    record_path = path.expanduser().resolve()
    if not record_path.exists():
        raise FileNotFoundError(f"Record file does not exist: {record_path}")

    suffix = record_path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        return _load_episode_state_pickle(record_path)
    if suffix in ("", ".jsonl"):
        return _load_episode_state_jsonl(record_path)
    raise ValueError(
        f"Unsupported record file suffix {suffix!r}; expected .pkl or .jsonl."
    )


def _load_episode_state_pickle(record_path: Path) -> dict[str, Any]:
    with record_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Record pickle payload must be a mapping: {record_path}")

    metadata = payload.get("metadata")
    states = payload.get("states")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Record pickle has no metadata mapping: {record_path}")
    if "resolved_episode" not in metadata:
        raise ValueError(
            f"Record pickle metadata has no resolved_episode payload: {record_path}"
        )
    if not isinstance(states, list) or not states:
        raise ValueError(f"Record pickle has no episode_state rows: {record_path}")
    return {"metadata": dict(metadata), "states": states}



def _load_episode_state_jsonl(record_path: Path) -> dict[str, Any]:
    metadata = None
    states: list[dict[str, Any]] = []
    with record_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event = row.get("event")
            if event == "metadata":
                metadata = row
            elif event == "episode_state":
                states.append(row)
            else:
                raise ValueError(
                    f"Unsupported record row event at {record_path}:{line_number}: {event!r}"
                )

    if metadata is None:
        raise ValueError(f"Record JSONL has no metadata row: {record_path}")
    if "resolved_episode" not in metadata:
        raise ValueError(
            f"Record JSONL metadata has no resolved_episode payload: {record_path}"
        )
    if not states:
        raise ValueError(f"Record JSONL has no episode_state rows: {record_path}")
    return {"metadata": metadata, "states": states}


def _resolve_static_episode(args: argparse.Namespace):
    from src.task.specs.episode import resolve_episode
    from src.task.specs.loader import load_task_spec
    from src.task.specs.schema import ResolvedBackgroundSpec

    task_spec = load_task_spec(args.task_dir.expanduser().resolve())
    return resolve_episode(
        task_spec,
        args.variation_id,
        asset_registry_root=args.asset_registry_root.expanduser().resolve(),
        episode_seed=int(args.episode_seed),
        background=(
            _resolve_background(args, ResolvedBackgroundSpec)
            if bool(args.load_background)
            else None
        ),
        prim_root=str(args.prim_root),
    )


def _resolve_background(args: argparse.Namespace, background_cls: Any) -> Any:
    return background_cls(
        scene_id=_resolve_background_scene_id(args),
        material_id=args.background_material_id,
        layout_id=args.background_layout_id,
        registry_path=str(args.background_registry.expanduser().resolve()),
        prim_path=str(args.background_prim_path),
    )


def _resolve_background_scene_id(args: argparse.Namespace) -> str:
    if args.background_scene_id is not None:
        return str(args.background_scene_id)
    payload = json.loads(args.background_registry.expanduser().resolve().read_text(encoding="utf-8"))
    return str(payload["default_scene_id"])


def _remove_prim_subtree(stage: Any, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        return
    stage.RemovePrim(prim_path)


def _ensure_xform(stage: Any, prim_path: str) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        UsdGeom.Xform.Define(stage, prim_path)


def _add_local_ground_plane(stage: Any, prim_path: str = "/World/GroundPlane") -> None:
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


def _add_local_lighting(stage: Any) -> None:
    from pxr import Gf, UsdGeom, UsdLux

    _ensure_xform(stage, "/World/DefaultLights")

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


def _set_or_add_xform_op(xformable: Any, op_type: Any, value: Any) -> None:
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


def _set_reference_prim_position(stage: Any, prim_path: str, position: Any) -> None:
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


def _np_array(values: Any):
    import numpy as np

    return np.asarray(values, dtype=np.float32)


def _disable_stage_gravity(stage: Any) -> None:
    from pxr import Gf

    prim = stage.GetPrimAtPath("/World/physicsScene")
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        return
    if prim.HasAttribute("gravityDirection"):
        prim.GetAttribute("gravityDirection").Set(Gf.Vec3f(0.0, 0.0, 0.0))
    if prim.HasAttribute("gravityMagnitude"):
        prim.GetAttribute("gravityMagnitude").Set(0.0)


def _create_cameras(stage: Any, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from omni.isaac.sensor import Camera
    from isaacsim.core.utils.prims import define_prim, is_prim_path_valid

    _ensure_xform(stage, "/World/Cameras")
    cameras: dict[str, Any] = {}
    for camera_spec in _selected_camera_specs(args):
        camera_id = str(camera_spec["camera_id"])
        prim_path = str(camera_spec["prim_path"])
        if not is_prim_path_valid(prim_path):
            define_prim(prim_path, "Camera")

        camera = Camera(
            prim_path=prim_path,
            name=camera_id,
            frequency=float(args.render_fps),
            resolution=(
                int(camera_spec["resolution"][0]),
                int(camera_spec["resolution"][1]),
            ),
        )
        camera.set_focal_length(float(camera_spec["focal_length"]))
        camera.set_horizontal_aperture(float(camera_spec["horizontal_aperture"]))
        camera.set_vertical_aperture(float(camera_spec["vertical_aperture"]))
        camera.set_clipping_range(
            float(camera_spec["clipping_range"][0]),
            float(camera_spec["clipping_range"][1]),
        )
        camera.set_world_pose(
            position=np.asarray(camera_spec["position"], dtype=float),
            orientation=np.asarray(camera_spec["orientation_wxyz"], dtype=float),
            camera_axes="usd",
        )
        camera.initialize()
        cameras[camera_id] = camera
    return cameras


def _selected_camera_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for camera_name in _requested_camera_names(args):
        if camera_name == "custom":
            specs.append(
                {
                    "camera_id": str(args.camera_id),
                    "prim_path": str(args.camera_prim_path),
                    "resolution": (
                        int(args.camera_resolution[0]),
                        int(args.camera_resolution[1]),
                    ),
                    "position": tuple(float(v) for v in args.camera_position),
                    "orientation_wxyz": tuple(float(v) for v in args.camera_orientation_wxyz),
                    "focal_length": float(args.camera_focal_length),
                    "horizontal_aperture": float(args.camera_horizontal_aperture),
                    "vertical_aperture": float(args.camera_vertical_aperture),
                    "clipping_range": (
                        float(args.camera_clipping_range[0]),
                        float(args.camera_clipping_range[1]),
                    ),
                }
            )
            continue

        preset = CAMERA_PRESETS[camera_name]
        specs.append({"camera_id": camera_name, **preset})
    return specs


def _restore_state_row(
    *,
    args: argparse.Namespace,
    stage: Any,
    episode: Any,
    robot: Any,
    state_row: Mapping[str, Any],
) -> dict[str, Any]:
    from src.scene.episode_runtime import (
        restore_episode_object_poses,
        runtime_object_poses_from_dict,
    )

    object_payload = state_row.get("objects")
    if not isinstance(object_payload, Mapping):
        raise ValueError("episode_state row is missing object pose mapping")
    restore_episode_object_poses(
        stage,
        episode,
        poses=runtime_object_poses_from_dict(object_payload),
    )

    robot_payload = state_row.get("robot")
    if not isinstance(robot_payload, dict):
        raise ValueError("episode_state row is missing robot runtime state")

    report = robot.restore_runtime_state(
        robot_payload,
        require_all_joints=not bool(args.allow_partial_joints),
        restore_velocities=not bool(args.zero_velocities),
        zero_missing_velocities=True,
        mirror_position_targets=True,
    )
    if bool(args.zero_velocities):
        _zero_robot_velocities(robot)
        _zero_episode_object_velocities(stage, episode)
    return report


def _zero_robot_velocities(robot: Any) -> None:
    joint_positions = robot.get_joint_positions()
    robot.set_joint_velocities({joint_name: 0.0 for joint_name in joint_positions})


def _zero_episode_object_velocities(stage: Any, episode: Any) -> int:
    from pxr import Gf, Usd, UsdPhysics

    count = 0
    zero = Gf.Vec3f(0.0, 0.0, 0.0)
    for obj in episode.objects.values():
        root_prim = stage.GetPrimAtPath(obj.prim_path)
        if root_prim is None or (hasattr(root_prim, "IsValid") and not root_prim.IsValid()):
            continue
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body = UsdPhysics.RigidBodyAPI(prim)
                rigid_body.CreateVelocityAttr().Set(zero)
                rigid_body.CreateAngularVelocityAttr().Set(zero)
                count += 1
    return count


def _render_frame(
    world: Any,
    *,
    step_physics: bool,
    physics_steps_per_render: int,
) -> None:
    if step_physics:
        for _ in range(max(1, int(physics_steps_per_render))):
            world.step(render=True, step_sim=True)
        return
    # Articulation state restored through Isaac APIs is not reliably pushed to
    # Hydra/Fabric by world.render() alone. Keep physics paused, but use the
    # World step path to synchronize robot joint poses first. In Isaac Sim 5,
    # step_sim=False does not always produce a fresh Camera RGB annotator frame,
    # so follow with world.render() for sensor output.
    world.step(render=True, step_sim=False)
    if hasattr(world, "render"):
        world.render()


def _render_startup(
    *,
    args: argparse.Namespace,
    world: Any,
    stage: Any,
    episode: Any,
    robot: Any,
    cameras: Mapping[str, Any],
    states: list[dict[str, Any]],
) -> list[int]:
    frame_count = max(0, int(args.startup_frames))
    if frame_count <= 0:
        return []

    sampled_indices = _startup_state_indices(
        state_count=len(states),
        frame_count=frame_count,
        sampling=str(args.startup_state_sampling),
        seed=int(args.startup_seed),
    )
    for frame_index in range(frame_count):
        if sampled_indices:
            state_row = states[sampled_indices[frame_index]]
            _restore_state_row(
                args=args,
                stage=stage,
                episode=episode,
                robot=robot,
                state_row=state_row,
            )
        _render_frame(
            world,
            step_physics=bool(args.step_physics),
            physics_steps_per_render=1,
        )
        _prime_camera_buffers(cameras)
    return sampled_indices


def _startup_state_indices(
    *,
    state_count: int,
    frame_count: int,
    sampling: str,
    seed: int,
) -> list[int]:
    if state_count <= 0 or frame_count <= 0:
        return []
    if sampling == "uniform":
        return [
            min(state_count - 1, int(frame_index * state_count / frame_count))
            for frame_index in range(frame_count)
        ]
    if sampling != "random":
        raise ValueError(f"Unsupported startup state sampling mode: {sampling}")
    rng = random.Random(int(seed))
    if frame_count <= state_count:
        indices = rng.sample(range(state_count), k=frame_count)
        rng.shuffle(indices)
        return indices
    indices = list(range(state_count))
    rng.shuffle(indices)
    while len(indices) < frame_count:
        indices.append(rng.randrange(state_count))
    return indices


def _render_warmup_frames(
    world: Any,
    *,
    frame_count: int,
    step_physics: bool,
    physics_steps_per_render: int,
    cameras: Mapping[str, Any],
) -> None:
    for _ in range(max(0, int(frame_count))):
        _render_frame(
            world,
            step_physics=bool(step_physics),
            physics_steps_per_render=int(physics_steps_per_render),
        )
        _prime_camera_buffers(cameras)


def _force_renderer_flush(
    world: Any,
    *,
    stage: Any,
    root_prim_path: str,
    warmup_frames: int,
    step_physics: bool,
    physics_steps_per_render: int,
    cameras: Mapping[str, Any],
) -> None:
    import carb.settings
    from pxr import UsdGeom

    settings = carb.settings.get_settings()
    settings.set("/rtx/resetPtAccum", True)

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if root_prim is not None and (not hasattr(root_prim, "IsValid") or root_prim.IsValid()):
        imageable = UsdGeom.Imageable(root_prim)
        visibility_attr = imageable.CreateVisibilityAttr()
        previous = visibility_attr.Get() or "inherited"
        visibility_attr.Set("invisible")
        _render_frame(
            world,
            step_physics=bool(step_physics),
            physics_steps_per_render=int(physics_steps_per_render),
        )
        _prime_camera_buffers(cameras)
        visibility_attr.Set(previous)
        _render_frame(
            world,
            step_physics=bool(step_physics),
            physics_steps_per_render=int(physics_steps_per_render),
        )
        _prime_camera_buffers(cameras)

    settings.set("/rtx/resetPtAccum", True)
    for _ in range(max(0, int(warmup_frames))):
        _render_frame(
            world,
            step_physics=bool(step_physics),
            physics_steps_per_render=int(physics_steps_per_render),
        )
        _prime_camera_buffers(cameras)


def _prime_camera_buffers(cameras: Mapping[str, Any]) -> None:
    for camera in cameras.values():
        camera.get_rgba()


def _selected_state_indices(
    *,
    state_count: int,
    stride: int,
    max_frames: int,
) -> list[int]:
    if state_count <= 0:
        return []
    indices = list(range(0, state_count, max(1, int(stride))))
    if int(max_frames) > 0:
        indices = indices[: int(max_frames)]
    return indices


def _camera_rgb(camera: Any):
    import numpy as np

    rgba = camera.get_rgba()
    if rgba is None:
        raise RuntimeError("Camera returned no RGBA data.")
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"Expected RGBA/RGB camera array, got shape {array.shape}.")
    rgb = array[:, :, :3]
    if rgb.dtype.kind == "f":
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _video_fps(args: argparse.Namespace) -> float:
    fps = float(args.render_fps if args.video_fps is None else args.video_fps)
    if fps <= 0.0:
        raise SystemExit("[render_episode] video fps must be > 0")
    return fps


def _video_paths(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    camera_ids: list[str],
) -> dict[str, Path]:
    video_dir = (
        out_dir / "videos"
        if args.video_dir is None
        else args.video_dir.expanduser().resolve()
    )
    return {camera_id: video_dir / f"{camera_id}.mp4" for camera_id in camera_ids}


def _preferred_video_backend() -> str | None:
    if importlib.util.find_spec("imageio") is not None and (
        importlib.util.find_spec("imageio_ffmpeg") is not None
        or importlib.util.find_spec("av") is not None
    ):
        return "imageio"
    if importlib.util.find_spec("cv2") is not None:
        return "opencv"
    return None


def _manifest_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class _VideoWriterSet:
    def __init__(self, writers: dict[str, Any]) -> None:
        self._writers = dict(writers)

    @classmethod
    def open(
        cls,
        paths: Mapping[str, Path],
        *,
        fps: float,
        quality: int,
    ) -> "_VideoWriterSet":
        if not paths:
            return cls({})
        backend = _preferred_video_backend()
        if backend is None:
            raise RuntimeError(
                "MP4 writing requires either imageio with an ffmpeg/pyav backend "
                "or OpenCV. Install one of those or pass --no-video."
            )
        if backend == "opencv":
            return cls(
                {
                    camera_id: _OpenCvVideoWriter(path, fps=float(fps))
                    for camera_id, path in paths.items()
                }
            )

        import imageio.v2 as imageio

        writers: dict[str, Any] = {}
        try:
            for camera_id, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    path.unlink()
                writers[camera_id] = imageio.get_writer(
                    str(path),
                    fps=float(fps),
                    codec="libx264",
                    quality=int(quality),
                    macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )
        except BaseException:
            for writer in writers.values():
                writer.close()
            raise
        return cls(writers)

    def append(self, camera_id: str, rgb: Any) -> None:
        writer = self._writers.get(camera_id)
        if writer is None:
            raise KeyError(f"No video writer for camera: {camera_id}")
        writer.append_data(rgb)

    def close(self) -> None:
        errors: list[BaseException] = []
        for writer in self._writers.values():
            try:
                writer.close()
            except BaseException as exc:
                errors.append(exc)
        self._writers.clear()
        if errors:
            raise RuntimeError(f"Failed to close {len(errors)} video writer(s).") from errors[0]


class _OpenCvVideoWriter:
    def __init__(self, path: Path, *, fps: float) -> None:
        self._path = path
        self._fps = float(fps)
        self._writer = None
        self._frame_shape: tuple[int, int] | None = None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._path.unlink()

    def append_data(self, rgb: Any) -> None:
        import cv2
        import numpy as np

        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] != 3:
            raise RuntimeError(f"Expected RGB frame with shape HxWx3, got {array.shape}.")
        height, width = int(array.shape[0]), int(array.shape[1])
        if self._writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._path),
                fourcc,
                self._fps,
                (width, height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"OpenCV failed to open MP4 writer: {self._path}")
            self._frame_shape = (height, width)
        elif self._frame_shape != (height, width):
            raise RuntimeError(
                f"Video frame size changed for {self._path}: "
                f"expected {self._frame_shape}, got {(height, width)}."
            )

        frame_bgr = np.ascontiguousarray(array[:, :, ::-1])
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _write_rgb_png(path: Path, rgb: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("PIL") is not None:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    if importlib.util.find_spec("imageio") is not None:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
        return
    if importlib.util.find_spec("cv2") is not None:
        import cv2

        cv2.imwrite(str(path), rgb[:, :, ::-1])
        return
    raise RuntimeError("Saving PNG requires one of: Pillow, imageio, or OpenCV.")


def _append_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
