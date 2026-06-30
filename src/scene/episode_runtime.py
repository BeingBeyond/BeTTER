from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.task.specs.schema import ResolvedBackgroundSpec, ResolvedEpisodeSpec


CreatePrimFn = Callable[..., Any]
XformPrimFactory = Callable[[str], Any]
ScaleReader = Callable[[Any, str], tuple[float, ...]]
ScaleWriter = Callable[[Any, str, tuple[float, ...]], None]
SceneManagerFactory = Callable[[str | Path], Any]


@dataclass(frozen=True)
class EpisodeRuntimeLoadConfig:
    load_background: bool = True
    load_objects: bool = True
    restore_object_poses: bool = True
    validate_asset_paths: bool = True
    capture_light_baseline: bool = True
    require_background_registry: bool = False


@dataclass(frozen=True)
class RuntimeObjectPose:
    position: tuple[float, ...]
    rotation: tuple[float, ...]
    scale: tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class RuntimeObjectHandle:
    instance_id: str
    prim_path: str
    asset_path: str


@dataclass(frozen=True)
class RuntimeEpisodeHandle:
    episode: ResolvedEpisodeSpec
    objects: dict[str, RuntimeObjectHandle]
    background_scene: Any | None = None


def load_resolved_episode_into_stage(
    stage: Any,
    episode: ResolvedEpisodeSpec,
    *,
    config: EpisodeRuntimeLoadConfig | None = None,
    create_prim: CreatePrimFn | None = None,
    xform_prim_factory: XformPrimFactory | None = None,
    scale_writer: ScaleWriter | None = None,
    scene_manager_factory: SceneManagerFactory | None = None,
) -> RuntimeEpisodeHandle:
    """Load background and foreground objects for a resolved episode."""

    cfg = config or EpisodeRuntimeLoadConfig()
    background_scene = None
    if cfg.load_background:
        background_scene = load_episode_background(
            stage,
            episode.background,
            capture_light_baseline=cfg.capture_light_baseline,
            require_registry=cfg.require_background_registry,
            scene_manager_factory=scene_manager_factory,
        )

    object_handles: dict[str, RuntimeObjectHandle] = {}
    if cfg.load_objects:
        create = create_prim or _default_create_prim
        for instance_id, obj in sorted(episode.objects.items()):
            asset_path = Path(obj.asset.asset_path).expanduser()
            if cfg.validate_asset_paths and not asset_path.exists():
                raise FileNotFoundError(f"Resolved episode asset path does not exist: {asset_path}")

            create(prim_path=obj.prim_path, usd_path=str(asset_path))
            object_handles[instance_id] = RuntimeObjectHandle(
                instance_id=instance_id,
                prim_path=obj.prim_path,
                asset_path=str(asset_path),
            )

    if cfg.restore_object_poses:
        restore_episode_object_poses(
            stage,
            episode,
            xform_prim_factory=xform_prim_factory,
            scale_writer=scale_writer,
        )

    return RuntimeEpisodeHandle(
        episode=episode,
        objects=object_handles,
        background_scene=background_scene,
    )


def load_episode_background(
    stage: Any,
    background: ResolvedBackgroundSpec | None,
    *,
    capture_light_baseline: bool = True,
    require_registry: bool = False,
    scene_manager_factory: SceneManagerFactory | None = None,
) -> Any | None:
    if background is None or background.scene_id is None:
        return None
    if not background.registry_path:
        if require_registry:
            raise ValueError("Resolved background has no registry_path")
        return None

    factory = scene_manager_factory or _default_scene_manager_factory
    manager = factory(background.registry_path)
    return manager.load_scene(
        stage=stage,
        scene_id=background.scene_id,
        material_id=background.material_id,
        layout_id=background.layout_id,
        prim_path=background.prim_path,
        capture_light_baseline=capture_light_baseline,
    )


def restore_episode_object_poses(
    stage: Any,
    episode: ResolvedEpisodeSpec,
    *,
    poses: Mapping[str, RuntimeObjectPose | Mapping[str, object]] | None = None,
    xform_prim_factory: XformPrimFactory | None = None,
    scale_writer: ScaleWriter | None = None,
) -> None:
    pose_overrides = poses or {}
    for instance_id, obj in sorted(episode.objects.items()):
        pose = _coerce_pose(
            pose_overrides.get(instance_id),
            default=RuntimeObjectPose(
                position=obj.position,
                rotation=obj.rotation,
                scale=obj.scale or (1.0, 1.0, 1.0),
            ),
        )
        set_object_pose(
            stage,
            obj.prim_path,
            pose,
            xform_prim_factory=xform_prim_factory,
            scale_writer=scale_writer,
        )


def capture_episode_object_poses(
    stage: Any,
    episode: ResolvedEpisodeSpec,
    *,
    xform_prim_factory: XformPrimFactory | None = None,
    scale_reader: ScaleReader | None = None,
) -> dict[str, RuntimeObjectPose]:
    return {
        instance_id: get_object_pose(
            stage,
            obj.prim_path,
            xform_prim_factory=xform_prim_factory,
            scale_reader=scale_reader,
            fallback_scale=obj.scale or (1.0, 1.0, 1.0),
        )
        for instance_id, obj in sorted(episode.objects.items())
    }


def set_object_pose(
    stage: Any,
    prim_path: str,
    pose: RuntimeObjectPose,
    *,
    xform_prim_factory: XformPrimFactory | None = None,
    scale_writer: ScaleWriter | None = None,
) -> None:
    xform = (xform_prim_factory or _default_xform_prim_factory)(prim_path)
    xform.set_world_pose(position=list(pose.position), orientation=list(pose.rotation))
    writer = scale_writer or _default_scale_writer
    writer(stage, prim_path, pose.scale)


def get_object_pose(
    stage: Any,
    prim_path: str,
    *,
    xform_prim_factory: XformPrimFactory | None = None,
    scale_reader: ScaleReader | None = None,
    fallback_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> RuntimeObjectPose:
    xform = (xform_prim_factory or _default_xform_prim_factory)(prim_path)
    position, rotation = xform.get_world_pose()
    reader = scale_reader or _default_scale_reader
    try:
        scale = reader(stage, prim_path)
    except Exception:
        scale = fallback_scale
    return RuntimeObjectPose(
        position=tuple(float(value) for value in position),
        rotation=tuple(float(value) for value in rotation),
        scale=tuple(float(value) for value in scale),
    )


def runtime_object_poses_to_dict(poses: Mapping[str, RuntimeObjectPose]) -> dict[str, dict[str, list[float]]]:
    return {
        instance_id: {
            "position": list(pose.position),
            "rotation": list(pose.rotation),
            "scale": list(pose.scale),
        }
        for instance_id, pose in sorted(poses.items())
    }


def runtime_object_poses_from_dict(payload: Mapping[str, Mapping[str, object]]) -> dict[str, RuntimeObjectPose]:
    return {
        str(instance_id): _coerce_pose(pose_payload)
        for instance_id, pose_payload in payload.items()
    }


def _coerce_pose(
    payload: RuntimeObjectPose | Mapping[str, object] | None,
    *,
    default: RuntimeObjectPose | None = None,
) -> RuntimeObjectPose:
    if payload is None:
        if default is None:
            raise ValueError("pose payload is required")
        return default
    if isinstance(payload, RuntimeObjectPose):
        return payload
    return RuntimeObjectPose(
        position=_float_tuple(payload["position"]),
        rotation=_float_tuple(payload["rotation"]),
        scale=_float_tuple(payload.get("scale", [1.0, 1.0, 1.0])),
    )


def _float_tuple(payload: object) -> tuple[float, ...]:
    if not isinstance(payload, (list, tuple)):
        raise ValueError("pose fields must be list or tuple")
    return tuple(float(value) for value in payload)


def _default_create_prim(**kwargs: Any) -> Any:
    from omni.isaac.core.utils.prims import create_prim  # type: ignore[import]

    return create_prim(**kwargs)


def _default_xform_prim_factory(prim_path: str) -> Any:
    from omni.isaac.core.prims import XFormPrim  # type: ignore[import]

    return XFormPrim(prim_path)


def _default_scene_manager_factory(registry_path: str | Path) -> Any:
    from .background import SceneManager

    return SceneManager.from_registry_file(registry_path)


def _default_scale_writer(stage: Any, prim_path: str, scale: tuple[float, ...]) -> None:
    from pxr import Gf, UsdGeom  # type: ignore[import]

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        raise RuntimeError(f"Invalid prim for scale write: {prim_path}")

    value = Gf.Vec3d(*[float(v) for v in scale])
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(value)
            return
    xformable.AddScaleOp().Set(value)


def _default_scale_reader(stage: Any, prim_path: str) -> tuple[float, ...]:
    from pxr import UsdGeom  # type: ignore[import]

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        raise RuntimeError(f"Invalid prim for scale read: {prim_path}")

    scale = [1.0, 1.0, 1.0]
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() != UsdGeom.XformOp.TypeScale:
            continue
        value = op.Get()
        if value is None:
            continue
        scale[0] *= float(value[0])
        scale[1] *= float(value[1])
        scale[2] *= float(value[2])
    return tuple(scale)
