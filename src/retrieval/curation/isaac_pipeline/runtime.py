from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, Optional

from .canonical_alignment import apply_icp_canonical_alignment
from .canonicalization import bake_stage_xforms_to_mesh_geometry, strip_xform_specs_from_layer
from .collision import apply_collision
from .materials import bind_physics_material
from .metadata import write_custom_metadata
from .normalization import (
    apply_centering,
    apply_scale,
    apply_z_up_orientation,
    check_size_and_center,
)
from .physx import apply_physx_collision_attrs
from .stage_builder import create_stage_and_reference, refresh_prims


async def _convert_glb_to_temp_usd(
    glb_path: Path,
    temp_usd_path: Path,
    timeout_seconds: float,
) -> None:
    import omni.kit.asset_converter

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = False
    converter_context.ignore_animations = True
    converter_context.single_mesh = True
    converter_context.ignore_camera = True
    converter_context.ignore_light = True
    converter_context.keep_all_materials = True
    converter_context.merge_all_meshes = False
    converter_context.use_meter_as_world_unit = True

    instance = omni.kit.asset_converter.get_instance()
    progress_state = {"last_pct": -1, "calls": 0}

    def _progress_callback(progress: int, total_steps: int):
        progress_state["calls"] += 1
        if total_steps <= 0:
            return True
        pct = int((progress / total_steps) * 100)
        if pct >= progress_state["last_pct"] + 10 or pct == 100:
            print(
                f"[convert] glb={glb_path.name} progress={pct}% "
                f"({progress}/{total_steps})"
            )
            progress_state["last_pct"] = pct
        return True

    task = instance.create_converter_task(
        str(glb_path),
        str(temp_usd_path),
        _progress_callback,
        converter_context,
    )

    try:
        success = await asyncio.wait_for(task.wait_until_finished(), timeout=timeout_seconds)
    except asyncio.TimeoutError as e:
        raise TimeoutError(
            "converter timeout "
            f"after {timeout_seconds}s "
            f"(progress_calls={progress_state['calls']} last_pct={progress_state['last_pct']})"
        ) from e

    if not success:
        raise RuntimeError(task.get_error_message() or "converter task failed")

    if not temp_usd_path.exists():
        raise RuntimeError("converter success but temp usd not found")


def _compute_mass(mass_range) -> float:
    return (float(mass_range[0]) + float(mass_range[1])) / 2.0


def _apply_physics_properties(
    stage,
    root_prim,
    mesh_prim,
    metadata: Dict,
    root_prim_path: str,
    final_min_dim: float,
    use_coacd: bool,
    coacd_threshold: float,
):
    from pxr import UsdPhysics

    physics_type = metadata.get("physics_type", "rigid")
    if physics_type != "rigid":
        write_custom_metadata(root_prim, metadata)
        stage.Save()
        return {
            "physics_applied": False,
            "collision_strategy": "skipped_non_rigid",
            "coacd_used": False,
            "analysis": {},
            "material_path": None,
        }

    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(_compute_mass(metadata["mass"]))

    collision_strategy, analysis = apply_collision(
        stage=stage,
        mesh_prim=mesh_prim,
        final_min_dim=final_min_dim,
        use_coacd=use_coacd,
        coacd_threshold=coacd_threshold,
    )

    material_path, _ = bind_physics_material(
        stage=stage,
        mesh_prim=mesh_prim,
        root_prim_path=root_prim_path,
        materials_list=metadata.get("materials", []),
    )

    apply_physx_collision_attrs(mesh_prim)
    write_custom_metadata(root_prim, metadata)
    stage.Save()

    return {
        "physics_applied": True,
        "collision_strategy": collision_strategy,
        "coacd_used": collision_strategy == "coacd",
        "analysis": analysis,
        "material_path": material_path,
    }


def preprocess_asset(
    uid: str,
    glb_path: Path,
    output_usd_path: Path,
    *,
    scale_range,
    mass_range,
    physics_type: str,
    materials,
    use_coacd: bool,
    coacd_threshold: float,
    convert_timeout_seconds: float,
    temp_usd_dir: Path,
    cleanup_temp: bool,
    keep_temp_on_failure: bool,
    bake_transforms_to_mesh: bool,
    normal_mode: str,
    enable_icp_alignment: bool,
    canonical_usd_path: Optional[Path],
    canonical_mesh_prim_path: Optional[str],
    icp_max_iterations: int,
    icp_tolerance: float,
    icp_sample_points: int,
    icp_max_corr_distance: float,
    icp_seed: int,
    icp_rmse_threshold: float,
) -> Dict:
    print(f"[preprocess] uid={uid} start glb={glb_path}")
    t0 = time.perf_counter()
    temp_usd_dir.mkdir(parents=True, exist_ok=True)
    output_usd_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "uid": uid,
        "scale": [float(scale_range[0]), float(scale_range[1])],
        "mass": [float(mass_range[0]), float(mass_range[1])],
        "physics_type": physics_type,
        "materials": list(materials or []),
    }
    icp_result: Dict[str, object] = {
        "enabled": False,
        "canonical_usd_path": None,
        "canonical_mesh_prim_path": canonical_mesh_prim_path,
    }
    canonicalization_result: Dict[str, object] = {
        "enabled": False,
        "normal_mode": normal_mode,
    }

    temp_usd_path = temp_usd_dir / f"{uid}_mesh.usd"
    timings_ms: Dict[str, float] = {}

    try:
        t_convert = time.perf_counter()
        print(
            f"[preprocess] uid={uid} convert start "
            f"temp_usd={temp_usd_path} timeout={convert_timeout_seconds}s"
        )
        asyncio.get_event_loop().run_until_complete(
            _convert_glb_to_temp_usd(
                glb_path=glb_path,
                temp_usd_path=temp_usd_path,
                timeout_seconds=convert_timeout_seconds,
            )
        )
        timings_ms["convert"] = (time.perf_counter() - t_convert) * 1000
        print(f"[preprocess] uid={uid} convert done elapsed={timings_ms['convert']:.1f}ms")

        t_stage = time.perf_counter()
        print(f"[preprocess] uid={uid} stage build start")
        stage, root_prim, mesh_prim, root_prim_path, mesh_prim_path = create_stage_and_reference(
            output_usd_path=output_usd_path,
            temp_usd_path=temp_usd_path,
            prim_name="Object",
        )
        timings_ms["stage_build"] = (time.perf_counter() - t_stage) * 1000
        print(f"[preprocess] uid={uid} stage build done elapsed={timings_ms['stage_build']:.1f}ms")

        t_norm = time.perf_counter()
        print(f"[preprocess] uid={uid} normalize start")
        stage, scale_factor = apply_scale(
            stage=stage,
            root_prim_path=root_prim_path,
            mesh_prim_path=mesh_prim_path,
            scale_range=scale_range,
        )
        if enable_icp_alignment:
            if canonical_usd_path is None:
                raise RuntimeError("enable_icp_alignment=True requires canonical_usd_path")
            t_icp = time.perf_counter()
            stage, icp_result = apply_icp_canonical_alignment(
                stage=stage,
                root_prim_path=root_prim_path,
                mesh_prim_path=mesh_prim_path,
                canonical_usd_path=canonical_usd_path,
                canonical_mesh_prim_path=canonical_mesh_prim_path,
                icp_max_iterations=icp_max_iterations,
                icp_tolerance=icp_tolerance,
                icp_sample_points=icp_sample_points,
                icp_max_corr_distance=icp_max_corr_distance,
                icp_seed=icp_seed,
                icp_rmse_threshold=icp_rmse_threshold,
            )
            timings_ms["icp_align"] = (time.perf_counter() - t_icp) * 1000

        stage = apply_centering(
            stage=stage,
            root_prim_path=root_prim_path,
            mesh_prim_path=mesh_prim_path,
            scale_factor=scale_factor,
        )
        apply_z_up_orientation(stage=stage, root_prim_path=root_prim_path)
        if bake_transforms_to_mesh:
            t_bake = time.perf_counter()
            print(
                f"[preprocess] uid={uid} canonicalize mesh bake start "
                f"normal_mode={normal_mode}"
            )
            stage, canonicalization_result = bake_stage_xforms_to_mesh_geometry(
                stage=stage,
                root_prim_path=root_prim_path,
                normal_mode=normal_mode,
                flatten_stage=True,
            )
            timings_ms["canonicalize_mesh_bake"] = (time.perf_counter() - t_bake) * 1000
            print(
                f"[preprocess] uid={uid} canonicalize mesh bake done "
                f"elapsed={timings_ms['canonicalize_mesh_bake']:.1f}ms"
            )
            post_bbox = canonicalization_result["post_bbox"]
            bbox_min = post_bbox["min"]
            bbox_max = post_bbox["max"]
            final_min_dim = min(float(bbox_max[i]) - float(bbox_min[i]) for i in range(3))
        else:
            stage, final_min_dim = check_size_and_center(
                stage=stage,
                root_prim_path=root_prim_path,
                mesh_prim_path=mesh_prim_path,
            )
        timings_ms["normalize"] = (time.perf_counter() - t_norm) * 1000
        print(f"[preprocess] uid={uid} normalize done elapsed={timings_ms['normalize']:.1f}ms")

        root_prim, mesh_prim = refresh_prims(stage, root_prim_path, mesh_prim_path)

        t_phys = time.perf_counter()
        print(f"[preprocess] uid={uid} physics start type={physics_type}")
        physics_result = _apply_physics_properties(
            stage=stage,
            root_prim=root_prim,
            mesh_prim=mesh_prim,
            metadata=metadata,
            root_prim_path=root_prim_path,
            final_min_dim=final_min_dim,
            use_coacd=use_coacd,
            coacd_threshold=coacd_threshold,
        )
        timings_ms["physics"] = (time.perf_counter() - t_phys) * 1000
        print(f"[preprocess] uid={uid} physics done elapsed={timings_ms['physics']:.1f}ms")

        t_flatten = time.perf_counter()
        print(f"[preprocess] uid={uid} flatten/export start")
        flattened_layer = stage.Flatten()
        final_xform_cleanup = None
        if bake_transforms_to_mesh:
            final_xform_cleanup = strip_xform_specs_from_layer(
                flattened_layer,
                root_prim_path=root_prim_path,
            )
        flattened_layer.Export(str(output_usd_path))
        timings_ms["flatten_export"] = (time.perf_counter() - t_flatten) * 1000
        print(f"[preprocess] uid={uid} flatten/export done elapsed={timings_ms['flatten_export']:.1f}ms")

        timings_ms["total"] = (time.perf_counter() - t0) * 1000
        print(f"[preprocess] uid={uid} done total={timings_ms['total']:.1f}ms")
        return {
            "uid": uid,
            "source_glb_path": str(glb_path),
            "prepared_usd_path": str(output_usd_path),
            "temp_usd_path": str(temp_usd_path),
            "root_prim_path": root_prim_path,
            "mesh_prim_path": mesh_prim_path,
            "scale_factor": scale_factor,
            "collision_strategy": physics_result["collision_strategy"],
            "coacd_used": physics_result["coacd_used"],
            "physics_applied": physics_result["physics_applied"],
            "analysis": physics_result.get("analysis", {}),
            "icp": icp_result,
            "canonicalization": canonicalization_result,
            "final_xform_cleanup": final_xform_cleanup,
            "timings_ms": timings_ms,
            "warnings": [],
        }
    except Exception as e:
        raise e
    finally:
        timings_ms["total"] = (time.perf_counter() - t0) * 1000
        if cleanup_temp and temp_usd_path.exists() and not keep_temp_on_failure:
            temp_usd_path.unlink()
