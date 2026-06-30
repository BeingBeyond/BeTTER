from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..contracts import EditorSessionAsset, EditorValidationResult
from .canonical_alignment import (
    load_reference_points,
    run_point_to_point_icp,
    sample_stage_points,
)
from .canonicalization import bake_stage_xforms_to_mesh_geometry, strip_xform_specs_from_layer
from .editor_validation import validate_editor_session_asset
from .normalization import apply_centering, apply_scale, apply_z_up_orientation, check_size_and_center


def _open_stage(stage_path: Path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"failed to open stage: {stage_path}")
    return stage


def _save_stage(stage) -> None:
    stage.Save()


def _matrix4_from_rotation_translation(rotation_matrix: np.ndarray, translation: Optional[np.ndarray]):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix
    if translation is not None:
        matrix[:3, 3] = translation
    return matrix


def _apply_transform_to_root(stage, *, root_prim_path: str, rotation_matrix: np.ndarray, translation: Optional[np.ndarray]) -> None:
    from pxr import Gf, UsdGeom

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"invalid root prim path: {root_prim_path}")

    xformable = UsdGeom.Xformable(root_prim)
    matrix = _matrix4_from_rotation_translation(rotation_matrix, translation)
    gf_matrix = Gf.Matrix4d(*matrix.T.flatten().tolist())
    op = xformable.AddTransformOp()
    op.Set(gf_matrix)
    _save_stage(stage)


def save_editor_session(session_asset: EditorSessionAsset) -> Dict[str, Any]:
    stage = _open_stage(session_asset.stage_path)
    _save_stage(stage)
    return {
        "stage_path": str(session_asset.stage_path),
        "root_prim_path": session_asset.root_prim_path,
        "saved": True,
    }


def normalize_editor_session(
    session_asset: EditorSessionAsset,
    *,
    scale_range,
    apply_scale_to_asset: bool = True,
    apply_centering_to_asset: bool = True,
    apply_z_up_to_asset: bool = True,
) -> Dict[str, Any]:
    stage = _open_stage(session_asset.stage_path)
    scale_factor = 1.0

    if apply_scale_to_asset:
        stage, scale_factor = apply_scale(
            stage=stage,
            root_prim_path=session_asset.root_prim_path,
            mesh_prim_path=session_asset.mesh_prim_path,
            scale_range=scale_range,
        )
    if apply_centering_to_asset:
        stage = apply_centering(
            stage=stage,
            root_prim_path=session_asset.root_prim_path,
            mesh_prim_path=session_asset.mesh_prim_path,
            scale_factor=scale_factor,
        )
    if apply_z_up_to_asset:
        apply_z_up_orientation(stage=stage, root_prim_path=session_asset.root_prim_path)

    stage, min_dim = check_size_and_center(
        stage=stage,
        root_prim_path=session_asset.root_prim_path,
        mesh_prim_path=session_asset.mesh_prim_path,
    )
    _save_stage(stage)

    return {
        "stage_path": str(session_asset.stage_path),
        "root_prim_path": session_asset.root_prim_path,
        "mesh_prim_path": session_asset.mesh_prim_path,
        "scale_factor": float(scale_factor),
        "final_min_dim": float(min_dim),
    }


def solve_icp_for_editor_session(
    session_asset: EditorSessionAsset,
    *,
    canonical_usd_path: Path,
    canonical_mesh_prim_path: Optional[str],
    icp_max_iterations: int,
    icp_tolerance: float,
    icp_sample_points: int,
    icp_max_corr_distance: float,
    icp_seed: int,
    icp_rmse_threshold: float,
) -> Dict[str, Any]:
    stage = _open_stage(session_asset.stage_path)
    reference_points = load_reference_points(
        canonical_usd_path=canonical_usd_path,
        canonical_mesh_prim_path=canonical_mesh_prim_path,
        sample_points=icp_sample_points,
        seed=icp_seed,
    )
    source_points = sample_stage_points(
        stage,
        root_prim_path=session_asset.root_prim_path,
        mesh_prim_path=session_asset.mesh_prim_path,
        sample_points=icp_sample_points,
        seed=icp_seed,
    )
    rotation_matrix, translation, metrics = run_point_to_point_icp(
        source_points=source_points,
        target_points=reference_points,
        max_iterations=icp_max_iterations,
        tolerance=icp_tolerance,
        max_correspondence_distance=icp_max_corr_distance,
    )
    rmse = float(metrics["rmse"])
    if rmse > icp_rmse_threshold:
        raise RuntimeError(
            "icp failed quality threshold "
            f"(rmse={rmse:.6f} threshold={icp_rmse_threshold:.6f})"
        )

    return {
        "canonical_usd_path": str(canonical_usd_path),
        "canonical_mesh_prim_path": canonical_mesh_prim_path,
        "rotation_matrix": rotation_matrix.tolist(),
        "translation": translation.tolist(),
        "rmse": rmse,
        "inlier_ratio": float(metrics["inlier_ratio"]),
        "iterations": int(metrics["iterations"]),
        "converged": bool(metrics["converged"] > 0.5),
        "det_r": float(metrics["det_r"]),
        "translation_norm": float(np.linalg.norm(translation)),
    }


def apply_icp_to_editor_session(
    session_asset: EditorSessionAsset,
    *,
    rotation_matrix,
    translation,
) -> Dict[str, Any]:
    stage = _open_stage(session_asset.stage_path)
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    move = np.asarray(translation, dtype=np.float64)
    _apply_transform_to_root(
        stage,
        root_prim_path=session_asset.root_prim_path,
        rotation_matrix=rotation,
        translation=move,
    )
    return {
        "stage_path": str(session_asset.stage_path),
        "root_prim_path": session_asset.root_prim_path,
        "translation_norm": float(np.linalg.norm(move)),
    }


def bake_editor_session_to_geometry(
    session_asset: EditorSessionAsset,
    *,
    normal_mode: str = "preserve",
) -> Dict[str, Any]:
    fallback_mode = getattr(session_asset, "fallback_mode", None) or session_asset.metadata.get("fallback_mode", "none")
    if fallback_mode == "referenced":
        raise RuntimeError(
            "cannot bake a referenced editable session; source asset could not be materialized locally"
        )

    stage = _open_stage(session_asset.stage_path)
    stage, canonicalization_result = bake_stage_xforms_to_mesh_geometry(
        stage=stage,
        root_prim_path=session_asset.root_prim_path,
        normal_mode=normal_mode,
        flatten_stage=True,
    )
    root_layer = stage.GetRootLayer()
    xform_cleanup = strip_xform_specs_from_layer(
        root_layer,
        root_prim_path=session_asset.root_prim_path,
    )
    root_layer.Save()
    return {
        "stage_path": str(session_asset.stage_path),
        "canonicalization": canonicalization_result,
        "final_xform_cleanup": xform_cleanup,
    }


def publish_editor_session_asset(
    session_asset: EditorSessionAsset,
    *,
    output_usd_path: Optional[Path] = None,
) -> Dict[str, Any]:
    validation = validate_editor_session_asset(session_asset)
    if not validation.is_publish_ready:
        raise RuntimeError(
            "editor session is not publish-ready: "
            + ", ".join(validation.issues)
        )

    registry_target = session_asset.metadata.get("registry_usd_path")
    target_path = Path(str(registry_target)).expanduser() if registry_target else output_usd_path
    if target_path is None:
        raise RuntimeError("no publish target resolved for editor session")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(session_asset.stage_path, target_path)

    return {
        "published_usd_path": str(target_path),
        "session_stage_path": str(session_asset.stage_path),
        "validation_issues": list(validation.issues),
        "overwrote_registry_usd": bool(registry_target),
    }


def get_editor_session_validation(session_asset: EditorSessionAsset) -> EditorValidationResult:
    return validate_editor_session_asset(session_asset)
