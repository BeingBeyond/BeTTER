from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..contracts import EditorSessionAsset, EditorValidationResult


def _iter_mesh_prims(root_prim) -> List:
    from pxr import Usd, UsdGeom

    meshes = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
    return meshes


def _iter_path_to_root(prim, root_prim) -> Iterable:
    cursor = prim
    root_path = root_prim.GetPath()
    while cursor and cursor.IsValid():
        yield cursor
        if cursor.GetPath() == root_path:
            return
        cursor = cursor.GetParent()


def _mesh_is_locally_owned(mesh_prim, root_prim) -> bool:
    for prim in _iter_path_to_root(mesh_prim, root_prim):
        if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
            return False
    return True


def _count_xform_ops(root_prim) -> int:
    from pxr import Usd

    total = 0
    for prim in Usd.PrimRange(root_prim):
        for prop in prim.GetAuthoredProperties():
            if prop.GetName().startswith("xformOp"):
                total += 1
    return total


def validate_editor_stage(
    stage,
    *,
    session_id: str,
    stage_path: Path,
    root_prim_path: str,
) -> EditorValidationResult:
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"invalid root prim path: {root_prim_path}")

    mesh_prims = _iter_mesh_prims(root_prim)
    local_mesh_prims = [mesh for mesh in mesh_prims if _mesh_is_locally_owned(mesh, root_prim)]
    xform_op_count = _count_xform_ops(root_prim)
    default_prim = stage.GetDefaultPrim()
    passes_default_prim = bool(default_prim and default_prim.IsValid() and default_prim.GetPath() == root_prim.GetPath())
    passes_local_mesh_ownership = len(mesh_prims) > 0 and len(local_mesh_prims) == len(mesh_prims)
    passes_no_xform_ops = xform_op_count == 0

    issues = []
    if not mesh_prims:
        issues.append("no_mesh_prims")
    if not passes_local_mesh_ownership:
        issues.append("non_local_mesh_ownership")
    if not passes_no_xform_ops:
        issues.append("authored_xform_ops_present")
    if not passes_default_prim:
        issues.append("default_prim_mismatch")

    return EditorValidationResult(
        session_id=session_id,
        stage_path=stage_path,
        root_prim_path=root_prim_path,
        mesh_count=len(mesh_prims),
        local_mesh_count=len(local_mesh_prims),
        xform_op_count=xform_op_count,
        passes_local_mesh_ownership=passes_local_mesh_ownership,
        passes_no_xform_ops=passes_no_xform_ops,
        passes_default_prim=passes_default_prim,
        is_publish_ready=passes_local_mesh_ownership and passes_no_xform_ops and passes_default_prim,
        issues=tuple(issues),
        stats={
            "mesh_paths": [str(mesh.GetPath()) for mesh in mesh_prims],
            "local_mesh_paths": [str(mesh.GetPath()) for mesh in local_mesh_prims],
        },
    )


def validate_editor_session_asset(session_asset: EditorSessionAsset) -> EditorValidationResult:
    from dataclasses import replace
    from pxr import Usd

    stage = Usd.Stage.Open(str(session_asset.stage_path))
    if stage is None:
        raise RuntimeError(f"failed to open editor stage: {session_asset.stage_path}")

    result = validate_editor_stage(
        stage,
        session_id=session_asset.session_id,
        stage_path=session_asset.stage_path,
        root_prim_path=session_asset.root_prim_path,
    )

    fallback_mode = getattr(session_asset, "fallback_mode", None) or session_asset.metadata.get("fallback_mode", "none")
    issues = list(result.issues)
    stats = dict(result.stats)
    stats["fallback_mode"] = fallback_mode

    if session_asset.mode == "editable" and fallback_mode == "referenced":
        if "referenced_editable_session" not in issues:
            issues.append("referenced_editable_session")
        return replace(
            result,
            is_publish_ready=False,
            issues=tuple(issues),
            stats=stats,
        )

    return replace(result, stats=stats)
