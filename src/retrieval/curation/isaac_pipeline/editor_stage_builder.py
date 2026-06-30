from __future__ import annotations

import traceback
from pathlib import Path

from .stage_builder import create_stage_and_reference, refresh_prims


def _reopen_stage(stage_path: Path):
    from pxr import Usd

    reopened = Usd.Stage.Open(str(stage_path))
    if reopened is None:
        raise RuntimeError(f"failed to reopen stage: {stage_path}")
    return reopened


def _find_source_root_prim(stage):
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim

    pseudo_root = stage.GetPseudoRoot()
    children = [prim for prim in pseudo_root.GetChildren() if prim and prim.IsValid()]
    if len(children) != 1:
        raise RuntimeError("source stage must have a default prim or exactly one root prim")
    return children[0]


def _find_first_mesh_path(stage, root_prim_path: str) -> str:
    from pxr import Usd, UsdGeom

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"invalid root prim path: {root_prim_path}")

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            return str(prim.GetPath())

    raise RuntimeError(f"no UsdGeom.Mesh found under root {root_prim_path}")


def _copy_source_subtree_to_stage(*, source_layer, source_root_path, output_usd_path: Path, prim_name: str, fallback_mode: str):
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(output_usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    dest_root_path = Sdf.Path(f"/{prim_name}")
    if not Sdf.CopySpec(source_layer, source_root_path, stage.GetRootLayer(), dest_root_path):
        raise RuntimeError(
            f"failed to copy source subtree from {source_root_path} to {dest_root_path}"
        )

    root_prim = stage.GetPrimAtPath(str(dest_root_path))
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"materialized root prim missing after copy: {dest_root_path}")

    stage.SetDefaultPrim(root_prim)
    stage.Save()
    reopened = _reopen_stage(output_usd_path)
    mesh_prim_path = _find_first_mesh_path(reopened, str(dest_root_path))
    root_prim = reopened.GetPrimAtPath(str(dest_root_path))
    mesh_prim = reopened.GetPrimAtPath(mesh_prim_path)
    return reopened, root_prim, mesh_prim, str(dest_root_path), mesh_prim_path, fallback_mode


def _materialize_source_to_editable_stage(*, output_usd_path: Path, source_usd_path: Path, prim_name: str):
    from pxr import Usd

    source_stage = Usd.Stage.Open(str(source_usd_path))
    if source_stage is None:
        raise RuntimeError(f"failed to open source usd stage: {source_usd_path}")

    source_root_prim = _find_source_root_prim(source_stage)
    try:
        flattened_layer = source_stage.Flatten()
        return _copy_source_subtree_to_stage(
            source_layer=flattened_layer,
            source_root_path=source_root_prim.GetPath(),
            output_usd_path=output_usd_path,
            prim_name=prim_name,
            fallback_mode="none",
        )
    except Exception:
        print(
            "[AssetsEditor] Editable stage flatten failed; falling back to direct root-layer copy.\n"
            f"source_stage={source_usd_path}\n"
            f"source_root={source_root_prim.GetPath()}\n"
            f"traceback=\n{traceback.format_exc()}",
            flush=True,
        )
        source_layer = source_stage.GetRootLayer()
        return _copy_source_subtree_to_stage(
            source_layer=source_layer,
            source_root_path=source_root_prim.GetPath(),
            output_usd_path=output_usd_path,
            prim_name=prim_name,
            fallback_mode="direct_copy",
        )


def create_preview_stage(
    *,
    output_usd_path: Path,
    source_usd_path: Path,
    prim_name: str = "Object",
):
    stage, root_prim, mesh_prim, root_prim_path, mesh_prim_path = create_stage_and_reference(
        output_usd_path=output_usd_path,
        temp_usd_path=source_usd_path,
        prim_name=prim_name,
    )
    stage.Save()
    return stage, root_prim, mesh_prim, root_prim_path, mesh_prim_path


def create_editable_stage(
    *,
    output_usd_path: Path,
    source_usd_path: Path,
    prim_name: str = "Object",
    flatten_for_edit: bool = True,
):
    if flatten_for_edit:
        try:
            return _materialize_source_to_editable_stage(
                output_usd_path=output_usd_path,
                source_usd_path=source_usd_path,
                prim_name=prim_name,
            )
        except Exception:
            print(
                "[AssetsEditor] Editable stage materialization failed; falling back to referenced editable stage.\n"
                f"source_stage={source_usd_path}\n"
                f"traceback=\n{traceback.format_exc()}",
                flush=True,
            )

    stage, _, _, root_prim_path, mesh_prim_path = create_stage_and_reference(
        output_usd_path=output_usd_path,
        temp_usd_path=source_usd_path,
        prim_name=prim_name,
    )
    stage.Save()
    root_prim, mesh_prim = refresh_prims(stage, root_prim_path, mesh_prim_path)
    return stage, root_prim, mesh_prim, root_prim_path, mesh_prim_path, "referenced"
