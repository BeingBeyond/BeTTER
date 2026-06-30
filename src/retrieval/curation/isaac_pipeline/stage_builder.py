from __future__ import annotations

from pathlib import Path
from typing import Tuple


def create_stage_and_reference(output_usd_path: Path, temp_usd_path: Path, prim_name: str = "Object"):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(output_usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim_path = f"/{prim_name}"
    root_prim = UsdGeom.Xform.Define(stage, root_prim_path)
    stage.SetDefaultPrim(root_prim.GetPrim())

    mesh_prim_path = f"{root_prim_path}/Mesh"
    mesh_xform = UsdGeom.Xform.Define(stage, mesh_prim_path)
    mesh_prim = mesh_xform.GetPrim()
    mesh_prim.GetReferences().AddReference(str(temp_usd_path))

    return stage, root_prim.GetPrim(), mesh_prim, root_prim_path, mesh_prim_path


def update_stage(stage):
    from pxr import Usd

    stage.Save()
    return Usd.Stage.Open(stage.GetRootLayer().realPath)


def refresh_prims(stage, root_prim_path: str, mesh_prim_path: str):
    root_prim = stage.GetPrimAtPath(root_prim_path)
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    return root_prim, mesh_prim
