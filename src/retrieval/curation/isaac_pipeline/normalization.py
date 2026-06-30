from __future__ import annotations

import math

from .stage_builder import refresh_prims, update_stage


def apply_scale(stage, root_prim_path: str, mesh_prim_path: str, scale_range):
    from pxr import Usd, UsdGeom

    stage = update_stage(stage)
    root_prim, mesh_prim = refresh_prims(stage, root_prim_path, mesh_prim_path)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(mesh_prim)
    bbox_range = bbox.ComputeAlignedRange()
    if bbox_range.IsEmpty():
        raise RuntimeError("bbox is empty after stage reference")

    dimensions = bbox_range.GetMax() - bbox_range.GetMin()
    max_dim = max(dimensions[0], dimensions[1], dimensions[2])
    if max_dim <= 1e-9:
        raise RuntimeError("invalid max dimension for scale")

    target_size = (float(scale_range[0]) + float(scale_range[1])) / 2.0
    scale_factor = target_size / max_dim

    UsdGeom.XformCommonAPI(root_prim).SetScale((scale_factor, scale_factor, scale_factor))
    return stage, scale_factor


def apply_centering(stage, root_prim_path: str, mesh_prim_path: str, scale_factor: float):
    from pxr import Usd, UsdGeom, Gf

    stage = update_stage(stage)
    _, mesh_prim = refresh_prims(stage, root_prim_path, mesh_prim_path)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(mesh_prim)
    bbox_range = bbox.ComputeAlignedRange()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()

    center_x = (min_point[0] + max_point[0]) / 2.0
    center_y = min_point[1]
    center_z = (min_point[2] + max_point[2]) / 2.0

    center_x = center_x / scale_factor
    center_y = center_y / scale_factor
    center_z = center_z / scale_factor

    mesh_xformable = UsdGeom.Xformable(mesh_prim)
    translate_attr = mesh_prim.GetAttribute("xformOp:translate")
    if not translate_attr:
        mesh_xformable.AddTranslateOp()
        translate_attr = mesh_prim.GetAttribute("xformOp:translate")
    translate_attr.Set(Gf.Vec3d(-center_x, -center_y, -center_z))
    return stage


def _get_or_add_orient_op(xformable):
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            return op
    return xformable.AddOrientOp()


def apply_z_up_orientation(stage, root_prim_path: str):
    from pxr import UsdGeom, Gf

    root_prim = stage.GetPrimAtPath(root_prim_path)
    root_xformable = UsdGeom.Xformable(root_prim)
    orient_attr = _get_or_add_orient_op(root_xformable)
    angle = math.radians(90)
    w = math.cos(angle / 2)
    x = math.sin(angle / 2)
    orient_attr.Set(Gf.Quatf(w, x, 0, 0))
    stage.Save()


def check_size_and_center(stage, root_prim_path: str, mesh_prim_path: str):
    from pxr import Usd, UsdGeom

    stage = update_stage(stage)
    _, mesh_prim = refresh_prims(stage, root_prim_path, mesh_prim_path)

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(mesh_prim)
    bbox_range = bbox.ComputeAlignedRange()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()

    dimensions = max_point - min_point
    final_min_dim = min(dimensions[0], dimensions[1], dimensions[2])
    return stage, float(final_min_dim)
