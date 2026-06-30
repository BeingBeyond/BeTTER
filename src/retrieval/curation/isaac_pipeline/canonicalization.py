from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from .stage_builder import update_stage


def _flatten_stage_to_root_layer(stage):
    from pxr import Usd

    stage.Save()
    layer = stage.GetRootLayer()
    layer_path = layer.realPath or layer.identifier
    if not layer_path:
        raise RuntimeError("cannot flatten anonymous stage without a root layer path")

    target_path = Path(layer_path)
    temp_path = target_path.with_name(f"{target_path.stem}.flattened_tmp{target_path.suffix}")

    flattened_layer = stage.Flatten()
    flattened_layer.Export(str(temp_path))

    reopened = Usd.Stage.Open(str(temp_path))
    if reopened is None:
        raise RuntimeError(f"failed to reopen flattened stage: {temp_path}")

    reopened.Save()
    os.replace(temp_path, target_path)

    final_stage = Usd.Stage.Open(str(target_path))
    if final_stage is None:
        raise RuntimeError(f"failed to reopen flattened stage: {target_path}")
    return final_stage


def _find_mesh_prims(stage, root_prim) -> List:
    from pxr import Usd, UsdGeom

    meshes = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
    return meshes


def _transform_normals(normals, matrix4):
    from pxr import Gf

    linear = Gf.Matrix3d(
        matrix4[0][0],
        matrix4[0][1],
        matrix4[0][2],
        matrix4[1][0],
        matrix4[1][1],
        matrix4[1][2],
        matrix4[2][0],
        matrix4[2][1],
        matrix4[2][2],
    )
    normal_matrix = linear.GetInverse().GetTranspose()

    out = []
    for normal in normals:
        vec = Gf.Vec3d(float(normal[0]), float(normal[1]), float(normal[2]))
        transformed = normal_matrix * vec
        if transformed.GetLength() > 1e-12:
            transformed = transformed.GetNormalized()
        out.append(Gf.Vec3f(float(transformed[0]), float(transformed[1]), float(transformed[2])))
    return out


def _compute_bbox(points) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    min_x = min(float(p[0]) for p in points)
    min_y = min(float(p[1]) for p in points)
    min_z = min(float(p[2]) for p in points)
    max_x = max(float(p[0]) for p in points)
    max_y = max(float(p[1]) for p in points)
    max_z = max(float(p[2]) for p in points)
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def _clear_xform_ops(root_prim) -> Dict[str, int]:
    from pxr import Usd, UsdGeom

    xformable_count = 0
    removed_attr_count = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Xformable):
            continue

        xformable_count += 1

        for prop in list(prim.GetAuthoredProperties()):
            name = prop.GetName()
            if name.startswith("xformOp"):
                if prim.RemoveProperty(name):
                    removed_attr_count += 1

    return {
        "xformable_count": xformable_count,
        "removed_attr_count": removed_attr_count,
    }


def clear_stage_xform_ops(stage, *, root_prim_path: str):
    """Remove authored xform ops under an already-canonicalized asset root."""

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"invalid root prim path: {root_prim_path}")

    cleared = _clear_xform_ops(root_prim)
    stage.Save()
    return stage, cleared


def strip_xform_specs_from_layer(layer, *, root_prim_path: str) -> Dict[str, int]:
    """Remove xform op property specs directly from an Sdf layer.

    This is used on flattened layers before export so we do not have to reopen
    the generated crate file inside Isaac Sim, which has proven crash-prone for
    a few Objaverse conversions.
    """

    root_spec = layer.GetPrimAtPath(root_prim_path)
    if root_spec is None:
        raise RuntimeError(f"invalid root prim path in flattened layer: {root_prim_path}")

    prim_count = 0
    removed_attr_count = 0

    def _children(prim_spec):
        name_children = prim_spec.nameChildren
        values = getattr(name_children, "values", None)
        if values is not None:
            return list(values())
        return list(name_children)

    def _walk(prim_spec):
        nonlocal prim_count, removed_attr_count

        prim_count += 1
        prop_specs = list(prim_spec.properties)
        for prop_spec in prop_specs:
            name = prop_spec.name
            if name.startswith("xformOp"):
                prim_spec.RemoveProperty(prop_spec)
                removed_attr_count += 1

        for child_spec in _children(prim_spec):
            _walk(child_spec)

    _walk(root_spec)

    return {
        "prim_count": prim_count,
        "removed_attr_count": removed_attr_count,
    }


def bake_stage_xforms_to_mesh_geometry(
    stage,
    *,
    root_prim_path: str,
    normal_mode: str = "preserve",
    flatten_stage: bool = True,
):
    """Bake composed USD transforms into Mesh.points and clear xform ops.

    The curation pipeline intentionally computes scale, centering, optional ICP,
    and upright orientation with normal USD xform ops first. This function is the
    final canonicalization step: it converts that composed world-space result
    into authored mesh geometry, recenters the local origin to bottom-center, and
    removes the transform ops from the asset hierarchy.
    """

    from pxr import Gf, Usd, UsdGeom

    if normal_mode not in {"preserve", "drop"}:
        raise ValueError("normal_mode must be either 'preserve' or 'drop'")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    stage = update_stage(stage)
    if flatten_stage:
        stage = _flatten_stage_to_root_layer(stage)

    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"invalid root prim path: {root_prim_path}")

    mesh_prims = _find_mesh_prims(stage, root_prim)
    if not mesh_prims:
        raise RuntimeError(f"no UsdGeom.Mesh found under root {root_prim_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_data = []
    all_baked_points = []

    for mesh_prim in mesh_prims:
        mesh = UsdGeom.Mesh(mesh_prim)
        points_attr = mesh.GetPointsAttr()
        points = points_attr.Get() if points_attr else None
        if not points:
            continue

        local_to_world = cache.GetLocalToWorldTransform(mesh_prim)
        baked_points = [
            local_to_world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            for p in points
        ]

        normals_attr = mesh.GetNormalsAttr()
        normals = normals_attr.Get() if normals_attr and normals_attr.HasAuthoredValue() else None
        had_normals = normals is not None and len(normals) > 0
        normals_interp = mesh.GetNormalsInterpolation() if had_normals else None
        baked_normals = None
        if had_normals and normal_mode == "preserve":
            baked_normals = _transform_normals(normals, local_to_world)

        mesh_data.append(
            {
                "mesh_prim": mesh_prim,
                "mesh": mesh,
                "points_attr": points_attr,
                "normals_attr": normals_attr,
                "had_normals": had_normals,
                "normals_interp": normals_interp,
                "baked_points": baked_points,
                "baked_normals": baked_normals,
            }
        )
        all_baked_points.extend(baked_points)

    if not mesh_data:
        raise RuntimeError("all mesh point arrays are empty")

    pre_min, pre_max = _compute_bbox(all_baked_points)
    center_x = (pre_min[0] + pre_max[0]) / 2.0
    center_y = (pre_min[1] + pre_max[1]) / 2.0
    bottom_z = pre_min[2]

    final_points = []
    for data in mesh_data:
        recentered = [
            Gf.Vec3f(
                float(p[0] - center_x),
                float(p[1] - center_y),
                float(p[2] - bottom_z),
            )
            for p in data["baked_points"]
        ]
        data["points_attr"].Set(recentered)

        if normal_mode == "preserve" and data["had_normals"] and data["baked_normals"]:
            data["normals_attr"].Set(data["baked_normals"])
            if data["normals_interp"]:
                data["mesh"].SetNormalsInterpolation(data["normals_interp"])
        elif normal_mode == "drop" and data["had_normals"]:
            data["mesh_prim"].RemoveProperty("normals")

        UsdGeom.Boundable(data["mesh_prim"]).CreateExtentAttr().Set(
            UsdGeom.PointBased.ComputeExtent(recentered)
        )
        final_points.extend(recentered)

    cleared = _clear_xform_ops(root_prim)

    if stage.GetDefaultPrim() != root_prim:
        stage.SetDefaultPrim(root_prim)

    stage.Save()

    final_min, final_max = _compute_bbox(final_points)
    return stage, {
        "enabled": True,
        "root_prim_path": root_prim_path,
        "mesh_count": len(mesh_data),
        "normal_mode": normal_mode,
        "flattened_before_bake": bool(flatten_stage),
        "pre_recenter_bbox": {"min": list(pre_min), "max": list(pre_max)},
        "post_bbox": {"min": list(final_min), "max": list(final_max)},
        "recenter_offset": [center_x, center_y, bottom_z],
        "cleared_xforms": cleared,
    }
