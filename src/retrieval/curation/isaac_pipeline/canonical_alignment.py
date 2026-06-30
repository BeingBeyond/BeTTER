from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .stage_builder import update_stage


def _sample_points(points: np.ndarray, sample_points: int, seed: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must have shape (N, 3)")
    if points.shape[0] < 3:
        raise ValueError("point cloud must contain at least 3 points")
    if sample_points <= 0:
        raise ValueError("sample_points must be positive")

    if points.shape[0] <= sample_points:
        return points.astype(np.float64, copy=True)

    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=sample_points, replace=False)
    return points[indices].astype(np.float64, copy=False)


def _extract_mesh_vertices(mesh_prim) -> np.ndarray:
    from pxr import UsdGeom

    from .mesh_analyzer import MeshAnalyzer

    analyzer = MeshAnalyzer()
    data = analyzer.extract_mesh_data(UsdGeom.Mesh(mesh_prim))
    if data is None:
        raise RuntimeError(f"mesh prim has no geometry data: {mesh_prim.GetPath()}")

    vertices, _ = data
    if vertices is None or vertices.shape[0] < 3:
        raise RuntimeError(f"mesh prim has insufficient vertices: {mesh_prim.GetPath()}")
    return vertices


def _extract_mesh_world_vertices(mesh_prim) -> np.ndarray:
    from pxr import Gf, Usd, UsdGeom

    vertices = _extract_mesh_vertices(mesh_prim).astype(np.float64, copy=False)
    local_to_world = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    world_vertices = np.empty((vertices.shape[0], 3), dtype=np.float64)
    for idx, point in enumerate(vertices):
        world_point = local_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        world_vertices[idx] = (float(world_point[0]), float(world_point[1]), float(world_point[2]))
    return world_vertices



def _find_mesh_prim(stage, root_prim_path: Optional[str], mesh_prim_path: Optional[str]):
    from pxr import UsdGeom

    if mesh_prim_path:
        mesh_root = stage.GetPrimAtPath(mesh_prim_path)
        if not mesh_root or not mesh_root.IsValid():
            raise RuntimeError(f"mesh prim path not found: {mesh_prim_path}")
    else:
        default_prim = stage.GetDefaultPrim()
        mesh_root = default_prim if default_prim and default_prim.IsValid() else stage.GetPseudoRoot()

    if mesh_root.IsA(UsdGeom.Mesh):
        return mesh_root

    for prim in mesh_root.GetChildren():
        if prim.IsA(UsdGeom.Mesh):
            return prim

    for prim in stage.Traverse():
        if root_prim_path and not str(prim.GetPath()).startswith(root_prim_path):
            continue
        if prim.IsA(UsdGeom.Mesh):
            return prim

    scope = mesh_prim_path or root_prim_path or "<default>"
    raise RuntimeError(f"no UsdGeom.Mesh found under scope {scope}")


def _extract_all_meshes_world_vertices(stage, root_prim_path: Optional[str]) -> np.ndarray:
    from pxr import UsdGeom
    
    if root_prim_path:
        root_prim = stage.GetPrimAtPath(root_prim_path)
    else:
        default_prim = stage.GetDefaultPrim()
        root_prim = default_prim if default_prim and default_prim.IsValid() else stage.GetPseudoRoot()

    all_vertices = []
    
    for prim in root_prim.GetChildren():
        if prim.IsA(UsdGeom.Mesh):
            try:
                verts = _extract_mesh_world_vertices(prim)
                all_vertices.append(verts)
            except RuntimeError:
                pass
        for child in prim.GetDescendants():
            if child.IsA(UsdGeom.Mesh):
                try:
                    verts = _extract_mesh_world_vertices(child)
                    all_vertices.append(verts)
                except RuntimeError:
                    pass

    # If root is mesh
    if root_prim.IsA(UsdGeom.Mesh):
        try:
            verts = _extract_mesh_world_vertices(root_prim)
            all_vertices.append(verts)
        except RuntimeError:
            pass

    if not all_vertices:
        scope = root_prim_path or "<default>"
        raise RuntimeError(f"no UsdGeom.Mesh found under scope {scope}")

    return np.vstack(all_vertices)


def load_reference_points(
    canonical_usd_path: Path,
    canonical_mesh_prim_path: Optional[str],
    sample_points: int,
    seed: int,
) -> np.ndarray:
    from pxr import Usd

    stage = Usd.Stage.Open(str(canonical_usd_path))
    if stage is None:
        raise RuntimeError(f"failed to open canonical usd: {canonical_usd_path}")

    vertices = _extract_all_meshes_world_vertices(stage, None)
    return _sample_points(vertices, sample_points=sample_points, seed=seed)


def find_mesh_prim_path(stage, root_prim_path: Optional[str], mesh_prim_path: Optional[str]) -> str:
    mesh_prim = _find_mesh_prim(stage=stage, root_prim_path=root_prim_path, mesh_prim_path=mesh_prim_path)
    return str(mesh_prim.GetPath())


def sample_stage_points(
    stage,
    *,
    root_prim_path: Optional[str],
    mesh_prim_path: Optional[str],
    sample_points: int,
    seed: int,
) -> np.ndarray:
    vertices = _extract_all_meshes_world_vertices(stage, root_prim_path)
    return _sample_points(vertices, sample_points=sample_points, seed=seed)


def extract_target_points(stage, root_prim_path: str, mesh_prim_path: str, sample_points: int, seed: int) -> np.ndarray:
    stage = update_stage(stage)
    return sample_stage_points(
        stage=stage,
        root_prim_path=root_prim_path,
        mesh_prim_path=mesh_prim_path,
        sample_points=sample_points,
        seed=seed,
    )


def _estimate_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_center = source_points.mean(axis=0)
    dst_center = target_points.mean(axis=0)

    src_centered = source_points - src_center
    dst_centered = target_points - dst_center

    h_mat = src_centered.T @ dst_centered
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    r_mat = vt_mat.T @ u_mat.T

    if np.linalg.det(r_mat) < 0:
        vt_mat[-1, :] *= -1.0
        r_mat = vt_mat.T @ u_mat.T

    t_vec = dst_center - (r_mat @ src_center)
    return r_mat, t_vec


def run_point_to_point_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
    max_correspondence_distance: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be positive")

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)

    if source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError("source_points must have shape (N>=3, 3)")
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] < 3:
        raise ValueError("target_points must have shape (N>=3, 3)")

    tree = cKDTree(target)
    
    src_center = source.mean(axis=0)
    dst_center = target.mean(axis=0)
    total_t = dst_center - src_center
    transformed = source + total_t
    
    total_r = np.eye(3, dtype=np.float64)

    prev_rmse = np.inf
    inlier_ratio = 0.0
    converged = False
    iterations = 0

    for it in range(1, max_iterations + 1):
        distances, indices = tree.query(transformed, k=1)
        mask = distances <= max_correspondence_distance
        inliers = int(mask.sum())
        if inliers < 3:
            raise RuntimeError("icp failed: insufficient correspondences")

        src_corr = transformed[mask]
        dst_corr = target[indices[mask]]
        delta_r, delta_t = _estimate_rigid_transform(src_corr, dst_corr)

        transformed = (transformed @ delta_r.T) + delta_t
        total_r = delta_r @ total_r
        total_t = (delta_r @ total_t) + delta_t

        residual = transformed[mask] - dst_corr
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        inlier_ratio = float(inliers / transformed.shape[0])

        iterations = it
        if abs(prev_rmse - rmse) < tolerance:
            converged = True
            prev_rmse = rmse
            break
        prev_rmse = rmse

    if not np.isfinite(total_r).all() or not np.isfinite(total_t).all():
        raise RuntimeError("icp produced non-finite transform")

    det_r = float(np.linalg.det(total_r))
    if det_r <= 0.0:
        raise RuntimeError("icp produced reflection or singular rotation")

    return total_r, total_t, {
        "rmse": float(prev_rmse),
        "inlier_ratio": inlier_ratio,
        "iterations": float(iterations),
        "converged": float(1.0 if converged else 0.0),
        "det_r": det_r,
    }


def _get_or_add_orient_op(xformable):
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            return op
    return xformable.AddOrientOp()


def _get_or_add_translate_op(xformable):
    from pxr import UsdGeom

    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def apply_icp_transform_to_mesh(
    stage,
    mesh_prim_path: str,
    rotation_matrix: np.ndarray,
    translation: Optional[np.ndarray] = None,
    *,
    apply_translation: bool = True,
) -> None:
    from pxr import Gf, UsdGeom

    # We use the root prim instead of the mesh prim so we don't tear objects apart
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        root_prim = stage.GetPseudoRoot()
        
    xformable = UsdGeom.Xformable(root_prim)
    if not xformable:
        # If the root is not an xformable, wrap the operation or make it one, but usually it is.
        pass

    # We calculate the transform matrix for ICP
    import numpy as np
    M_icp = np.eye(4)
    M_icp[:3, :3] = rotation_matrix
    if apply_translation and translation is not None:
        M_icp[:3, 3] = translation

    gf_mat_icp = Gf.Matrix4d(*M_icp.T.flatten().tolist())

    # We add it as a new Xform op at the beginning (world space transform applied to the local) or just add a transform op
    transform_op = xformable.AddTransformOp()
    transform_op.Set(gf_mat_icp)
def apply_icp_canonical_alignment(
    stage,
    *,
    root_prim_path: str,
    mesh_prim_path: str,
    canonical_usd_path: Path,
    canonical_mesh_prim_path: Optional[str],
    icp_max_iterations: int,
    icp_tolerance: float,
    icp_sample_points: int,
    icp_max_corr_distance: float,
    icp_seed: int,
    icp_rmse_threshold: float,
):
    reference_points = load_reference_points(
        canonical_usd_path=canonical_usd_path,
        canonical_mesh_prim_path=canonical_mesh_prim_path,
        sample_points=icp_sample_points,
        seed=icp_seed,
    )

    target_points = extract_target_points(
        stage=stage,
        root_prim_path=root_prim_path,
        mesh_prim_path=mesh_prim_path,
        sample_points=icp_sample_points,
        seed=icp_seed,
    )

    rotation_matrix, translation, icp_metrics = run_point_to_point_icp(
        source_points=target_points,
        target_points=reference_points,
        max_iterations=icp_max_iterations,
        tolerance=icp_tolerance,
        max_correspondence_distance=icp_max_corr_distance,
    )

    rmse = float(icp_metrics["rmse"])
    if rmse > icp_rmse_threshold:
        raise RuntimeError(
            "icp failed quality threshold "
            f"(rmse={rmse:.6f} threshold={icp_rmse_threshold:.6f})"
        )

    apply_icp_transform_to_mesh(
        stage=stage,
        mesh_prim_path=mesh_prim_path,
        rotation_matrix=rotation_matrix,
        translation=translation,
        apply_translation=True,
    )
    stage.Save()

    return stage, {
        "enabled": True,
        "canonical_usd_path": str(canonical_usd_path),
        "canonical_mesh_prim_path": canonical_mesh_prim_path,
        "rmse": rmse,
        "inlier_ratio": float(icp_metrics["inlier_ratio"]),
        "iterations": int(icp_metrics["iterations"]),
        "converged": bool(icp_metrics["converged"] > 0.5),
        "det_r": float(icp_metrics["det_r"]),
        "translation_norm": float(np.linalg.norm(translation)),
    }
