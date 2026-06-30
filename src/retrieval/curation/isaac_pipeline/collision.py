from __future__ import annotations

import logging

from .mesh_analyzer import MeshAnalyzer

logger = logging.getLogger(__name__)


class CollisionGenerator:
    def __init__(self, use_coacd: bool = True, coacd_threshold: float = 0.05):
        self.use_coacd = use_coacd
        self.coacd_threshold = coacd_threshold
        self.analyzer = MeshAnalyzer()
        self._coacd_available = None

    def generate_collision_mesh(self, stage, mesh_prim, analysis, collision_prim_path):
        recommendation = analysis.get("recommendation", "simple")

        if recommendation == "simple":
            return self._generate_native_collision(stage, mesh_prim, analysis, collision_prim_path)

        if recommendation == "coacd" and self.use_coacd:
            success = self._generate_coacd_collision(stage, mesh_prim, analysis, collision_prim_path)
            if success:
                return True
            logger.warning("CoACD generation failed, falling back to native method")

        return self._generate_native_collision(stage, mesh_prim, analysis, collision_prim_path)

    def _generate_native_collision(self, stage, mesh_prim, analysis, collision_prim_path):
        from pxr import UsdGeom, UsdPhysics

        collision_prim = stage.DefinePrim(collision_prim_path, "Mesh")
        collision_prim.GetReferences().AddReference(mesh_prim.GetPrim().GetPath())
        UsdPhysics.CollisionAPI.Apply(collision_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)

        min_dim = analysis.get("min_dimension", 0.05)
        if min_dim < 0.01:
            mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        else:
            mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)

        UsdGeom.Imageable(collision_prim).CreateVisibilityAttr("invisible")
        return True

    def _generate_coacd_collision(self, stage, mesh_prim, analysis, collision_prim_path):
        import coacd
        from pxr import Gf, UsdGeom, UsdPhysics

        mesh_data = self.analyzer.extract_mesh_data(mesh_prim)
        if mesh_data is None:
            return False

        vertices, faces = mesh_data
        coacd_mesh = coacd.Mesh(vertices, faces)

        min_dim = analysis.get("min_dimension", 0.05)
        max_dim = analysis.get("max_dimension", 1.0)
        adaptive_threshold = self.coacd_threshold
        if min_dim < 0.01:
            adaptive_threshold = self.coacd_threshold * 0.5
        elif max_dim > 1.0:
            adaptive_threshold = self.coacd_threshold * 1.5

        parts = coacd.run_coacd(
            coacd_mesh,
            threshold=adaptive_threshold,
            max_convex_hulls=64,
            preprocess_mode="auto",
        )
        if not parts:
            return False

        if collision_prim_path.name == "Collisions":
            collision_scope_path = collision_prim_path
        else:
            collision_scope_path = collision_prim_path.AppendChild("Collisions")

        collision_scope = UsdGeom.Scope.Define(stage, collision_scope_path)
        UsdGeom.Imageable(collision_scope).CreateVisibilityAttr("invisible")

        for i, (part_vertices, part_faces) in enumerate(parts):
            part_path = collision_scope_path.AppendChild(f"collision_{i}")
            part_mesh = UsdGeom.Mesh.Define(stage, part_path)

            points = [Gf.Vec3f(v[0], v[1], v[2]) for v in part_vertices]
            part_mesh.GetPointsAttr().Set(points)

            face_vertex_indices = []
            face_vertex_counts = []
            for face in part_faces:
                face_vertex_indices.extend([int(face[0]), int(face[1]), int(face[2])])
                face_vertex_counts.append(3)

            part_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
            part_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)

            UsdPhysics.CollisionAPI.Apply(part_mesh)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(part_mesh)
            mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            UsdGeom.Imageable(part_mesh).CreateVisibilityAttr("invisible")

        return True


def apply_native_collision_fallback(stage, mesh_prim, final_min_dim: float):
    from pxr import UsdPhysics

    prim = mesh_prim if hasattr(mesh_prim, "Apply") else mesh_prim.GetPrim() if hasattr(mesh_prim, "GetPrim") else mesh_prim
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

    if final_min_dim < 0.01:
        mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
    else:
        mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)


def apply_collision(stage, mesh_prim, final_min_dim: float, use_coacd: bool, coacd_threshold: float):
    analyzer = MeshAnalyzer()
    analysis = analyzer.analyze_mesh(mesh_prim)

    # Use the conservative native fallback until advanced collision generation
    # is enabled for this pipeline.
    apply_native_collision_fallback(stage, mesh_prim.GetPrim(), final_min_dim)
    return "native_fallback", analysis
