from __future__ import annotations

import numpy as np
from pxr import UsdGeom, Usd


class MeshAnalyzer:
    THIN_WALL_THRESHOLD = 0.01
    ELONGATION_RATIO = 5.0

    def analyze_mesh(self, mesh_prim: UsdGeom.Mesh):
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        bbox = bbox_cache.ComputeWorldBound(mesh_prim)
        bbox_range = bbox.ComputeAlignedRange()

        if bbox_range.IsEmpty():
            return {
                "is_thin_walled": False,
                "is_elongated": False,
                "is_complex": False,
                "min_dimension": 0.0,
                "max_dimension": 0.0,
                "mid_dimension": 0.0,
                "aspect_ratio": 1.0,
                "bbox": None,
                "recommendation": "simple",
            }

        min_point = bbox_range.GetMin()
        max_point = bbox_range.GetMax()
        dimensions = max_point - min_point

        max_dim = max(dimensions[0], dimensions[1], dimensions[2])
        min_dim = min(dimensions[0], dimensions[1], dimensions[2])
        mid_dim = sorted([dimensions[0], dimensions[1], dimensions[2]])[1]

        is_thin_walled = min_dim < self.THIN_WALL_THRESHOLD
        aspect_ratio = max_dim / min_dim if min_dim > 1e-6 else float("inf")
        is_elongated = aspect_ratio > self.ELONGATION_RATIO
        is_complex = is_thin_walled or is_elongated or (mid_dim / max_dim < 0.3)

        if is_complex or is_thin_walled or is_elongated or min_dim < 0.01:
            recommendation = "coacd"
        else:
            recommendation = "simple"

        return {
            "is_thin_walled": is_thin_walled,
            "is_elongated": is_elongated,
            "is_complex": is_complex,
            "min_dimension": min_dim,
            "max_dimension": max_dim,
            "mid_dimension": mid_dim,
            "aspect_ratio": aspect_ratio,
            "bbox": {"min": list(min_point), "max": list(max_point)},
            "dimensions": list(dimensions),
            "recommendation": recommendation,
        }

    def extract_mesh_data(self, mesh_prim: UsdGeom.Mesh):
        points_attr = mesh_prim.GetPointsAttr()
        if not points_attr:
            return None

        points = points_attr.Get()
        if not points:
            return None

        vertices = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)

        face_vertex_indices_attr = mesh_prim.GetFaceVertexIndicesAttr()
        face_vertex_counts_attr = mesh_prim.GetFaceVertexCountsAttr()
        if not face_vertex_indices_attr or not face_vertex_counts_attr:
            return None

        face_vertex_indices = face_vertex_indices_attr.Get()
        face_vertex_counts = face_vertex_counts_attr.Get()
        if not face_vertex_indices or not face_vertex_counts:
            return None

        faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 1],
                    face_vertex_indices[idx + 2],
                ])
                idx += 3
            elif count == 4:
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 1],
                    face_vertex_indices[idx + 2],
                ])
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 2],
                    face_vertex_indices[idx + 3],
                ])
                idx += 4
            else:
                for i in range(1, count - 1):
                    faces.append([
                        face_vertex_indices[idx],
                        face_vertex_indices[idx + i],
                        face_vertex_indices[idx + i + 1],
                    ])
                idx += count

        return vertices, np.array(faces, dtype=np.int32)
