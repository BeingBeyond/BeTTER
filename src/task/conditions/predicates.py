from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.core.quaternion import wxyz_to_xyzw

from .context import ConditionContext


def _bbox_bounds(obj) -> tuple[np.ndarray, np.ndarray]:
    bbox = obj.get_bounding_box()
    if hasattr(bbox, "get_min_bound") and hasattr(bbox, "get_max_bound"):
        min_bound = np.asarray(bbox.get_min_bound(), dtype=float)
        max_bound = np.asarray(bbox.get_max_bound(), dtype=float)
        return min_bound, max_bound

    if isinstance(bbox, (tuple, list)) and len(bbox) == 2:
        min_bound = np.asarray(bbox[0], dtype=float)
        max_bound = np.asarray(bbox[1], dtype=float)
        return min_bound, max_bound

    raise TypeError("Unsupported bounding box type; expected Open3D AABB or (min,max) tuple.")


def _xy_overlap_ratio(min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray) -> float:
    overlap_x = max(0.0, min(max_a[0], max_b[0]) - max(min_a[0], min_b[0]))
    overlap_y = max(0.0, min(max_a[1], max_b[1]) - max(min_a[1], min_b[1]))
    overlap = overlap_x * overlap_y
    area_a = max(1e-8, (max_a[0] - min_a[0]) * (max_a[1] - min_a[1]))
    return overlap / area_a


def _bbox_intersection_volume_ratio(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    contained_min = np.maximum(min_a, min_b)
    contained_max = np.minimum(max_a, max_b)
    contained_volume = float(np.prod(np.maximum(0.0, contained_max - contained_min)))
    volume_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    return contained_volume / max(volume_a, 1.0e-8)


def _bbox_gap_distance(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    gap = np.maximum(0.0, np.maximum(min_a - max_b, min_b - max_a))
    return float(np.linalg.norm(gap))


def _axis_direction(quat_wxyz: object, axis: str) -> np.ndarray:
    quat_xyzw = wxyz_to_xyzw(quat_wxyz)
    matrix = R.from_quat(quat_xyzw).as_matrix()
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    direction = np.asarray(matrix[:, axis_index], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm > 1.0e-8:
        direction = direction / norm
    return direction


def evaluate_node_condition(ctx: ConditionContext, condition: dict[str, Any]) -> bool:
    object_id = condition.get("subject_id") or condition.get("subject")
    if object_id is None:
        raise ValueError(f"Node condition missing subject_id: {condition}")

    attribute = condition.get("attribute")
    if attribute is None:
        raise ValueError(f"Node condition missing attribute: {condition}")

    axis_name = str(condition.get("axis", "z"))
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    if axis_name not in axis_to_idx:
        raise ValueError(f"Unsupported axis '{axis_name}'. Expected one of x/y/z.")
    axis_idx = axis_to_idx[axis_name]

    obj = ctx.get_object(str(object_id))
    pos, quat_wxyz = obj.get_world_pose()

    min_value = condition.get("min")
    max_value = condition.get("max")
    if min_value is None or max_value is None:
        raise ValueError(f"Node condition requires both min and max: {condition}")

    if attribute == "position":
        value = float(np.asarray(pos, dtype=float)[axis_idx])
    elif attribute in {"rotation_deg", "euler_deg"}:
        quat_xyzw = wxyz_to_xyzw(quat_wxyz)
        euler_xyz_deg = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        value = float(euler_xyz_deg[axis_idx])
    else:
        raise ValueError(f"Unsupported node attribute '{attribute}'")

    return float(min_value) <= value <= float(max_value)


def evaluate_edge_condition(ctx: ConditionContext, condition: dict[str, Any]) -> bool:
    subject_id = condition.get("subject_id") or condition.get("subject")
    target_id = condition.get("target_id") or condition.get("object")
    relation = condition.get("relation")

    if subject_id is None or target_id is None or relation is None:
        raise ValueError(f"Edge condition requires subject_id, target_id, relation: {condition}")

    subject = ctx.get_object(str(subject_id))
    target = ctx.get_object(str(target_id))

    s_min, s_max = _bbox_bounds(subject)
    t_min, t_max = _bbox_bounds(target)

    s_center = (s_min + s_max) * 0.5
    t_center = (t_min + t_max) * 0.5

    thresholds = ctx.thresholds
    rel = str(relation)

    if rel == "near":
        return float(np.linalg.norm(s_center - t_center)) <= float(thresholds.near_distance)

    if rel == "touching":
        return _bbox_gap_distance(s_min, s_max, t_min, t_max) < float(thresholds.touching_distance)

    if rel in {"left_of", "left"}:
        return float(s_min[1]) >= float(t_max[1] - thresholds.left_right_tolerance)

    if rel in {"right_of", "right"}:
        return float(s_max[1]) <= float(t_min[1] + thresholds.left_right_tolerance)

    if rel in {"front_of", "front"}:
        subject_pos, _subject_quat = subject.get_world_pose()
        target_pos, _target_quat = target.get_world_pose()
        return float(np.asarray(subject_pos, dtype=float)[0]) > float(np.asarray(target_pos, dtype=float)[0])

    if rel in {"behind", "back"}:
        subject_pos, _subject_quat = subject.get_world_pose()
        target_pos, _target_quat = target.get_world_pose()
        return float(np.asarray(subject_pos, dtype=float)[0]) < float(np.asarray(target_pos, dtype=float)[0])

    if rel == "above":
        overlap = _xy_overlap_ratio(s_min, s_max, t_min, t_max)
        return float(s_min[2]) > float(t_max[2]) and overlap >= float(thresholds.above_overlap_ratio)

    if rel in {"on", "top"}:
        subject_pos, _subject_quat = subject.get_world_pose()
        target_pos, _target_quat = target.get_world_pose()
        z_diff = float(np.asarray(subject_pos, dtype=float)[2] - np.asarray(target_pos, dtype=float)[2])
        overlap = _xy_overlap_ratio(s_min, s_max, t_min, t_max)
        return (
            z_diff >= float(thresholds.on_z_offset_min)
            and z_diff <= float(thresholds.on_z_offset_max)
            and overlap >= float(thresholds.on_overlap_ratio)
        )

    if rel == "in":
        containment_ratio = _bbox_intersection_volume_ratio(s_min, s_max, t_min, t_max)
        threshold = getattr(thresholds, "in_containment_ratio", thresholds.contain_xy_ratio)
        return containment_ratio >= float(threshold)

    if rel == "vertical_align":
        xy_distance = float(np.linalg.norm(s_center[:2] - t_center[:2]))
        return xy_distance < float(thresholds.vertical_align_distance)

    if rel in {"axis_align_x", "axis_align_y", "axis_align_z"}:
        axis = rel[-1]
        _subject_pos, subject_quat = subject.get_world_pose()
        _target_pos, target_quat = target.get_world_pose()
        subject_axis = _axis_direction(subject_quat, axis)
        target_axis = _axis_direction(target_quat, axis)
        dot = float(np.clip(np.dot(subject_axis, target_axis), -1.0, 1.0))
        if dot < 0.0:
            return False
        angle = float(np.arccos(dot))
        return angle < float(thresholds.axis_align_angle)

    raise ValueError(f"Unsupported edge relation '{relation}'")
