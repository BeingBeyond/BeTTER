from __future__ import annotations

from typing import Any, Dict, Optional


def _path_to_str(path_obj: Any) -> str:
    if hasattr(path_obj, "pathString"):
        return str(path_obj.pathString)
    return str(path_obj)


def _get_attr(prim: Any, name: str):
    if not hasattr(prim, "GetAttribute"):
        return None
    attr = prim.GetAttribute(name)
    if attr is None:
        return None
    if hasattr(attr, "IsValid") and not attr.IsValid():
        return None
    return attr


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes, int, float, bool)):
        return value
    if hasattr(value, "path"):
        return str(value.path)
    if hasattr(value, "real") and hasattr(value, "imag"):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "__iter__"):
        try:
            return [x for x in value]
        except TypeError:
            return value
    return value


def _get_attr_value(prim: Any, name: str) -> Any:
    attr = _get_attr(prim, name)
    if attr is None:
        return None
    value = attr.Get()
    return _normalize_value(value)


def _set_attr_value(prim: Any, name: str, value: Any) -> None:
    attr = _get_attr(prim, name)
    if attr is None:
        return
    attr.Set(value)


def _find_rotate_z_attr_name(prim: Any) -> Optional[str]:
    if not hasattr(prim, "GetAttributes"):
        return None
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if attr_name.startswith("xformOp:rotateZ"):
            return attr_name
    return None


def _is_light_prim(prim: Any) -> bool:
    type_name = ""
    if hasattr(prim, "GetTypeName"):
        type_name = str(prim.GetTypeName())
    if type_name.endswith("Light"):
        return True
    return type_name in {"DomeLight", "DistantLight", "SphereLight", "DiskLight", "RectLight", "CylinderLight"}


def collect_light_state(stage: Any, root_prim_path: str) -> Dict[str, Dict[str, Any]]:
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if root_prim is None:
        raise RuntimeError(f"Root prim not found: {root_prim_path}")

    root_prefix = root_prim_path.rstrip("/")
    light_state: Dict[str, Dict[str, Any]] = {}

    for prim in stage.Traverse():
        prim_path = _path_to_str(prim.GetPath())
        if prim_path != root_prefix and not prim_path.startswith(root_prefix + "/"):
            continue
        if not _is_light_prim(prim):
            continue

        type_name = str(prim.GetTypeName()) if hasattr(prim, "GetTypeName") else ""
        entry: Dict[str, Any] = {
            "type_name": type_name,
            "intensity": _get_attr_value(prim, "intensity"),
            "exposure": _get_attr_value(prim, "exposure"),
            "color": _get_attr_value(prim, "color"),
            "enableColorTemperature": _get_attr_value(prim, "enableColorTemperature"),
            "colorTemperature": _get_attr_value(prim, "colorTemperature"),
        }

        if type_name == "DomeLight":
            entry["texture:file"] = _get_attr_value(prim, "texture:file")
            rotate_attr = _find_rotate_z_attr_name(prim)
            entry["rotate_z_attr"] = rotate_attr
            if rotate_attr is not None:
                entry["rotate_z"] = _get_attr_value(prim, rotate_attr)

        light_state[prim_path] = entry

    return light_state


def apply_light_state(stage: Any, state_map: Dict[str, Dict[str, Any]]) -> None:
    for prim_path, state in state_map.items():
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None:
            continue

        for attr_name in ["intensity", "exposure", "color", "enableColorTemperature", "colorTemperature", "texture:file"]:
            value = state.get(attr_name)
            if value is not None:
                _set_attr_value(prim, attr_name, value)

        rotate_attr_name = state.get("rotate_z_attr") or _find_rotate_z_attr_name(prim)
        rotate_value = state.get("rotate_z")
        if rotate_attr_name is not None and rotate_value is not None:
            _set_attr_value(prim, rotate_attr_name, rotate_value)


def scale_light_intensities(stage: Any, baseline_state: Dict[str, Dict[str, Any]], factor: float) -> None:
    for prim_path, state in baseline_state.items():
        baseline_intensity = state.get("intensity")
        if baseline_intensity is None:
            continue

        prim = stage.GetPrimAtPath(prim_path)
        if prim is None:
            continue

        _set_attr_value(prim, "intensity", float(baseline_intensity) * float(factor))
