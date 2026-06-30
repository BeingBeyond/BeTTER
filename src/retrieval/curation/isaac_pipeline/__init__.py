"""Isaac preprocessing pipeline for full-fidelity GLB->USD asset preparation."""


def preprocess_asset(*args, **kwargs):
    from .runtime import preprocess_asset as _preprocess_asset

    return _preprocess_asset(*args, **kwargs)


def open_editor_session(*args, **kwargs):
    from .editor_session import open_editor_session as _open_editor_session

    return _open_editor_session(*args, **kwargs)


def validate_editor_session_asset(*args, **kwargs):
    from .editor_validation import validate_editor_session_asset as _validate_editor_session_asset

    return _validate_editor_session_asset(*args, **kwargs)


def validate_editor_stage(*args, **kwargs):
    from .editor_validation import validate_editor_stage as _validate_editor_stage

    return _validate_editor_stage(*args, **kwargs)


def normalize_editor_session(*args, **kwargs):
    from .editor_operations import normalize_editor_session as _normalize_editor_session

    return _normalize_editor_session(*args, **kwargs)


def solve_icp_for_editor_session(*args, **kwargs):
    from .editor_operations import solve_icp_for_editor_session as _solve_icp_for_editor_session

    return _solve_icp_for_editor_session(*args, **kwargs)


def apply_icp_to_editor_session(*args, **kwargs):
    from .editor_operations import apply_icp_to_editor_session as _apply_icp_to_editor_session

    return _apply_icp_to_editor_session(*args, **kwargs)


def save_editor_session(*args, **kwargs):
    from .editor_operations import save_editor_session as _save_editor_session

    return _save_editor_session(*args, **kwargs)


def bake_editor_session_to_geometry(*args, **kwargs):
    from .editor_operations import bake_editor_session_to_geometry as _bake_editor_session_to_geometry

    return _bake_editor_session_to_geometry(*args, **kwargs)


def publish_editor_session_asset(*args, **kwargs):
    from .editor_operations import publish_editor_session_asset as _publish_editor_session_asset

    return _publish_editor_session_asset(*args, **kwargs)


def get_editor_session_validation(*args, **kwargs):
    from .editor_operations import get_editor_session_validation as _get_editor_session_validation

    return _get_editor_session_validation(*args, **kwargs)


__all__ = [
    "preprocess_asset",
    "open_editor_session",
    "validate_editor_session_asset",
    "validate_editor_stage",
    "normalize_editor_session",
    "solve_icp_for_editor_session",
    "apply_icp_to_editor_session",
    "save_editor_session",
    "bake_editor_session_to_geometry",
    "publish_editor_session_asset",
    "get_editor_session_validation",
]
