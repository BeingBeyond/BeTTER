from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from ..contracts import EditorSessionAsset, EditorSessionRequest
from .editor_stage_builder import create_editable_stage, create_preview_stage
from .editor_validation import validate_editor_stage
from .runtime import _convert_glb_to_temp_usd


_USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def _run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        raise RuntimeError("cannot run editor asset conversion while an event loop is already running")
    return loop.run_until_complete(coro)


def _resolve_session_id(request: EditorSessionRequest) -> str:
    return request.session_id or f"{request.uid}-{request.mode}-{uuid4().hex[:8]}"


def _prepare_source_usd(request: EditorSessionRequest, session_id: str) -> Path | None:
    source_path = request.source_asset_path
    if source_path.suffix.lower() in _USD_SUFFIXES:
        return source_path

    temp_usd_dir = request.temp_usd_dir or (request.session_dir / session_id / "temp")
    temp_usd_dir.mkdir(parents=True, exist_ok=True)
    temp_usd_path = temp_usd_dir / f"{request.uid}_source.usd"
    _run_sync(
        _convert_glb_to_temp_usd(
            glb_path=source_path,
            temp_usd_path=temp_usd_path,
            timeout_seconds=request.convert_timeout_seconds,
        )
    )
    return temp_usd_path


def open_editor_session(request: EditorSessionRequest) -> EditorSessionAsset:
    from pxr import Usd

    session_id = _resolve_session_id(request)
    session_root = request.session_dir / session_id
    session_root.mkdir(parents=True, exist_ok=True)
    stage_path = session_root / f"{request.mode}_session.usda"
    source_usd_path = _prepare_source_usd(request, session_id)

    if source_usd_path is None:
        raise RuntimeError(f"failed to resolve source asset for session: {request.source_asset_path}")

    fallback_mode = "none"
    if request.mode == "preview":
        stage, _, _, root_prim_path, mesh_prim_path = create_preview_stage(
            output_usd_path=stage_path,
            source_usd_path=source_usd_path,
            prim_name=request.root_prim_name,
        )
    else:
        stage, _, _, root_prim_path, mesh_prim_path, fallback_mode = create_editable_stage(
            output_usd_path=stage_path,
            source_usd_path=source_usd_path,
            prim_name=request.root_prim_name,
            flatten_for_edit=request.flatten_for_edit,
        )

    stage.Save()
    reopened_stage = Usd.Stage.Open(str(stage_path))
    if reopened_stage is None:
        raise RuntimeError(f"failed to reopen editor session stage: {stage_path}")

    validation = validate_editor_stage(
        reopened_stage,
        session_id=session_id,
        stage_path=stage_path,
        root_prim_path=root_prim_path,
    )

    temp_usd_path = None
    if request.source_asset_path.suffix.lower() not in _USD_SUFFIXES:
        temp_usd_path = source_usd_path

    return EditorSessionAsset(
        uid=request.uid,
        session_id=session_id,
        mode=request.mode,
        source_asset_path=request.source_asset_path,
        stage_path=stage_path,
        root_prim_path=root_prim_path,
        mesh_prim_path=mesh_prim_path,
        local_mesh_owned=validation.passes_local_mesh_ownership,
        temp_usd_path=temp_usd_path,
        metadata={
            **dict(request.metadata),
            "flatten_for_edit": request.flatten_for_edit,
            "fallback_mode": fallback_mode,
            "source_usd_path": str(source_usd_path),
            "publish_ready": validation.is_publish_ready,
            "validation_issues": list(validation.issues),
        },
    )
