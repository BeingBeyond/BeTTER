from __future__ import annotations

import asyncio
import json
import shutil
import sys
import traceback
from pathlib import Path
from uuid import uuid4

import omni.ext
import yaml


BACKEND_MODULE = "src.retrieval.curation"
EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXTENSION_ROOT.parents[2]
SCAN_ROOT = EXTENSION_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "src" / "retrieval" / "curation" / "isaac_pipeline"
TASKS_ROOT = REPO_ROOT / "assets" / "tasks"
REGISTRY_ROOT = REPO_ROOT / "assets" / "objects" / "registry"
DEFAULT_SESSION_DIR = REPO_ROOT / "outputs" / "assets_editor_sessions"
EDITOR_ROOT_PRIM_PATH = "/World/AssetsEditor"
PREVIEW_MOUNT_PRIM_PATH = f"{EDITOR_ROOT_PRIM_PATH}/PreviewMount"
EDITABLE_MOUNT_PRIM_PATH = f"{EDITOR_ROOT_PRIM_PATH}/EditableMount"
PUBLISHED_SESSION_DIR = DEFAULT_SESSION_DIR / "published"
BACKGROUND_REGISTRY_PATH = REPO_ROOT / "assets" / "scenes" / "backgrounds" / "registry.v2.json"
BACKGROUND_PRIM_PATH = "/World/Background"
BACKGROUND_TRANSLATE = (-0.5, 0.0, 0.0)
RETRIEVAL_SERVER_URL = "http://127.0.0.1:8001"
COMPARE_ROOT_PRIM_PATH = f"{EDITOR_ROOT_PRIM_PATH}/Compare"
COMPARE_MAX_CANDIDATES = 4
COMPARE_SLOT_SPACING = 0.375
DEFAULT_ASSET_SCALE_RANGE = (0.1, 0.3)
DEFAULT_ASSET_MASS_RANGE = (0.1, 1.0)
OBJECT_UNIVERSE_GROUP_KEYS = ("container", "goal_objects", "fail_objects", "decor_objects")
WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 1080


class AssetsEditorExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        import omni.ui as ui

        self._ensure_repo_on_sys_path()
        self._ui = ui
        self._ext_id = ext_id
        self._session_asset = None
        self._selected_task_key = None
        self._selected_instance_id = None
        self._selected_candidate_index = 0
        self._selected_staging_index = 0
        self._remove_candidate_dialog = None
        self._active_asset_source_kind = "candidate"
        self._status_model = ui.SimpleStringModel(
            "Extension loaded.\nUI stays thin; editor logic remains in the external backend."
        )
        self._session_dir_model = ui.SimpleStringModel(str(DEFAULT_SESSION_DIR))
        self._session_mode_model = ui.SimpleStringModel("No active session")
        self._session_stage_model = ui.SimpleStringModel("")
        self._validation_model = ui.SimpleStringModel("No validation run yet")
        self._resolved_asset_path_model = ui.SimpleStringModel("")
        self._retrieval_server_url_model = ui.SimpleStringModel(RETRIEVAL_SERVER_URL)
        self._retrieval_status_model = ui.SimpleStringModel("Unchecked")
        self._staging_assets = []
        self._retrieval_offsets = {}
        self._retrieval_batch_size = 5
        self._staging_prepare_dir = DEFAULT_SESSION_DIR / "staging_prepared"
        self._staging_download_dir = DEFAULT_SESSION_DIR / "staging_downloads"
        self._background_manager = None
        self._active_background_scene = None
        self._active_background_variant = None
        self._background_registry_path = BACKGROUND_REGISTRY_PATH
        self._background_prim_path = BACKGROUND_PRIM_PATH
        self._background_translate = BACKGROUND_TRANSLATE
        self._preview_mount_prim_path = PREVIEW_MOUNT_PRIM_PATH
        self._editable_mount_prim_path = EDITABLE_MOUNT_PRIM_PATH
        self._compare_root_prim_path = COMPARE_ROOT_PRIM_PATH
        self._compare_slot_spacing = COMPARE_SLOT_SPACING
        self._compare_mode_model = ui.SimpleStringModel("No compare view")
        self._compared_candidates = []
        self._compare_session_assets = []
        # persistent host stage state
        self._host_stage_ready = False
        self._host_stage_identifier = ""
        self._preview_session_asset = None
        self._editable_session_asset = None
        self._active_mode = None
        self._active_session_layer_path = None
        # legacy fields kept for backward compat with any stale references
        self._active_mount_prim_path = ""
        self._active_focus_prim_path = ""
        self._tasks = self._discover_tasks()
        if self._tasks:
            self._selected_task_key = next(iter(self._tasks.keys()))
            selected_task = self._tasks[self._selected_task_key]
            if selected_task["instances"]:
                self._selected_instance_id = selected_task["instances"][0]["instance_id"]
        self._window = ui.Window("BeTTER Assets Editor", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self._window.frame.set_build_fn(self._build_ui)
        self._request_rebuild()

    def _ensure_repo_on_sys_path(self):
        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.append(repo_path)

    def _load_backend_api(self):
        from src.retrieval.curation import (
            EditorSessionRequest,
            bake_editor_session_to_geometry,
            get_editor_session_validation,
            open_editor_session,
            publish_editor_session_asset,
            save_editor_session,
        )

        return (
            EditorSessionRequest,
            open_editor_session,
            get_editor_session_validation,
            save_editor_session,
            bake_editor_session_to_geometry,
            publish_editor_session_asset,
        )

    def _load_task_spec_api(self):
        from src.task.specs import load_task_spec

        return load_task_spec

    def _schedule_async(self, coro):
        asyncio.ensure_future(coro)

    async def _open_stage_in_viewport(self, stage_path: Path, mode: str):
        import omni.usd

        context = omni.usd.get_context()
        result, error = await context.open_stage_async(str(stage_path))
        if not result:
            raise RuntimeError(f"failed to open {mode} stage in Isaac Sim: {stage_path} (error={error})")
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"stage unavailable after opening {mode} stage: {stage_path}")
        return context, stage

    async def _ensure_host_stage(self):
        import omni.usd

        context = omni.usd.get_context()
        if self._host_stage_ready:
            stage = context.get_stage()
            if stage is not None:
                root_layer = stage.GetRootLayer()
                current_id = getattr(root_layer, "realPath", "") or root_layer.identifier
                if current_id == self._host_stage_identifier:
                    return context, stage
        result, error = await context.new_stage_async()
        if not result:
            raise RuntimeError(f"failed to create editor host stage: error={error}")
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("editor host stage unavailable after creation")
        root_layer = stage.GetRootLayer()
        self._host_stage_identifier = getattr(root_layer, "realPath", "") or root_layer.identifier
        self._host_stage_ready = True
        self._load_background_into_stage(stage)
        return context, stage

    def _mount_session_payload(self, stage, mount_prim_path: str, session_stage_path: Path):
        prim = stage.GetPrimAtPath(mount_prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            prim = stage.DefinePrim(mount_prim_path, "Xform")
        payloads = prim.GetPayloads()
        payloads.ClearPayloads()
        payloads.AddPayload(str(session_stage_path))
        if hasattr(prim, "SetActive"):
            prim.SetActive(True)
        if hasattr(prim, "Load"):
            prim.Load()

    def _unmount_session_payload(self, stage, mount_prim_path: str):
        prim = stage.GetPrimAtPath(mount_prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            return
        if hasattr(prim, "SetActive"):
            prim.SetActive(False)
        if hasattr(prim, "Unload"):
            prim.Unload()
        payloads = prim.GetPayloads() if hasattr(prim, "GetPayloads") else None
        if payloads is not None and hasattr(payloads, "ClearPayloads"):
            payloads.ClearPayloads()

    def _deactivate_mount_prim(self, stage, mount_prim_path: str):
        prim = stage.GetPrimAtPath(mount_prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            return
        if hasattr(prim, "SetActive"):
            prim.SetActive(False)

    def _activate_mount_prim(self, stage, mount_prim_path: str):
        prim = stage.GetPrimAtPath(mount_prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            return
        if hasattr(prim, "SetActive"):
            prim.SetActive(True)
        if hasattr(prim, "Load"):
            prim.Load()

    def _set_edit_target_to_session_layer(self, stage, mount_prim_path: str, session_stage_path: Path):
        from pxr import Sdf, Usd

        session_path_str = str(session_stage_path)
        layer = Sdf.Layer.FindOrOpen(session_path_str)
        if layer is None:
            return False

        mount_prim = stage.GetPrimAtPath(mount_prim_path)
        if mount_prim is None or (hasattr(mount_prim, "IsValid") and not mount_prim.IsValid()):
            return False

        composition_query = Usd.PrimCompositionQuery(mount_prim)
        for arc in composition_query.GetCompositionArcs():
            target_layer = arc.GetTargetLayer()
            if target_layer is None:
                continue
            target_layer_id = getattr(target_layer, "realPath", "") or target_layer.identifier
            if target_layer_id != session_path_str:
                continue
            target_node = arc.GetTargetNode()
            if target_node is None:
                continue
            edit_target = Usd.EditTarget(layer, target_node)
            if edit_target is None or edit_target.IsNull():
                continue
            stage.SetEditTarget(edit_target)
            self._active_session_layer_path = session_path_str
            return True
        return False

    def _remap_prim_path(self, session_prim_path: str, mount_prim_path: str, session_root_prim_path: str) -> str:
        relative = session_prim_path[len(session_root_prim_path):]
        return mount_prim_path + relative

    def _get_asset_identity_signatures(self, asset: dict) -> set[tuple[str, str]]:
        signatures: set[tuple[str, str]] = set()
        if not asset:
            return signatures

        for key in ("source_uid", "asset_key", "registry_usd"):
            value = str(asset.get(key) or "").strip()
            if value:
                signatures.add((key, value))

        path_candidates = [
            asset.get("asset_path"),
            asset.get("registry_usd_path"),
            asset.get("prepared_asset_path"),
            asset.get("downloaded_path"),
        ]
        for value in path_candidates:
            if not value:
                continue
            resolved = Path(str(value)).expanduser().resolve()
            signatures.add(("path", str(resolved)))

        return signatures

    def _get_compare_candidates(self):
        instance = self._get_selected_instance()
        active_asset = self._get_active_asset()
        if instance is None or active_asset is None:
            return []
        active_signatures = self._get_asset_identity_signatures(active_asset)
        selected_index = active_asset.get("index") if active_asset.get("selection_source") == "candidate" else None
        ordered = []
        for candidate in instance["candidates"]:
            if selected_index is not None and candidate["index"] == selected_index:
                continue
            if active_signatures and self._get_asset_identity_signatures(candidate) & active_signatures:
                continue
            ordered.append(candidate)
        return ordered[:COMPARE_MAX_CANDIDATES]

    def _get_compare_mount_prim_path(self, index: int) -> str:
        return f"{self._compare_root_prim_path}/Slot_{index:02d}"

    def _layout_compare_slots(self, stage, count: int):
        from pxr import Gf, UsdGeom

        half = count // 2
        for index in range(count):
            mount_prim_path = self._get_compare_mount_prim_path(index)
            prim = stage.GetPrimAtPath(mount_prim_path)
            if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
                continue
            xformable = UsdGeom.Xformable(prim)
            translate_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
            if index < half:
                y_offset = -(half - index) * self._compare_slot_spacing
            else:
                y_offset = (index - half + 1) * self._compare_slot_spacing
            translate_op.Set(Gf.Vec3d(0.0, y_offset, 0.0))

    def _clear_compare_mounts(self, stage):
        if stage is None:
            self._compared_candidates = []
            self._compare_session_assets = []
            self._compare_mode_model.set_value("No compare view")
            return
        for index in range(COMPARE_MAX_CANDIDATES):
            self._unmount_session_payload(stage, self._get_compare_mount_prim_path(index))
        self._compared_candidates = []
        self._compare_session_assets = []
        self._compare_mode_model.set_value("No compare view")

    def _open_compare_view(self):
        async def _run():
            try:
                if self._active_mode == "editable" and not self._host_stage_ready:
                    raise RuntimeError("compare overlay is unavailable while editable is opened as a direct session stage")
                active_asset = self._get_active_asset()
                if active_asset is None:
                    raise RuntimeError("no active asset available for compare")
                active_asset_path = active_asset.get("asset_path")
                if active_asset_path is None:
                    raise RuntimeError("active asset has no USD path for compare")
                candidates = self._get_compare_candidates()
                if not candidates:
                    raise RuntimeError("no compare candidates available")
                (
                    EditorSessionRequest,
                    open_editor_session,
                    _,
                    _,
                    _,
                    _,
                ) = self._load_backend_api()
                session_dir = Path(self._session_dir_model.as_string).expanduser()
                session_dir.mkdir(parents=True, exist_ok=True)
                context, stage = await self._ensure_host_stage()
                self._clear_compare_mounts(stage)

                compare_sources = [active_asset, *candidates]
                compare_session_assets = []
                for index, candidate in enumerate(compare_sources[:COMPARE_MAX_CANDIDATES]):
                    source_path = Path(candidate["asset_path"]).expanduser()
                    request = EditorSessionRequest(
                        uid=(candidate.get("source_uid") or source_path.stem or f"compare_{index}"),
                        source_asset_path=source_path,
                        session_dir=session_dir,
                        mode="preview",
                        metadata={
                            "registry_usd_ref": candidate.get("registry_usd") or "",
                            "registry_usd_path": str(source_path),
                            "registry_meta_ref": candidate.get("registry_meta") or "",
                            "registry_meta_path": str(candidate.get("registry_meta_path") or ""),
                            "task_key": self._selected_task_key or "",
                            "instance_id": self._selected_instance_id or "",
                            "candidate_index": candidate.get("index", -1),
                            "asset_key": candidate.get("asset_key") or "",
                            "source_uid": candidate.get("source_uid") or "",
                            "compare_slot_index": index,
                            "selection_source": candidate.get("selection_source") or "candidate",
                        },
                    )
                    session_asset = open_editor_session(request)
                    compare_session_assets.append(session_asset)
                    mount_prim_path = self._get_compare_mount_prim_path(index)
                    self._mount_session_payload(stage, mount_prim_path, session_asset.stage_path)

                self._layout_compare_slots(stage, len(compare_session_assets))
                self._compared_candidates = compare_sources[:COMPARE_MAX_CANDIDATES]
                self._compare_session_assets = compare_session_assets
                self._compare_mode_model.set_value(
                    ", ".join((candidate.get("display_label") or candidate.get("source_uid") or candidate.get("asset_key") or str(index)) for index, candidate in enumerate(compare_sources[:COMPARE_MAX_CANDIDATES]))
                )
                self._select_and_frame_prim(self._compare_root_prim_path)
                self._set_status(
                    "Opened compare overlay.\n"
                    f"instance={self._selected_instance_id}\n"
                    f"compared={len(compare_session_assets)}\n"
                    f"active_mode={self._active_mode or 'none'}"
                )
                self._render_ui()
            except Exception:
                self._set_status(f"Open compare failed.\n{traceback.format_exc()}")
                self._render_ui()

        self._schedule_async(_run())

    def _flush_mounted_session_layer(self) -> bool:
        from pxr import Sdf

        if self._active_session_layer_path is None:
            return False
        session_path_str = self._active_session_layer_path
        layer = Sdf.Layer.FindOrOpen(session_path_str)
        if layer is None:
            return False
        layer.Save()
        return True

    def _flush_live_session_stage(self) -> bool:
        # Delegate to mount-aware flush; fall back to root-layer identity check
        # for any remaining direct-stage paths during transition.
        if self._active_session_layer_path is not None:
            return self._flush_mounted_session_layer()
        import omni.usd

        if self._session_asset is None:
            return False
        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            return False
        root_layer = stage.GetRootLayer()
        stage_identifier = getattr(root_layer, "realPath", "") or root_layer.identifier
        session_path = str(self._session_asset.stage_path)
        if stage_identifier != session_path:
            return False
        stage.Save()
        root_layer.Save()
        return True

    def _get_background_manager(self):
        if self._background_manager is None:
            from src.scene.background.manager_v2 import SceneManagerV2

            self._background_manager = SceneManagerV2.from_registry_file(self._background_registry_path)
        return self._background_manager

    def _resolve_background_variant(self):
        manager = self._get_background_manager()
        scene = manager.get_scene()
        layout_id = scene.layouts[0].layout_id if scene.layouts else None
        return manager.resolve_variant(layout_id=layout_id)

    def _load_background_into_stage(self, stage):
        from pxr import Gf, UsdGeom

        manager = self._get_background_manager()
        self._unload_background_from_stage(stage)
        resolved_variant = self._resolve_background_variant()
        background_scene = manager.load_runtime_scene(
            resolved_variant=resolved_variant,
            stage=stage,
            prim_path=self._background_prim_path,
        )
        background_prim = stage.GetPrimAtPath(self._background_prim_path)
        if background_prim is None or (hasattr(background_prim, "IsValid") and not background_prim.IsValid()):
            raise RuntimeError(f"background prim unavailable after load: {self._background_prim_path}")
        xformable = UsdGeom.Xformable(background_prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*self._background_translate))
        self._active_background_scene = background_scene
        self._active_background_variant = resolved_variant
        return resolved_variant

    def _unload_background_from_stage(self, stage):
        background_scene = self._active_background_scene
        self._active_background_scene = None
        self._active_background_variant = None
        if background_scene is None or stage is None:
            return False

        prim = stage.GetPrimAtPath(background_scene.prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            return False

        background_scene.unload(stage)
        return True

    async def _clear_viewport_stage(self):
        import omni.usd

        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is not None and self._host_stage_ready:
            self._clear_compare_mounts(stage)
            self._unmount_session_payload(stage, self._preview_mount_prim_path)
            self._unmount_session_payload(stage, self._editable_mount_prim_path)
            self._unload_background_from_stage(stage)
            try:
                stage.SetEditTarget(stage.GetRootLayer())
            except Exception:
                pass
        self._active_mode = None
        self._active_session_layer_path = None
        self._active_mount_prim_path = ""
        self._active_focus_prim_path = ""
        self._host_stage_ready = False
        self._host_stage_identifier = ""
        result, error = await context.new_stage_async()
        if not result:
            raise RuntimeError(f"failed to clear editor host stage: error={error}")
        return context, context.get_stage()

    async def _unstage_current_session(self):
        if self._active_mode is None and not self._host_stage_ready:
            return False
        import omni.usd

        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is not None and self._host_stage_ready:
            self._clear_compare_mounts(stage)
            self._unmount_session_payload(stage, self._preview_mount_prim_path)
            self._unmount_session_payload(stage, self._editable_mount_prim_path)
            try:
                stage.SetEditTarget(stage.GetRootLayer())
            except Exception:
                pass
        self._session_asset = None
        self._preview_session_asset = None
        self._editable_session_asset = None
        self._compared_candidates = []
        self._compare_session_assets = []
        self._compare_mode_model.set_value("No compare view")
        self._active_mode = None
        self._active_session_layer_path = None
        self._active_mount_prim_path = ""
        self._active_focus_prim_path = ""
        return True

    def _select_and_frame_prim(self, prim_path: str):
        import omni.kit.commands
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport

        context = omni.usd.get_context()
        stage = context.get_stage()
        selection = context.get_selection()
        selection.set_selected_prim_paths([prim_path], True)

        active_viewport = get_active_viewport()
        if active_viewport is None:
            return

        previous_edit_target = None
        if stage is not None:
            previous_edit_target = stage.GetEditTarget()
            try:
                stage.SetEditTarget(stage.GetRootLayer())
            except Exception:
                previous_edit_target = None

        try:
            resolution = active_viewport.resolution or (1, 1)
            aspect_ratio = resolution[0] / max(resolution[1], 1)
            omni.kit.commands.execute(
                "FramePrimsCommand",
                prim_to_move=active_viewport.camera_path,
                prims_to_frame=[prim_path],
                time_code=active_viewport.time,
                aspect_ratio=aspect_ratio,
                zoom=0.6,
            )
        finally:
            if stage is not None and previous_edit_target is not None:
                try:
                    stage.SetEditTarget(previous_edit_target)
                except Exception:
                    pass

    async def _open_session_in_viewport(self, mode: str, stage_path: Path):
        try:
            if mode == "editable":
                await self._open_stage_in_viewport(stage_path, mode)
                self._host_stage_ready = False
                self._host_stage_identifier = ""
                self._active_mode = mode
                self._active_mount_prim_path = ""
                self._active_focus_prim_path = self._session_asset.root_prim_path
                self._active_session_layer_path = None
                self._compare_mode_model.set_value("No compare view")
                self._select_and_frame_prim(self._session_asset.root_prim_path)
                self._set_status(
                    f"Opened {mode} session directly.\n"
                    f"session_stage={stage_path}\n"
                    f"focus={self._session_asset.root_prim_path}\n"
                    f"local_mesh_owned={self._session_asset.local_mesh_owned}"
                )
                self._render_ui()
                return

            context, stage = await self._ensure_host_stage()
            mount_prim_path = self._preview_mount_prim_path
            other_mount_prim_path = self._editable_mount_prim_path

            self._mount_session_payload(stage, mount_prim_path, stage_path)
            self._deactivate_mount_prim(stage, other_mount_prim_path)

            edit_target_set = self._set_edit_target_to_session_layer(stage, mount_prim_path, stage_path)

            session_root = self._session_asset.root_prim_path
            focus_prim_path = self._remap_prim_path(session_root, mount_prim_path, session_root)

            self._active_mode = mode
            self._active_mount_prim_path = mount_prim_path
            self._active_focus_prim_path = focus_prim_path

            self._select_and_frame_prim(focus_prim_path)
            bg = self._active_background_variant
            self._set_status(
                f"Opened {mode} session in host stage.\n"
                f"host_stage={self._host_stage_identifier}\n"
                f"session_stage={stage_path}\n"
                f"mount={mount_prim_path}\n"
                f"focus={focus_prim_path}\n"
                f"edit_target_set={edit_target_set}\n"
                f"background_scene={bg.scene_id if bg else 'none'}"
            )
            self._render_ui()
        except Exception:
            self._set_status(f"Viewport open failed for {mode}.\n{traceback.format_exc()}")
            self._render_ui()

    async def _run_session_mutation(self, action_name: str, callback):
        import omni.usd

        if self._session_asset is None:
            raise RuntimeError("no active session")

        session_mode = self._session_asset.mode
        session_stage_path = self._session_asset.stage_path
        mount_prim_path = self._active_mount_prim_path or (
            self._preview_mount_prim_path if session_mode == "preview" else self._editable_mount_prim_path
        )
        session_was_open = self._active_mode is not None and self._host_stage_ready

        context = omni.usd.get_context()
        stage = context.get_stage() if self._host_stage_ready else None

        if session_was_open:
            self._flush_mounted_session_layer()
            if stage is not None:
                self._unmount_session_payload(stage, mount_prim_path)
                try:
                    stage.SetEditTarget(stage.GetRootLayer())
                except Exception:
                    pass

        try:
            result = callback()
        except Exception:
            if session_was_open and stage is not None:
                try:
                    self._mount_session_payload(stage, mount_prim_path, session_stage_path)
                    self._set_edit_target_to_session_layer(stage, mount_prim_path, session_stage_path)
                except Exception:
                    pass
            raise

        if session_was_open and stage is not None:
            self._mount_session_payload(stage, mount_prim_path, session_stage_path)
            self._set_edit_target_to_session_layer(stage, mount_prim_path, session_stage_path)
            session_root = self._session_asset.root_prim_path
            focus_prim_path = self._remap_prim_path(session_root, mount_prim_path, session_root)
            self._active_focus_prim_path = focus_prim_path
            self._select_and_frame_prim(focus_prim_path)
        return result

    def _discover_tasks(self):
        load_task_spec = self._load_task_spec_api()
        tasks = {}
        for task_yaml in sorted(TASKS_ROOT.glob("*/*/task.yaml")):
            task_dir = task_yaml.parent
            asset_bindings_path = task_dir / "asset_bindings.yaml"
            object_universe_path = task_dir / "object_universe.yaml"
            task_spec = load_task_spec(task_dir)
            task_payload = self._load_yaml(task_yaml)
            universe_payload = self._load_yaml(object_universe_path)
            universe_objects = self._object_universe_by_instance(universe_payload)
            bindings_payload = self._load_yaml(asset_bindings_path) if asset_bindings_path.exists() else {}
            bindings = bindings_payload.get("bindings") or {}
            task_id = str(task_spec.task_id or task_payload.get("task_id") or task_dir.name)
            template_type = str(task_spec.template_type or task_payload.get("template_type") or task_dir.parent.name)
            task_key = f"{template_type}/{task_id}"

            ordered_instance_ids = []
            for group_name in ("container", "target", "distractor", "decor"):
                for instance_id in task_spec.semantic_groups.get(group_name, ()):
                    if instance_id not in ordered_instance_ids:
                        ordered_instance_ids.append(instance_id)
            for instance_id in sorted(task_spec.object_instances.keys()):
                if instance_id not in ordered_instance_ids:
                    ordered_instance_ids.append(instance_id)

            instances = []
            for instance_id in ordered_instance_ids:
                object_spec = task_spec.object_instances.get(instance_id)
                if object_spec is None:
                    continue
                object_payload = universe_objects.get(instance_id, {})
                scale_range = self._resolve_object_range(
                    object_payload,
                    range_key="target_size_range",
                    midpoint_key="target_size",
                    fallback=DEFAULT_ASSET_SCALE_RANGE,
                )
                mass_range = self._resolve_object_range(
                    object_payload,
                    range_key="mass_range",
                    midpoint_key="target_mass",
                    fallback=DEFAULT_ASSET_MASS_RANGE,
                )
                binding = bindings.get(instance_id) or {}
                candidates = []
                for binding_index, candidate in enumerate(binding.get("candidates") or []):
                    if not isinstance(candidate, dict):
                        continue
                    registry_usd = candidate.get("registry_usd")
                    if not registry_usd:
                        continue
                    asset_path = (REGISTRY_ROOT / str(registry_usd)).resolve()
                    candidates.append(
                        {
                            "index": len(candidates),
                            "binding_index": binding_index,
                            "asset_key": str(candidate.get("asset_key") or ""),
                            "source_uid": str(candidate.get("source_uid") or ""),
                            "registry_usd": str(registry_usd),
                            "registry_meta": str(candidate.get("registry_meta") or ""),
                            "asset_path": asset_path,
                        }
                    )
                instances.append(
                    {
                        "instance_id": str(instance_id),
                        "semantic_name": str(binding.get("semantic_name") or object_spec.semantic_name or instance_id),
                        "retrieval_query": str(object_spec.retrieval_query or ""),
                        "description": str(object_spec.description or ""),
                        "role": str(object_spec.role or ""),
                        "group_key": str(object_payload.get("_group_key") or ""),
                        "target_size_range": [float(scale_range[0]), float(scale_range[1])],
                        "target_size": self._range_midpoint(scale_range),
                        "mass_range": [float(mass_range[0]), float(mass_range[1])],
                        "target_mass": self._range_midpoint(mass_range),
                        "physics_type": str(object_payload.get("physics_type") or "rigid"),
                        "materials": self._coerce_string_list(object_payload.get("materials")),
                        "candidates": candidates,
                    }
                )

            if not instances:
                continue

            tasks[task_key] = {
                "task_key": task_key,
                "task_id": task_id,
                "template_type": template_type,
                "task_dir": task_dir,
                "instruction": str(task_payload.get("instruction") or ""),
                "instances": instances,
            }
        return tasks

    def _load_yaml(self, path: Path):
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _object_universe_by_instance(self, universe_payload: dict) -> dict[str, dict]:
        objects = {}
        for group_key in OBJECT_UNIVERSE_GROUP_KEYS:
            for item in universe_payload.get(group_key, []) or []:
                if not isinstance(item, dict) or not item.get("instance_id"):
                    continue
                object_payload = dict(item)
                object_payload["_group_key"] = group_key
                objects[str(item["instance_id"])] = object_payload
        return objects

    def _coerce_float_pair(self, values, fallback: tuple[float, float]) -> tuple[float, float]:
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            try:
                return (float(values[0]), float(values[1]))
            except (TypeError, ValueError):
                return fallback
        return fallback

    def _coerce_string_list(self, values) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        if isinstance(values, (list, tuple)):
            return [str(value) for value in values if str(value)]
        return []

    def _resolve_object_range(
        self,
        object_payload: dict,
        *,
        range_key: str,
        midpoint_key: str,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        if range_key in object_payload:
            return self._coerce_float_pair(object_payload.get(range_key), fallback)
        if midpoint_key in object_payload:
            try:
                value = float(object_payload[midpoint_key])
            except (TypeError, ValueError):
                return fallback
            return (value, value)
        return fallback

    def _range_midpoint(self, values: tuple[float, float]) -> float:
        return (float(values[0]) + float(values[1])) / 2.0

    def _get_selected_task(self):
        if self._selected_task_key is None:
            return None
        return self._tasks.get(self._selected_task_key)

    def _get_selected_instance(self):
        task = self._get_selected_task()
        if task is None:
            return None
        for instance in task["instances"]:
            if instance["instance_id"] == self._selected_instance_id:
                return instance
        return task["instances"][0] if task["instances"] else None

    def _get_selected_candidate(self):
        instance = self._get_selected_instance()
        if instance is None or not instance["candidates"]:
            return None
        index = min(max(self._selected_candidate_index, 0), len(instance["candidates"]) - 1)
        self._selected_candidate_index = index
        candidate = dict(instance["candidates"][index])
        candidate["selection_source"] = "candidate"
        candidate["display_label"] = candidate["source_uid"] or candidate["asset_key"] or f"candidate_{candidate['index']}"
        return candidate

    def _candidate_identity_fields(self, candidate: dict) -> dict[str, str]:
        return {
            key: str(candidate.get(key) or "")
            for key in ("asset_key", "registry_usd", "registry_meta", "source_uid")
            if str(candidate.get(key) or "")
        }

    def _candidate_identity_matches(self, item: dict, candidate: dict) -> bool:
        fields = self._candidate_identity_fields(candidate)
        if not fields:
            return False
        return all(str(item.get(key) or "") == value for key, value in fields.items())

    def _resolve_candidate_binding_index(self, existing: list[dict], candidate: dict) -> int | None:
        binding_index = candidate.get("binding_index")
        if isinstance(binding_index, int) and 0 <= binding_index < len(existing):
            if self._candidate_identity_matches(existing[binding_index], candidate):
                return binding_index

        matches = [
            index
            for index, item in enumerate(existing)
            if self._candidate_identity_matches(item, candidate)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"candidate identity is ambiguous in asset_bindings.yaml: {self._candidate_identity_fields(candidate)}")

        if isinstance(binding_index, int) and 0 <= binding_index < len(existing) and not self._candidate_identity_fields(candidate):
            return binding_index
        return None

    def _get_selected_staging_asset(self):
        if not self._staging_assets:
            return None
        index = min(max(self._selected_staging_index, 0), len(self._staging_assets) - 1)
        self._selected_staging_index = index
        asset = dict(self._staging_assets[index])
        asset["selection_source"] = "staging"
        asset["staging_index"] = index
        asset["display_label"] = asset.get("label") or asset.get("uid") or asset.get("prim_path") or f"staging_{index + 1}"
        registry_usd = asset.get("registry_usd")
        registry_usd_path = asset.get("registry_usd_path")
        prepared_asset_path = asset.get("prepared_asset_path")
        downloaded_path = asset.get("downloaded_path")
        asset_path = None
        if registry_usd:
            asset_path = (REGISTRY_ROOT / str(registry_usd)).resolve()
        elif registry_usd_path:
            asset_path = Path(str(registry_usd_path)).expanduser().resolve()
        elif prepared_asset_path:
            asset_path = Path(str(prepared_asset_path)).expanduser().resolve()
        elif downloaded_path:
            asset_path = Path(str(downloaded_path)).expanduser().resolve()
        asset["asset_path"] = asset_path
        asset.setdefault("source_uid", str(asset.get("uid") or ""))
        asset.setdefault("asset_key", str(asset.get("asset_key") or ""))
        asset.setdefault("registry_usd", str(asset.get("registry_usd") or ""))
        return asset

    def _get_active_asset(self):
        if self._active_asset_source_kind == "staging":
            staging_asset = self._get_selected_staging_asset()
            if staging_asset is not None:
                return staging_asset
        candidate = self._get_selected_candidate()
        if candidate is not None:
            return candidate
        return self._get_selected_staging_asset()

    def _refresh_resolved_asset_path(self):
        asset = self._get_active_asset()
        asset_path = asset.get("asset_path") if asset else None
        self._resolved_asset_path_model.set_value(str(asset_path) if asset_path else "")

    def _render_ui(self):
        self._request_rebuild()

    def _request_rebuild(self):
        if getattr(self, "_window", None) is not None:
            self._refresh_resolved_asset_path()
            self._window.frame.rebuild()

    def _build_ui(self):
        ui = self._ui
        selected_task = self._get_selected_task()
        selected_instance = self._get_selected_instance()
        active_asset = self._get_active_asset()

        with self._window.frame:
            with ui.VStack(spacing=8):
                self._build_header_bar(ui, selected_task, selected_instance, active_asset)
                with ui.HStack(spacing=10):
                    self._build_navigation_pane(ui, selected_task)
                    self._build_candidate_pane(ui, selected_instance, active_asset)
                    self._build_detail_pane(ui, selected_task, selected_instance, active_asset)

    def _build_header_bar(self, ui, selected_task, selected_instance, active_asset):
        with ui.ZStack(height=62):
            ui.Rectangle()
            with ui.VStack(spacing=4):
                ui.Spacer(height=4)
                with ui.HStack(spacing=8, height=24):
                    ui.Label("BeTTER Assets Editor", width=170)
                    ui.Label("Task", width=34)
                    task_model = self._ui.SimpleStringModel(selected_task["task_key"] if selected_task else "")
                    task_field = ui.StringField(model=task_model, height=22)
                    task_field.enabled = False
                    ui.Label("Session dir", width=72)
                    session_field = ui.StringField(model=self._session_dir_model, height=22)
                    session_field.enabled = True
                    ui.Button("Reload", width=72, clicked_fn=self._reload_tasks)
                with ui.HStack(spacing=14, height=20):
                    ui.Label(f"tasks {len(self._tasks)}")
                    ui.Label(f"instances {len(selected_task['instances']) if selected_task else 0}")
                    ui.Label(f"candidates {len(selected_instance['candidates']) if selected_instance else 0}")
                    ui.Label(f"selected {active_asset['display_label'] if active_asset else '-'}")

    def _build_navigation_pane(self, ui, selected_task):
        with ui.VStack(spacing=6, width=320):
            self._build_pane_title(ui, "Tasks")
            self._build_compact_text_block(ui, selected_task["instruction"] if selected_task else "No task selected.", height=54)
            with ui.HStack(spacing=8, height=20):
                ui.Label("Registry", width=52)
                ui.Label(str(TASKS_ROOT), word_wrap=False)
            with ui.ZStack(height=300):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=3):
                        ui.Spacer(height=3)
                        for task_key in self._tasks.keys():
                            self._build_selectable_button(
                                ui,
                                label=task_key,
                                selected=task_key == self._selected_task_key,
                                clicked_fn=lambda key=task_key: self._select_task(key),
                                height=24,
                            )
                        ui.Spacer(height=3)
            self._build_pane_title(ui, "Instances")
            with ui.ZStack(height=560):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=3):
                        ui.Spacer(height=3)
                        if selected_task is None:
                            ui.Label("No registered task found.")
                        else:
                            for instance in selected_task["instances"]:
                                instance_id = instance["instance_id"]
                                semantic = instance["semantic_name"]
                                count = len(instance["candidates"])
                                self._build_selectable_button(
                                    ui,
                                    label=f"{semantic}  |  {instance_id}  |  {count}",
                                    selected=instance_id == self._selected_instance_id,
                                    clicked_fn=lambda value=instance_id: self._select_instance(value),
                                    height=24,
                                )
                        ui.Spacer(height=3)

    def _build_candidate_pane(self, ui, selected_instance, active_asset):
        with ui.VStack(spacing=6, width=500):
            self._build_pane_title(ui, "Candidates")
            with ui.HStack(spacing=8, height=18):
                ui.Label("Instance", width=50)
                ui.Label(selected_instance["instance_id"] if selected_instance else "-")
            with ui.HStack(spacing=8, height=18):
                ui.Label("Semantic", width=50)
                ui.Label(selected_instance["semantic_name"] if selected_instance else "-")
            with ui.ZStack(height=240):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=4):
                        ui.Spacer(height=4)
                        if selected_instance is None:
                            ui.Label("No instance selected.")
                        elif not selected_instance["candidates"]:
                            ui.Label("No candidates bound yet. Use Retrieve 5, Download, Prepare, and Adopt.")
                        else:
                            for candidate in selected_instance["candidates"]:
                                candidate_asset = dict(candidate)
                                candidate_asset["selection_source"] = "candidate"
                                selected = (
                                    active_asset is not None
                                    and active_asset.get("selection_source") == "candidate"
                                    and candidate["index"] == active_asset.get("index")
                                )
                                title = candidate["source_uid"] or candidate["asset_key"] or f"candidate_{candidate['index']}"
                                self._build_candidate_row(ui, title, candidate["asset_key"] or "-", candidate["registry_usd"], selected, lambda idx=candidate["index"]: self._select_candidate(idx))
                        ui.Spacer(height=4)
            self._build_pane_title(ui, "Assets")
            self._build_compact_row(ui, "Server", self._retrieval_status_model.as_string)
            self._build_compact_row(ui, "URL", self._retrieval_server_url_model.as_string)
            state_key = self._get_retrieval_state_key()
            offset = self._retrieval_offsets.get(state_key, 0)
            self._build_compact_row(ui, "Offset", str(offset))
            with ui.HStack(spacing=8, height=30):
                ui.Button("Check Server", clicked_fn=self._check_retrieval_server)
                ui.Button("Retrieve 5", clicked_fn=self._retrieve_staging_assets)
            with ui.HStack(spacing=8, height=30):
                ui.Button("Reset Offset", clicked_fn=self._reset_retrieval_progress)
                ui.Button("Save Scene Selection", clicked_fn=self._save_scene_selection_to_staging)
            with ui.HStack(spacing=8, height=30):
                ui.Button("Download", clicked_fn=self._download_staging_assets)
                ui.Button("Prepare", clicked_fn=self._prepare_staging_assets)
            with ui.HStack(spacing=8, height=30):
                ui.Button("Adopt", clicked_fn=self._adopt_selected_staging_asset)
                ui.Button("Adopt All", clicked_fn=self._adopt_staging_assets)
            with ui.HStack(spacing=8, height=30):
                ui.Button("Clear Staging", clicked_fn=self._clear_staging_assets)
            with ui.HStack(spacing=8, height=30):
                ui.Button("Remove Staging", clicked_fn=self._remove_selected_staging_asset)
            self._build_pane_title(ui, "Staging")
            with ui.ZStack(height=260):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=3):
                        ui.Spacer(height=3)
                        if not self._staging_assets:
                            ui.Label("No staging assets yet.")
                        else:
                            for index, asset in enumerate(self._staging_assets):
                                selected = (
                                    active_asset is not None
                                    and active_asset.get("selection_source") == "staging"
                                    and index == active_asset.get("staging_index")
                                )
                                title = asset.get("label") or asset.get("uid") or asset.get("prim_path") or f"staging_{index + 1}"
                                detail = self._format_staging_asset_detail(asset)
                                self._build_staging_row(ui, title, detail, selected, lambda idx=index: self._select_staging_asset(idx))
                        ui.Spacer(height=3)

    def _build_detail_pane(self, ui, selected_task, selected_instance, active_asset):
        with ui.VStack(spacing=6, width=620):
            self._build_pane_title(ui, "Selection")
            self._build_compact_row(ui, "Task", selected_task["task_key"] if selected_task else "-")
            self._build_compact_row(ui, "Instance", selected_instance["instance_id"] if selected_instance else "-")
            self._build_compact_row(ui, "Semantic", selected_instance["semantic_name"] if selected_instance else "-")
            self._build_compact_row(ui, "Source", active_asset.get("selection_source", "-") if active_asset else "-")
            self._build_compact_row(ui, "Source UID", active_asset["source_uid"] if active_asset and active_asset.get("source_uid") else "-")
            self._build_compact_row(ui, "Mode", self._session_mode_model.as_string)
            self._build_compact_row(ui, "Compare", self._compare_mode_model.as_string)
            self._build_compact_row(ui, "Validation", self._validation_model.as_string)
            with ui.HStack(spacing=10):
                with ui.VStack(spacing=6, width=305):
                    self._build_pane_title(ui, "Resolved asset")
                    self._build_compact_text_block(ui, self._resolved_asset_path_model.as_string or "-", height=88)
                    self._build_pane_title(ui, "Session")
                    self._build_compact_text_block(ui, self._session_stage_model.as_string or "No active session", height=88)
                with ui.VStack(spacing=6, width=305):
                    self._build_pane_title(ui, "View")
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Preview", clicked_fn=self._open_preview_session)
                        ui.Button("Compare", clicked_fn=self._open_compare_session)
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Unstage View", clicked_fn=self._unstage_session)
                        ui.Button("Remove Candidate", clicked_fn=self._confirm_remove_selected_candidate)
                    self._build_pane_title(ui, "Edit")
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Open Editable", clicked_fn=self._open_editable_session)
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Save", clicked_fn=self._save_current_session)
                        ui.Button("Bake", clicked_fn=self._bake_current_session)
                    with ui.HStack(spacing=8, height=30):
                        ui.Button("Validate", clicked_fn=self._validate_current_session)
                        ui.Button("Publish", clicked_fn=self._publish_current_session)
                    self._build_pane_title(ui, "Notes")
                    self._build_compact_text_block(
                        ui,
                        "View opens host-stage inspection tools with background support. Edit opens the session USD directly so viewport edits land in the same file used by Save, Bake, and Publish. Assets lives under Candidates because append/expand operations extend the current candidate pool.",
                        height=126,
                    )
            self._build_pane_title(ui, "Status")
            self._build_compact_text_block(ui, self._status_model.as_string, height=560)

    def _build_pane_title(self, ui, title: str):
        ui.Label(title, height=22)

    def _build_compact_row(self, ui, label: str, value: str):
        with ui.HStack(spacing=8, height=24):
            ui.Label(label, width=78)
            model = self._ui.SimpleStringModel(value or "")
            field = ui.StringField(model=model, height=22)
            field.enabled = False

    def _build_compact_text_block(self, ui, text: str, height: int):
        with ui.ZStack(height=height):
            ui.Rectangle()
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=0):
                    ui.Spacer(height=4)
                    ui.Label(text or "", word_wrap=True)
                    ui.Spacer(height=4)

    def _build_candidate_row(self, ui, title: str, asset_key: str, registry_usd: str, selected: bool, clicked_fn):
        with ui.ZStack(height=26):
            ui.Rectangle()
            with ui.VStack(spacing=0):
                ui.Spacer(height=2)
                self._build_selectable_button(ui, title, selected, clicked_fn, height=22)

    def _build_staging_row(self, ui, title: str, detail: str, selected: bool, clicked_fn):
        with ui.ZStack(height=42):
            ui.Rectangle()
            with ui.VStack(spacing=1):
                ui.Spacer(height=2)
                self._build_selectable_button(ui, title, selected, clicked_fn, height=22)
                self._build_meta_line(ui, "state", detail)

    def _format_staging_asset_detail(self, asset: dict) -> str:
        status_bits = []
        if asset.get("downloaded_path"):
            status_bits.append("downloaded")
        if asset.get("prepared_asset_path"):
            status_bits.append("prepared")
        if asset.get("adopted"):
            status_bits.append("adopted")
        if not status_bits:
            status_bits.append(asset.get("source") or "unknown")
        return ", ".join(status_bits)

    def _build_selectable_button(self, ui, label: str, selected: bool, clicked_fn, height: int):
        button_label = f"> {label}" if selected else label
        ui.Button(button_label, height=height, clicked_fn=clicked_fn)

    def _build_meta_line(self, ui, label: str, value: str):
        with ui.HStack(spacing=6, height=16):
            ui.Label(label, width=38)
            field = ui.StringField(model=self._ui.SimpleStringModel(value or ""), height=16)
            field.enabled = False

    def _get_retrieval_prompt(self) -> str:
        selected_instance = self._get_selected_instance()
        asset = self._get_active_asset()
        semantic_name = selected_instance["semantic_name"] if selected_instance else "asset"
        retrieval_query = selected_instance.get("retrieval_query", "") if selected_instance else ""
        source_uid = asset.get("source_uid") if asset else ""
        return retrieval_query.strip() or semantic_name.replace("_", " ").strip() or source_uid or "asset"

    def _get_retrieval_state_key(self) -> str:
        prompt = self._get_retrieval_prompt()
        instance_id = self._selected_instance_id or ""
        task_key = self._selected_task_key or ""
        return f"{task_key}|{instance_id}|{prompt}"

    def _get_excluded_retrieval_uids(self) -> list[str]:
        excluded = []
        seen = set()
        selected_instance = self._get_selected_instance()
        if selected_instance is not None:
            for candidate in selected_instance["candidates"]:
                uid = (candidate.get("source_uid") or "").strip()
                if uid and uid not in seen:
                    excluded.append(uid)
                    seen.add(uid)
        for asset in self._staging_assets:
            uid = (asset.get("uid") or "").strip()
            if uid and uid not in seen:
                excluded.append(uid)
                seen.add(uid)
        return excluded

    def _append_staging_assets(self, assets: list[dict]):
        existing_keys = {
            (
                asset.get("source") or "",
                asset.get("uid") or "",
                asset.get("prim_path") or "",
                asset.get("label") or "",
            )
            for asset in self._staging_assets
        }
        for asset in assets:
            key = (
                asset.get("source") or "",
                asset.get("uid") or "",
                asset.get("prim_path") or "",
                asset.get("label") or "",
            )
            if key in existing_keys:
                continue
            self._staging_assets.append(asset)
            existing_keys.add(key)

    def _next_retrieval_batch(self, client, prompt: str, batch_size: int) -> tuple[list[str], int, int]:
        state_key = self._get_retrieval_state_key()
        offset = self._retrieval_offsets.get(state_key, 0)
        excluded_uids = self._get_excluded_retrieval_uids()

        uids = client.search(
            prompt=prompt,
            top_k=batch_size,
            offset=offset,
            exclude_uids=excluded_uids,
        )
        next_offset = offset + len(uids)

        if not uids and offset > 0:
            offset = 0
            uids = client.search(
                prompt=prompt,
                top_k=batch_size,
                offset=offset,
                exclude_uids=excluded_uids,
            )
            next_offset = len(uids)

        self._retrieval_offsets[state_key] = next_offset
        return uids, offset, next_offset

    def _reset_retrieval_progress(self):
        state_key = self._get_retrieval_state_key()
        self._retrieval_offsets[state_key] = 0
        self._set_status(f"Reset retrieval offset for {state_key}.")
        self._render_ui()

    def _get_staging_retrieval_assets(self) -> list[dict]:
        return [asset for asset in self._staging_assets if asset.get("source") == "retrieval"]

    def _format_staging_assets(self) -> str:
        if not self._staging_assets:
            return "No staging assets yet."
        lines = []
        for index, asset in enumerate(self._staging_assets, start=1):
            label = asset.get("label") or asset.get("uid") or asset.get("prim_path") or f"staging_{index}"
            source = asset.get("source") or "unknown"
            prompt = asset.get("prompt")
            status_bits = []
            if asset.get("downloaded_path"):
                status_bits.append("downloaded")
            if asset.get("prepared_asset_path"):
                status_bits.append("prepared")
            if asset.get("adopted"):
                status_bits.append("adopted")
            suffix = f" [{source}]"
            if prompt:
                suffix += f" prompt={prompt}"
            if status_bits:
                suffix += f" status={','.join(status_bits)}"
            lines.append(f"{index}. {label}{suffix}")
        return "\n".join(lines)

    def _load_retriever_client_class(self):
        import importlib

        retrieval_client_module = importlib.import_module("src.retrieval.client")
        retrieval_client_module = importlib.reload(retrieval_client_module)
        return retrieval_client_module.RetrieverClient

    def _build_staging_asset_key(self, semantic_name: str, uid: str) -> str:
        semantic_slug = (semantic_name or "asset").strip().replace(" ", "_") or "asset"
        return f"{uid}__{semantic_slug}__{uuid4().hex[:8]}"

    def _build_staging_registry_relpaths(self, asset_key: str) -> tuple[str, str, Path, Path]:
        selected_task = self._get_selected_task()
        if selected_task is None:
            raise RuntimeError("no selected task for staging adoption")
        registry_dir = REGISTRY_ROOT / selected_task["template_type"] / selected_task["task_id"]
        registry_dir.mkdir(parents=True, exist_ok=True)
        usd_path = registry_dir / f"{asset_key}.usd"
        meta_path = registry_dir / f"{asset_key}.meta.json"
        return (
            str(usd_path.relative_to(REGISTRY_ROOT)),
            str(meta_path.relative_to(REGISTRY_ROOT)),
            usd_path,
            meta_path,
        )

    def _download_staging_assets(self):
        try:
            RetrieverClient = self._load_retriever_client_class()
            client = RetrieverClient(self._retrieval_server_url_model.as_string)
            retrieval_assets = [asset for asset in self._get_staging_retrieval_assets() if asset.get("uid") and not asset.get("downloaded_path")]
            if not retrieval_assets:
                raise RuntimeError("no retrieval staging assets pending download")
            download_dir = self._staging_download_dir
            download_dir.mkdir(parents=True, exist_ok=True)
            uids = [str(asset["uid"]) for asset in retrieval_assets]
            paths = client.download(uids=uids, download_dir=str(download_dir))
            downloaded = 0
            for asset in retrieval_assets:
                path = paths.get(str(asset["uid"]))
                if not path:
                    continue
                asset["downloaded_path"] = str(path)
                downloaded += 1
            self._set_status(
                "Downloaded staging assets.\n"
                f"requested={len(uids)}\n"
                f"downloaded={downloaded}\n"
                f"download_dir={download_dir}"
            )
        except Exception as exc:
            self._set_status(f"Download staging assets failed.\n{exc}")
        self._render_ui()

    def _prepare_staging_assets(self):
        try:
            from src.retrieval.curation import IsaacSimPhysicsAdapter, PhysicsPreprocessInput

            selected_instance = self._get_selected_instance()
            semantic_name = selected_instance["semantic_name"] if selected_instance else "asset"
            scale_range = self._coerce_float_pair(
                selected_instance.get("target_size_range") if selected_instance else None,
                DEFAULT_ASSET_SCALE_RANGE,
            )
            mass_range = self._coerce_float_pair(
                selected_instance.get("mass_range") if selected_instance else None,
                DEFAULT_ASSET_MASS_RANGE,
            )
            physics_type = str((selected_instance or {}).get("physics_type") or "rigid")
            materials = self._coerce_string_list((selected_instance or {}).get("materials"))
            preprocessing_metadata = {
                "task_key": self._selected_task_key or "",
                "instance_id": self._selected_instance_id or "",
                "semantic_name": semantic_name,
                "target_size_range": [float(scale_range[0]), float(scale_range[1])],
                "target_size": self._range_midpoint(scale_range),
                "mass_range": [float(mass_range[0]), float(mass_range[1])],
                "target_mass": self._range_midpoint(mass_range),
                "physics_type": physics_type,
                "materials": materials,
            }
            to_prepare = [asset for asset in self._get_staging_retrieval_assets() if asset.get("downloaded_path") and not asset.get("prepared_asset_path")]
            if not to_prepare:
                raise RuntimeError("no downloaded staging assets pending preparation")
            output_dir = self._staging_prepare_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            adapter = IsaacSimPhysicsAdapter()
            prepared = 0
            for asset in to_prepare:
                uid = str(asset.get("uid") or "").strip()
                if not uid:
                    continue
                preprocess_input = PhysicsPreprocessInput(
                    uid=uid,
                    glb_path=Path(str(asset["downloaded_path"])).expanduser(),
                    output_dir=output_dir,
                    scale_range=scale_range,
                    mass_range=mass_range,
                    physics_type=physics_type,
                    materials=materials,
                    metadata={
                        "prompt": asset.get("prompt"),
                        "source": asset.get("source"),
                        "task_object": preprocessing_metadata,
                    },
                )
                prepared_asset = adapter.preprocess(preprocess_input)
                asset_key = self._build_staging_asset_key(semantic_name, uid)
                registry_usd, registry_meta, registry_usd_path, registry_meta_path = self._build_staging_registry_relpaths(asset_key)
                shutil.copy2(prepared_asset.prepared_asset_path, registry_usd_path)
                meta_payload = {
                    "asset_key": asset_key,
                    "source": {
                        "source_uid": uid,
                        "semantic_name": semantic_name,
                        "prompt": asset.get("prompt"),
                        "downloaded_path": str(asset.get("downloaded_path") or ""),
                        "task_object": preprocessing_metadata,
                    },
                    "prepared": {
                        "prepared_asset_path": str(prepared_asset.prepared_asset_path),
                        "physics_applied": prepared_asset.physics_applied,
                        "adapter_name": prepared_asset.adapter_name,
                        "metadata": prepared_asset.metadata,
                    },
                }
                registry_meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
                asset["asset_key"] = asset_key
                asset["registry_usd"] = registry_usd
                asset["registry_meta"] = registry_meta
                asset["prepared_asset_path"] = str(prepared_asset.prepared_asset_path)
                asset["registry_usd_path"] = str(registry_usd_path)
                asset["registry_meta_path"] = str(registry_meta_path)
                prepared += 1
            self._set_status(
                "Prepared staging assets.\n"
                f"prepared={prepared}\n"
                f"target_size={preprocessing_metadata['target_size']} range={preprocessing_metadata['target_size_range']}\n"
                f"target_mass={preprocessing_metadata['target_mass']} range={preprocessing_metadata['mass_range']}\n"
                f"output_dir={output_dir}"
            )
        except Exception as exc:
            self._set_status(f"Prepare staging assets failed.\n{traceback.format_exc()}" if isinstance(exc, Exception) else str(exc))
        self._render_ui()

    def _adopt_assets_into_selected_instance(self, assets: list[dict], *, action_label: str):
        selected_task = self._get_selected_task()
        selected_instance = self._get_selected_instance()
        if selected_task is None or selected_instance is None:
            raise RuntimeError("no selected task/instance for staging adoption")
        task_dir = Path(selected_task["task_dir"])
        asset_bindings_path = task_dir / "asset_bindings.yaml"
        if asset_bindings_path.exists():
            payload = self._load_yaml(asset_bindings_path)
        else:
            payload = {
                "task_id": selected_task["task_id"],
                "template_type": selected_task["template_type"],
                "bindings": {},
            }
        bindings = payload.setdefault("bindings", {})
        binding = bindings.get(self._selected_instance_id)
        if not isinstance(binding, dict):
            binding = {
                "semantic_name": selected_instance.get("semantic_name") or self._selected_instance_id,
                "candidates": [],
            }
            bindings[self._selected_instance_id] = binding
        else:
            binding.setdefault("semantic_name", selected_instance.get("semantic_name") or self._selected_instance_id)
        candidates = list(binding.get("candidates") or [])
        existing_asset_keys = {str(candidate.get("asset_key") or "") for candidate in candidates}
        adoptable = [asset for asset in assets if asset.get("registry_usd") and asset.get("asset_key") and not asset.get("adopted")]
        if not adoptable:
            raise RuntimeError("no prepared staging assets pending adoption")
        added = 0
        for asset in adoptable:
            asset_key = str(asset.get("asset_key") or "")
            if not asset_key or asset_key in existing_asset_keys:
                asset["adopted"] = True
                continue
            candidates.append(
                {
                    "asset_key": asset_key,
                    "registry_usd": str(asset.get("registry_usd") or ""),
                    "registry_meta": str(asset.get("registry_meta") or ""),
                    "source_uid": str(asset.get("uid") or ""),
                }
            )
            existing_asset_keys.add(asset_key)
            asset["adopted"] = True
            added += 1
        binding["candidates"] = candidates
        asset_bindings_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        self._reload_tasks()
        self._set_status(
            f"{action_label}.\n"
            f"instance={self._selected_instance_id}\n"
            f"added={added}\n"
            f"binding={asset_bindings_path}"
        )

    def _adopt_selected_staging_asset(self):
        try:
            asset = self._get_selected_staging_asset()
            if asset is None:
                raise RuntimeError("no staging asset selected")
            self._adopt_assets_into_selected_instance([self._staging_assets[int(asset['staging_index'])]], action_label="Adopted selected staging asset")
        except Exception:
            self._set_status(f"Adopt selected staging asset failed.\n{traceback.format_exc()}")
        self._render_ui()

    def _adopt_staging_assets(self):
        try:
            self._adopt_assets_into_selected_instance(self._get_staging_retrieval_assets(), action_label="Adopted all staging assets")
        except Exception:
            self._set_status(f"Adopt staging assets failed.\n{traceback.format_exc()}")
        self._render_ui()

    def _clear_staging_assets(self):
        self._staging_assets = []
        self._set_status("Cleared staging assets.")
        self._render_ui()

    def _check_retrieval_server(self):
        from urllib.parse import urljoin

        import requests

        server_url = self._retrieval_server_url_model.as_string.rstrip("/")
        health_url = urljoin(f"{server_url}/", "health")
        try:
            response = requests.get(health_url, timeout=1.5)
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "ok":
                raise RuntimeError(f"unexpected health payload: {payload}")
            self._retrieval_status_model.set_value("Alive")
            self._set_status(f"Retrieval server healthy at {health_url}.")
        except Exception as exc:
            self._retrieval_status_model.set_value("Offline")
            self._set_status(f"Retrieval server check failed.\n{exc}")
        self._render_ui()

    def _retrieve_staging_assets(self):
        try:
            RetrieverClient = self._load_retriever_client_class()

            selected_instance = self._get_selected_instance()
            if selected_instance is None:
                raise RuntimeError("no selected instance to infer a retrieval prompt")

            prompt = self._get_retrieval_prompt()

            client = RetrieverClient(self._retrieval_server_url_model.as_string)
            uids, offset_before, offset_after = self._next_retrieval_batch(
                client=client,
                prompt=prompt,
                batch_size=self._retrieval_batch_size,
            )
            new_assets = [
                {
                    "label": uid,
                    "uid": uid,
                    "prompt": prompt,
                    "source": "retrieval",
                }
                for uid in uids
            ]
            self._append_staging_assets(new_assets)
            self._set_status(
                "Retrieved staging assets.\n"
                f"server={self._retrieval_server_url_model.as_string}\n"
                f"prompt={prompt}\n"
                f"offset={offset_before}->{offset_after}\n"
                f"excluded={len(self._get_excluded_retrieval_uids())}\n"
                f"added={len(new_assets)}\n"
                f"staging_total={len(self._staging_assets)}"
            )
        except Exception as exc:
            self._set_status(f"Retrieve staging assets failed.\n{exc}")
        self._render_ui()

    def _save_scene_selection_to_staging(self):
        import omni.usd

        context = omni.usd.get_context()
        selection = context.get_selection()
        prim_paths = list(selection.get_selected_prim_paths()) if selection is not None else []
        if not prim_paths:
            self._set_status("Save scene selection failed.\nNo selected prims in the current scene.")
            self._render_ui()
            return
        self._append_staging_assets(
            [
                {
                    "label": prim_path.rsplit("/", 1)[-1] or prim_path,
                    "prim_path": prim_path,
                    "source": "scene",
                }
                for prim_path in prim_paths
            ]
        )
        self._set_status(
            "Saved scene selection to staging.\n"
            f"count={len(prim_paths)}\n"
            f"staging_total={len(self._staging_assets)}"
        )
        self._render_ui()

    def _set_status(self, message: str):
        self._status_model.set_value(message)
        print(f"[AssetsEditor] {message}", flush=True)

    def _reload_tasks(self):
        previous_task_key = self._selected_task_key
        previous_instance_id = self._selected_instance_id
        self._tasks = self._discover_tasks()
        if previous_task_key in self._tasks:
            self._selected_task_key = previous_task_key
        else:
            self._selected_task_key = next(iter(self._tasks.keys())) if self._tasks else None
        selected_task = self._get_selected_task()
        instance_ids = [instance["instance_id"] for instance in selected_task["instances"]] if selected_task else []
        if previous_instance_id in instance_ids:
            self._selected_instance_id = previous_instance_id
        else:
            self._selected_instance_id = instance_ids[0] if instance_ids else None
        self._selected_candidate_index = 0
        self._set_status(f"Reloaded task registry. tasks={len(self._tasks)}")
        self._render_ui()

    def _select_task(self, task_key: str):
        self._selected_task_key = task_key
        task = self._tasks[task_key]
        self._selected_instance_id = task["instances"][0]["instance_id"] if task["instances"] else None
        self._selected_candidate_index = 0
        self._active_asset_source_kind = "candidate"
        self._render_ui()

    def _select_instance(self, instance_id: str):
        self._selected_instance_id = instance_id
        self._selected_candidate_index = 0
        self._active_asset_source_kind = "candidate"
        self._render_ui()

    def _select_candidate(self, candidate_index: int):
        self._selected_candidate_index = candidate_index
        self._active_asset_source_kind = "candidate"
        self._render_ui()

    def _select_staging_asset(self, staging_index: int):
        self._selected_staging_index = staging_index
        self._active_asset_source_kind = "staging"
        self._render_ui()

    def _remove_selected_staging_asset(self):
        if not self._staging_assets:
            self._set_status("Remove staging asset failed.\nNo staging assets available.")
            self._render_ui()
            return
        asset = self._get_selected_staging_asset()
        if asset is None:
            self._set_status("Remove staging asset failed.\nNo staging asset selected.")
            self._render_ui()
            return
        index = int(asset["staging_index"])
        removed = self._staging_assets.pop(index)
        self._selected_staging_index = max(0, min(index, len(self._staging_assets) - 1)) if self._staging_assets else 0
        self._active_asset_source_kind = "candidate"
        self._set_status(f"Removed staging asset.\nlabel={removed.get('label') or removed.get('uid') or removed.get('prim_path')}")
        self._render_ui()

    def _confirm_remove_selected_candidate(self):
        asset = self._get_selected_candidate()
        if asset is None:
            self._set_status("Remove candidate failed.\nNo candidate selected.")
            self._render_ui()
            return

        ui = self._ui
        label = asset.get("source_uid") or asset.get("asset_key") or "candidate"
        dialog = ui.Window("Confirm Candidate Removal", width=420, height=180, flags=ui.WINDOW_FLAGS_NO_RESIZE)
        self._remove_candidate_dialog = dialog

        def _close_dialog():
            dialog.visible = False

        def _confirm():
            _close_dialog()
            self._remove_selected_candidate(asset)

        with dialog.frame:
            with ui.VStack(spacing=10):
                ui.Spacer(height=8)
                ui.Label("Remove this candidate binding from the current instance?", height=24)
                self._build_compact_text_block(ui, f"instance={self._selected_instance_id or '-'}\nasset={label}", height=56)
                with ui.HStack(spacing=8, height=30):
                    ui.Button("Cancel", clicked_fn=_close_dialog)
                    ui.Button("Remove", clicked_fn=_confirm)

    def _remove_selected_candidate(self, candidate: dict | None = None):
        try:
            candidate = dict(candidate or self._get_selected_candidate() or {})
            if not candidate:
                raise RuntimeError("no candidate selected")
            selected_task = self._get_selected_task()
            if selected_task is None:
                raise RuntimeError("no selected task")
            task_dir = Path(selected_task["task_dir"])
            asset_bindings_path = task_dir / "asset_bindings.yaml"
            payload = self._load_yaml(asset_bindings_path)
            bindings = payload.setdefault("bindings", {})
            binding = bindings.get(self._selected_instance_id)
            if not isinstance(binding, dict):
                raise RuntimeError(f"binding missing for instance {self._selected_instance_id}")
            existing = list(binding.get("candidates") or [])
            remove_index = self._resolve_candidate_binding_index(existing, candidate)
            if remove_index is None:
                raise RuntimeError("selected candidate not found in asset_bindings.yaml")
            removed = existing.pop(remove_index)
            kept = existing
            binding["candidates"] = kept
            asset_bindings_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
            removed_display_index = int(candidate.get("index") or 0)
            self._reload_tasks()
            self._active_asset_source_kind = "candidate"
            selected_instance = self._get_selected_instance()
            remaining_count = len(selected_instance["candidates"]) if selected_instance else 0
            self._selected_candidate_index = max(0, min(removed_display_index, remaining_count - 1)) if remaining_count else 0
            self._set_status(
                "Removed candidate binding.\n"
                f"instance={self._selected_instance_id}\n"
                f"asset_key={removed.get('asset_key') or ''}\n"
                f"registry_usd={removed.get('registry_usd') or ''}\n"
                f"binding={asset_bindings_path}"
            )
        except Exception:
            self._set_status(f"Remove candidate failed.\n{traceback.format_exc()}")
            self._render_ui()

    def _open_session(self, mode: str):
        try:
            asset = self._get_active_asset()
            if asset is None:
                raise RuntimeError("no asset selected")
            asset_path = asset.get("asset_path")
            if asset_path is None:
                raise RuntimeError("selected asset has no resolved path")
            (
                EditorSessionRequest,
                open_editor_session,
                _,
                _,
                _,
                _,
            ) = self._load_backend_api()
            source_path = Path(asset_path).expanduser()
            session_dir = Path(self._session_dir_model.as_string).expanduser()
            session_dir.mkdir(parents=True, exist_ok=True)
            request = EditorSessionRequest(
                uid=asset.get("source_uid") or source_path.stem or "asset",
                source_asset_path=source_path,
                session_dir=session_dir,
                mode=mode,
                metadata={
                    "registry_usd_ref": asset.get("registry_usd") or "",
                    "registry_usd_path": str(source_path),
                    "registry_meta_ref": asset.get("registry_meta") or "",
                    "registry_meta_path": str(asset.get("registry_meta_path") or ""),
                    "task_key": self._selected_task_key or "",
                    "instance_id": self._selected_instance_id or "",
                    "candidate_index": asset.get("index", -1),
                    "asset_key": asset.get("asset_key") or "",
                    "source_uid": asset.get("source_uid") or "",
                    "selection_source": asset.get("selection_source") or "candidate",
                },
            )
            self._session_asset = open_editor_session(request)
            if mode == "preview":
                self._preview_session_asset = self._session_asset
            else:
                self._editable_session_asset = self._session_asset
            self._session_mode_model.set_value(self._session_asset.mode)
            self._session_stage_model.set_value(str(self._session_asset.stage_path))
            self._validation_model.set_value(
                "publish-ready"
                if self._session_asset.metadata.get("publish_ready")
                else ", ".join(self._session_asset.metadata.get("validation_issues", [])) or "not publish-ready"
            )
            self._active_focus_prim_path = ""
            fallback_mode = getattr(self._session_asset, "fallback_mode", None) or self._session_asset.metadata.get("fallback_mode", "none")
            self._set_status(
                f"Opened {mode} session.\n"
                f"task={self._selected_task_key}\n"
                f"instance={self._selected_instance_id}\n"
                f"asset={source_path}\n"
                f"stage={self._session_asset.stage_path}\n"
                f"local_mesh_owned={self._session_asset.local_mesh_owned}\n"
                f"fallback_mode={fallback_mode}"
            )
            self._render_ui()
            self._schedule_async(self._open_session_in_viewport(mode, self._session_asset.stage_path))
        except Exception:
            self._set_status(f"Open {mode} session failed.\n{traceback.format_exc()}")
            self._render_ui()

    def _unstage_session(self):
        async def _run():
            try:
                removed = await self._unstage_current_session()
                if removed:
                    self._set_status("Unstaged current session mount.")
                else:
                    self._set_status("No mounted session to unstage.")
                self._render_ui()
            except Exception:
                self._set_status(f"Unstage failed.\n{traceback.format_exc()}")
                self._render_ui()

        self._schedule_async(_run())

    def _open_preview_session(self):
        self._open_session("preview")

    def _open_editable_session(self):
        self._open_session("editable")

    def _open_compare_session(self):
        self._open_compare_view()

    def _validate_current_session(self):
        try:
            if self._session_asset is None:
                raise RuntimeError("no active session")
            (
                _,
                _,
                get_editor_session_validation,
                _,
                _,
                _,
            ) = self._load_backend_api()
            validation = get_editor_session_validation(self._session_asset)
            validation_text = "publish-ready" if validation.is_publish_ready else ", ".join(validation.issues) or "not publish-ready"
            self._validation_model.set_value(validation_text)
            self._set_status(
                "Validation completed.\n"
                f"publish_ready={validation.is_publish_ready}\n"
                f"issues={list(validation.issues)}"
            )
            self._render_ui()
        except Exception:
            self._set_status(f"Validation failed.\n{traceback.format_exc()}")
            self._render_ui()

    def _save_current_session(self):
        async def _run():
            try:
                if self._session_asset is None:
                    raise RuntimeError("no active session")
                (
                    _,
                    _,
                    get_editor_session_validation,
                    save_editor_session,
                    _,
                    _,
                ) = self._load_backend_api()
                flushed = self._flush_mounted_session_layer()
                save_result = save_editor_session(self._session_asset)
                validation = get_editor_session_validation(self._session_asset)
                validation_text = "publish-ready" if validation.is_publish_ready else ", ".join(validation.issues) or "not publish-ready"
                self._validation_model.set_value(validation_text)
                self._set_status(
                    "Session saved.\n"
                    f"stage={save_result['stage_path']}\n"
                    f"flushed_mounted_layer={flushed}\n"
                    f"publish_ready={validation.is_publish_ready}\n"
                    f"issues={list(validation.issues)}"
                )
                self._render_ui()
            except Exception:
                self._set_status(f"Save session failed.\n{traceback.format_exc()}")
                self._render_ui()

        self._schedule_async(_run())

    def _bake_current_session(self):
        async def _run():
            try:
                if self._session_asset is None:
                    raise RuntimeError("no active session")
                (
                    _,
                    _,
                    get_editor_session_validation,
                    _,
                    bake_editor_session_to_geometry,
                    _,
                ) = self._load_backend_api()
                bake_result = await self._run_session_mutation(
                    "bake",
                    lambda: bake_editor_session_to_geometry(self._session_asset),
                )
                validation = get_editor_session_validation(self._session_asset)
                validation_text = "publish-ready" if validation.is_publish_ready else ", ".join(validation.issues) or "not publish-ready"
                self._validation_model.set_value(validation_text)
                self._set_status(
                    "Bake completed on session USD.\n"
                    f"stage={self._session_asset.stage_path}\n"
                    f"xform_cleanup={bake_result.get('final_xform_cleanup')}\n"
                    f"publish_ready={validation.is_publish_ready}\n"
                    f"issues={list(validation.issues)}"
                )
                self._render_ui()
            except Exception:
                self._set_status(f"Bake session failed.\n{traceback.format_exc()}")
                self._render_ui()

        self._schedule_async(_run())

    def _publish_current_session(self):
        async def _run():
            try:
                if self._session_asset is None:
                    raise RuntimeError("no active session")
                (
                    _,
                    _,
                    get_editor_session_validation,
                    _,
                    _,
                    publish_editor_session_asset,
                ) = self._load_backend_api()
                validation = get_editor_session_validation(self._session_asset)
                validation_text = "publish-ready" if validation.is_publish_ready else ", ".join(validation.issues) or "not publish-ready"
                self._validation_model.set_value(validation_text)
                if not validation.is_publish_ready:
                    raise RuntimeError("session is not publish-ready")
                publish_result = await self._run_session_mutation(
                    "publish",
                    lambda: publish_editor_session_asset(self._session_asset),
                )
                self._set_status(
                    "Publish completed.\n"
                    f"session_stage={self._session_asset.stage_path}\n"
                    f"published_usd={publish_result['published_usd_path']}\n"
                    f"overwrote_registry_usd={publish_result['overwrote_registry_usd']}"
                )
                self._render_ui()
            except Exception:
                self._set_status(f"Publish failed.\n{traceback.format_exc()}")
                self._render_ui()

        self._schedule_async(_run())

    def on_shutdown(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and not loop.is_running():
            loop.run_until_complete(self._clear_viewport_stage())
        else:
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                if stage is not None and self._host_stage_ready:
                    self._clear_compare_mounts(stage)
                    self._unmount_session_payload(stage, self._preview_mount_prim_path)
                    self._unmount_session_payload(stage, self._editable_mount_prim_path)
                    self._unload_background_from_stage(stage)
            except Exception:
                pass

        if getattr(self, "_window", None) is not None:
            self._window = None
        self._session_asset = None
        self._preview_session_asset = None
        self._editable_session_asset = None
        self._active_mode = None
        self._active_session_layer_path = None
        self._active_background_scene = None
        self._active_background_variant = None
        self._background_manager = None
        self._host_stage_ready = False
        self._host_stage_identifier = ""
        self._active_mount_prim_path = ""
        self._active_focus_prim_path = ""
        self._ext_id = None
