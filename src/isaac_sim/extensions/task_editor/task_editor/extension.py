from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import carb
import carb
import omni.ext
import yaml


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXTENSION_ROOT.parents[2]
TASKS_ROOT = REPO_ROOT / "assets" / "tasks"
TEMPLATES_ROOT = REPO_ROOT / "assets" / "task_templates"
ENDPOINTS_FILE = TEMPLATES_ROOT / "_endpoints.yaml"

WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 1080
LEFT_PANE_WIDTH = 300
MIDDLE_PANE_WIDTH = 440
RIGHT_PANE_WIDTH = 680
ACTION_BUTTON_WIDTH = 148

GROUP_KEYS = ("container", "goal_objects", "fail_objects", "decor_objects")
GROUP_LABELS = {
    "container": "Container",
    "goal_objects": "Goal objects (target)",
    "fail_objects": "Fail objects (distractor)",
    "decor_objects": "Decor objects",
}


class TaskEditorExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        import omni.ui as ui
        import carb

        self._ensure_repo_on_sys_path()
        self._ui = ui
        self._ext_id = ext_id

        # Selection state
        self._selected_task_key = None
        self._selected_object_id = None   # instance_id of selected object
        self._right_pane_mode = "object"  # "object" | "task_info" | "generate"

        # Generate form state
        self._templates = self._load_templates()
        self._endpoints = self._load_endpoints()
        self._selected_template_index = 0
        self._selected_template_combo = None
        self._selected_endpoint_index = 0
        self._selected_endpoint_combo = None
        self._template_combo_subscription = None
        self._endpoint_combo_subscription = None
        self._gen_task_id_model = ui.SimpleStringModel("")
        self._gen_model_name_model = ui.SimpleStringModel("gpt-4o")
        self._gen_guidance_model = ui.SimpleStringModel("")

        # Task info edit models
        self._task_id_model = ui.SimpleStringModel("")
        self._task_instruction_model = ui.SimpleStringModel("")
        self._task_goal_relation_model = ui.SimpleStringModel("")
        self._task_description_model = ui.SimpleStringModel("")

        # Object edit models
        self._obj_semantic_name_model = ui.SimpleStringModel("")
        self._obj_description_model = ui.SimpleStringModel("")
        self._obj_retrieval_query_model = ui.SimpleStringModel("")
        self._obj_mass_min_model = ui.SimpleFloatModel(0.1)
        self._obj_mass_max_model = ui.SimpleFloatModel(0.5)
        self._obj_size_min_model = ui.SimpleFloatModel(0.05)
        self._obj_size_max_model = ui.SimpleFloatModel(0.2)
        self._obj_tags_model = ui.SimpleStringModel("")

        # Add object form
        self._add_obj_mode = False
        self._add_obj_group_key = "goal_objects"
        self._add_obj_semantic_model = ui.SimpleStringModel("")
        self._add_obj_query_model = ui.SimpleStringModel("")
        self._add_obj_description_model = ui.SimpleStringModel("")
        self._add_obj_mass_min_model = ui.SimpleFloatModel(0.1)
        self._add_obj_mass_max_model = ui.SimpleFloatModel(0.5)
        self._add_obj_size_min_model = ui.SimpleFloatModel(0.05)
        self._add_obj_size_max_model = ui.SimpleFloatModel(0.2)
        self._add_obj_tags_model = ui.SimpleStringModel("")

        self._status_model = ui.SimpleStringModel("Task Editor loaded.")
        self._dirty: dict[str, dict] = {}  # task_key -> {"task": {...}, "universe": {...}}

        self._tasks = self._discover_tasks()
        if self._tasks:
            self._selected_task_key = next(iter(self._tasks))

        self._window = ui.Window("BeTTER Task Editor", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self._window.frame.set_build_fn(self._build_ui)
        self._request_rebuild()

    def on_shutdown(self):
        self._window = None

    # ------------------------------------------------------------------
    # sys.path / lazy imports
    # ------------------------------------------------------------------

    def _ensure_repo_on_sys_path(self):
        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.append(repo_path)

    def _load_generation_api(self):
        from src.task.specs import (
            OpenAITaskDraftGenerator,
            TaskTemplateSlotSpec,
            TaskTemplateSpec,
            write_task_authoring_bundle,
        )
        return OpenAITaskDraftGenerator, TaskTemplateSlotSpec, TaskTemplateSpec, write_task_authoring_bundle

    # ------------------------------------------------------------------
    # Template / endpoint registry
    # ------------------------------------------------------------------

    def _load_templates(self) -> list[dict]:
        templates = []
        if not TEMPLATES_ROOT.exists():
            return templates
        for path in sorted(TEMPLATES_ROOT.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                data["_path"] = path
                templates.append(data)
            except Exception:
                pass
        return templates

    def _load_endpoints(self) -> list[dict]:
        if not ENDPOINTS_FILE.exists():
            return [{"name": "OpenAI (default)", "base_url": ""}]
        try:
            data = yaml.safe_load(ENDPOINTS_FILE.read_text(encoding="utf-8")) or {}
            return data.get("endpoints") or [{"name": "OpenAI (default)", "base_url": ""}]
        except Exception:
            return [{"name": "OpenAI (default)", "base_url": ""}]

    def _template_names(self) -> list[str]:
        return [t.get("template_id") or str(t.get("_path", "?")) for t in self._templates]

    def _endpoint_names(self) -> list[str]:
        return [e.get("name", "?") for e in self._endpoints]

    def _selected_template(self) -> dict | None:
        if not self._templates:
            return None
        idx = self._selected_template_index
        if self._selected_template_combo is not None:
            idx = int(self._selected_template_combo.model.get_item_value_model().as_int)
        idx = max(0, min(idx, len(self._templates) - 1))
        self._selected_template_index = idx
        return self._templates[idx]

    def _selected_endpoint(self) -> dict:
        if not self._endpoints:
            return {"name": "OpenAI (default)", "base_url": ""}
        idx = self._selected_endpoint_index
        if self._selected_endpoint_combo is not None:
            idx = int(self._selected_endpoint_combo.model.get_item_value_model().as_int)
        idx = max(0, min(idx, len(self._endpoints) - 1))
        self._selected_endpoint_index = idx
        return self._endpoints[idx]

    def _on_template_index_changed(self, model):
        self._selected_template_index = int(model.as_int)
        self._request_rebuild()

    def _on_endpoint_index_changed(self, model):
        self._selected_endpoint_index = int(model.as_int)
        self._request_rebuild()

    # ------------------------------------------------------------------
    # Task discovery
    # ------------------------------------------------------------------

    def _discover_tasks(self) -> dict[str, dict]:
        tasks = {}
        for task_yaml in sorted(TASKS_ROOT.glob("*/*/task.yaml")):
            task_dir = task_yaml.parent
            try:
                task_data = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            task_id = task_data.get("task_id") or task_dir.name
            template_type = task_data.get("template_type") or task_dir.parent.name
            task_key = f"{template_type}/{task_id}"

            universe_path = task_dir / "object_universe.yaml"
            universe_data: dict = {}
            if universe_path.exists():
                try:
                    universe_data = yaml.safe_load(universe_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    pass

            tasks[task_key] = {
                "task_key": task_key,
                "task_dir": task_dir,
                "task_id": task_id,
                "template_type": template_type,
                "task_data": task_data,
                "universe_data": universe_data,
            }
        return tasks

    def _get_selected_task(self) -> dict | None:
        if self._selected_task_key is None:
            return None
        return self._tasks.get(self._selected_task_key)

    # ------------------------------------------------------------------
    # Dirty / in-memory edit layer
    # ------------------------------------------------------------------

    def _get_dirty_task(self) -> dict | None:
        if self._selected_task_key is None:
            return None
        entry = self._dirty.get(self._selected_task_key)
        if entry is not None:
            return entry
        task = self._get_selected_task()
        if task is None:
            return None
        import copy
        entry = {
            "task": copy.deepcopy(task["task_data"]),
            "universe": copy.deepcopy(task["universe_data"]),
        }
        self._dirty[self._selected_task_key] = entry
        return entry

    def _all_objects_flat(self, entry: dict) -> list[dict]:
        universe = entry["universe"]
        result = []
        for group_key in GROUP_KEYS:
            for obj in universe.get(group_key) or []:
                result.append({**obj, "_group_key": group_key})
        return result

    def _get_selected_object(self) -> dict | None:
        entry = self._get_dirty_task()
        if entry is None or self._selected_object_id is None:
            return None
        for obj in self._all_objects_flat(entry):
            if obj.get("instance_id") == self._selected_object_id:
                return obj
        return None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _select_task(self, task_key: str):
        self._selected_task_key = task_key
        self._selected_object_id = None
        self._add_obj_mode = False
        self._right_pane_mode = "object"
        self._load_task_info_models()
        self._set_status(f"Selected {task_key}.")
        self._request_rebuild()

    def _select_object(self, instance_id: str):
        self._selected_object_id = instance_id
        self._add_obj_mode = False
        self._right_pane_mode = "object"
        self._load_object_models()
        self._set_status(f"Selected object {instance_id}.")
        self._request_rebuild()

    def _open_task_info(self):
        self._right_pane_mode = "task_info"
        self._add_obj_mode = False
        self._load_task_info_models()
        self._request_rebuild()

    def _open_remove_task_dialog(self):
        task = self._get_selected_task()
        if task is None:
            self._set_status("No task selected.")
            return
        dialog = self._ui.Window("Confirm Task Removal", width=440, height=220, flags=self._ui.WINDOW_FLAGS_NO_RESIZE)

        def _close_dialog():
            dialog.visible = False

        def _confirm():
            _close_dialog()
            self._remove_selected_task()

        with dialog.frame:
            with self._ui.VStack(spacing=10):
                self._ui.Spacer(height=8)
                self._ui.Label("Remove this task?", height=24)
                self._build_compact_block(
                    self._ui,
                    f"task={task['task_key']}\npath={task['task_dir']}",
                    height=72,
                )
                with self._ui.HStack(spacing=8, height=30):
                    self._ui.Button("Cancel", clicked_fn=_close_dialog)
                    self._ui.Button("Remove", clicked_fn=_confirm)

    def _remove_selected_task(self):
        task = self._get_selected_task()
        if task is None:
            self._set_status("No task selected.")
            return
        task_dir = task["task_dir"]
        if not task_dir.exists():
            self._set_status(f"Task directory is missing: {task_dir}")
            return
        shutil.rmtree(task_dir)
        self._tasks = self._discover_tasks()
        self._selected_task_key = next(iter(self._tasks), None)
        self._selected_object_id = None
        self._right_pane_mode = "object"
        if self._selected_task_key is not None:
            self._load_task_info_models()
            self._load_object_models()
        self._set_status(f"Removed task '{task['task_key']}'.")
        self._request_rebuild()

    def _open_generate_form(self):
        self._right_pane_mode = "generate"
        self._add_obj_mode = False
        self._selected_template_combo = None
        self._selected_endpoint_combo = None
        self._template_combo_subscription = None
        self._endpoint_combo_subscription = None
        self._request_rebuild()

    def _open_add_object_form(self, group_key: str):
        self._add_obj_mode = True
        self._add_obj_group_key = group_key
        self._add_obj_semantic_model.set_value("")
        self._add_obj_query_model.set_value("")
        self._add_obj_description_model.set_value("")
        self._add_obj_mass_min_model.set_value(0.1)
        self._add_obj_mass_max_model.set_value(0.5)
        self._add_obj_size_min_model.set_value(0.05)
        self._add_obj_size_max_model.set_value(0.2)
        self._add_obj_tags_model.set_value("")
        self._request_rebuild()

    def _cancel_add_object(self):
        self._add_obj_mode = False
        self._request_rebuild()

    def _confirm_add_object(self):
        entry = self._get_dirty_task()
        if entry is None:
            self._set_status("No task selected.")
            return
        semantic = self._add_obj_semantic_model.as_string.strip()
        if not semantic:
            self._set_status("Semantic name cannot be empty.")
            return
        query = self._add_obj_query_model.as_string.strip()
        if not query:
            self._set_status("Retrieval query cannot be empty.")
            return

        import hashlib
        task = self._get_selected_task()
        salt = f"{task['task_id']}:{self._add_obj_group_key}:{semantic}:new"
        suffix = hashlib.md5(salt.encode()).hexdigest()[:4]
        instance_id = f"{semantic.replace(' ', '_')}_new_{suffix}"

        tags_text = self._add_obj_tags_model.as_string.strip()
        tags = [t.strip() for t in tags_text.split(",") if t.strip()]

        new_obj = {
            "instance_id": instance_id,
            "semantic_name": semantic,
            "description": self._add_obj_description_model.as_string.strip(),
            "retrieval_query": query,
            "mass_range": [
                round(float(self._add_obj_mass_min_model.as_float), 4),
                round(float(self._add_obj_mass_max_model.as_float), 4),
            ],
            "target_size_range": [
                round(float(self._add_obj_size_min_model.as_float), 4),
                round(float(self._add_obj_size_max_model.as_float), 4),
            ],
            "tags": tags,
        }
        if self._add_obj_group_key == "container":
            new_obj["role"] = "container"

        universe = entry["universe"]
        universe.setdefault(self._add_obj_group_key, []).append(new_obj)

        self._add_obj_mode = False
        self._selected_object_id = instance_id
        self._load_object_models()
        self._set_status(f"Added '{semantic}' to {GROUP_LABELS.get(self._add_obj_group_key, self._add_obj_group_key)}.")
        self._request_rebuild()

    def _remove_selected_object(self):
        entry = self._get_dirty_task()
        if entry is None or self._selected_object_id is None:
            self._set_status("No object selected.")
            return
        universe = entry["universe"]
        for group_key in GROUP_KEYS:
            group = universe.get(group_key) or []
            for i, obj in enumerate(group):
                if obj.get("instance_id") == self._selected_object_id:
                    group.pop(i)
                    universe[group_key] = group
                    removed_id = self._selected_object_id
                    self._selected_object_id = None
                    self._set_status(f"Removed '{removed_id}'.")
                    self._request_rebuild()
                    return
        self._set_status(f"Object '{self._selected_object_id}' not found.")

    def _save_object_edits(self):
        entry = self._get_dirty_task()
        if entry is None or self._selected_object_id is None:
            self._set_status("No object selected.")
            return
        universe = entry["universe"]
        for group_key in GROUP_KEYS:
            group = universe.get(group_key) or []
            for obj in group:
                if obj.get("instance_id") == self._selected_object_id:
                    obj["semantic_name"] = self._obj_semantic_name_model.as_string.strip()
                    obj["description"] = self._obj_description_model.as_string.strip()
                    obj["retrieval_query"] = self._obj_retrieval_query_model.as_string.strip()
                    mass_min = round(float(self._obj_mass_min_model.as_float), 4)
                    mass_max = round(float(self._obj_mass_max_model.as_float), 4)
                    size_min = round(float(self._obj_size_min_model.as_float), 4)
                    size_max = round(float(self._obj_size_max_model.as_float), 4)
                    if "mass_range" in obj or mass_min != 0.1:
                        obj["mass_range"] = [mass_min, mass_max]
                    if "target_size_range" in obj or size_min != 0.05:
                        obj["target_size_range"] = [size_min, size_max]
                    tags_text = self._obj_tags_model.as_string.strip()
                    obj["tags"] = [t.strip() for t in tags_text.split(",") if t.strip()]
                    self._set_status(f"Saved edits for '{self._selected_object_id}'.")
                    self._request_rebuild()
                    return
        self._set_status(f"Object '{self._selected_object_id}' not found in universe.")

    def _save_task_info_edits(self):
        entry = self._get_dirty_task()
        if entry is None:
            self._set_status("No task selected.")
            return
        task = entry["task"]
        task["task_id"] = self._task_id_model.as_string.strip()
        task["instruction"] = self._task_instruction_model.as_string.strip()
        task["description"] = self._task_description_model.as_string.strip()
        defaults = task.setdefault("defaults", {})
        goal_rel = self._task_goal_relation_model.as_string.strip()
        if goal_rel:
            defaults["primary_goal_relation"] = goal_rel
        self._set_status("Saved task info.")
        self._request_rebuild()

    def _write_task_to_disk(self):
        entry = self._get_dirty_task()
        task_obj = self._get_selected_task()
        if entry is None or task_obj is None:
            self._set_status("No task selected.")
            return
        task_dir = task_obj["task_dir"]
        task_path = task_dir / "task.yaml"
        universe_path = task_dir / "object_universe.yaml"
        task_path.write_text(yaml.safe_dump(entry["task"], sort_keys=False, allow_unicode=False), encoding="utf-8")
        universe_path.write_text(yaml.safe_dump(entry["universe"], sort_keys=False, allow_unicode=False), encoding="utf-8")
        # Refresh cache
        self._tasks = self._discover_tasks()
        self._dirty.pop(self._selected_task_key, None)
        self._set_status(f"Written to {task_dir}.")
        self._request_rebuild()

    def _reload_tasks(self):
        self._dirty.clear()
        self._tasks = self._discover_tasks()
        self._templates = self._load_templates()
        self._endpoints = self._load_endpoints()
        if self._selected_task_key not in self._tasks:
            self._selected_task_key = next(iter(self._tasks), None)
        self._selected_object_id = None
        self._add_obj_mode = False
        self._set_status("Reloaded.")
        self._request_rebuild()

    def _generate_task(self):
        tmpl = self._selected_template()
        if tmpl is None:
            self._set_status("No templates found. Add YAML files to assets/task_templates/.")
            return
        task_id = self._gen_task_id_model.as_string.strip()
        if not task_id:
            self._set_status("Task ID cannot be empty.")
            return
        guidance = self._gen_guidance_model.as_string.strip()
        template_type = tmpl.get("template_type") or ""
        target_dir = TASKS_ROOT / template_type / task_id
        if target_dir.exists():
            self._set_status(f"Task '{template_type}/{task_id}' already exists.")
            return
        endpoint = self._selected_endpoint()
        base_url = endpoint.get("base_url") or None
        model_name = self._gen_model_name_model.as_string.strip() or "gpt-4o"
        self._set_status(f"Generating '{task_id}' with {model_name} ...")
        self._request_rebuild()
        try:
            OpenAITaskDraftGenerator, TaskTemplateSlotSpec, TaskTemplateSpec, write_task_authoring_bundle = self._load_generation_api()
            slots = []
            for s in (tmpl.get("slots") or []):
                slots.append(TaskTemplateSlotSpec(
                    slot_name=str(s.get("slot_name") or ""),
                    group_hint=str(s.get("group_hint") or ""),
                    description=str(s.get("description") or ""),
                    min_count=int(s.get("min_count", 1)),
                    max_count=int(s.get("max_count", 1)),
                    tags=tuple(str(t) for t in (s.get("tags") or [])),
                ))
            template = TaskTemplateSpec(
                template_id=str(tmpl.get("template_id") or ""),
                template_type=template_type,
                slots=tuple(slots),
                allowed_relations=tuple(str(r) for r in (tmpl.get("allowed_relations") or ["in"])),
                primary_goal_relation=str(tmpl.get("primary_goal_relation") or "in"),
                fail_relations=tuple(str(r) for r in (tmpl.get("fail_relations") or ["in", "on"])),
            )
            generator = OpenAITaskDraftGenerator(model=model_name, base_url=base_url)
            bundle = generator.generate_task_authoring_bundle(task_id=task_id, template=template, guidance=guidance)
            write_task_authoring_bundle(bundle, target_dir)
        except Exception as exc:
            tb = traceback.format_exc().strip()
            message = f"Generation failed: {exc}\n\nTraceback:\n{tb}"
            print(message, file=sys.stderr)
            sys.stderr.flush()
            carb.log_error(message)
            self._set_status(message)
            self._request_rebuild()
            return
        self._tasks = self._discover_tasks()
        new_key = f"{template_type}/{task_id}"
        self._selected_task_key = new_key if new_key in self._tasks else next(iter(self._tasks), None)
        self._selected_object_id = None
        self._right_pane_mode = "task_info"
        self._load_task_info_models()
        self._set_status(f"Generated '{template_type}/{task_id}'. Review and save.")
        self._request_rebuild()

    # ------------------------------------------------------------------
    # Model loaders
    # ------------------------------------------------------------------

    def _load_task_info_models(self):
        entry = self._get_dirty_task()
        if entry is None:
            return
        task = entry["task"]
        self._task_id_model.set_value(str(task.get("task_id") or ""))
        self._task_instruction_model.set_value(str(task.get("instruction") or ""))
        self._task_description_model.set_value(str(task.get("description") or ""))
        defaults = task.get("defaults") or {}
        self._task_goal_relation_model.set_value(str(defaults.get("primary_goal_relation") or ""))

    def _load_object_models(self):
        obj = self._get_selected_object()
        if obj is None:
            return
        self._obj_semantic_name_model.set_value(str(obj.get("semantic_name") or ""))
        self._obj_description_model.set_value(str(obj.get("description") or ""))
        self._obj_retrieval_query_model.set_value(str(obj.get("retrieval_query") or ""))
        mass_range = obj.get("mass_range") or [0.1, 0.5]
        self._obj_mass_min_model.set_value(float(mass_range[0]))
        self._obj_mass_max_model.set_value(float(mass_range[1]))
        size_range = obj.get("target_size_range") or [0.05, 0.2]
        self._obj_size_min_model.set_value(float(size_range[0]))
        self._obj_size_max_model.set_value(float(size_range[1]))
        tags = obj.get("tags") or []
        self._obj_tags_model.set_value(", ".join(str(t) for t in tags))

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        self._status_model.set_value(text)

    def _request_rebuild(self):
        if getattr(self, "_window", None) is not None:
            self._window.frame.rebuild()

    def _build_pane_title(self, ui, title: str):
        with ui.HStack(height=20):
            ui.Label(title)

    def _build_row(self, ui, label: str, model, width: int = 90):
        with ui.HStack(spacing=6, height=24):
            ui.Label(label, width=width)
            ui.StringField(model=model, height=22)

    def _build_float_range_row(self, ui, label: str, min_model, max_model, width: int = 90):
        with ui.HStack(spacing=6, height=24):
            ui.Label(label, width=width)
            ui.Label("min", width=28)
            ui.FloatField(model=min_model, height=22)
            ui.Label("max", width=28)
            ui.FloatField(model=max_model, height=22)

    def _build_readonly_row(self, ui, label: str, value: str, width: int = 90):
        with ui.HStack(spacing=6, height=24):
            ui.Label(label, width=width)
            m = self._ui.SimpleStringModel(value)
            f = ui.StringField(model=m, height=22)
            f.enabled = False

    def _build_compact_block(self, ui, text: str, height: int = 60):
        with ui.ZStack(height=height):
            ui.Rectangle()
            with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                with ui.VStack(spacing=0):
                    ui.Spacer(height=4)
                    ui.Label(text or "", word_wrap=True)
                    ui.Spacer(height=4)

    def _build_selectable_button(self, ui, label: str, selected: bool, clicked_fn, height: int = 24):
        btn_label = f"> {label}" if selected else f"  {label}"
        ui.Button(btn_label, height=height, clicked_fn=clicked_fn)

    # ------------------------------------------------------------------
    # Top-level UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        ui = self._ui
        with self._window.frame:
            with ui.VStack(spacing=6):
                self._build_header(ui)
                with ui.HStack(spacing=8):
                    self._build_left_pane(ui)
                    self._build_middle_pane(ui)
                    self._build_right_pane(ui)

    def _build_header(self, ui):
        with ui.ZStack(height=50):
            ui.Rectangle()
            with ui.VStack(spacing=2):
                ui.Spacer(height=4)
                with ui.HStack(spacing=10, height=24):
                    ui.Label("BeTTER Task Editor", width=180)
                    ui.Label("Tasks root:", width=72)
                    m = self._ui.SimpleStringModel(str(TASKS_ROOT))
                    f = ui.StringField(model=m, height=22)
                    f.enabled = False
                    ui.Button("Reload", width=72, clicked_fn=self._reload_tasks)
                with ui.HStack(spacing=14, height=18):
                    ui.Label(f"tasks: {len(self._tasks)}")
                    ui.Label(f"templates: {len(self._templates)}")
                    ui.Label(self._status_model.as_string, word_wrap=False)

    # ------------------------------------------------------------------
    # Left pane: task list
    # ------------------------------------------------------------------

    def _build_left_pane(self, ui):
        with ui.VStack(spacing=6, width=LEFT_PANE_WIDTH):
            self._build_pane_title(ui, "Tasks")
            ui.Button("+ Generate New Task", height=26, clicked_fn=self._open_generate_form)
            with ui.ZStack(height=820):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=3):
                        ui.Spacer(height=3)
                        for task_key in self._tasks:
                            is_dirty = task_key in self._dirty
                            label = f"{'* ' if is_dirty else ''}{task_key}"
                            self._build_selectable_button(
                                ui,
                                label=label,
                                selected=task_key == self._selected_task_key,
                                clicked_fn=lambda k=task_key: self._select_task(k),
                            )
                        ui.Spacer(height=3)

    # ------------------------------------------------------------------
    # Middle pane: object list
    # ------------------------------------------------------------------

    def _build_middle_pane(self, ui):
        with ui.VStack(spacing=6, width=MIDDLE_PANE_WIDTH):
            task = self._get_selected_task()
            entry = self._get_dirty_task() if task else None
            self._build_pane_title(ui, "Objects")
            if task is None:
                ui.Label("No task selected.")
                return
            with ui.HStack(spacing=6, height=24):
                ui.Button("Task Info", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_task_info)
                ui.Button("Remove Task", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_remove_task_dialog)
                ui.Button("Save to disk", width=ACTION_BUTTON_WIDTH, clicked_fn=self._write_task_to_disk)

            for group_key in GROUP_KEYS:
                group_label = GROUP_LABELS.get(group_key, group_key)
                objects = (entry["universe"].get(group_key) or []) if entry else []
                self._build_pane_title(ui, group_label)
                with ui.HStack(spacing=4, height=22):
                    ui.Label(f"{len(objects)} objects", width=80)
                    ui.Button(
                        "+ Add",
                        width=64,
                        clicked_fn=lambda gk=group_key: self._open_add_object_form(gk),
                    )
                if objects:
                    with ui.ZStack(height=min(24 * len(objects) + 10, 120)):
                        ui.Rectangle()
                        with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                            with ui.VStack(spacing=2):
                                ui.Spacer(height=3)
                                for obj in objects:
                                    iid = obj.get("instance_id") or ""
                                    sname = obj.get("semantic_name") or iid
                                    query = obj.get("retrieval_query") or ""
                                    lbl = f"{sname}  ({query})"
                                    self._build_selectable_button(
                                        ui,
                                        label=lbl,
                                        selected=iid == self._selected_object_id,
                                        clicked_fn=lambda i=iid: self._select_object(i),
                                    )
                                ui.Spacer(height=3)
                else:
                    ui.Label("  (empty)", height=20)

    # ------------------------------------------------------------------
    # Right pane: editor
    # ------------------------------------------------------------------

    def _build_right_pane(self, ui):
        with ui.VStack(spacing=6, width=RIGHT_PANE_WIDTH):
            if self._add_obj_mode:
                self._build_add_object_form(ui)
                return
            if self._right_pane_mode == "generate":
                self._build_generate_form(ui)
                return
            if self._right_pane_mode == "task_info":
                self._build_task_info_editor(ui)
                return
            self._build_object_editor(ui)

    def _build_generate_form(self, ui):
        self._build_pane_title(ui, "Generate New Task")
        # Template selector
        with ui.HStack(spacing=6, height=24):
            ui.Label("Template", width=90)
            names = self._template_names()
            if names:
                idx = min(self._selected_template_index, len(names) - 1)
                self._selected_template_combo = ui.ComboBox(idx, *names, height=22)
                template_model = self._selected_template_combo.model.get_item_value_model()
                self._template_combo_subscription = template_model.subscribe_value_changed_fn(self._on_template_index_changed)
            else:
                self._selected_template_combo = None
                self._template_combo_subscription = None
                m = self._ui.SimpleStringModel("No templates found")
                ui.StringField(model=m, height=22)
        # Endpoint selector
        with ui.HStack(spacing=6, height=24):
            ui.Label("Endpoint", width=90)
            ep_names = self._endpoint_names()
            if ep_names:
                idx = min(self._selected_endpoint_index, len(ep_names) - 1)
                self._selected_endpoint_combo = ui.ComboBox(idx, *ep_names, height=22)
                endpoint_model = self._selected_endpoint_combo.model.get_item_value_model()
                self._endpoint_combo_subscription = endpoint_model.subscribe_value_changed_fn(self._on_endpoint_index_changed)
            else:
                self._selected_endpoint_combo = None
                self._endpoint_combo_subscription = None
        self._build_row(ui, "Model", self._gen_model_name_model)
        self._build_row(ui, "Task ID", self._gen_task_id_model)
        self._build_pane_title(ui, "Guidance")
        ui.StringField(model=self._gen_guidance_model, multiline=True, height=180)
        # Show selected template summary
        tmpl = self._selected_template()
        if tmpl:
            lines = [
                f"template_type: {tmpl.get('template_type', '-')}",
                f"goal_relation: {tmpl.get('primary_goal_relation', '-')}",
                f"slots: {', '.join(s.get('slot_name','?') for s in (tmpl.get('slots') or []))}",
                tmpl.get("description") or "",
            ]
            self._build_pane_title(ui, "Template preview")
            self._build_compact_block(ui, "\n".join(lines), height=80)
        with ui.HStack(spacing=8, height=26):
            ui.Button("Generate", width=ACTION_BUTTON_WIDTH, clicked_fn=self._generate_task)
            ui.Button("Cancel", width=ACTION_BUTTON_WIDTH, clicked_fn=lambda: self._set_mode("object"))
        self._build_compact_block(ui, self._status_model.as_string, height=80)

    def _build_task_info_editor(self, ui):
        entry = self._get_dirty_task()
        task_obj = self._get_selected_task()
        if entry is None or task_obj is None:
            ui.Label("No task selected.")
            return
        self._build_pane_title(ui, "Task Info")
        self._build_readonly_row(ui, "Task dir", str(task_obj["task_dir"]))
        self._build_row(ui, "Task ID", self._task_id_model)
        self._build_row(ui, "Instruction", self._task_instruction_model)
        self._build_row(ui, "Description", self._task_description_model)
        self._build_row(ui, "Goal relation", self._task_goal_relation_model)
        with ui.HStack(spacing=8, height=26):
            ui.Button("Apply", width=ACTION_BUTTON_WIDTH, clicked_fn=self._save_task_info_edits)
            ui.Button("Remove Task", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_remove_task_dialog)
            ui.Button("Save to disk", width=ACTION_BUTTON_WIDTH, clicked_fn=self._write_task_to_disk)
        self._build_compact_block(ui, self._status_model.as_string, height=60)

    def _build_object_editor(self, ui):
        obj = self._get_selected_object()
        if obj is None:
            self._build_pane_title(ui, "Object Editor")
            ui.Label("Select an object from the middle pane.")
            return
        group_key = obj.get("_group_key", "")
        self._build_pane_title(ui, f"Object Editor  —  {GROUP_LABELS.get(group_key, group_key)}")
        self._build_readonly_row(ui, "Instance ID", obj.get("instance_id") or "")
        self._build_row(ui, "Semantic name", self._obj_semantic_name_model)
        self._build_row(ui, "Description", self._obj_description_model)
        self._build_row(ui, "Retrieval query", self._obj_retrieval_query_model)
        self._build_float_range_row(ui, "Mass range (kg)", self._obj_mass_min_model, self._obj_mass_max_model)
        self._build_float_range_row(ui, "Size range (m)", self._obj_size_min_model, self._obj_size_max_model)
        self._build_row(ui, "Tags", self._obj_tags_model)
        ui.Label("Tags: comma-separated (e.g. content, fragile)", height=18)
        with ui.HStack(spacing=8, height=26):
            ui.Button("Apply", width=ACTION_BUTTON_WIDTH, clicked_fn=self._save_object_edits)
            ui.Button("Remove object", width=ACTION_BUTTON_WIDTH, clicked_fn=self._remove_selected_object)
        self._build_compact_block(ui, self._status_model.as_string, height=60)

    def _build_add_object_form(self, ui):
        group_label = GROUP_LABELS.get(self._add_obj_group_key, self._add_obj_group_key)
        self._build_pane_title(ui, f"Add Object  —  {group_label}")
        self._build_row(ui, "Semantic name", self._add_obj_semantic_model)
        self._build_row(ui, "Description", self._add_obj_description_model)
        self._build_row(ui, "Retrieval query", self._add_obj_query_model)
        self._build_float_range_row(ui, "Mass range (kg)", self._add_obj_mass_min_model, self._add_obj_mass_max_model)
        self._build_float_range_row(ui, "Size range (m)", self._add_obj_size_min_model, self._add_obj_size_max_model)
        self._build_row(ui, "Tags", self._add_obj_tags_model)
        with ui.HStack(spacing=8, height=26):
            ui.Button("Confirm", width=ACTION_BUTTON_WIDTH, clicked_fn=self._confirm_add_object)
            ui.Button("Cancel", width=ACTION_BUTTON_WIDTH, clicked_fn=self._cancel_add_object)
        self._build_compact_block(ui, self._status_model.as_string, height=60)

    def _set_mode(self, mode: str):
        self._right_pane_mode = mode
        self._add_obj_mode = False
        self._request_rebuild()
