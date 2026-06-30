from __future__ import annotations

import asyncio
import sys
import traceback
from pathlib import Path

import omni.ext
import yaml
from pxr import Sdf, UsdGeom


ALL_CANDIDATES_ROOT = "/World/AllCandidates"


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXTENSION_ROOT.parents[2]
TASKS_ROOT = REPO_ROOT / "assets" / "tasks"
ASSET_REGISTRY_ROOT = REPO_ROOT / "assets" / "objects" / "registry"
WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 1080
LEFT_PANE_WIDTH = 320
MIDDLE_PANE_WIDTH = 500
RIGHT_PANE_WIDTH = 620
ACTION_BUTTON_WIDTH = 156
VECTOR_LABELS = ("x", "y", "z")
QUAT_LABELS = ("w", "x", "y", "z")
COMMON_RELATION_CHOICES = (
    "in",
    "on",
    "intersect",
    "contain",
    "contains",
    "inside",
    "over",
    "under",
    "left_of",
    "right_of",
    "behind",
    "in_front_of",
    "next_to",
    "touching",
)


class VariationEditorExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        import omni.ui as ui

        self._ensure_repo_on_sys_path()
        self._ui = ui
        self._ext_id = ext_id
        self._selected_task_key = None
        self._selected_variation_id = None
        self._selected_instance_id = None
        self._layout_view_mode = "resolved"
        self._show_active_only = False
        self._show_inherited_slots = True
        self._status_model = ui.SimpleStringModel("Variation Editor loaded.")
        self._task_path_model = ui.SimpleStringModel(str(TASKS_ROOT))
        self._layout_mode_model = ui.SimpleStringModel("Layout (resolved)")
        self._dirty_model = ui.SimpleStringModel("")
        self._variation_enabled_model = ui.SimpleBoolModel(True)
        self._variation_instruction_model = ui.SimpleStringModel("")
        self._goal_override_relation_model = ui.SimpleStringModel("")
        self._goal_override_target_model = ui.SimpleStringModel("")
        self._goal_relation_model = ui.SimpleStringModel("")
        self._container_instance_id_model = ui.SimpleStringModel("")
        self._group_target_model = ui.SimpleStringModel("")
        self._group_distractor_model = ui.SimpleStringModel("")
        self._group_decor_model = ui.SimpleStringModel("")
        self._preview_model = ui.SimpleStringModel("No resolved preview yet.")
        self._goal_override_editor_mode = False
        self._condition_override_editor_kind = "goal"
        self._goal_override_subject_model = ui.SimpleStringModel("")
        self._goal_override_relation_model = ui.SimpleStringModel("")
        self._goal_override_target_model = ui.SimpleStringModel("")
        self._create_variation_mode = False
        self._new_variation_id_model = ui.SimpleStringModel("")
        self._new_variation_extends_index = 0
        self._new_variation_extends_combo = None
        self._goal_override_subject_index = 0
        self._goal_override_subject_combo = None
        self._goal_override_relation_index = 0
        self._goal_override_relation_combo = None
        self._goal_override_target_index = 0
        self._goal_override_target_combo = None
        self._remove_variation_dialog = None
        self._loaded_episode = None
        self._slot_position_models = [ui.SimpleFloatModel(0.0) for _ in range(3)]
        self._slot_rotation_models = [ui.SimpleFloatModel(0.0) for _ in range(4)]
        self._slot_scale_models = [ui.SimpleFloatModel(1.0) for _ in range(3)]
        self._tasks = self._discover_tasks()
        self._dirty_variations: dict[tuple[str | None, str | None], dict] = {}
        if self._tasks:
            self._selected_task_key = next(iter(self._tasks.keys()))
            task = self._tasks[self._selected_task_key]
            self._selected_variation_id = task["variation_ids"][0] if task["variation_ids"] else None
            self._sync_selected_instance()
            self._load_editor_models()
            self._update_preview_summary()
        self._window = ui.Window("BeTTER Variation Editor", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        self._window.frame.set_build_fn(self._build_ui)
        self._request_rebuild()

    def on_shutdown(self):
        self._window = None

    def _ensure_repo_on_sys_path(self):
        repo_path = str(REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.append(repo_path)

    def _load_task_spec_api(self):
        from src.task.specs import load_task_spec, resolve_variation

        return load_task_spec, resolve_variation

    def _schedule_async(self, coro):
        asyncio.ensure_future(coro)

    def _dirty_key(self) -> tuple[str | None, str | None]:
        return self._selected_task_key, self._selected_variation_id

    def _get_dirty_payload(self) -> dict | None:
        return self._dirty_variations.get(self._dirty_key())

    def _ensure_dirty_payload(self) -> dict:
        key = self._dirty_key()
        payload = self._dirty_variations.get(key)
        if payload is None:
            variation = self._get_selected_variation()
            slot_item = self._get_selected_slot_item()
            slot = None if slot_item is None else (slot_item["authored_slot"] or slot_item["resolved_slot"])
            payload = {
                "enabled": True if variation is None else bool(variation.enabled),
                "instruction": "" if variation is None else str(variation.instruction or ""),
                "goal_relation": "" if variation is None else str(variation.goal_relation or ""),
                "container_instance_id": "" if variation is None else str(variation.container_instance_id or ""),
                "group_target": self._resolved_group_members_to_text("target"),
                "group_distractor": self._resolved_group_members_to_text("distractor"),
                "group_decor": self._resolved_group_members_to_text("decor"),
                "goal_override_items": [] if variation is None else self._goal_override_items_from_predicates(variation.policy_overrides.success.append),
                "fail_override_items": [] if variation is None else self._goal_override_items_from_predicates(variation.policy_overrides.fail.append),
                "slot_models_instance_id": None if slot_item is None else slot_item["instance_id"],
                "slot_position": [0.0, 0.0, 0.0] if slot is None else list(slot.position),
                "slot_rotation": [1.0, 0.0, 0.0, 0.0] if slot is None else list(slot.rotation),
                "slot_scale": [1.0, 1.0, 1.0] if slot is None else list(slot.scale),
            }
            self._dirty_variations[key] = payload
        return payload

    def _discover_tasks(self):
        load_task_spec, resolve_variation = self._load_task_spec_api()
        tasks = {}
        for task_yaml in sorted(TASKS_ROOT.glob("*/*/task.yaml")):
            task_dir = task_yaml.parent
            try:
                task_spec = load_task_spec(task_dir)
            except Exception:
                continue
            task_key = f"{task_spec.template_type}/{task_spec.task_id}"
            variation_ids = [task_spec.base_variation.variation_id, *sorted(task_spec.variations.keys())]
            resolved_variations = {}
            for variation_id in variation_ids:
                try:
                    resolved_variations[variation_id] = resolve_variation(task_spec, variation_id)
                except Exception:
                    resolved_variations[variation_id] = None
            tasks[task_key] = {
                "task_key": task_key,
                "task_dir": task_dir,
                "task_spec": task_spec,
                "task_id": task_spec.task_id,
                "template_type": task_spec.template_type,
                "instruction": task_spec.base_variation.instruction,
                "variation_ids": variation_ids,
                "resolved_variations": resolved_variations,
            }
        return tasks

    def _get_selected_task(self):
        if self._selected_task_key is None:
            return None
        return self._tasks.get(self._selected_task_key)

    def _reload_current_task_cache(self):
        task = self._get_selected_task()
        if task is None:
            return None
        load_task_spec, resolve_variation = self._load_task_spec_api()
        try:
            task_spec = load_task_spec(task["task_dir"])
        except Exception:
            return task.get("task_spec")
        variation_ids = [task_spec.base_variation.variation_id, *sorted(task_spec.variations.keys())]
        resolved_variations = {}
        for variation_id in variation_ids:
            try:
                resolved_variations[variation_id] = resolve_variation(task_spec, variation_id)
            except Exception:
                resolved_variations[variation_id] = None
        task["task_spec"] = task_spec
        task["instruction"] = task_spec.base_variation.instruction
        task["variation_ids"] = variation_ids
        task["resolved_variations"] = resolved_variations
        if self._selected_variation_id not in variation_ids:
            self._selected_variation_id = variation_ids[0] if variation_ids else None
        return task_spec

    def _get_selected_task_spec(self):
        task = self._get_selected_task()
        return None if task is None else task["task_spec"]

    def _get_selected_variation(self):
        task_spec = self._get_selected_task_spec()
        if task_spec is None or self._selected_variation_id is None:
            return None
        if self._selected_variation_id == task_spec.base_variation.variation_id:
            return task_spec.base_variation
        return task_spec.variations.get(self._selected_variation_id)

    def _get_selected_resolved_variation(self):
        task = self._get_selected_task()
        if task is None or self._selected_variation_id is None:
            return None
        return task["resolved_variations"].get(self._selected_variation_id)

    def _get_slot_items(self):
        task_spec = self._get_selected_task_spec()
        variation = self._get_selected_variation()
        resolved = self._get_selected_resolved_variation()
        if task_spec is None or variation is None:
            return []
        authored_slots = variation.layout_slots
        resolved_slots = {} if resolved is None else resolved.layout_slots
        active_ids = self._staged_active_instance_ids()
        items = []
        for instance_id, object_spec in sorted(task_spec.object_instances.items()):
            authored = authored_slots.get(instance_id)
            resolved_slot = resolved_slots.get(instance_id)
            is_active = instance_id in active_ids
            is_overridden = authored is not None
            is_staged_override = self._is_staged_slot_override(instance_id, authored, resolved_slot)
            group_status = self._staged_group_status_text(instance_id, object_spec.role)
            if self._show_active_only and not is_active:
                continue
            if not self._show_inherited_slots and not is_overridden and self._layout_view_mode == "override":
                continue
            slot = authored if self._layout_view_mode == "override" else resolved_slot
            if slot is None and self._layout_view_mode == "override":
                continue
            items.append(
                {
                    "instance_id": instance_id,
                    "semantic_name": object_spec.semantic_name,
                    "role": object_spec.role or "-",
                    "is_active": is_active,
                    "is_overridden": is_overridden,
                    "is_staged_override": is_staged_override,
                    "group_status": group_status,
                    "slot_source": self._slot_source_text(is_overridden, is_staged_override),
                    "slot": slot,
                    "resolved_slot": resolved_slot,
                    "authored_slot": authored,
                }
            )
        return items

    def _get_selected_slot_item(self):
        items = self._get_slot_items()
        if not items:
            return None
        for item in items:
            if item["instance_id"] == self._selected_instance_id:
                return item
        self._selected_instance_id = items[0]["instance_id"]
        return items[0]

    def _sync_selected_instance(self):
        item = self._get_selected_slot_item()
        if item is not None:
            self._selected_instance_id = item["instance_id"]

    def _load_editor_models(self):
        variation = self._get_selected_variation()
        slot_item = self._get_selected_slot_item()
        payload = self._ensure_dirty_payload()
        if variation is not None:
            self._variation_enabled_model.set_value(bool(payload["enabled"]))
            self._variation_instruction_model.set_value(str(payload["instruction"]))
            self._goal_override_subject_model.set_value(self._selected_instance_id or "")
            self._goal_override_relation_model.set_value("")
            self._goal_override_target_model.set_value("")
            self._goal_relation_model.set_value(str(payload["goal_relation"]))
            self._container_instance_id_model.set_value(str(payload["container_instance_id"]))
            self._group_target_model.set_value(str(payload["group_target"]))
            self._group_distractor_model.set_value(str(payload["group_distractor"]))
            self._group_decor_model.set_value(str(payload["group_decor"]))
        slot = None
        if slot_item is not None:
            slot = slot_item["authored_slot"] or slot_item["resolved_slot"]
        slot_instance_id = None if slot_item is None else slot_item["instance_id"]
        if payload.get("slot_models_instance_id") != slot_instance_id:
            payload["slot_models_instance_id"] = slot_instance_id
            payload["slot_position"] = [0.0, 0.0, 0.0] if slot is None else list(slot.position)
            payload["slot_rotation"] = [1.0, 0.0, 0.0, 0.0] if slot is None else list(slot.rotation)
            payload["slot_scale"] = [1.0, 1.0, 1.0] if slot is None else list(slot.scale)
        for model, value in zip(self._slot_position_models, payload["slot_position"]):
            model.set_value(float(value))
        for model, value in zip(self._slot_rotation_models, payload["slot_rotation"]):
            model.set_value(float(value))
        for model, value in zip(self._slot_scale_models, payload["slot_scale"]):
            model.set_value(float(value))
        self._refresh_dirty_label()

    def _refresh_dirty_label(self):
        dirty_keys = [key for key in self._dirty_variations.keys() if key[0] is not None and key[1] is not None]
        self._dirty_model.set_value(f"dirty {len(dirty_keys)}" if dirty_keys else "")

    def _request_rebuild(self):
        if getattr(self, "_window", None) is not None:
            self._layout_mode_model.set_value(
                "Layout (resolved)" if self._layout_view_mode == "resolved" else "Layout overrides"
            )
            self._window.frame.rebuild()

    def _set_status(self, text: str):
        self._status_model.set_value(text)
        print(f"[VariationEditor] {text}", flush=True)

    def _reload_tasks(self):
        self._tasks = self._discover_tasks()
        if self._selected_task_key not in self._tasks:
            self._selected_task_key = next(iter(self._tasks.keys()), None)
        task = self._get_selected_task()
        if task is not None and self._selected_variation_id not in task["variation_ids"]:
            self._selected_variation_id = task["variation_ids"][0] if task["variation_ids"] else None
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._update_preview_summary()
        self._set_status("Reloaded task registry.")
        self._request_rebuild()

    def _select_task(self, task_key: str):
        self._selected_task_key = task_key
        task = self._get_selected_task()
        self._selected_variation_id = task["variation_ids"][0] if task and task["variation_ids"] else None
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._update_preview_summary()
        self._set_status(f"Selected task {task_key}.")
        self._request_rebuild()

    def _select_variation(self, variation_id: str):
        self._selected_variation_id = variation_id
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._update_preview_summary()
        self._set_status(f"Selected variation {variation_id}.")
        self._request_rebuild()

    def _select_instance(self, instance_id: str):
        self._selected_instance_id = instance_id
        self._load_editor_models()
        self._set_status(f"Selected layout slot {instance_id}.")
        self._request_rebuild()

    def _set_layout_view_mode(self, mode: str):
        self._layout_view_mode = mode
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._set_status(f"Switched layout view to {mode}.")
        self._request_rebuild()

    def _toggle_show_active_only(self):
        self._show_active_only = not self._show_active_only
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._request_rebuild()

    def _toggle_show_inherited_slots(self):
        self._show_inherited_slots = not self._show_inherited_slots
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._request_rebuild()

    def _capture_variation_fields(self):
        payload = self._ensure_dirty_payload()
        payload["enabled"] = bool(self._variation_enabled_model.as_bool)
        payload["instruction"] = str(self._variation_instruction_model.as_string)
        self._refresh_dirty_label()

    def _capture_slot_fields(self):
        payload = self._ensure_dirty_payload()
        payload["slot_models_instance_id"] = self._selected_instance_id
        payload["slot_position"] = [float(model.as_float) for model in self._slot_position_models]
        payload["slot_rotation"] = [float(model.as_float) for model in self._slot_rotation_models]
        payload["slot_scale"] = [float(model.as_float) for model in self._slot_scale_models]
        self._refresh_dirty_label()

    def _revert_slot_override(self):
        slot_item = self._get_selected_slot_item()
        if slot_item is None:
            self._set_status("No selected slot.")
            return
        payload = self._ensure_dirty_payload()
        resolved_slot = slot_item["resolved_slot"]
        payload["slot_models_instance_id"] = self._selected_instance_id
        payload["slot_position"] = [0.0, 0.0, 0.0] if resolved_slot is None else list(resolved_slot.position)
        payload["slot_rotation"] = [1.0, 0.0, 0.0, 0.0] if resolved_slot is None else list(resolved_slot.rotation)
        payload["slot_scale"] = [1.0, 1.0, 1.0] if resolved_slot is None else list(resolved_slot.scale)
        self._load_editor_models()
        self._set_status(f"Reverted staged override values for {self._selected_instance_id}.")
        self._request_rebuild()

    def _payload_path_for_selected_variation(self) -> Path | None:
        task = self._get_selected_task()
        if task is None or self._selected_variation_id is None:
            return None
        if self._selected_variation_id == "BASE":
            return task["task_dir"] / "base_variation.yaml"
        return task["task_dir"] / "variations" / f"{self._selected_variation_id}.yaml"

    def _load_variation_payload(self) -> dict:
        path = self._payload_path_for_selected_variation()
        if path is None or not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _slot_payload_from_models(self) -> dict:
        self._capture_slot_fields()
        payload = self._ensure_dirty_payload()
        return {
            "position": [float(v) for v in payload["slot_position"]],
            "rotation": [float(v) for v in payload["slot_rotation"]],
            "scale": [float(v) for v in payload["slot_scale"]],
        }

    def _group_members_to_text(self, variation, group_name: str) -> str:
        if variation is None:
            return ""
        override = variation.group_overrides.get(group_name)
        if override is None:
            return ""
        if override.set_members is not None:
            return ", ".join(override.set_members)
        members = list(override.add_members)
        if override.remove_members:
            members.extend(f"-{item}" for item in override.remove_members)
        return ", ".join(members)

    def _resolved_group_members_to_text(self, group_name: str) -> str:
        resolved = self._get_selected_resolved_variation()
        if resolved is not None:
            return ", ".join(resolved.semantic_groups.get(group_name, ()))
        return self._group_members_to_text(self._get_selected_variation(), group_name)

    def _staged_group_members(self, group_name: str) -> tuple[str, ...]:
        payload = self._get_dirty_payload()
        key = f"group_{group_name}"
        if payload is not None and key in payload:
            return tuple(self._parse_group_members_text(str(payload[key])))
        resolved = self._get_selected_resolved_variation()
        if resolved is not None:
            return tuple(resolved.semantic_groups.get(group_name, ()))
        return ()

    def _staged_active_instance_ids(self) -> set[str]:
        resolved = self._get_selected_resolved_variation()
        active_ids = set(() if resolved is None else resolved.active_instance_ids)
        payload = self._get_dirty_payload()
        if payload is None:
            return active_ids

        grouped_ids = set()
        for group_name in ("target", "distractor", "decor"):
            grouped_ids.update(self._staged_group_members(group_name))

        slot_ids = set(() if resolved is None else resolved.layout_slots.keys())
        slot_instance_id = payload.get("slot_models_instance_id")
        if slot_instance_id:
            slot_ids.add(str(slot_instance_id))

        container_instance_id = str(payload.get("container_instance_id") or "").strip()
        if not container_instance_id and resolved is not None:
            container_instance_id = resolved.container_instance_id

        active_ids = grouped_ids | slot_ids
        if container_instance_id:
            active_ids.add(container_instance_id)
        return active_ids

    def _staged_group_status_text(self, instance_id: str, role: str | None) -> str:
        if role == "container":
            return "container"
        assignable_group = self._semantic_group_for_instance(instance_id)
        if assignable_group not in {"target", "distractor", "decor"}:
            return "no-group"
        state = "in" if instance_id in self._staged_group_members(assignable_group) else "out"
        return f"{assignable_group}:{state}"

    def _slot_source_text(self, is_overridden: bool, is_staged_override: bool) -> str:
        if is_staged_override:
            return "staged override"
        return "override" if is_overridden else "inherited"

    def _is_staged_slot_override(self, instance_id: str, authored_slot, resolved_slot) -> bool:
        payload = self._get_dirty_payload()
        if payload is None or payload.get("slot_models_instance_id") != instance_id:
            return False
        reference_slot = authored_slot or resolved_slot
        if reference_slot is None:
            return True
        expected = {
            "slot_position": tuple(reference_slot.position),
            "slot_rotation": tuple(reference_slot.rotation),
            "slot_scale": tuple(reference_slot.scale),
        }
        for key, reference_values in expected.items():
            staged_values = tuple(float(value) for value in payload.get(key, ()))
            if len(staged_values) != len(reference_values):
                return True
            for staged, reference in zip(staged_values, reference_values):
                if abs(float(staged) - float(reference)) > 1e-9:
                    return True
        return False

    def _goal_override_items_from_predicates(self, predicates) -> list[dict[str, str]]:
        return [
            {
                "subject_id": predicate.subject_id,
                "relation": predicate.relation,
                "target_id": predicate.target_id,
            }
            for predicate in sorted(predicates, key=lambda item: (item.subject_id, item.relation, item.target_id))
        ]

    def _condition_override_items_key(self, kind: str) -> str:
        return "goal_override_items" if kind == "goal" else "fail_override_items"

    def _condition_label(self, kind: str) -> str:
        return "goal" if kind == "goal" else "fail"

    def _resolved_condition_items(self, kind: str) -> list[dict[str, str]]:
        resolved = self._get_selected_resolved_variation()
        if resolved is None:
            return []
        predicates = resolved.success_predicates if kind == "goal" else resolved.fail_predicates
        return [
            {
                "subject_id": predicate.subject_id,
                "relation": predicate.relation,
                "target_id": predicate.target_id,
            }
            for predicate in predicates
        ]

    def _staged_condition_override_items(self, kind: str) -> list[dict[str, str]]:
        payload = self._ensure_dirty_payload()
        items = payload.get(self._condition_override_items_key(kind)) or []
        return [
            {
                "subject_id": str(item.get("subject_id") or "").strip(),
                "relation": str(item.get("relation") or "").strip(),
                "target_id": str(item.get("target_id") or "").strip(),
            }
            for item in items
            if str(item.get("subject_id") or "").strip()
            and str(item.get("relation") or "").strip()
            and str(item.get("target_id") or "").strip()
        ]

    def _goal_override_summary_text(self) -> str:
        items = self._staged_condition_override_items("goal")
        if not items:
            return "-"
        return " | ".join(f"{item['subject_id']} {item['relation']} {item['target_id']}" for item in items)

    def _fail_override_summary_text(self) -> str:
        items = self._staged_condition_override_items("fail")
        if not items:
            return "-"
        return " | ".join(f"{item['subject_id']} {item['relation']} {item['target_id']}" for item in items)

    def _goal_override_relation_choices(self) -> list[str]:
        task_spec = self._get_selected_task_spec()
        relations: set[str] = set(COMMON_RELATION_CHOICES)
        if task_spec is not None:
            relations.update(task_spec.allowed_relations)
        resolved = self._get_selected_resolved_variation()
        if resolved is not None:
            relations.update(predicate.relation for predicate in resolved.success_predicates)
            relations.update(predicate.relation for predicate in resolved.fail_predicates)
        relations.update(item["relation"] for item in self._staged_condition_override_items("goal") if item.get("relation"))
        relations.update(item["relation"] for item in self._staged_condition_override_items("fail") if item.get("relation"))
        current_relation = str(self._goal_override_relation_model.as_string).strip()
        if current_relation:
            relations.add(current_relation)
        return sorted(relations)

    def _goal_override_target_choices(self) -> list[str]:
        resolved = self._get_selected_resolved_variation()
        targets: set[str] = set()
        if resolved is not None:
            targets.update(predicate.target_id for predicate in resolved.success_predicates)
            targets.update(predicate.target_id for predicate in resolved.fail_predicates)
            if resolved.container_instance_id:
                targets.add(resolved.container_instance_id)
        targets.update(item["target_id"] for item in self._staged_condition_override_items("goal") if item.get("target_id"))
        targets.update(item["target_id"] for item in self._staged_condition_override_items("fail") if item.get("target_id"))
        current_target = str(self._goal_override_target_model.as_string).strip()
        if current_target:
            targets.add(current_target)
        return sorted(targets)

    def _open_condition_override_editor(self, kind: str):
        payload = self._ensure_dirty_payload()
        items_key = self._condition_override_items_key(kind)
        payload.setdefault(items_key, self._staged_condition_override_items(kind))
        subject_choices = self._goal_override_subject_choices()
        relation_choices = self._goal_override_relation_choices()
        target_choices = self._goal_override_target_choices()
        selected_subject = self._selected_instance_id or (subject_choices[0] if subject_choices else "")
        try:
            self._goal_override_subject_index = subject_choices.index(selected_subject)
        except ValueError:
            self._goal_override_subject_index = 0
        self._goal_override_subject_model.set_value(selected_subject)
        self._goal_override_subject_combo = None
        self._goal_override_relation_index = 0
        self._goal_override_relation_model.set_value(relation_choices[0] if relation_choices else "")
        self._goal_override_relation_combo = None
        self._goal_override_target_index = 0
        self._goal_override_target_model.set_value(target_choices[0] if target_choices else "")
        self._goal_override_target_combo = None
        self._condition_override_editor_kind = kind
        self._goal_override_editor_mode = True
        self._set_status(f"Edit variation {self._condition_label(kind)}-condition overrides.")
        self._request_rebuild()

    def _open_goal_override_editor(self):
        self._open_condition_override_editor("goal")

    def _open_fail_override_editor(self):
        self._open_condition_override_editor("fail")

    def _close_goal_override_editor(self):
        self._goal_override_editor_mode = False
        self._request_rebuild()

    def _add_goal_override_item(self):
        kind = self._condition_override_editor_kind
        subject_choices = self._goal_override_subject_choices()
        relation_choices = self._goal_override_relation_choices()
        target_choices = self._goal_override_target_choices()
        subject_id = str(self._goal_override_subject_model.as_string).strip()
        relation = str(self._goal_override_relation_model.as_string).strip()
        target_id = str(self._goal_override_target_model.as_string).strip()
        if self._goal_override_subject_combo is not None and subject_choices:
            subject_index = int(self._goal_override_subject_combo.model.get_item_value_model().as_int)
            if 0 <= subject_index < len(subject_choices):
                self._goal_override_subject_index = subject_index
                subject_id = subject_choices[subject_index]
        if self._goal_override_relation_combo is not None and relation_choices:
            relation_index = int(self._goal_override_relation_combo.model.get_item_value_model().as_int)
            if 0 <= relation_index < len(relation_choices):
                self._goal_override_relation_index = relation_index
                relation = relation_choices[relation_index]
        if self._goal_override_target_combo is not None and target_choices:
            target_index = int(self._goal_override_target_combo.model.get_item_value_model().as_int)
            if 0 <= target_index < len(target_choices):
                self._goal_override_target_index = target_index
                target_id = target_choices[target_index]
        self._goal_override_subject_model.set_value(subject_id)
        self._goal_override_relation_model.set_value(relation)
        self._goal_override_target_model.set_value(target_id)
        if not subject_id or not relation or not target_id:
            self._set_status(f"{self._condition_label(kind).capitalize()} override subject, relation, and target are required.")
            return
        payload = self._ensure_dirty_payload()
        items = self._staged_condition_override_items(kind)
        entry = {
            "subject_id": subject_id,
            "relation": relation,
            "target_id": target_id,
        }
        if entry in items:
            self._set_status(f"{self._condition_label(kind).capitalize()} override already exists.")
            return
        items.append(entry)
        payload[self._condition_override_items_key(kind)] = items
        self._refresh_dirty_label()
        self._set_status(f"Added {self._condition_label(kind)} override for {subject_id}.")
        self._request_rebuild()

    def _remove_goal_override_item(self, index: int):
        kind = self._condition_override_editor_kind
        payload = self._ensure_dirty_payload()
        items = self._staged_condition_override_items(kind)
        if index < 0 or index >= len(items):
            self._set_status(f"{self._condition_label(kind).capitalize()} override index is invalid.")
            return
        removed = items.pop(index)
        payload[self._condition_override_items_key(kind)] = items
        self._refresh_dirty_label()
        self._set_status(
            f"Removed {self._condition_label(kind)} override {removed['subject_id']} {removed['relation']} {removed['target_id']}."
        )
        self._request_rebuild()

    def _goal_override_subject_choices(self) -> list[str]:
        resolved = self._get_selected_resolved_variation()
        subjects: set[str] = set()
        if resolved is not None:
            subjects.update(resolved.active_instance_ids)
            subjects.update(predicate.subject_id for predicate in resolved.success_predicates)
            subjects.update(predicate.subject_id for predicate in resolved.fail_predicates)
        subjects.update(item["subject_id"] for item in self._staged_condition_override_items("goal") if item.get("subject_id"))
        subjects.update(item["subject_id"] for item in self._staged_condition_override_items("fail") if item.get("subject_id"))
        current_subject = str(self._goal_override_subject_model.as_string).strip()
        if current_subject:
            subjects.add(current_subject)
        return sorted(subjects)

    def _build_resolved_condition_summary(self, kind: str) -> str:
        resolved = self._get_selected_resolved_variation()
        items = self._resolved_condition_items(kind)
        if kind == "goal" and resolved is not None:
            header = [
                f"container: {resolved.container_instance_id or '-'}",
                f"goal_relation: {resolved.goal_relation or '-'}",
                f"active_targets: {', '.join(resolved.semantic_groups.get('target', ())) or '-'}",
                "",
            ]
        else:
            header = []
        if not items:
            return "\n".join(header + [f"No resolved {self._condition_label(kind)} conditions."])
        return "\n".join(
            header
            + [
                f"{index}. {item['subject_id']}  {item['relation']}  {item['target_id']}"
                for index, item in enumerate(items, start=1)
            ]
        )

    def _build_goal_override_editor(self, ui):
        kind = self._condition_override_editor_kind
        label = self._condition_label(kind).capitalize()
        resolved_items = self._resolved_condition_items(kind)
        items = self._staged_condition_override_items(kind)
        self._build_pane_title(ui, f"{label} Condition Overrides")
        self._build_compact_row(ui, "Variation", self._selected_variation_id or "")
        self._build_pane_title(ui, f"Resolved {label} Conditions")
        self._build_compact_row(ui, "Count", str(len(resolved_items)))
        with ui.ZStack(height=180):
            ui.Rectangle()
            with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                with ui.VStack(spacing=4):
                    ui.Spacer(height=4)
                    if not resolved_items:
                        ui.Label(f"No resolved {self._condition_label(kind)} conditions.")
                    else:
                        for index, item in enumerate(resolved_items, start=1):
                            ui.Label(f"{index}. {item['subject_id']}  {item['relation']}  {item['target_id']}")
                    ui.Spacer(height=4)
        self._build_pane_title(ui, "Override Append Rules")
        with ui.ZStack(height=180):
            ui.Rectangle()
            with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                with ui.VStack(spacing=4):
                    ui.Spacer(height=4)
                    if not items:
                        ui.Label(f"No {self._condition_label(kind)} overrides.")
                    else:
                        for index, item in enumerate(items):
                            with ui.HStack(spacing=8, height=24):
                                ui.Label(f"{item['subject_id']}  {item['relation']}  {item['target_id']}")
                                ui.Button(
                                    "Remove",
                                    width=ACTION_BUTTON_WIDTH,
                                    clicked_fn=lambda value=index: self._remove_goal_override_item(value),
                                )
                    ui.Spacer(height=4)
        subject_choices = self._goal_override_subject_choices()
        relation_choices = self._goal_override_relation_choices()
        target_choices = self._goal_override_target_choices()
        with ui.HStack(spacing=8, height=24):
            ui.Label("Subject", width=78)
            if subject_choices:
                selected_index = min(max(int(self._goal_override_subject_index), 0), len(subject_choices) - 1)
                self._goal_override_subject_combo = ui.ComboBox(selected_index, *subject_choices, height=22)
                self._goal_override_subject_model.set_value(subject_choices[selected_index])
            else:
                self._goal_override_subject_combo = None
                ui.StringField(model=self._goal_override_subject_model, height=22)
        with ui.HStack(spacing=8, height=24):
            ui.Label("Relation", width=78)
            if relation_choices:
                selected_index = min(max(int(self._goal_override_relation_index), 0), len(relation_choices) - 1)
                self._goal_override_relation_combo = ui.ComboBox(selected_index, *relation_choices, height=22)
                self._goal_override_relation_model.set_value(relation_choices[selected_index])
            else:
                self._goal_override_relation_combo = None
                ui.StringField(model=self._goal_override_relation_model, height=22)
        with ui.HStack(spacing=8, height=24):
            ui.Label("Target", width=78)
            if target_choices:
                selected_index = min(max(int(self._goal_override_target_index), 0), len(target_choices) - 1)
                self._goal_override_target_combo = ui.ComboBox(selected_index, *target_choices, height=22)
                self._goal_override_target_model.set_value(target_choices[selected_index])
            else:
                self._goal_override_target_combo = None
                ui.StringField(model=self._goal_override_target_model, height=22)
        with ui.HStack(spacing=8, height=24):
            ui.Button("Add", width=ACTION_BUTTON_WIDTH, clicked_fn=self._add_goal_override_item)
            ui.Button("Back", width=ACTION_BUTTON_WIDTH, clicked_fn=self._close_goal_override_editor)

    def _toggle_selected_instance_active(self):
        if self._selected_instance_id is None:
            self._set_status("No selected instance.")
            return
        task_spec = self._get_selected_task_spec()
        variation = self._get_selected_variation()
        if task_spec is None or variation is None:
            self._set_status("No selected variation.")
            return
        object_spec = task_spec.object_instances.get(self._selected_instance_id)
        if object_spec is None:
            self._set_status(f"Unknown instance '{self._selected_instance_id}'.")
            return
        group_name = "container" if object_spec.role == "container" else self._semantic_group_for_instance(self._selected_instance_id)
        if group_name not in {"target", "distractor", "decor"}:
            self._set_status(f"Instance '{self._selected_instance_id}' is not variation-toggleable.")
            return

        payload = self._ensure_dirty_payload()
        current_members = self._parse_group_members_text(str(payload[f"group_{group_name}"]))
        if self._selected_instance_id in current_members:
            current_members.remove(self._selected_instance_id)
            action = "Deactivated"
        else:
            current_members.append(self._selected_instance_id)
            action = "Activated"
        payload[f"group_{group_name}"] = ", ".join(sorted(current_members))
        self._refresh_dirty_label()
        self._set_status(f"{action} {self._selected_instance_id} in {group_name}.")
        self._request_rebuild()

    def _is_selected_instance_active_in_staged_payload(self) -> bool:
        if self._selected_instance_id is None:
            return False
        group_name = self._semantic_group_for_instance(self._selected_instance_id)
        if group_name not in {"target", "distractor", "decor"}:
            resolved = self._get_selected_resolved_variation()
            return bool(resolved and self._selected_instance_id in resolved.active_instance_ids)
        payload = self._ensure_dirty_payload()
        return self._selected_instance_id in self._parse_group_members_text(str(payload[f"group_{group_name}"]))

    def _semantic_group_for_instance(self, instance_id: str) -> str | None:
        task_spec = self._get_selected_task_spec()
        if task_spec is None:
            return None
        for group_name in ("target", "distractor", "decor"):
            if instance_id in task_spec.semantic_groups.get(group_name, ()):
                return group_name
        return None

    def _parse_group_members_text(self, value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip() and not item.strip().startswith("-")]

    async def _get_or_create_stage(self):
        import omni.usd

        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is not None:
            return stage

        result, error = await context.new_stage_async()
        if not result:
            raise RuntimeError(f"failed to create a new USD stage: {error}")
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("USD stage unavailable after new_stage_async")
        return stage

    def _remove_stage_prim_if_present(self, stage, prim_path: str) -> bool:
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            return False
        stage.RemovePrim(Sdf.Path(prim_path))
        return True

    def _clear_variation_preview_roots(self, stage, *prim_paths: str) -> int:
        removed = 0
        for prim_path in prim_paths:
            if self._remove_stage_prim_if_present(stage, prim_path):
                removed += 1
        return removed

    def _load_selected_variation_into_stage(self):
        self._schedule_async(self._load_selected_variation_into_stage_async())

    async def _load_selected_variation_into_stage_async(self):
        task = self._get_selected_task()
        if task is None or self._selected_variation_id is None:
            self._set_status("No variation selected for stage load.")
            return
        self._capture_variation_fields()
        self._write_selected_variation()
        try:
            import omni.usd
            from src.scene.episode_runtime import EpisodeRuntimeLoadConfig, load_resolved_episode_into_stage
            from src.task.specs import resolve_episode
        except Exception as exc:
            self._set_status(f"Runtime imports failed: {exc}")
            self._request_rebuild()
            return

        try:
            stage = await self._get_or_create_stage()
            episode = resolve_episode(
                task["task_spec"],
                self._selected_variation_id,
                asset_registry_root=ASSET_REGISTRY_ROOT,
                episode_seed=0,
                prim_root="/World/Objects",
            )
            removed = self._clear_variation_preview_roots(stage, "/World/Objects")
            load_resolved_episode_into_stage(
                stage,
                episode,
                config=EpisodeRuntimeLoadConfig(
                    load_background=False,
                    capture_light_baseline=False,
                    require_background_registry=False,
                ),
            )
            self._loaded_episode = episode
        except Exception:
            self._set_status(f"Load variation layout failed.\n{traceback.format_exc()}")
            self._request_rebuild()
            return

        self._set_status(
            f"Loaded variation '{self._selected_variation_id}' into current stage.\n"
            f"objects={len(episode.objects)}\n"
            f"cleared_roots={removed}\n"
            f"root=/World/Objects"
        )
        self._request_rebuild()

    def _load_all_candidates_into_stage(self):
        self._schedule_async(self._load_all_candidates_into_stage_async())

    async def _load_all_candidates_into_stage_async(self):
        task = self._get_selected_task()
        variation_id = self._selected_variation_id
        if task is None or variation_id is None:
            self._set_status("No variation selected for all-candidates stage load.")
            return
        self._capture_variation_fields()
        self._write_selected_variation()
        try:
            import omni.usd
            from src.scene.episode_runtime import EpisodeRuntimeLoadConfig, load_resolved_episode_into_stage
            from src.task.specs import resolve_variation
            from src.task.specs.asset_registry import AssetRegistryIndex
            from src.task.specs.schema import ResolvedEpisodeSpec, ResolvedObjectSpec
        except Exception as exc:
            self._set_status(f"Runtime imports failed: {exc}")
            self._request_rebuild()
            return

        task_spec = task["task_spec"]
        try:
            stage = await self._get_or_create_stage()
            resolved_variation = resolve_variation(task_spec, variation_id)
            registry = AssetRegistryIndex(ASSET_REGISTRY_ROOT)
        except Exception:
            self._set_status(f"Resolve variation failed.\n{traceback.format_exc()}")
            self._request_rebuild()
            return

        objects = {}
        for instance_id in resolved_variation.active_instance_ids:
            slot = resolved_variation.layout_slots.get(instance_id)
            if slot is None:
                continue
            candidates = task_spec.asset_bindings.get(instance_id, ())
            if not candidates:
                continue
            object_spec = task_spec.object_instances.get(instance_id)
            semantic_name = instance_id if object_spec is None else object_spec.semantic_name
            role = None if object_spec is None else object_spec.role
            for candidate in candidates:
                try:
                    asset = registry.resolve_candidate(task_spec, instance_id, candidate).to_spec()
                except Exception:
                    self._set_status(f"Resolve candidate failed for {instance_id}.\n{traceback.format_exc()}")
                    self._request_rebuild()
                    return
                object_id = f"{instance_id}__{asset.asset_key}"
                objects[object_id] = ResolvedObjectSpec(
                    instance_id=object_id,
                    semantic_name=semantic_name,
                    role=role,
                    prim_path=f"{ALL_CANDIDATES_ROOT.rstrip('/')}/{object_id}",
                    asset=asset,
                    position=tuple(float(value) for value in slot.position),
                    rotation=tuple(float(value) for value in slot.rotation),
                    scale=tuple(float(value) for value in slot.scale),
                )

        if not objects:
            self._set_status(f"Variation '{variation_id}' has no candidates to load.")
            self._request_rebuild()
            return

        episode = ResolvedEpisodeSpec(
            task_id=task_spec.task_id,
            template_type=task_spec.template_type,
            variation_id=f"{variation_id}__all_candidates",
            instruction=resolved_variation.instruction,
            episode_seed=0,
            background=None,
            objects=dict(sorted(objects.items())),
            success_predicates=resolved_variation.success_predicates,
            fail_predicates=resolved_variation.fail_predicates,
            semantic_groups=resolved_variation.semantic_groups,
        )
        try:
            removed = self._clear_variation_preview_roots(stage, ALL_CANDIDATES_ROOT)
            load_resolved_episode_into_stage(
                stage,
                episode,
                config=EpisodeRuntimeLoadConfig(
                    load_background=False,
                    capture_light_baseline=False,
                    require_background_registry=False,
                ),
            )
            self._loaded_episode = episode
        except Exception:
            self._set_status(f"Load all candidates failed.\n{traceback.format_exc()}")
            self._request_rebuild()
            return

        self._set_status(
            f"Loaded {len(objects)} candidate instances for variation '{variation_id}' under {ALL_CANDIDATES_ROOT}.\n"
            f"cleared_roots={removed}"
        )
        self._request_rebuild()

    def _update_preview_summary(self):
        task = self._get_selected_task()
        variation = self._get_selected_resolved_variation()
        if task is None or variation is None:
            self._preview_model.set_value("No resolved preview available.")
            return
        lines = [
            f"variation_id: {variation.variation_id}",
            f"enabled: {variation.enabled}",
            f"instruction: {variation.instruction}",
            f"goal_relation: {variation.goal_relation}",
            f"container_instance_id: {variation.container_instance_id}",
            f"active_instance_count: {len(variation.active_instance_ids)}",
            f"layout_slot_count: {len(variation.layout_slots)}",
            f"success_predicates: {len(variation.success_predicates)}",
            f"fail_predicates: {len(variation.fail_predicates)}",
            "",
            f"target: {', '.join(variation.semantic_groups.get('target', ())) or '-'}",
            f"distractor: {', '.join(variation.semantic_groups.get('distractor', ())) or '-'}",
            f"decor: {', '.join(variation.semantic_groups.get('decor', ())) or '-'}",
        ]
        self._preview_model.set_value("\n".join(lines))

    def _capture_loaded_stage_layout(self):
        if self._loaded_episode is None:
            self._set_status("No loaded stage layout to capture.")
            return
        if self._selected_instance_id is None:
            self._set_status("No selected slot to capture from stage.")
            return

        try:
            import omni.usd
            from omni.isaac.core.prims import XFormPrim
        except Exception as exc:
            self._set_status(f"Stage capture imports failed: {exc}")
            self._request_rebuild()
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._set_status("No open USD stage is available.")
            self._request_rebuild()
            return

        obj = self._loaded_episode.objects.get(self._selected_instance_id)
        if obj is None:
            self._set_status(f"Selected slot '{self._selected_instance_id}' is not in the loaded episode.")
            self._request_rebuild()
            return

        prim = stage.GetPrimAtPath(obj.prim_path)
        if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
            self._set_status(f"Loaded prim does not exist in stage: {obj.prim_path}")
            self._request_rebuild()
            return

        try:
            xform = XFormPrim(obj.prim_path)
            position, rotation = xform.get_world_pose()
            scale = [1.0, 1.0, 1.0]
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                    continue
                value = op.Get()
                if value is None:
                    continue
                scale = [float(value[0]), float(value[1]), float(value[2])]
                break
        except Exception as exc:
            self._set_status(f"Capture stage pose failed: {exc}")
            self._request_rebuild()
            return

        payload = self._ensure_dirty_payload()
        payload["slot_models_instance_id"] = self._selected_instance_id
        payload["slot_position"] = [float(value) for value in position]
        payload["slot_rotation"] = [float(value) for value in rotation]
        payload["slot_scale"] = scale
        self._load_editor_models()
        self._set_status(f"Captured stage pose for {self._selected_instance_id}.")
        self._request_rebuild()

    def _refresh_selected_variation_preview(self):
        task = self._get_selected_task()
        if task is None or self._selected_variation_id is None:
            self._preview_model.set_value("No resolved preview available.")
            self._set_status("No selected variation to resolve.")
            return
        task_spec = self._reload_current_task_cache()
        if task_spec is None:
            self._preview_model.set_value("No resolved preview available.")
            self._set_status("Reload failed for selected task.")
            return
        resolved = task["resolved_variations"].get(self._selected_variation_id)
        if resolved is None:
            self._preview_model.set_value("Resolve failed: no resolved variation available.")
            self._set_status(f"Resolve failed for {self._selected_variation_id}.")
            self._request_rebuild()
            return
        self._update_preview_summary()
        self._set_status(f"Resolved variation {self._selected_variation_id}.")
        self._request_rebuild()

    def _write_selected_variation(self):
        if self._selected_variation_id is None:
            self._set_status("No selected variation.")
            return
        path = self._payload_path_for_selected_variation()
        if path is None:
            self._set_status("No payload path for selected variation.")
            return
        payload = self._load_variation_payload()
        self._capture_variation_fields()
        dirty = self._ensure_dirty_payload()
        payload["enabled"] = bool(dirty["enabled"])
        payload["instruction"] = str(dirty["instruction"])
        self._write_group_override_payload(payload, "goal_objects", str(dirty["group_target"]))
        self._write_group_override_payload(payload, "fail_objects", str(dirty["group_distractor"]))
        self._write_group_override_payload(payload, "decor_objects", str(dirty["group_decor"]))
        self._write_goal_override_payload(payload, dirty)
        if self._selected_instance_id is not None:
            pose_overrides = dict(payload.get("pose_overrides") or {})
            slot_payload = self._slot_payload_from_models()
            pose_overrides[self._selected_instance_id] = slot_payload
            payload["pose_overrides"] = pose_overrides
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        key = self._dirty_key()
        self._dirty_variations.pop(key, None)
        self._reload_current_task_cache()
        self._sync_selected_instance()
        self._load_editor_models()
        self._update_preview_summary()
        self._set_status(f"Saved {path}.")
        self._request_rebuild()

    def _write_group_override_payload(self, payload: dict, prefix: str, text: str):
        members = self._parse_group_members_text(text)
        payload[f"set_{prefix}"] = members
        payload.pop(f"add_{prefix}", None)
        payload.pop(f"remove_{prefix}", None)
        selected_key = {
            "goal_objects": "selected_goal_objects",
            "fail_objects": "selected_fail_objects",
            "decor_objects": "selected_decor_objects",
        }[prefix]
        payload.pop(selected_key, None)

    def _write_goal_override_payload(self, payload: dict, dirty: dict):
        policy_overrides = dict(payload.get("policy_overrides") or {})
        success_payload = dict(policy_overrides.get("success") or {})
        fail_payload = dict(policy_overrides.get("fail") or {})
        goal_items = [
            {
                "subject_id": item["subject_id"],
                "relation": item["relation"],
                "target_id": item["target_id"],
            }
            for item in (dirty.get("goal_override_items") or [])
            if item.get("subject_id") and item.get("relation") and item.get("target_id")
        ]
        fail_items = [
            {
                "subject_id": item["subject_id"],
                "relation": item["relation"],
                "target_id": item["target_id"],
            }
            for item in (dirty.get("fail_override_items") or [])
            if item.get("subject_id") and item.get("relation") and item.get("target_id")
        ]
        if goal_items:
            success_payload["append"] = goal_items
        else:
            success_payload.pop("append", None)
        if fail_items:
            fail_payload["append"] = fail_items
        else:
            fail_payload.pop("append", None)
        if success_payload:
            policy_overrides["success"] = success_payload
        else:
            policy_overrides.pop("success", None)
        if fail_payload:
            policy_overrides["fail"] = fail_payload
        else:
            policy_overrides.pop("fail", None)
        if policy_overrides:
            payload["policy_overrides"] = policy_overrides
        else:
            payload.pop("policy_overrides", None)

    def _clear_selected_slot_override(self):
        if self._selected_instance_id is None:
            self._set_status("No selected slot.")
            return
        path = self._payload_path_for_selected_variation()
        if path is None:
            self._set_status("No payload path for selected variation.")
            return
        payload = self._load_variation_payload()
        pose_overrides = dict(payload.get("pose_overrides") or {})
        if self._selected_instance_id in pose_overrides:
            pose_overrides.pop(self._selected_instance_id, None)
            payload["pose_overrides"] = pose_overrides
            path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
            self._reload_current_task_cache()
            self._sync_selected_instance()
            key = self._dirty_key()
            self._dirty_variations.pop(key, None)
            self._load_editor_models()
            self._update_preview_summary()
            self._set_status(f"Removed override for {self._selected_instance_id}.")
            self._request_rebuild()
            return
        self._set_status(f"No authored override exists for {self._selected_instance_id}.")

    def _open_remove_variation_dialog(self):
        task = self._get_selected_task()
        if task is None or self._selected_variation_id is None:
            self._set_status("No variation selected.")
            return
        if self._selected_variation_id == task["task_spec"].base_variation.variation_id:
            self._set_status("BASE variation cannot be removed.")
            return
        ui = self._ui
        dialog = ui.Window("Confirm Variation Removal", width=420, height=180, flags=ui.WINDOW_FLAGS_NO_RESIZE)

        def _close_dialog():
            dialog.visible = False

        def _confirm():
            _close_dialog()
            self._remove_selected_variation()

        with dialog.frame:
            with ui.VStack(spacing=10):
                ui.Spacer(height=8)
                ui.Label("Remove this variation?", height=24)
                self._build_compact_text_block(
                    ui,
                    f"task={task['task_key']}\nvariation={self._selected_variation_id}",
                    height=56,
                )
                with ui.HStack(spacing=8, height=30):
                    ui.Button("Cancel", clicked_fn=_close_dialog)
                    ui.Button("Remove", clicked_fn=_confirm)

    def _remove_selected_variation(self):
        task = self._get_selected_task()
        if task is None or self._selected_variation_id is None:
            self._set_status("Remove variation failed. No variation selected.")
            return
        if self._selected_variation_id == task["task_spec"].base_variation.variation_id:
            self._set_status("Remove variation failed. BASE variation cannot be removed.")
            return
        removed_variation_id = self._selected_variation_id
        path = task["task_dir"] / "variations" / f"{removed_variation_id}.yaml"
        if not path.exists():
            self._set_status(f"Remove variation failed. Missing file: {path}")
            return
        path.unlink()
        self._reload_current_task_cache()
        refreshed_task = self._get_selected_task()
        if refreshed_task is not None:
            self._selected_variation_id = refreshed_task["variation_ids"][0] if refreshed_task["variation_ids"] else None
        else:
            self._selected_variation_id = None
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._update_preview_summary()
        self._set_status(f"Removed variation '{removed_variation_id}'.")
        self._request_rebuild()

    def _open_create_variation_form(self):
        task = self._get_selected_task()
        self._create_variation_mode = True
        self._new_variation_id_model.set_value("")
        variation_ids = [] if task is None else list(task["variation_ids"])
        selected_parent = self._selected_variation_id or "BASE"
        try:
            self._new_variation_extends_index = variation_ids.index(selected_parent)
        except ValueError:
            self._new_variation_extends_index = 0
        self._new_variation_extends_combo = None
        self._set_status("Enter new variation ID and parent, then Confirm.")
        self._request_rebuild()

    def _cancel_create_variation(self):
        self._create_variation_mode = False
        self._request_rebuild()

    def _confirm_create_variation(self):
        task = self._get_selected_task()
        if task is None:
            self._set_status("No task selected.")
            return
        new_id = self._new_variation_id_model.as_string.strip()
        if not new_id:
            self._set_status("Variation ID cannot be empty.")
            return
        if new_id in task["variation_ids"]:
            self._set_status(f"Variation '{new_id}' already exists.")
            return
        variation_ids = list(task["variation_ids"])
        if not variation_ids:
            self._set_status("No parent variation is available.")
            return
        extends_index = int(self._new_variation_extends_index)
        if self._new_variation_extends_combo is not None:
            extends_index = int(self._new_variation_extends_combo.model.get_item_value_model().as_int)
        if extends_index < 0 or extends_index >= len(variation_ids):
            self._set_status("Selected parent variation is invalid.")
            return
        extends = variation_ids[extends_index]
        variations_dir = task["task_dir"] / "variations"
        variations_dir.mkdir(exist_ok=True)
        path = variations_dir / f"{new_id}.yaml"
        path.write_text(
            yaml.safe_dump({"variation_id": new_id, "extends": extends, "instruction": "", "enabled": True}, sort_keys=False),
            encoding="utf-8",
        )
        self._create_variation_mode = False
        self._new_variation_extends_combo = None
        self._reload_current_task_cache()
        self._selected_variation_id = new_id
        self._selected_instance_id = None
        self._sync_selected_instance()
        self._load_editor_models()
        self._update_preview_summary()
        self._set_status(f"Created variation '{new_id}' extending '{extends}'.")
        self._request_rebuild()

    def _build_create_variation_form(self, ui):
        task = self._get_selected_task()
        variation_ids = [] if task is None else list(task["variation_ids"])
        known = ", ".join(variation_ids) if variation_ids else "-"
        self._build_pane_title(ui, "Create Variation")
        self._build_editable_string_row(ui, "New ID", self._new_variation_id_model)
        with ui.HStack(spacing=8, height=24):
            ui.Label("Extends", width=78)
            if variation_ids:
                selected_index = min(max(int(self._new_variation_extends_index), 0), len(variation_ids) - 1)
                self._new_variation_extends_combo = ui.ComboBox(selected_index, *variation_ids, height=22)
            else:
                self._new_variation_extends_combo = None
                empty_model = self._ui.SimpleStringModel("No variations available")
                field = ui.StringField(model=empty_model, height=22)
                field.enabled = False
        with ui.HStack(spacing=8, height=24):
            ui.Button("Confirm", width=ACTION_BUTTON_WIDTH, clicked_fn=self._confirm_create_variation)
            ui.Button("Cancel", width=ACTION_BUTTON_WIDTH, clicked_fn=self._cancel_create_variation)
        self._build_compact_row(ui, "Known", known)

    def _build_ui(self):
        ui = self._ui
        selected_task = self._get_selected_task()
        selected_variation = self._get_selected_variation()
        selected_slot_item = self._get_selected_slot_item()
        with self._window.frame:
            with ui.VStack(spacing=8):
                self._build_header_bar(ui, selected_task, selected_variation)
                with ui.HStack(spacing=10):
                    self._build_navigation_pane(ui, selected_task)
                    self._build_layout_pane(ui, selected_task, selected_variation)
                    self._build_detail_pane(ui, selected_task, selected_variation, selected_slot_item)

    def _build_header_bar(self, ui, selected_task, selected_variation):
        with ui.ZStack(height=62):
            ui.Rectangle()
            with ui.VStack(spacing=4):
                ui.Spacer(height=4)
                with ui.HStack(spacing=8, height=24):
                    ui.Label("BeTTER Variation Editor", width=190)
                    ui.Label("Task", width=34)
                    task_model = self._ui.SimpleStringModel(selected_task["task_key"] if selected_task else "")
                    task_field = ui.StringField(model=task_model, height=22)
                    task_field.enabled = False
                    ui.Label("Registry", width=52)
                    registry_field = ui.StringField(model=self._task_path_model, height=22)
                    registry_field.enabled = False
                    ui.Button("Reload", width=72, clicked_fn=self._reload_tasks)
                    if self._dirty_model.as_string:
                        ui.Label(self._dirty_model.as_string, width=72)
                with ui.HStack(spacing=14, height=20):
                    ui.Label(f"tasks {len(self._tasks)}")
                    ui.Label(f"variations {len(selected_task['variation_ids']) if selected_task else 0}")
                    ui.Label(f"layout view {self._layout_mode_model.as_string}")
                    ui.Label(f"selected {selected_variation.variation_id if selected_variation else '-'}")

    def _build_navigation_pane(self, ui, selected_task):
        with ui.VStack(spacing=6, width=LEFT_PANE_WIDTH):
            self._build_pane_title(ui, "Tasks")
            self._build_compact_text_block(ui, selected_task["instruction"] if selected_task else "No task selected.", height=54)
            with ui.HStack(spacing=8, height=20):
                ui.Label("Root", width=52)
                ui.Label(str(TASKS_ROOT), word_wrap=False)
            with ui.ZStack(height=760):
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

    def _build_layout_pane(self, ui, selected_task, selected_variation):
        with ui.VStack(spacing=6, width=MIDDLE_PANE_WIDTH):
            self._build_pane_title(ui, "Variations")
            with ui.ZStack(height=170):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=3):
                        ui.Spacer(height=3)
                        if selected_task is None:
                            ui.Label("No task selected.")
                        else:
                            for variation_id in selected_task["variation_ids"]:
                                label = variation_id if variation_id != "BASE" else "BASE  |  base"
                                self._build_selectable_button(
                                    ui,
                                    label=label,
                                    selected=variation_id == self._selected_variation_id,
                                    clicked_fn=lambda value=variation_id: self._select_variation(value),
                                    height=24,
                                )
                        ui.Spacer(height=3)
            self._build_pane_title(ui, "Variation Actions")
            with ui.HStack(spacing=8, height=24):
                ui.Button("Create", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_create_variation_form)
                ui.Button("Remove", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_remove_variation_dialog)
            self._build_pane_title(ui, "Layouts")
            with ui.HStack(spacing=8, height=24):
                ui.Button("Layout (resolved)", width=ACTION_BUTTON_WIDTH, clicked_fn=lambda: self._set_layout_view_mode("resolved"))
                ui.Button("Layout overrides", width=ACTION_BUTTON_WIDTH, clicked_fn=lambda: self._set_layout_view_mode("override"))
            with ui.HStack(spacing=8, height=24):
                active_label = "Hide inactive" if self._show_active_only else "Show active only"
                inherited_label = "Hide inherited" if self._show_inherited_slots else "Show inherited"
                ui.Button(active_label, width=ACTION_BUTTON_WIDTH, clicked_fn=self._toggle_show_active_only)
                ui.Button(inherited_label, width=ACTION_BUTTON_WIDTH, clicked_fn=self._toggle_show_inherited_slots)
            with ui.HStack(spacing=8, height=18):
                ui.Label("Variation", width=58)
                ui.Label(selected_variation.variation_id if selected_variation else "-")
            with ui.HStack(spacing=8, height=18):
                ui.Label("Mode", width=58)
                ui.Label(self._layout_mode_model.as_string)
            with ui.ZStack(height=560):
                ui.Rectangle()
                with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    with ui.VStack(spacing=4):
                        ui.Spacer(height=4)
                        if selected_task is None or selected_variation is None:
                            ui.Label("No variation selected.")
                        else:
                            for item in self._get_slot_items():
                                status_bits = ["active" if item["is_active"] else "inactive"]
                                status_bits.append(item["group_status"])
                                status_bits.append(item["slot_source"])
                                label = f"{item['semantic_name']}  |  {item['instance_id']}  |  {', '.join(status_bits)}"
                                self._build_selectable_button(
                                    ui,
                                    label=label,
                                    selected=item["instance_id"] == self._selected_instance_id,
                                    clicked_fn=lambda value=item["instance_id"]: self._select_instance(value),
                                    height=24,
                                )
                        ui.Spacer(height=4)

    def _build_detail_pane(self, ui, selected_task, selected_variation, selected_slot_item):
        with ui.VStack(spacing=6, width=RIGHT_PANE_WIDTH):
            if self._create_variation_mode:
                self._build_create_variation_form(ui)
                return
            if self._goal_override_editor_mode:
                self._build_goal_override_editor(ui)
                return
            self._build_variation_detail_pane(ui, selected_task, selected_variation, selected_slot_item)

    def _build_variation_detail_pane(self, ui, selected_task, selected_variation, selected_slot_item):
        self._build_pane_title(ui, "Variation Metadata")
        self._build_compact_row(ui, "Variation", selected_variation.variation_id if selected_variation else "")
        self._build_editable_string_row(ui, "Instruction", self._variation_instruction_model)
        self._build_compact_row(ui, "Goal relation", self._variation_goal_relation_text(selected_task, selected_variation))
        self._build_editable_bool_row(ui, "Enabled", self._variation_enabled_model)
        with ui.HStack(spacing=8, height=24):
            ui.Button("Load into stage", width=ACTION_BUTTON_WIDTH, clicked_fn=self._load_selected_variation_into_stage)
            ui.Button("Load all to stage", width=ACTION_BUTTON_WIDTH, clicked_fn=self._load_all_candidates_into_stage)
            ui.Button("Save variation", width=ACTION_BUTTON_WIDTH, clicked_fn=self._write_selected_variation)
            ui.Button("Resolve preview", width=ACTION_BUTTON_WIDTH, clicked_fn=self._refresh_selected_variation_preview)
        self._build_pane_title(ui, "Variation Semantics")
        self._build_compact_row(ui, "Group", self._semantic_group_for_instance(selected_slot_item["instance_id"]) if selected_slot_item else "")
        self._build_compact_row(ui, "Group status", selected_slot_item["group_status"] if selected_slot_item else "")
        self._build_compact_row(ui, "Goal overrides", self._goal_override_summary_text())
        self._build_compact_row(ui, "Fail overrides", self._fail_override_summary_text())
        self._build_pane_title(ui, "Resolved Goal Conditions")
        self._build_compact_text_block(ui, self._build_resolved_condition_summary("goal"), height=110)
        with ui.HStack(spacing=8, height=24):
            toggle_label = "Deactivate" if self._is_selected_instance_active_in_staged_payload() else "Activate"
            ui.Button(toggle_label, width=ACTION_BUTTON_WIDTH, clicked_fn=self._toggle_selected_instance_active)
            ui.Button("Edit Goal Overrides", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_goal_override_editor)
        with ui.HStack(spacing=8, height=24):
            ui.Button("Edit Fail Overrides", width=ACTION_BUTTON_WIDTH, clicked_fn=self._open_fail_override_editor)
        self._build_pane_title(ui, "Slot Override Editor")
        self._build_compact_row(ui, "Instance", selected_slot_item["instance_id"] if selected_slot_item else "")
        self._build_compact_row(ui, "Semantic", selected_slot_item["semantic_name"] if selected_slot_item else "")
        self._build_compact_row(ui, "Role", selected_slot_item["role"] if selected_slot_item else "")
        self._build_compact_row(ui, "Slot source", selected_slot_item["slot_source"] if selected_slot_item else "")
        self._build_vector_editor(ui, "Position", self._slot_position_models, VECTOR_LABELS)
        self._build_vector_editor(ui, "Rotation", self._slot_rotation_models, QUAT_LABELS)
        self._build_vector_editor(ui, "Scale", self._slot_scale_models, VECTOR_LABELS)
        with ui.HStack(spacing=8, height=24):
            ui.Button("Capture from stage", width=ACTION_BUTTON_WIDTH, clicked_fn=self._capture_loaded_stage_layout)
            ui.Button("Reset form", width=ACTION_BUTTON_WIDTH, clicked_fn=self._revert_slot_override)
            ui.Button("Clear override", width=ACTION_BUTTON_WIDTH, clicked_fn=self._clear_selected_slot_override)
        self._build_pane_title(ui, "Status")
        self._build_compact_text_block(ui, self._status_model.as_string, height=90)

    def _variation_goal_relation_text(self, selected_task, selected_variation) -> str:
        if selected_variation is None:
            return ""
        goal_relation = str(selected_variation.goal_relation or "")
        if goal_relation:
            return goal_relation
        if selected_task is None:
            return ""
        return str(selected_task["task_spec"].primary_goal_relation or "")

    def _build_pane_title(self, ui, title: str):
        with ui.HStack(height=20):
            ui.Label(title)

    def _build_compact_row(self, ui, label: str, value: str):
        with ui.HStack(spacing=8, height=24):
            ui.Label(label, width=78)
            model = self._ui.SimpleStringModel(value or "")
            field = ui.StringField(model=model, height=22)
            field.enabled = False

    def _build_editable_string_row(self, ui, label: str, model):
        with ui.HStack(spacing=8, height=24):
            ui.Label(label, width=78)
            ui.StringField(model=model, height=22)

    def _build_editable_bool_row(self, ui, label: str, model):
        with ui.HStack(spacing=8, height=24):
            ui.Label(label, width=78)
            ui.CheckBox(model=model, width=22)
            ui.Label("true" if model.as_bool else "false")

    def _build_vector_editor(self, ui, label: str, models, labels):
        with ui.VStack(spacing=2):
            ui.Label(label, height=18)
            with ui.HStack(spacing=4, height=24):
                for axis_label, model in zip(labels, models):
                    ui.Label(axis_label, width=14)
                    ui.FloatField(model=model, height=22)

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

    def _build_selectable_button(self, ui, label: str, selected: bool, clicked_fn, height: int):
        button_label = f"> {label}" if selected else label
        ui.Button(button_label, height=height, clicked_fn=clicked_fn)
