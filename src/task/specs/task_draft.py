from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
import re

import yaml


_ALLOWED_GROUP_HINTS = ("target", "distractor", "decor", "container")


@dataclass(frozen=True)
class TaskTemplateSlotSpec:
    slot_name: str
    group_hint: str
    description: str = ""
    min_count: int = 1
    max_count: int = 1
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskTemplateSpec:
    template_id: str
    template_type: str
    slots: tuple[TaskTemplateSlotSpec, ...]
    allowed_relations: tuple[str, ...] = ("in",)
    primary_goal_relation: str = "in"
    fail_relations: tuple[str, ...] = ("in", "on")


@dataclass(frozen=True)
class TaskDraftObjectSpec:
    slot_name: str
    semantic_name: str
    description: str
    retrieval_query: str
    mass_range: tuple[float, float]
    target_size_range: tuple[float, float]
    group_hint: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskDraftPolicyHintsSpec:
    success_hints: tuple[str, ...] = ()
    fail_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskDraftSpec:
    guidance: str
    objects: tuple[TaskDraftObjectSpec, ...]
    base_variation_instruction: str
    container_slot: str | None
    primary_goal_relation: str
    fail_relations: tuple[str, ...]
    policy_hints: TaskDraftPolicyHintsSpec = field(default_factory=TaskDraftPolicyHintsSpec)


@dataclass(frozen=True)
class CompiledTaskAuthoringBundle:
    task_yaml: dict[str, object]
    object_universe_yaml: dict[str, object]
    base_variation_yaml: dict[str, object]
    variations_dir_entries: dict[str, dict[str, object]] = field(default_factory=dict)


def compile_task_draft(
    *,
    task_id: str,
    template: TaskTemplateSpec,
    draft: TaskDraftSpec,
) -> CompiledTaskAuthoringBundle:
    _validate_template(template)
    _validate_draft(template, draft)

    instance_ids_by_slot: dict[str, list[str]] = {}
    object_universe = {
        "container": [],
        "goal_objects": [],
        "fail_objects": [],
        "decor_objects": [],
    }
    set_goal_objects: list[str] = []
    set_fail_objects: list[str] = []
    set_decor_objects: list[str] = []

    slot_occurrence_index: dict[str, int] = {}
    container_instance_id: str | None = None

    for draft_object in draft.objects:
        slot_occurrence_index[draft_object.slot_name] = slot_occurrence_index.get(draft_object.slot_name, 0) + 1
        occurrence = slot_occurrence_index[draft_object.slot_name]
        instance_id = _build_instance_id(
            semantic_name=draft_object.semantic_name,
            occurrence=occurrence,
            salt=f"{task_id}:{draft_object.slot_name}:{draft_object.semantic_name}:{occurrence}",
        )
        instance_ids_by_slot.setdefault(draft_object.slot_name, []).append(instance_id)

        entry = {
            "instance_id": instance_id,
            "semantic_name": _normalize_token(draft_object.semantic_name),
            "description": draft_object.description,
            "retrieval_query": draft_object.retrieval_query,
            "mass_range": [float(draft_object.mass_range[0]), float(draft_object.mass_range[1])],
            "target_size_range": [float(draft_object.target_size_range[0]), float(draft_object.target_size_range[1])],
            "target_mass": _range_midpoint(draft_object.mass_range),
            "target_size": _range_midpoint(draft_object.target_size_range),
            "tags": list(draft_object.tags),
        }

        if draft_object.group_hint == "container":
            entry["role"] = "container"
            object_universe["container"].append(entry)
            if draft.container_slot == draft_object.slot_name and container_instance_id is None:
                container_instance_id = instance_id
        elif draft_object.group_hint == "target":
            object_universe["goal_objects"].append(entry)
            set_goal_objects.append(instance_id)
        elif draft_object.group_hint == "distractor":
            object_universe["fail_objects"].append(entry)
            set_fail_objects.append(instance_id)
        elif draft_object.group_hint == "decor":
            object_universe["decor_objects"].append(entry)
            set_decor_objects.append(instance_id)

    if container_instance_id is None:
        raise ValueError("Current task schema requires a resolved container slot for task defaults.")

    task_yaml = {
        "task_id": task_id,
        "template_type": template.template_type,
        "instruction": "",
        "guidance": draft.guidance,
        "defaults": {
            "container_instance_id": container_instance_id,
            "primary_goal_relation": draft.primary_goal_relation,
            "allowed_goal_relations": list(template.allowed_relations),
            "fail_policy": {
                "distractor_relation_to_container_is_fail": True,
                "fail_on_relations": list(draft.fail_relations),
            },
        },
        "draft_policy_hints": {
            "success_hints": list(draft.policy_hints.success_hints),
            "fail_hints": list(draft.policy_hints.fail_hints),
        },
        "variation_registry": {
            "train": ["BASE"],
            "eval": [],
        },
    }

    base_variation_yaml = {
        "variation_id": "BASE",
        "enabled": True,
        "instruction": draft.base_variation_instruction,
        "goal_relation": draft.primary_goal_relation,
        "container_instance_id": container_instance_id,
        "set_goal_objects": set_goal_objects,
        "set_fail_objects": set_fail_objects,
        "set_decor_objects": set_decor_objects,
    }

    return CompiledTaskAuthoringBundle(
        task_yaml=task_yaml,
        object_universe_yaml=object_universe,
        base_variation_yaml=base_variation_yaml,
        variations_dir_entries={},
    )


def write_task_authoring_bundle(bundle: CompiledTaskAuthoringBundle, task_dir: str | Path) -> None:
    out_dir = Path(task_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "variations").mkdir(exist_ok=True)
    _write_yaml(out_dir / "task.yaml", bundle.task_yaml)
    _write_yaml(out_dir / "object_universe.yaml", bundle.object_universe_yaml)
    _write_yaml(out_dir / "base_variation.yaml", bundle.base_variation_yaml)
    for variation_id, payload in bundle.variations_dir_entries.items():
        _write_yaml(out_dir / "variations" / f"{variation_id}.yaml", payload)


def _validate_template(template: TaskTemplateSpec) -> None:
    if not template.template_id.strip():
        raise ValueError("Template id cannot be empty.")
    if not template.template_type.strip():
        raise ValueError("Template type cannot be empty.")
    if not template.slots:
        raise ValueError("Template must define at least one slot.")
    for slot in template.slots:
        if slot.group_hint not in _ALLOWED_GROUP_HINTS:
            raise ValueError(f"Unsupported slot group hint: {slot.group_hint}")
        if slot.min_count < 0 or slot.max_count < slot.min_count:
            raise ValueError(f"Invalid count range for slot '{slot.slot_name}'.")
    if template.primary_goal_relation not in template.allowed_relations:
        raise ValueError("Template primary_goal_relation must be present in allowed_relations.")


def _validate_draft(template: TaskTemplateSpec, draft: TaskDraftSpec) -> None:
    slot_specs = {slot.slot_name: slot for slot in template.slots}
    seen_counts: dict[str, int] = {}

    if not draft.base_variation_instruction.strip():
        raise ValueError("Base variation instruction cannot be empty.")
    if draft.primary_goal_relation not in template.allowed_relations:
        raise ValueError(f"Primary goal relation '{draft.primary_goal_relation}' is not allowed by template.")
    if draft.container_slot is not None and draft.container_slot not in slot_specs:
        raise ValueError(f"Unknown container slot '{draft.container_slot}'.")

    for relation in draft.fail_relations:
        if relation not in template.allowed_relations and relation not in ("on",):
            raise ValueError(f"Fail relation '{relation}' is not supported by template.")

    for draft_object in draft.objects:
        slot = slot_specs.get(draft_object.slot_name)
        if slot is None:
            raise ValueError(f"Draft references unknown slot '{draft_object.slot_name}'.")
        seen_counts[draft_object.slot_name] = seen_counts.get(draft_object.slot_name, 0) + 1
        if draft_object.group_hint not in _ALLOWED_GROUP_HINTS:
            raise ValueError(f"Unsupported group hint '{draft_object.group_hint}'.")
        if slot.group_hint != draft_object.group_hint:
            raise ValueError(
                f"Slot '{draft_object.slot_name}' expects group '{slot.group_hint}', got '{draft_object.group_hint}'."
            )
        _validate_closed_range("mass_range", draft_object.mass_range)
        _validate_closed_range("target_size_range", draft_object.target_size_range)
        if not draft_object.semantic_name.strip():
            raise ValueError(f"Slot '{draft_object.slot_name}' has empty semantic_name.")
        if not draft_object.retrieval_query.strip():
            raise ValueError(f"Slot '{draft_object.slot_name}' has empty retrieval_query.")

    for slot_name, slot in slot_specs.items():
        count = seen_counts.get(slot_name, 0)
        if count < slot.min_count or count > slot.max_count:
            raise ValueError(
                f"Slot '{slot_name}' expects count in [{slot.min_count}, {slot.max_count}], got {count}."
            )

    if draft.container_slot is None:
        raise ValueError("Current task compiler requires container_slot to be set.")


def _validate_closed_range(name: str, values: tuple[float, float]) -> None:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    low = float(values[0])
    high = float(values[1])
    if high < low:
        raise ValueError(f"{name} must satisfy min <= max.")


def _build_instance_id(*, semantic_name: str, occurrence: int, salt: str) -> str:
    stem = _normalize_token(semantic_name) or "object"
    digest = sha1(salt.encode("utf-8")).hexdigest()[:4]
    return f"{stem}_{occurrence}_{digest}"


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return token.strip("_")


def _range_midpoint(values: tuple[float, float]) -> float:
    return (float(values[0]) + float(values[1])) / 2.0


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
