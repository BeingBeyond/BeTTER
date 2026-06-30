from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence

from .task_draft import (
    CompiledTaskAuthoringBundle,
    TaskDraftObjectSpec,
    TaskDraftPolicyHintsSpec,
    TaskDraftSpec,
    TaskTemplateSpec,
    compile_task_draft,
)


def task_draft_from_dict(payload: Mapping[str, object], *, guidance_override: str | None = None) -> TaskDraftSpec:
    guidance = guidance_override if guidance_override is not None else _require_string(payload, "guidance")
    base_variation_instruction = _require_string(payload, "base_variation_instruction")
    container_slot = _require_optional_string(payload, "container_slot")
    primary_goal_relation = _require_string(payload, "primary_goal_relation")
    fail_relations = _require_string_tuple(payload, "fail_relations")

    objects_payload = payload.get("objects")
    if not isinstance(objects_payload, Sequence) or isinstance(objects_payload, (str, bytes)):
        actual_type = type(objects_payload).__name__
        preview = repr(objects_payload)
        if len(preview) > 800:
            preview = preview[:800] + "..."
        payload_keys = sorted(str(key) for key in payload.keys())
        print(f"Task draft parse error: top-level keys={payload_keys}", file=sys.stderr)
        print(f"Task draft parse error: objects payload type={actual_type}", file=sys.stderr)
        print(f"Task draft parse error: objects payload preview={preview}", file=sys.stderr)
        sys.stderr.flush()
        raise ValueError(f"objects must be a sequence of object payloads, got {actual_type}: {preview}")
    objects = tuple(_task_draft_object_from_dict(item) for item in objects_payload)

    policy_hints_payload = payload.get("policy_hints")
    if not isinstance(policy_hints_payload, Mapping):
        raise ValueError("policy_hints must be an object.")
    policy_hints = TaskDraftPolicyHintsSpec(
        success_hints=_require_string_tuple(policy_hints_payload, "success_hints"),
        fail_hints=_require_string_tuple(policy_hints_payload, "fail_hints"),
    )

    return TaskDraftSpec(
        guidance=guidance,
        objects=objects,
        base_variation_instruction=base_variation_instruction,
        container_slot=container_slot,
        primary_goal_relation=primary_goal_relation,
        fail_relations=fail_relations,
        policy_hints=policy_hints,
    )


def task_draft_from_json(payload: str, *, guidance_override: str | None = None) -> TaskDraftSpec:
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid task draft JSON: {exc.msg}.") from exc
    if not isinstance(loaded, Mapping):
        print(f"Task draft parse error: decoded top-level type={type(loaded).__name__}", file=sys.stderr)
        sys.stderr.flush()
        raise ValueError("Task draft JSON must decode to an object.")
    return task_draft_from_dict(loaded, guidance_override=guidance_override)


def compile_task_draft_payload(
    *,
    task_id: str,
    template: TaskTemplateSpec,
    payload: Mapping[str, object] | str,
    guidance_override: str | None = None,
) -> CompiledTaskAuthoringBundle:
    draft = task_draft_from_json(payload, guidance_override=guidance_override) if isinstance(payload, str) else task_draft_from_dict(payload, guidance_override=guidance_override)
    return compile_task_draft(task_id=task_id, template=template, draft=draft)



def _task_draft_object_from_dict(payload: object) -> TaskDraftObjectSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("Each draft object must be an object.")
    return TaskDraftObjectSpec(
        slot_name=_require_string(payload, "slot_name"),
        semantic_name=_require_string(payload, "semantic_name"),
        description=_require_string(payload, "description"),
        retrieval_query=_require_string(payload, "retrieval_query"),
        mass_range=_require_number_pair(payload, "mass_range"),
        target_size_range=_require_number_pair(payload, "target_size_range"),
        group_hint=_require_string(payload, "group_hint"),
        tags=_require_optional_string_tuple(payload, "tags"),
    )



def _require_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value



def _require_optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null.")
    return value



def _require_string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain only strings.")
        result.append(item)
    return tuple(result)



def _require_optional_string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} must contain only strings.")
        result.append(item)
    return tuple(result)



def _require_number_pair(payload: Mapping[str, object], key: str) -> tuple[float, float]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{key} must be a length-2 sequence of numbers.")
    first, second = value
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        raise ValueError(f"{key} must be a length-2 sequence of numbers.")
    return (float(first), float(second))
