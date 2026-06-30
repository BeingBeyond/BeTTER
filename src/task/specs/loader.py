from __future__ import annotations

from pathlib import Path

import yaml

from .schema import (
    AssetCandidateSpec,
    DefaultPolicySpec,
    GroupOverrideSpec,
    LayoutSlotSpec,
    ObjectInstanceSpec,
    PolicyChannelSpec,
    PolicyOverrideChannelSpec,
    PolicyOverrideSpec,
    PredicateSpec,
    PredicateTemplateSpec,
    TaskSpec,
    VariationSpec,
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _as_string_tuple(values: list | tuple | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values if isinstance(value, str))


def _layout_slots_from_payload(payload: dict) -> dict[str, LayoutSlotSpec]:
    slots: dict[str, LayoutSlotSpec] = {}
    for instance_id, slot in (payload.get("pose_overrides", {}) or {}).items():
        if not isinstance(slot, dict):
            continue
        slots[str(instance_id)] = LayoutSlotSpec(
            position=tuple(float(value) for value in slot.get("position", []) or []),
            rotation=tuple(float(value) for value in slot.get("rotation", []) or []),
            scale=tuple(float(value) for value in slot.get("scale", []) or []),
        )
    return dict(sorted(slots.items()))


def _asset_bindings_from_payload(payload: dict) -> dict[str, tuple[AssetCandidateSpec, ...]]:
    bindings: dict[str, tuple[AssetCandidateSpec, ...]] = {}
    for instance_id, binding in (payload.get("bindings", {}) or {}).items():
        if not isinstance(binding, dict):
            continue
        semantic_name = str(binding.get("semantic_name") or "") or None
        candidates: list[AssetCandidateSpec] = []
        for candidate in binding.get("candidates", []) or []:
            if not isinstance(candidate, dict) or not candidate.get("source_uid"):
                continue
            candidates.append(
                AssetCandidateSpec(
                    source_uid=str(candidate["source_uid"]),
                    asset_key=str(candidate["asset_key"]) if candidate.get("asset_key") else None,
                    registry_usd=str(candidate["registry_usd"]) if candidate.get("registry_usd") else None,
                    registry_meta=str(candidate["registry_meta"]) if candidate.get("registry_meta") else None,
                    semantic_name=semantic_name,
                )
            )
        bindings[str(instance_id)] = tuple(
            sorted(candidates, key=lambda item: (item.source_uid, item.asset_key or ""))
        )
    return dict(sorted(bindings.items()))


def _group_override_from_prefix(payload: dict, prefix: str) -> GroupOverrideSpec | None:
    set_members = payload.get(f"set_{prefix}")
    add_members = _as_string_tuple(payload.get(f"add_{prefix}"))
    remove_members = _as_string_tuple(payload.get(f"remove_{prefix}"))
    if set_members is None and len(add_members) == 0 and len(remove_members) == 0:
        return None
    return GroupOverrideSpec(
        set_members=_as_string_tuple(set_members) if set_members is not None else None,
        add_members=add_members,
        remove_members=remove_members,
    )


def _resolve_groups(base_groups: dict[str, tuple[str, ...]], variation_payload: dict) -> dict[str, tuple[str, ...]]:
    mapping = {
        "target": _group_override_from_prefix(variation_payload, "goal_objects"),
        "distractor": _group_override_from_prefix(variation_payload, "fail_objects"),
        "decor": _group_override_from_prefix(variation_payload, "decor_objects"),
    }
    resolved: dict[str, tuple[str, ...]] = {}
    for name, override in mapping.items():
        current = set(base_groups.get(name, ()))
        if override is not None:
            if override.set_members is not None:
                current = set(override.set_members)
            current.update(override.add_members)
            current.difference_update(override.remove_members)
        resolved[name] = tuple(sorted(current))
    return resolved


def _base_group_members(payload: dict, prefix: str, legacy_key: str) -> tuple[str, ...]:
    override = _group_override_from_prefix(payload, prefix)
    if override is not None:
        current = set()
        if override.set_members is not None:
            current = set(override.set_members)
        current.update(override.add_members)
        current.difference_update(override.remove_members)
        return tuple(sorted(current))
    return _as_string_tuple(payload.get(legacy_key))


def _predicate_from_payload(payload: object) -> PredicateSpec | None:
    if not isinstance(payload, dict):
        return None
    subject_id = payload.get("subject_id")
    relation = payload.get("relation")
    target_id = payload.get("target_id")
    if not isinstance(subject_id, str) or not isinstance(relation, str) or not isinstance(target_id, str):
        return None
    return PredicateSpec(subject_id=subject_id, relation=relation, target_id=target_id)


def _policy_override_channel_from_payload(payload: object) -> PolicyOverrideChannelSpec:
    if not isinstance(payload, dict):
        return PolicyOverrideChannelSpec()
    replace_payload = payload.get("replace")
    replace = None
    if isinstance(replace_payload, list):
        replace_items = tuple(
            predicate
            for predicate in (_predicate_from_payload(item) for item in replace_payload)
            if predicate is not None
        )
        replace = tuple(sorted(set(replace_items), key=lambda item: (item.subject_id, item.relation, item.target_id)))
    append_items = tuple(
        predicate
        for predicate in (_predicate_from_payload(item) for item in payload.get("append", []) or [])
        if predicate is not None
    )
    remove_items = tuple(
        predicate
        for predicate in (_predicate_from_payload(item) for item in payload.get("remove", []) or [])
        if predicate is not None
    )
    return PolicyOverrideChannelSpec(
        replace=replace,
        append=tuple(sorted(set(append_items), key=lambda item: (item.subject_id, item.relation, item.target_id))),
        remove=tuple(sorted(set(remove_items), key=lambda item: (item.subject_id, item.relation, item.target_id))),
    )


def _policy_override_spec_from_payload(payload: object) -> PolicyOverrideSpec:
    if not isinstance(payload, dict):
        return PolicyOverrideSpec()
    return PolicyOverrideSpec(
        success=_policy_override_channel_from_payload(payload.get("success")),
        fail=_policy_override_channel_from_payload(payload.get("fail")),
    )


def load_task_spec(task_dir: Path) -> TaskSpec:
    task_yaml = _load_yaml(task_dir / "task.yaml")
    object_universe = _load_yaml(task_dir / "object_universe.yaml")
    asset_bindings_path = task_dir / "asset_bindings.yaml"
    asset_bindings = _load_yaml(asset_bindings_path) if asset_bindings_path.exists() else {}
    base_variation_yaml = _load_yaml(task_dir / "base_variation.yaml")
    task_instruction = str(task_yaml.get("instruction") or "")

    object_instances: dict[str, ObjectInstanceSpec] = {}

    for item in object_universe.get("container", []) or []:
        instance_id = str(item["instance_id"])
        object_instances[instance_id] = ObjectInstanceSpec(
            instance_id=instance_id,
            semantic_name=str(item.get("semantic_name") or item.get("role") or "container"),
            retrieval_query=str(item.get("retrieval_query") or ""),
            description=str(item.get("description") or ""),
            tags=_as_string_tuple(item.get("tags")),
            role=str(item.get("role") or "container"),
        )

    for key in ("goal_objects", "fail_objects", "decor_objects"):
        for item in object_universe.get(key, []) or []:
            instance_id = str(item["instance_id"])
            object_instances[instance_id] = ObjectInstanceSpec(
                instance_id=instance_id,
                semantic_name=str(item.get("semantic_name") or instance_id),
                retrieval_query=str(item.get("retrieval_query") or ""),
                description=str(item.get("description") or ""),
                tags=_as_string_tuple(item.get("tags")),
            )

    semantic_groups = {
        "container": tuple(str(item["instance_id"]) for item in object_universe.get("container", []) or []),
        "target": tuple(str(item["instance_id"]) for item in object_universe.get("goal_objects", []) or []),
        "distractor": tuple(str(item["instance_id"]) for item in object_universe.get("fail_objects", []) or []),
        "decor": tuple(str(item["instance_id"]) for item in object_universe.get("decor_objects", []) or []),
    }

    structured_asset_bindings = _asset_bindings_from_payload(asset_bindings)
    candidate_pools = {
        instance_id: tuple(candidate.source_uid for candidate in candidates)
        for instance_id, candidates in structured_asset_bindings.items()
    }

    defaults = task_yaml.get("defaults", {}) or {}
    container_instance_id = str(defaults.get("container_instance_id") or semantic_groups["container"][0])
    primary_goal_relation = str(defaults.get("primary_goal_relation") or "in")
    fail_relations = _as_string_tuple((defaults.get("fail_policy", {}) or {}).get("fail_on_relations")) or ("in", "on")

    default_policy = DefaultPolicySpec(
        success=PolicyChannelSpec(
            mode="all_of",
            predicates=(PredicateTemplateSpec(subjects_from="target", relation=primary_goal_relation, target_id=container_instance_id),),
        ),
        fail=PolicyChannelSpec(
            mode="any_of",
            predicates=(PredicateTemplateSpec(subjects_from="distractor", relations=fail_relations, target_id=container_instance_id),),
        ),
    )

    base_groups = {
        "target": _base_group_members(base_variation_yaml, "goal_objects", "selected_goal_objects"),
        "distractor": _base_group_members(base_variation_yaml, "fail_objects", "selected_fail_objects"),
        "decor": _base_group_members(base_variation_yaml, "decor_objects", "selected_decor_objects"),
    }
    base_variation = VariationSpec(
        variation_id=str(base_variation_yaml.get("variation_id") or "BASE"),
        instruction=str(base_variation_yaml.get("instruction") or task_instruction),
        enabled=bool(base_variation_yaml.get("enabled", True)),
        goal_relation=str(base_variation_yaml.get("goal_relation") or primary_goal_relation),
        container_instance_id=str(base_variation_yaml.get("container_instance_id") or container_instance_id),
        group_overrides={
            "target": GroupOverrideSpec(set_members=base_groups["target"]),
            "distractor": GroupOverrideSpec(set_members=base_groups["distractor"]),
            "decor": GroupOverrideSpec(set_members=base_groups["decor"]),
        },
        policy_overrides=_policy_override_spec_from_payload(base_variation_yaml.get("policy_overrides")),
        layout_slots=_layout_slots_from_payload(base_variation_yaml),
    )

    variations: dict[str, VariationSpec] = {}
    for variation_path in sorted((task_dir / "variations").glob("*.yaml")):
        payload = _load_yaml(variation_path)
        variation_id = str(payload["variation_id"])
        _ = _resolve_groups(base_groups, payload)
        goal_relation = str(payload.get("goal_relation") or base_variation.goal_relation or primary_goal_relation)
        variation_container_id = str(payload.get("container_instance_id") or base_variation.container_instance_id or container_instance_id)

        overrides = {
            name: override
            for name, override in {
                "target": _group_override_from_prefix(payload, "goal_objects"),
                "distractor": _group_override_from_prefix(payload, "fail_objects"),
                "decor": _group_override_from_prefix(payload, "decor_objects"),
            }.items()
            if override is not None
        }
        variations[variation_id] = VariationSpec(
            variation_id=variation_id,
            extends=str(payload.get("extends") or base_variation.variation_id),
            instruction=str(payload.get("instruction") or base_variation.instruction),
            enabled=bool(payload.get("enabled", True)),
            group_overrides=overrides,
            policy_overrides=_policy_override_spec_from_payload(payload.get("policy_overrides")),
            layout_slots=_layout_slots_from_payload(payload),
            goal_relation=goal_relation,
            container_instance_id=variation_container_id,
        )

    return TaskSpec(
        task_id=str(task_yaml.get("task_id") or task_dir.name),
        template_type=str(task_yaml.get("template_type") or task_dir.parent.name),
        allowed_relations=_as_string_tuple(defaults.get("allowed_goal_relations")) or (primary_goal_relation,),
        primary_goal_relation=primary_goal_relation,
        fail_relations=fail_relations,
        container_instance_id=container_instance_id,
        object_instances=dict(sorted(object_instances.items())),
        semantic_groups=semantic_groups,
        candidate_pools=candidate_pools,
        default_policy=default_policy,
        base_variation=base_variation,
        variations=dict(sorted(variations.items())),
        asset_bindings=structured_asset_bindings,
    )
