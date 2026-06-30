from __future__ import annotations

from .schema import (
    GroupOverrideSpec,
    LayoutSlotSpec,
    PolicyChannelSpec,
    PolicyOverrideChannelSpec,
    PredicateSpec,
    PredicateTemplateSpec,
    ResolvedVariationSpec,
    TaskSpec,
    VariationSpec,
)


def _ordered_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _apply_group_override(base_members: tuple[str, ...], override: GroupOverrideSpec | None) -> tuple[str, ...]:
    members = set(base_members)
    if override is None:
        return tuple(sorted(members))
    if override.set_members is not None:
        members = set(override.set_members)
    members.update(override.add_members)
    members.difference_update(override.remove_members)
    return tuple(sorted(members))


def _merge_layout_slots(
    parent_slots: dict[str, LayoutSlotSpec],
    override_slots: dict[str, LayoutSlotSpec],
) -> dict[str, LayoutSlotSpec]:
    merged = dict(parent_slots)
    merged.update(override_slots)
    return dict(sorted(merged.items()))


def _expand_templates(
    channel: PolicyChannelSpec,
    semantic_groups: dict[str, tuple[str, ...]],
    container_instance_id: str,
    goal_relation: str,
    fail_relations: tuple[str, ...],
) -> list[PredicateSpec]:
    compiled: list[PredicateSpec] = []
    for template in channel.predicates:
        compiled.extend(
            _expand_template(
                template=template,
                semantic_groups=semantic_groups,
                container_instance_id=container_instance_id,
                goal_relation=goal_relation,
                fail_relations=fail_relations,
            )
        )
    compiled.extend(channel.append)
    return sorted(set(compiled), key=lambda p: (p.subject_id, p.relation, p.target_id))


def _expand_template(
    template: PredicateTemplateSpec,
    semantic_groups: dict[str, tuple[str, ...]],
    container_instance_id: str,
    goal_relation: str,
    fail_relations: tuple[str, ...],
) -> list[PredicateSpec]:
    subjects: tuple[str, ...]
    if template.subject_id is not None:
        subjects = (template.subject_id,)
    elif template.subjects_from is not None:
        subjects = semantic_groups.get(template.subjects_from, ())
    else:
        subjects = ()

    target_id = template.target_id or container_instance_id
    if template.relation is not None:
        relations = (template.relation,)
    elif len(template.relations) > 0:
        relations = template.relations
    elif template.subjects_from == "target":
        relations = (goal_relation,)
    else:
        relations = fail_relations

    return [PredicateSpec(subject_id=subject_id, relation=relation, target_id=target_id) for subject_id in subjects for relation in relations]


def _apply_policy_override(
    base_predicates: list[PredicateSpec],
    override: PolicyOverrideChannelSpec,
) -> tuple[PredicateSpec, ...]:
    current = list(base_predicates)
    if override.replace is not None:
        current = list(override.replace)
    current.extend(override.append)
    if len(override.remove) > 0:
        removed = set(override.remove)
        current = [predicate for predicate in current if predicate not in removed]
    return tuple(sorted(set(current), key=lambda p: (p.subject_id, p.relation, p.target_id)))


def _resolved_semantic_groups(task_spec: TaskSpec, variation_spec: VariationSpec) -> dict[str, tuple[str, ...]]:
    groups = {name: tuple(members) for name, members in task_spec.semantic_groups.items()}
    for name, override in task_spec.base_variation.group_overrides.items():
        groups[name] = _apply_group_override(groups.get(name, ()), override)
    for name, override in variation_spec.group_overrides.items():
        groups[name] = _apply_group_override(groups.get(name, ()), override)
    return dict(sorted((name, _ordered_unique(list(members))) for name, members in groups.items()))


def resolve_variation(task_spec: TaskSpec, variation_id: str) -> ResolvedVariationSpec:
    if variation_id == task_spec.base_variation.variation_id:
        variation_spec = task_spec.base_variation
    else:
        variation_spec = task_spec.variations[variation_id]

    if variation_spec.extends not in (None, task_spec.base_variation.variation_id):
        raise ValueError(f"Unsupported stage-1 variation inheritance target: {variation_spec.extends}")

    base = task_spec.base_variation
    semantic_groups = _resolved_semantic_groups(task_spec, variation_spec)
    container_instance_id = variation_spec.container_instance_id or base.container_instance_id or task_spec.container_instance_id
    goal_relation = variation_spec.goal_relation or base.goal_relation or task_spec.primary_goal_relation
    instruction = variation_spec.instruction or base.instruction
    layout_slots = _merge_layout_slots(base.layout_slots, variation_spec.layout_slots)

    success_defaults = _expand_templates(
        channel=task_spec.default_policy.success,
        semantic_groups=semantic_groups,
        container_instance_id=container_instance_id,
        goal_relation=goal_relation,
        fail_relations=task_spec.fail_relations,
    )
    fail_defaults = _expand_templates(
        channel=task_spec.default_policy.fail,
        semantic_groups=semantic_groups,
        container_instance_id=container_instance_id,
        goal_relation=goal_relation,
        fail_relations=task_spec.fail_relations,
    )

    success_predicates = _apply_policy_override(success_defaults, variation_spec.policy_overrides.success)
    fail_predicates = _apply_policy_override(fail_defaults, variation_spec.policy_overrides.fail)

    active_instance_ids = sorted(
        set(layout_slots.keys())
        | {container_instance_id}
        | set(semantic_groups.get("target", ()))
        | set(semantic_groups.get("distractor", ()))
        | set(semantic_groups.get("decor", ()))
    )

    return ResolvedVariationSpec(
        variation_id=variation_id,
        enabled=variation_spec.enabled,
        instruction=instruction,
        container_instance_id=container_instance_id,
        goal_relation=goal_relation,
        semantic_groups=semantic_groups,
        layout_slots=layout_slots,
        success_predicates=success_predicates,
        fail_predicates=fail_predicates,
        active_instance_ids=tuple(active_instance_ids),
    )
