from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .resolver import resolve_variation
from .schema import (
    PolicyOverrideChannelSpec,
    PolicyOverrideSpec,
    PredicateSpec,
    ResolvedVariationSpec,
    TaskSpec,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _predicate_dict(predicate: PredicateSpec) -> dict[str, str]:
    return {
        "type": "edge",
        "subject_id": predicate.subject_id,
        "relation": predicate.relation,
        "target_id": predicate.target_id,
    }


def _predicates_from_expression(expression: dict) -> tuple[PredicateSpec, ...]:
    conditions = expression.get("conditions", []) or []
    out: list[PredicateSpec] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        subject_id = condition.get("subject_id")
        relation = condition.get("relation")
        target_id = condition.get("target_id")
        if isinstance(subject_id, str) and isinstance(relation, str) and isinstance(target_id, str):
            out.append(PredicateSpec(subject_id=subject_id, relation=relation, target_id=target_id))
    return tuple(sorted(set(out), key=lambda p: (p.subject_id, p.relation, p.target_id)))


def _default_success_predicates(resolved: ResolvedVariationSpec) -> tuple[PredicateSpec, ...]:
    return tuple(
        PredicateSpec(
            subject_id=subject_id,
            relation=resolved.goal_relation,
            target_id=resolved.container_instance_id,
        )
        for subject_id in sorted(resolved.semantic_groups.get("target", ()))
    )


def _default_fail_predicates(resolved: ResolvedVariationSpec, fail_relations: tuple[str, ...]) -> tuple[PredicateSpec, ...]:
    return tuple(
        PredicateSpec(
            subject_id=subject_id,
            relation=relation,
            target_id=resolved.container_instance_id,
        )
        for subject_id in sorted(resolved.semantic_groups.get("distractor", ()))
        for relation in fail_relations
    )


def infer_policy_override_from_compiled(
    task_dir: Path,
    variation_id: str,
    resolved_defaults: ResolvedVariationSpec,
    fail_relations: tuple[str, ...],
) -> PolicyOverrideSpec:
    compiled_dir = task_dir / "compiled" / variation_id
    if not compiled_dir.exists():
        return PolicyOverrideSpec()

    success_expr = _load_json(compiled_dir / "success.json")
    fail_expr = _load_json(compiled_dir / "fail.json")

    compiled_success = _predicates_from_expression(success_expr)
    compiled_fail = _predicates_from_expression(fail_expr)
    default_success = _default_success_predicates(resolved_defaults)
    default_fail = _default_fail_predicates(resolved_defaults, fail_relations)

    success_append = tuple(sorted(set(compiled_success) - set(default_success), key=lambda p: (p.subject_id, p.relation, p.target_id)))
    success_remove = tuple(sorted(set(default_success) - set(compiled_success), key=lambda p: (p.subject_id, p.relation, p.target_id)))
    fail_append = tuple(sorted(set(compiled_fail) - set(default_fail), key=lambda p: (p.subject_id, p.relation, p.target_id)))
    fail_remove = tuple(sorted(set(default_fail) - set(compiled_fail), key=lambda p: (p.subject_id, p.relation, p.target_id)))

    return PolicyOverrideSpec(
        success=PolicyOverrideChannelSpec(append=success_append, remove=success_remove),
        fail=PolicyOverrideChannelSpec(append=fail_append, remove=fail_remove),
    )


def apply_compiled_policy_overrides(task_spec: TaskSpec, task_dir: Path) -> TaskSpec:
    updated_variations = {}
    for variation_id, variation_spec in task_spec.variations.items():
        resolved_defaults = resolve_variation(task_spec, variation_id)
        policy_overrides = infer_policy_override_from_compiled(
            task_dir=task_dir,
            variation_id=variation_id,
            resolved_defaults=resolved_defaults,
            fail_relations=task_spec.fail_relations,
        )
        updated_variations[variation_id] = replace(variation_spec, policy_overrides=policy_overrides)
    return replace(task_spec, variations=dict(sorted(updated_variations.items())))


def build_success_expression(resolved: ResolvedVariationSpec) -> dict[str, object]:
    return {
        "op": "and",
        "conditions": [_predicate_dict(predicate) for predicate in resolved.success_predicates],
    }


def build_fail_expression(resolved: ResolvedVariationSpec) -> dict[str, object]:
    return {
        "op": "or",
        "conditions": [_predicate_dict(predicate) for predicate in resolved.fail_predicates],
    }


def build_logic_graph(task_name: str, resolved: ResolvedVariationSpec) -> dict[str, object]:
    return {
        "task_name": task_name,
        "variation_id": resolved.variation_id,
        "nodes": [
            {
                "node_id": predicate.subject_id.split("_1_")[0] if "_1_" in predicate.subject_id else predicate.subject_id,
                "type": "maintenance",
                "parents": [],
                "condition": {
                    "type": "spatial",
                    "subject_id": predicate.subject_id,
                    "relation": predicate.relation,
                    "target_id": predicate.target_id,
                },
                "target_object": predicate.subject_id,
            }
            for predicate in resolved.success_predicates
        ],
    }
