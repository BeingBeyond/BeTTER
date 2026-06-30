from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path

from .asset_registry import AssetRegistryIndex
from .resolver import resolve_variation
from .schema import (
    AssetCandidateSpec,
    PredicateSpec,
    ResolvedAssetSpec,
    ResolvedBackgroundSpec,
    ResolvedEpisodeSpec,
    ResolvedObjectSpec,
    TaskSpec,
)


def resolve_episode(
    task_spec: TaskSpec,
    variation_id: str,
    *,
    asset_registry_root: str | Path,
    episode_seed: int,
    background: ResolvedBackgroundSpec | Mapping[str, object] | None = None,
    prim_root: str = "/World/Objects",
    selected_asset_ids: Mapping[str, str] | None = None,
) -> ResolvedEpisodeSpec:
    """Compile a semantic variation into deterministic runtime episode state."""

    resolved_variation = resolve_variation(task_spec, variation_id)
    rng = random.Random(int(episode_seed))
    index = AssetRegistryIndex(asset_registry_root)
    selected_assets = dict(selected_asset_ids or {})
    objects: dict[str, ResolvedObjectSpec] = {}

    for instance_id in resolved_variation.active_instance_ids:
        slot = resolved_variation.layout_slots.get(instance_id)
        if slot is None:
            raise ValueError(
                f"Active instance '{instance_id}' has no resolved layout slot in variation '{variation_id}'"
            )

        candidates = _candidates_for_instance(task_spec, instance_id)
        if not candidates:
            raise ValueError(f"Active instance '{instance_id}' has no asset candidates")

        candidate = _select_candidate(
            instance_id=instance_id,
            candidates=candidates,
            rng=rng,
            selected_asset_ids=selected_assets,
        )
        asset = index.resolve_candidate(task_spec, instance_id, candidate).to_spec()
        object_spec = task_spec.object_instances.get(instance_id)
        semantic_name = object_spec.semantic_name if object_spec is not None else instance_id
        role = object_spec.role if object_spec is not None else None

        objects[instance_id] = ResolvedObjectSpec(
            instance_id=instance_id,
            semantic_name=semantic_name,
            role=role,
            prim_path=f"{prim_root.rstrip('/')}/{instance_id}",
            asset=asset,
            position=tuple(float(value) for value in slot.position),
            rotation=tuple(float(value) for value in slot.rotation),
            scale=tuple(float(value) for value in slot.scale),
        )

    return ResolvedEpisodeSpec(
        task_id=task_spec.task_id,
        template_type=task_spec.template_type,
        variation_id=resolved_variation.variation_id,
        instruction=resolved_variation.instruction,
        episode_seed=int(episode_seed),
        background=_coerce_background(background),
        objects=dict(sorted(objects.items())),
        success_predicates=resolved_variation.success_predicates,
        fail_predicates=resolved_variation.fail_predicates,
        semantic_groups=resolved_variation.semantic_groups,
    )


def resolved_episode_to_dict(episode: ResolvedEpisodeSpec) -> dict[str, object]:
    return {
        "schema_version": episode.schema_version,
        "task_id": episode.task_id,
        "template_type": episode.template_type,
        "variation_id": episode.variation_id,
        "instruction": episode.instruction,
        "episode_seed": int(episode.episode_seed),
        "background": _background_to_dict(episode.background),
        "semantic_groups": {
            name: list(members) for name, members in sorted(episode.semantic_groups.items())
        },
        "success_predicates": [_predicate_to_dict(predicate) for predicate in episode.success_predicates],
        "fail_predicates": [_predicate_to_dict(predicate) for predicate in episode.fail_predicates],
        "objects": {
            instance_id: _object_to_dict(obj)
            for instance_id, obj in sorted(episode.objects.items())
        },
    }


def resolved_episode_from_dict(payload: Mapping[str, object]) -> ResolvedEpisodeSpec:
    objects_payload = payload.get("objects", {})
    if not isinstance(objects_payload, Mapping):
        raise ValueError("Resolved episode payload field 'objects' must be a mapping")

    return ResolvedEpisodeSpec(
        schema_version=str(payload.get("schema_version") or "resolved_episode.v1"),
        task_id=str(payload["task_id"]),
        template_type=str(payload["template_type"]),
        variation_id=str(payload["variation_id"]),
        instruction=str(payload.get("instruction") or ""),
        episode_seed=int(payload.get("episode_seed") or 0),
        background=_background_from_payload(payload.get("background")),
        semantic_groups=_semantic_groups_from_payload(payload.get("semantic_groups")),
        success_predicates=_predicates_from_payload(payload.get("success_predicates")),
        fail_predicates=_predicates_from_payload(payload.get("fail_predicates")),
        objects={
            str(instance_id): _object_from_payload(str(instance_id), object_payload)
            for instance_id, object_payload in objects_payload.items()
        },
    )


def write_resolved_episode(episode: ResolvedEpisodeSpec, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(resolved_episode_to_dict(episode), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_resolved_episode(path: str | Path) -> ResolvedEpisodeSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Resolved episode file must contain a JSON object: {path}")
    return resolved_episode_from_dict(payload)


def _candidates_for_instance(
    task_spec: TaskSpec,
    instance_id: str,
) -> tuple[AssetCandidateSpec, ...]:
    if instance_id in task_spec.asset_bindings:
        return task_spec.asset_bindings[instance_id]
    source_uid_candidates = tuple(
        AssetCandidateSpec(source_uid=source_uid)
        for source_uid in task_spec.candidate_pools.get(instance_id, ())
    )
    if source_uid_candidates:
        return source_uid_candidates
    return ()


def _select_candidate(
    *,
    instance_id: str,
    candidates: tuple[AssetCandidateSpec, ...],
    rng: random.Random,
    selected_asset_ids: Mapping[str, str],
) -> AssetCandidateSpec:
    selected = selected_asset_ids.get(instance_id)
    if selected:
        for candidate in candidates:
            if selected in {candidate.source_uid, candidate.asset_key}:
                return candidate
        raise ValueError(
            f"Selected asset id for '{instance_id}' is not in its candidate pool: {selected}"
        )
    return rng.choice(candidates)


def _coerce_background(
    background: ResolvedBackgroundSpec | Mapping[str, object] | None,
) -> ResolvedBackgroundSpec | None:
    if background is None or isinstance(background, ResolvedBackgroundSpec):
        return background
    return _background_from_payload(background)


def _background_to_dict(background: ResolvedBackgroundSpec | None) -> dict[str, object] | None:
    if background is None:
        return None
    return {
        "scene_id": background.scene_id,
        "material_id": background.material_id,
        "layout_id": background.layout_id,
        "registry_path": background.registry_path,
        "prim_path": background.prim_path,
    }


def _background_from_payload(payload: object) -> ResolvedBackgroundSpec | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("Resolved episode background must be a mapping or null")
    return ResolvedBackgroundSpec(
        scene_id=str(payload["scene_id"]) if payload.get("scene_id") is not None else None,
        material_id=str(payload["material_id"]) if payload.get("material_id") is not None else None,
        layout_id=str(payload["layout_id"]) if payload.get("layout_id") is not None else None,
        registry_path=str(payload["registry_path"]) if payload.get("registry_path") is not None else None,
        prim_path=str(payload.get("prim_path") or "/World/Background"),
    )


def _object_to_dict(obj: ResolvedObjectSpec) -> dict[str, object]:
    return {
        "instance_id": obj.instance_id,
        "semantic_name": obj.semantic_name,
        "role": obj.role,
        "prim_path": obj.prim_path,
        "asset": {
            "source_uid": obj.asset.source_uid,
            "asset_key": obj.asset.asset_key,
            "asset_path": obj.asset.asset_path,
            "registry_usd": obj.asset.registry_usd,
            "registry_meta": obj.asset.registry_meta,
        },
        "pose": {
            "position": list(obj.position),
            "rotation": list(obj.rotation),
            "scale": list(obj.scale),
        },
    }


def _object_from_payload(instance_id: str, payload: object) -> ResolvedObjectSpec:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Resolved episode object '{instance_id}' must be a mapping")
    asset_payload = payload.get("asset")
    pose_payload = payload.get("pose")
    if not isinstance(asset_payload, Mapping):
        raise ValueError(f"Resolved episode object '{instance_id}'.asset must be a mapping")
    if not isinstance(pose_payload, Mapping):
        raise ValueError(f"Resolved episode object '{instance_id}'.pose must be a mapping")

    return ResolvedObjectSpec(
        instance_id=str(payload.get("instance_id") or instance_id),
        semantic_name=str(payload.get("semantic_name") or instance_id),
        role=str(payload["role"]) if payload.get("role") is not None else None,
        prim_path=str(payload["prim_path"]),
        asset=ResolvedAssetSpec(
            source_uid=str(asset_payload["source_uid"]),
            asset_key=str(asset_payload["asset_key"]),
            asset_path=str(asset_payload["asset_path"]),
            registry_usd=str(asset_payload["registry_usd"])
            if asset_payload.get("registry_usd") is not None
            else None,
            registry_meta=str(asset_payload["registry_meta"])
            if asset_payload.get("registry_meta") is not None
            else None,
        ),
        position=_float_tuple(pose_payload.get("position")),
        rotation=_float_tuple(pose_payload.get("rotation")),
        scale=_float_tuple(pose_payload.get("scale")),
    )


def _predicate_to_dict(predicate: PredicateSpec) -> dict[str, str]:
    return {
        "subject_id": predicate.subject_id,
        "relation": predicate.relation,
        "target_id": predicate.target_id,
    }


def _predicates_from_payload(payload: object) -> tuple[PredicateSpec, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError("Resolved episode predicates must be a list")
    return tuple(
        PredicateSpec(
            subject_id=str(item["subject_id"]),
            relation=str(item["relation"]),
            target_id=str(item["target_id"]),
        )
        for item in payload
        if isinstance(item, Mapping)
    )


def _semantic_groups_from_payload(payload: object) -> dict[str, tuple[str, ...]]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("Resolved episode semantic_groups must be a mapping")
    return {
        str(name): tuple(str(member) for member in members)
        for name, members in payload.items()
        if isinstance(members, list)
    }


def _float_tuple(payload: object) -> tuple[float, ...]:
    if not isinstance(payload, list):
        raise ValueError("Pose fields must be lists")
    return tuple(float(value) for value in payload)
