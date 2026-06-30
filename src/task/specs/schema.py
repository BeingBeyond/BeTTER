from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredicateSpec:
    subject_id: str
    relation: str
    target_id: str


@dataclass(frozen=True)
class PredicateTemplateSpec:
    subjects_from: str | None = None
    relations: tuple[str, ...] = ()
    relation: str | None = None
    subject_id: str | None = None
    target_id: str | None = None


@dataclass(frozen=True)
class PolicyChannelSpec:
    mode: str
    predicates: tuple[PredicateTemplateSpec, ...] = ()
    append: tuple[PredicateSpec, ...] = ()


@dataclass(frozen=True)
class DefaultPolicySpec:
    success: PolicyChannelSpec
    fail: PolicyChannelSpec


@dataclass(frozen=True)
class PolicyOverrideChannelSpec:
    replace: tuple[PredicateSpec, ...] | None = None
    append: tuple[PredicateSpec, ...] = ()
    remove: tuple[PredicateSpec, ...] = ()


@dataclass(frozen=True)
class PolicyOverrideSpec:
    success: PolicyOverrideChannelSpec = field(default_factory=PolicyOverrideChannelSpec)
    fail: PolicyOverrideChannelSpec = field(default_factory=PolicyOverrideChannelSpec)


@dataclass(frozen=True)
class GroupOverrideSpec:
    set_members: tuple[str, ...] | None = None
    add_members: tuple[str, ...] = ()
    remove_members: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutSlotSpec:
    position: tuple[float, ...]
    rotation: tuple[float, ...]
    scale: tuple[float, ...]


@dataclass(frozen=True)
class AssetCandidateSpec:
    source_uid: str
    asset_key: str | None = None
    registry_usd: str | None = None
    registry_meta: str | None = None
    semantic_name: str | None = None


@dataclass(frozen=True)
class ObjectInstanceSpec:
    instance_id: str
    semantic_name: str
    retrieval_query: str
    description: str = ""
    tags: tuple[str, ...] = ()
    role: str | None = None


@dataclass(frozen=True)
class VariationSpec:
    variation_id: str
    extends: str | None = None
    instruction: str = ""
    enabled: bool = True
    group_overrides: dict[str, GroupOverrideSpec] = field(default_factory=dict)
    policy_overrides: PolicyOverrideSpec = field(default_factory=PolicyOverrideSpec)
    layout_slots: dict[str, LayoutSlotSpec] = field(default_factory=dict)
    goal_relation: str | None = None
    container_instance_id: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    template_type: str
    allowed_relations: tuple[str, ...]
    primary_goal_relation: str
    fail_relations: tuple[str, ...]
    container_instance_id: str
    object_instances: dict[str, ObjectInstanceSpec]
    semantic_groups: dict[str, tuple[str, ...]]
    candidate_pools: dict[str, tuple[str, ...]]
    default_policy: DefaultPolicySpec
    base_variation: VariationSpec
    variations: dict[str, VariationSpec]
    asset_bindings: dict[str, tuple[AssetCandidateSpec, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedVariationSpec:
    variation_id: str
    enabled: bool
    instruction: str
    container_instance_id: str
    goal_relation: str
    semantic_groups: dict[str, tuple[str, ...]]
    layout_slots: dict[str, LayoutSlotSpec]
    success_predicates: tuple[PredicateSpec, ...]
    fail_predicates: tuple[PredicateSpec, ...]
    active_instance_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedAssetSpec:
    source_uid: str
    asset_key: str
    asset_path: str
    registry_usd: str | None = None
    registry_meta: str | None = None


@dataclass(frozen=True)
class ResolvedObjectSpec:
    instance_id: str
    semantic_name: str
    prim_path: str
    asset: ResolvedAssetSpec
    position: tuple[float, ...]
    rotation: tuple[float, ...]
    scale: tuple[float, ...]
    role: str | None = None


@dataclass(frozen=True)
class ResolvedBackgroundSpec:
    scene_id: str | None = None
    material_id: str | None = None
    layout_id: str | None = None
    registry_path: str | None = None
    prim_path: str = "/World/Background"


@dataclass(frozen=True)
class ResolvedEpisodeSpec:
    task_id: str
    template_type: str
    variation_id: str
    instruction: str
    episode_seed: int
    objects: dict[str, ResolvedObjectSpec]
    success_predicates: tuple[PredicateSpec, ...]
    fail_predicates: tuple[PredicateSpec, ...]
    semantic_groups: dict[str, tuple[str, ...]]
    background: ResolvedBackgroundSpec | None = None
    schema_version: str = "resolved_episode.v1"
