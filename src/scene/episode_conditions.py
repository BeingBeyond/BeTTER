from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.task.conditions import ConditionContext, ConditionKernel
from src.task.conditions.schema import RelationThresholds
from src.task.evaluation import StepEvaluation, TaskEvaluator, TaskEvaluatorConfig
from src.task.specs.schema import PredicateSpec, ResolvedEpisodeSpec


XformPrimFactory = Callable[[str], Any]
BoundingBoxReader = Callable[[Any, str], tuple[tuple[float, ...], tuple[float, ...]]]


@dataclass(frozen=True)
class RuntimeAABB:
    min_bound: tuple[float, ...]
    max_bound: tuple[float, ...]

    def get_min_bound(self) -> tuple[float, ...]:
        return self.min_bound

    def get_max_bound(self) -> tuple[float, ...]:
        return self.max_bound


@dataclass(frozen=True)
class EpisodeConditionObject:
    instance_id: str
    prim_path: str
    stage: Any
    xform_prim_factory: XformPrimFactory | None = None
    bounding_box_reader: BoundingBoxReader | None = None

    def get_world_pose(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        xform = (self.xform_prim_factory or _default_xform_prim_factory)(self.prim_path)
        position, orientation = xform.get_world_pose()
        return _float_tuple(position, length=3), _float_tuple(orientation, length=4)

    def get_bounding_box(self) -> RuntimeAABB:
        reader = self.bounding_box_reader or _default_bounding_box_reader
        min_bound, max_bound = reader(self.stage, self.prim_path)
        return RuntimeAABB(
            min_bound=_float_tuple(min_bound, length=3),
            max_bound=_float_tuple(max_bound, length=3),
        )


@dataclass(frozen=True)
class EpisodeConditionExpressions:
    goal_expression: dict[str, object] | None
    fail_expression: dict[str, object] | None


def predicate_to_condition_dict(predicate: PredicateSpec) -> dict[str, str]:
    return {
        "type": "edge",
        "subject_id": predicate.subject_id,
        "relation": predicate.relation,
        "target_id": predicate.target_id,
    }


def build_episode_condition_expressions(episode: ResolvedEpisodeSpec) -> EpisodeConditionExpressions:
    goal_expression: dict[str, object] | None = {
        "op": "and",
        "conditions": [predicate_to_condition_dict(predicate) for predicate in episode.success_predicates],
    }
    fail_expression: dict[str, object] | None = None
    if episode.fail_predicates:
        fail_expression = {
            "op": "or",
            "conditions": [predicate_to_condition_dict(predicate) for predicate in episode.fail_predicates],
        }
    return EpisodeConditionExpressions(
        goal_expression=goal_expression,
        fail_expression=fail_expression,
    )


def build_episode_condition_evaluator(
    episode: ResolvedEpisodeSpec,
    *,
    config: TaskEvaluatorConfig | Mapping[str, Any] | None = None,
) -> TaskEvaluator:
    expressions = build_episode_condition_expressions(episode)
    if config is None or isinstance(config, TaskEvaluatorConfig):
        evaluator_config = config
    else:
        evaluator_config = TaskEvaluator.from_config_dict(dict(config))
    return TaskEvaluator(
        goal_expression=expressions.goal_expression,
        fail_expression=expressions.fail_expression,
        config=evaluator_config,
    )


def build_episode_condition_context(
    stage: Any,
    episode: ResolvedEpisodeSpec,
    *,
    thresholds: RelationThresholds | None = None,
    xform_prim_factory: XformPrimFactory | None = None,
    bounding_box_reader: BoundingBoxReader | None = None,
) -> ConditionContext:
    objects = {
        instance_id: EpisodeConditionObject(
            instance_id=instance_id,
            prim_path=obj.prim_path,
            stage=stage,
            xform_prim_factory=xform_prim_factory,
            bounding_box_reader=bounding_box_reader,
        )
        for instance_id, obj in episode.objects.items()
    }
    return ConditionContext.from_objects(objects=objects, thresholds=thresholds)


def evaluate_episode_conditions(
    stage: Any,
    episode: ResolvedEpisodeSpec,
    *,
    evaluator: TaskEvaluator | None = None,
    thresholds: RelationThresholds | None = None,
    xform_prim_factory: XformPrimFactory | None = None,
    bounding_box_reader: BoundingBoxReader | None = None,
) -> StepEvaluation:
    active_evaluator = evaluator or build_episode_condition_evaluator(episode)
    ctx = build_episode_condition_context(
        stage,
        episode,
        thresholds=thresholds,
        xform_prim_factory=xform_prim_factory,
        bounding_box_reader=bounding_box_reader,
    )
    return active_evaluator.step(ctx)


def evaluate_episode_condition_details(
    stage: Any,
    episode: ResolvedEpisodeSpec,
    *,
    thresholds: RelationThresholds | None = None,
    xform_prim_factory: XformPrimFactory | None = None,
    bounding_box_reader: BoundingBoxReader | None = None,
) -> dict[str, list[dict[str, Any]]]:
    ctx = build_episode_condition_context(
        stage,
        episode,
        thresholds=thresholds,
        xform_prim_factory=xform_prim_factory,
        bounding_box_reader=bounding_box_reader,
    )
    kernel = ConditionKernel()
    return {
        "goal": [
            _predicate_detail(
                kernel=kernel,
                ctx=ctx,
                predicate=predicate,
                value_key="satisfied",
            )
            for predicate in episode.success_predicates
        ],
        "fail": [
            _predicate_detail(
                kernel=kernel,
                ctx=ctx,
                predicate=predicate,
                value_key="triggered",
            )
            for predicate in episode.fail_predicates
        ],
    }


def _predicate_detail(
    *,
    kernel: ConditionKernel,
    ctx: ConditionContext,
    predicate: PredicateSpec,
    value_key: str,
) -> dict[str, Any]:
    expression = predicate_to_condition_dict(predicate)
    result = kernel.evaluate(ctx, expression)
    return {
        "description": f"{predicate.subject_id} {predicate.relation} {predicate.target_id}",
        "subject_id": predicate.subject_id,
        "relation": predicate.relation,
        "target_id": predicate.target_id,
        value_key: bool(result.value),
    }


def _default_xform_prim_factory(prim_path: str) -> Any:
    from omni.isaac.core.prims import XFormPrim  # type: ignore[import]

    return XFormPrim(prim_path)


def _default_bounding_box_reader(stage: Any, prim_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    from pxr import Usd, UsdGeom  # type: ignore[import]

    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or (hasattr(prim, "IsValid") and not prim.IsValid()):
        raise RuntimeError(f"Invalid prim for condition bounding box read: {prim_path}")

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    aligned_box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return _float_tuple(aligned_box.GetMin(), length=3), _float_tuple(aligned_box.GetMax(), length=3)


def _float_tuple(value: object, *, length: int) -> tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        try:
            value = tuple(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"Expected sequence of length {length}.") from exc
    out = tuple(float(item) for item in value)
    if len(out) != length:
        raise ValueError(f"Expected sequence of length {length}, got {len(out)}.")
    return out
