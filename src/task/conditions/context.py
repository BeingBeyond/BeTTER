from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.scene.objects.base import SceneObject
else:
    SceneObject = Any

from .schema import RelationThresholds


@dataclass
class ConditionContext:
    objects: Dict[str, SceneObject]
    thresholds: RelationThresholds

    @classmethod
    def from_objects(
        cls,
        objects: Dict[str, SceneObject],
        thresholds: RelationThresholds | None = None,
    ) -> "ConditionContext":
        return cls(objects=dict(objects), thresholds=thresholds or RelationThresholds())

    def get_object(self, object_id: str) -> SceneObject:
        if object_id not in self.objects:
            raise KeyError(f"Unknown object id in condition: {object_id}")
        return self.objects[object_id]
