from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.task.conditions import ConditionContext, ConditionKernel

from .temporal import ConsecutiveCounter


@dataclass(frozen=True)
class TaskEvaluatorConfig:
    goal_consecutive_steps: int = 5
    fail_consecutive_steps: int = 1


@dataclass(frozen=True)
class StepEvaluation:
    goal_raw: bool
    goal_streak: int
    goal_passed: bool
    fail_raw: bool
    fail_streak: int
    fail_passed: bool
    terminated: bool
    termination_type: str
    reason: str


class TaskEvaluator:
    def __init__(
        self,
        goal_expression: Optional[dict[str, Any] | list[dict[str, Any]]],
        fail_expression: Optional[dict[str, Any] | list[dict[str, Any]]],
        config: TaskEvaluatorConfig | None = None,
        kernel: ConditionKernel | None = None,
    ) -> None:
        self._goal_expression = self._normalize_goal_expression(goal_expression)
        self._fail_expression = self._normalize_fail_expression(fail_expression)
        self._config = config or TaskEvaluatorConfig()
        self._kernel = kernel or ConditionKernel()
        self._goal_counter = ConsecutiveCounter(self._config.goal_consecutive_steps)
        self._fail_counter = ConsecutiveCounter(self._config.fail_consecutive_steps)

    def step(self, ctx: ConditionContext) -> StepEvaluation:
        goal_raw = True
        if self._goal_expression is not None:
            goal_raw = self._kernel.evaluate(ctx, self._goal_expression).value

        fail_raw = False
        if self._fail_expression is not None:
            fail_raw = self._kernel.evaluate(ctx, self._fail_expression).value

        goal_streak, goal_passed = self._goal_counter.update(goal_raw)
        fail_streak, fail_passed = self._fail_counter.update(fail_raw)

        if fail_passed:
            return StepEvaluation(
                goal_raw=goal_raw,
                goal_streak=goal_streak,
                goal_passed=goal_passed,
                fail_raw=fail_raw,
                fail_streak=fail_streak,
                fail_passed=fail_passed,
                terminated=True,
                termination_type="failure",
                reason="fail_condition_sustained",
            )

        if goal_passed:
            return StepEvaluation(
                goal_raw=goal_raw,
                goal_streak=goal_streak,
                goal_passed=goal_passed,
                fail_raw=fail_raw,
                fail_streak=fail_streak,
                fail_passed=fail_passed,
                terminated=True,
                termination_type="success",
                reason="goal_condition_sustained",
            )

        return StepEvaluation(
            goal_raw=goal_raw,
            goal_streak=goal_streak,
            goal_passed=goal_passed,
            fail_raw=fail_raw,
            fail_streak=fail_streak,
            fail_passed=fail_passed,
            terminated=False,
            termination_type="none",
            reason="running",
        )

    @staticmethod
    def from_config_dict(config: Dict[str, Any]) -> TaskEvaluatorConfig:
        return TaskEvaluatorConfig(
            goal_consecutive_steps=int(config.get("goal_consecutive_steps", 5)),
            fail_consecutive_steps=int(config.get("fail_consecutive_steps", 1)),
        )

    @staticmethod
    def _normalize_goal_expression(
        expression: Optional[dict[str, Any] | list[dict[str, Any]]]
    ) -> Optional[dict[str, Any] | list[dict[str, Any]]]:
        if isinstance(expression, list):
            return {"op": "and", "conditions": expression}
        return expression

    @staticmethod
    def _normalize_fail_expression(
        expression: Optional[dict[str, Any] | list[dict[str, Any]]]
    ) -> Optional[dict[str, Any] | list[dict[str, Any]]]:
        if isinstance(expression, list):
            return {"op": "or", "conditions": expression}
        return expression
