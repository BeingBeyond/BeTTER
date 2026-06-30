from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .context import ConditionContext
from .predicates import evaluate_edge_condition, evaluate_node_condition


@dataclass(frozen=True)
class EvalTraceEntry:
    kind: str
    expression: dict[str, Any]
    value: bool


@dataclass(frozen=True)
class EvalResult:
    value: bool
    trace: List[EvalTraceEntry]


class ConditionKernel:
    def evaluate(self, ctx: ConditionContext, expression: dict[str, Any] | list[dict[str, Any]]) -> EvalResult:
        trace: List[EvalTraceEntry] = []

        if isinstance(expression, list):
            value = self._eval_and(ctx, expression, trace)
            return EvalResult(value=value, trace=trace)

        value = self._eval_node(ctx, expression, trace)
        return EvalResult(value=value, trace=trace)

    def _eval_node(self, ctx: ConditionContext, node: dict[str, Any], trace: List[EvalTraceEntry]) -> bool:
        if "op" in node:
            op = str(node["op"])
            if op == "not":
                child = node.get("condition")
                if not isinstance(child, dict):
                    raise ValueError("NOT operator requires dict field 'condition'.")
                value = not self._eval_node(ctx, child, trace)
                trace.append(EvalTraceEntry(kind="logic", expression=node, value=value))
                return value

            if op == "and":
                conditions = node.get("conditions")
                if conditions is None:
                    conditions = []
                if not isinstance(conditions, list):
                    raise ValueError("AND operator requires list field 'conditions'.")
                value = self._eval_and(ctx, conditions, trace)
                trace.append(EvalTraceEntry(kind="logic", expression=node, value=value))
                return value

            if op == "or":
                conditions = node.get("conditions")
                if conditions is None:
                    conditions = []
                if not isinstance(conditions, list):
                    raise ValueError("OR operator requires list field 'conditions'.")
                value = self._eval_or(ctx, conditions, trace)
                trace.append(EvalTraceEntry(kind="logic", expression=node, value=value))
                return value

            raise ValueError(f"Unsupported logical operator '{op}'")

        node_type = node.get("type")
        if node_type == "node":
            value = evaluate_node_condition(ctx, node)
        elif node_type == "edge" or "relation" in node:
            value = evaluate_edge_condition(ctx, node)
        else:
            raise ValueError(f"Unsupported condition node type '{node_type}'")

        trace.append(EvalTraceEntry(kind="atomic", expression=node, value=bool(value)))
        return bool(value)

    def _eval_and(self, ctx: ConditionContext, nodes: list[dict[str, Any]], trace: List[EvalTraceEntry]) -> bool:
        if len(nodes) == 0:
            return True
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError(f"Condition node must be dict, got: {type(node)}")
            if not self._eval_node(ctx, node, trace):
                return False
        return True

    def _eval_or(self, ctx: ConditionContext, nodes: list[dict[str, Any]], trace: List[EvalTraceEntry]) -> bool:
        if len(nodes) == 0:
            return False
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError(f"Condition node must be dict, got: {type(node)}")
            if self._eval_node(ctx, node, trace):
                return True
        return False
