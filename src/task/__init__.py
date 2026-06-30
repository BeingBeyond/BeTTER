__all__ = [
    "ConditionContext",
    "ConditionKernel",
    "EvalResult",
    "RelationThresholds",
    "TaskEvaluator",
    "TaskEvaluatorConfig",
    "StepEvaluation",
]


def __getattr__(name: str):
    if name in {"ConditionContext", "ConditionKernel", "EvalResult", "RelationThresholds"}:
        from . import conditions

        return getattr(conditions, name)
    if name in {"TaskEvaluator", "TaskEvaluatorConfig", "StepEvaluation"}:
        from . import evaluation

        return getattr(evaluation, name)
    raise AttributeError(f"module 'src.task' has no attribute {name!r}")
