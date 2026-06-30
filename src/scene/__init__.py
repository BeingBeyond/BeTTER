from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "SceneObject": (".objects", "SceneObject"),
    "RigidObject": (".objects", "RigidObject"),
    "ArticulatedObject": (".objects", "ArticulatedObject"),
    "SpatialEdge": (".layout", "SpatialEdge"),
    "SpatialGraph": (".layout", "SpatialGraph"),
    "Workspace": (".layout", "Workspace"),
    "LayoutSampler": (".layout", "LayoutSampler"),
    "SceneManagerV2": (".background", "SceneManagerV2"),
    "SceneManager": (".background", "SceneManager"),
    "BackgroundScene": (".background", "BackgroundScene"),
    "load_registry": (".background", "load_registry"),
    "EpisodeRuntimeLoadConfig": (".episode_runtime", "EpisodeRuntimeLoadConfig"),
    "RuntimeEpisodeHandle": (".episode_runtime", "RuntimeEpisodeHandle"),
    "RuntimeObjectHandle": (".episode_runtime", "RuntimeObjectHandle"),
    "RuntimeObjectPose": (".episode_runtime", "RuntimeObjectPose"),
    "capture_episode_object_poses": (".episode_runtime", "capture_episode_object_poses"),
    "EpisodeConditionExpressions": (".episode_conditions", "EpisodeConditionExpressions"),
    "EpisodeConditionObject": (".episode_conditions", "EpisodeConditionObject"),
    "EpisodeObjectTrajectory": (".episode_trajectory", "EpisodeObjectTrajectory"),
    "EpisodeTrajectoryFrame": (".episode_trajectory", "EpisodeTrajectoryFrame"),
    "RuntimeAABB": (".episode_conditions", "RuntimeAABB"),
    "build_episode_condition_context": (".episode_conditions", "build_episode_condition_context"),
    "build_episode_condition_evaluator": (".episode_conditions", "build_episode_condition_evaluator"),
    "build_episode_condition_expressions": (".episode_conditions", "build_episode_condition_expressions"),
    "evaluate_episode_conditions": (".episode_conditions", "evaluate_episode_conditions"),
    "episode_trajectory_from_dict": (".episode_trajectory", "episode_trajectory_from_dict"),
    "episode_trajectory_to_dict": (".episode_trajectory", "episode_trajectory_to_dict"),
    "load_episode_trajectory": (".episode_trajectory", "load_episode_trajectory"),
    "load_episode_background": (".episode_runtime", "load_episode_background"),
    "load_resolved_episode_into_stage": (".episode_runtime", "load_resolved_episode_into_stage"),
    "restore_episode_object_poses": (".episode_runtime", "restore_episode_object_poses"),
    "runtime_object_poses_from_dict": (".episode_runtime", "runtime_object_poses_from_dict"),
    "runtime_object_poses_to_dict": (".episode_runtime", "runtime_object_poses_to_dict"),
    "write_episode_trajectory": (".episode_trajectory", "write_episode_trajectory"),
    "CameraAnnotation": (".camera", "CameraAnnotation"),
    "CameraManager": (".camera", "CameraManager"),
    "CameraPoseMode": (".camera", "CameraPoseMode"),
    "CameraFollowTargetMode": (".camera", "CameraFollowTargetMode"),
    "CameraSpec": (".camera", "CameraSpec"),
    "RuntimeCamera": (".camera", "RuntimeCamera"),
    "load_camera_registry": (".camera", "load_registry"),
    "load_single_camera_definition": (".camera", "load_single_camera_definition"),
    "RobotBase": (".robots", "RobotBase"),
    "ROBOT_RUNTIME_STATE_SCHEMA_VERSION": (
        ".robots",
        "ROBOT_RUNTIME_STATE_SCHEMA_VERSION",
    ),
    "EmbodimentSpec": (".robots", "EmbodimentSpec"),
    "FrameRoleSummary": (".robots", "FrameRoleSummary"),
    "FrankaRobotiqFrameResolver": (".robots", "FrankaRobotiqFrameResolver"),
    "EmbodimentRegistry": (".robots", "EmbodimentRegistry"),
    "apply_embodiment_runtime_control": (".robots", "apply_embodiment_runtime_control"),
    "apply_joint_controller_profile": (".robots", "apply_joint_controller_profile"),
    "load_embodiment_registry": (".robots", "load_registry"),
    "load_controller_profile": (".robots", "load_controller_profile"),
    "resolve_embodiment_paths": (".robots", "resolve_embodiment_paths"),
    "resolve_runtime_control": (".robots", "resolve_runtime_control"),
    "create_robot": (".robots", "create_robot"),
    "with_gain_overrides": (".robots", "with_gain_overrides"),
    "PandaRobot": (".robots", "PandaRobot"),
    "FrankaRobotiqRobot": (".robots", "FrankaRobotiqRobot"),
    "EMBODIMENT_SCHEMA_VERSION": (".embodiment_schema", "EMBODIMENT_SCHEMA_VERSION"),
    "FRAME_ROLE_LEGACY_EEF": (".embodiment_schema", "FRAME_ROLE_LEGACY_EEF"),
    "FRAME_ROLE_TASK_TCP": (".embodiment_schema", "FRAME_ROLE_TASK_TCP"),
    "FRAME_ROLE_TOOL_BASE": (".embodiment_schema", "FRAME_ROLE_TOOL_BASE"),
    "Pose7": (".embodiment_schema", "Pose7"),
    "TaskTcpDeltaAction": (".embodiment_schema", "TaskTcpDeltaAction"),
    "make_task_tcp_delta_action": (".embodiment_schema", "make_task_tcp_delta_action"),
    "migrate_legacy_action_schema": (
        ".embodiment_schema",
        "migrate_legacy_action_schema",
    ),
    "migrate_legacy_robot_state_schema": (
        ".embodiment_schema",
        "migrate_legacy_robot_state_schema",
    ),
    "pose7_from_payload": (".embodiment_schema", "pose7_from_payload"),
    "task_tcp_delta_action_from_mapping": (
        ".embodiment_schema",
        "task_tcp_delta_action_from_mapping",
    ),
    "CartesianDeltaAction": (".controllers", "CartesianDeltaAction"),
    "ControllerCommand": (".controllers", "ControllerCommand"),
    "DeltaIKControllerConfig": (".controllers", "DeltaIKControllerConfig"),
    "FrankaRobotiqDeltaIKController": (
        ".controllers",
        "FrankaRobotiqDeltaIKController",
    ),
    "JsonlRobotStateLogger": (".recording", "JsonlRobotStateLogger"),
    "ROBOT_STATE_LOG_SCHEMA_VERSION": (
        ".recording",
        "ROBOT_STATE_LOG_SCHEMA_VERSION",
    ),
    "RobotStateSnapshotter": (".recording", "RobotStateSnapshotter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
