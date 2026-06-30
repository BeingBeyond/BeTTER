from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


EMBODIMENT_SCHEMA_VERSION = "better.embodiment.schema.v1"

FRAME_ROLE_LEGACY_EEF = "legacy_eef_frame"
FRAME_ROLE_TOOL_BASE = "tool_base_frame"
FRAME_ROLE_TASK_TCP = "task_tcp_frame"

LEGACY_EEF_POSE_KEY = "eef_pose"
LEGACY_EE_POSE_KEY = "ee_pose"
LEGACY_ACTION_LABEL_KEY = "label"
LEGACY_EEF_DELTA_POSE_KEY = "legacy_eef_delta_pose"
LEGACY_TRAINING_LABEL_KEY = "legacy_training_label"
LEGACY_EEF_POSE_WORLD_KEY = "legacy_eef_pose_world"
TOOL_BASE_POSE_WORLD_KEY = "tool_base_pose_world"
TASK_TCP_POSE_WORLD_KEY = "task_tcp_pose_world"
DELTA_TASK_TCP_POSE_KEY = "delta_task_tcp_pose"
GRIPPER_COMMAND_KEY = "gripper_command"


@dataclass(frozen=True)
class Pose7:
    """World pose with Isaac-style quaternion order."""

    position: np.ndarray
    orientation_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _vector(self.position, name="position", length=3),
        )
        object.__setattr__(
            self,
            "orientation_wxyz",
            _vector(self.orientation_wxyz, name="orientation_wxyz", length=4),
        )

    def to_mapping(self) -> dict[str, np.ndarray]:
        return {
            "position": self.position.copy(),
            "orientation": self.orientation_wxyz.copy(),
        }


@dataclass(frozen=True)
class TaskTcpDeltaAction:
    """Canonical Cartesian action for task-TCP controllers."""

    delta_task_tcp_pose: np.ndarray
    gripper: float | None = None
    reference_frame: str = "task_tcp"
    frame_role: str = FRAME_ROLE_TASK_TCP
    schema_version: str = EMBODIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_task_tcp_pose",
            _vector(
                self.delta_task_tcp_pose,
                name=DELTA_TASK_TCP_POSE_KEY,
                length=6,
            ),
        )
        if self.gripper is not None:
            object.__setattr__(self, "gripper", float(self.gripper))

    @property
    def translation(self) -> np.ndarray:
        return self.delta_task_tcp_pose[:3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.delta_task_tcp_pose[3:].copy()

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            DELTA_TASK_TCP_POSE_KEY: self.delta_task_tcp_pose.copy(),
            "reference_frame": self.reference_frame,
            "frame_role": self.frame_role,
        }
        if self.gripper is not None:
            payload[GRIPPER_COMMAND_KEY] = {
                "mode": "legacy_binary",
                "value": float(self.gripper),
            }
        return payload


def make_task_tcp_delta_action(
    translation: Any,
    rotation: Any,
    *,
    gripper: float | None = None,
    reference_frame: str = "task_tcp",
) -> dict[str, Any]:
    delta = np.concatenate(
        [
            _vector(translation, name="translation", length=3),
            _vector(rotation, name="rotation", length=3),
        ]
    ).astype(np.float32)
    return TaskTcpDeltaAction(
        delta_task_tcp_pose=delta,
        gripper=gripper,
        reference_frame=reference_frame,
    ).to_mapping()


def task_tcp_delta_action_from_mapping(payload: Mapping[str, Any]) -> TaskTcpDeltaAction:
    if DELTA_TASK_TCP_POSE_KEY in payload:
        delta = _vector(payload[DELTA_TASK_TCP_POSE_KEY], name=DELTA_TASK_TCP_POSE_KEY, length=6)
    elif "translation" in payload and "rotation" in payload:
        delta = np.concatenate(
            [
                _vector(payload["translation"], name="translation", length=3),
                _vector(payload["rotation"], name="rotation", length=3),
            ]
        ).astype(np.float32)
    else:
        raise KeyError(
            "Task TCP delta action requires either 'delta_task_tcp_pose' or "
            "'translation' + 'rotation'."
        )

    return TaskTcpDeltaAction(
        delta_task_tcp_pose=delta,
        gripper=_extract_gripper(payload),
        reference_frame=str(payload.get("reference_frame", "task_tcp")),
        frame_role=str(payload.get("frame_role", FRAME_ROLE_TASK_TCP)),
        schema_version=str(payload.get("schema_version", EMBODIMENT_SCHEMA_VERSION)),
    )


def migrate_legacy_action_schema(
    action: Mapping[str, Any],
    *,
    legacy_frame_role: str = FRAME_ROLE_LEGACY_EEF,
) -> dict[str, Any]:
    """Return a schema-annotated action copy without mutating legacy data."""

    migrated = deepcopy(dict(action))
    migrated.setdefault("schema_version", EMBODIMENT_SCHEMA_VERSION)
    frame_schema = deepcopy(migrated.get("frame_schema", {}))

    delta_source_key = _first_present_key(
        action,
        (LEGACY_EEF_POSE_KEY, LEGACY_EE_POSE_KEY, LEGACY_ACTION_LABEL_KEY),
    )
    if delta_source_key is not None:
        legacy_delta = np.asarray(action[delta_source_key], dtype=np.float32).copy()
        migrated.setdefault(LEGACY_EEF_DELTA_POSE_KEY, legacy_delta)
        frame_schema[LEGACY_EEF_DELTA_POSE_KEY] = {
            "frame_role": legacy_frame_role,
            "source_key": delta_source_key,
            "status": "legacy_not_task_tcp",
        }
        migrated.setdefault("requires_task_tcp_action_migration", True)

    if LEGACY_ACTION_LABEL_KEY in action:
        migrated.setdefault(
            LEGACY_TRAINING_LABEL_KEY,
            np.asarray(action[LEGACY_ACTION_LABEL_KEY], dtype=np.float32).copy(),
        )

    if "gripper_closeness" in action and GRIPPER_COMMAND_KEY not in migrated:
        gripper = _vector(action["gripper_closeness"], name="gripper_closeness", length=1)
        migrated[GRIPPER_COMMAND_KEY] = {
            "mode": "closeness",
            "value": float(gripper[0]),
            "source_key": "gripper_closeness",
        }
    elif "gripper_action" in action and GRIPPER_COMMAND_KEY not in migrated:
        gripper = _vector(action["gripper_action"], name="gripper_action", length=1)
        migrated[GRIPPER_COMMAND_KEY] = {
            "mode": "legacy_binary",
            "value": float(gripper[0]),
            "source_key": "gripper_action",
        }
    elif isinstance(action.get("execution"), Mapping) and GRIPPER_COMMAND_KEY not in migrated:
        execution = action["execution"]
        if "gripper_action" in execution:
            gripper = _vector(
                execution["gripper_action"],
                name="execution.gripper_action",
                length=1,
            )
            migrated[GRIPPER_COMMAND_KEY] = {
                "mode": "legacy_binary",
                "value": float(gripper[0]),
                "source_key": "execution.gripper_action",
            }

    if frame_schema:
        migrated["frame_schema"] = frame_schema
    return migrated


def migrate_legacy_robot_state_schema(
    robot_state: Mapping[str, Any],
    *,
    frame_resolver: Any | None = None,
    legacy_eef_is_tool_base: bool = True,
) -> dict[str, Any]:
    """Return a schema-annotated robot-state copy without editing a trajectory."""

    migrated = deepcopy(dict(robot_state))
    migrated.setdefault("schema_version", EMBODIMENT_SCHEMA_VERSION)
    frame_schema = deepcopy(migrated.get("frame_schema", {}))

    if "eef_pose_world" in robot_state:
        migrated.setdefault(
            LEGACY_EEF_POSE_WORLD_KEY,
            _copy_pose_payload(robot_state["eef_pose_world"]),
        )
        frame_schema[LEGACY_EEF_POSE_WORLD_KEY] = {
            "frame_role": FRAME_ROLE_LEGACY_EEF,
            "source_key": "eef_pose_world",
            "status": "legacy_ambiguous",
        }

        if legacy_eef_is_tool_base:
            migrated.setdefault(
                TOOL_BASE_POSE_WORLD_KEY,
                _copy_pose_payload(robot_state["eef_pose_world"]),
            )
            frame_schema[TOOL_BASE_POSE_WORLD_KEY] = {
                "frame_role": FRAME_ROLE_TOOL_BASE,
                "source_key": "eef_pose_world",
            }

            if frame_resolver is not None and TASK_TCP_POSE_WORLD_KEY not in migrated:
                tool_base_pose = pose7_from_payload(robot_state["eef_pose_world"])
                tcp_position, tcp_orientation = frame_resolver.tool_base_pose_to_task_tcp_pose(
                    tool_base_pose.position,
                    tool_base_pose.orientation_wxyz,
                )
                migrated[TASK_TCP_POSE_WORLD_KEY] = Pose7(
                    tcp_position,
                    tcp_orientation,
                ).to_mapping()
                frame_schema[TASK_TCP_POSE_WORLD_KEY] = {
                    "frame_role": FRAME_ROLE_TASK_TCP,
                    "source_key": TOOL_BASE_POSE_WORLD_KEY,
                    "derived_by": type(frame_resolver).__name__,
                }
        else:
            migrated.setdefault("requires_task_tcp_pose_migration", True)

    if frame_schema:
        migrated["frame_schema"] = frame_schema
    return migrated


def pose7_from_payload(payload: Any) -> Pose7:
    if isinstance(payload, Mapping):
        if "position" not in payload:
            raise KeyError("Pose mapping is missing 'position'.")
        orientation_key = "orientation" if "orientation" in payload else "rotation"
        if orientation_key not in payload:
            raise KeyError("Pose mapping is missing 'orientation' or 'rotation'.")
        return Pose7(
            position=payload["position"],
            orientation_wxyz=payload[orientation_key],
        )

    values = np.asarray(payload, dtype=np.float32).reshape(-1)
    if values.shape != (7,):
        raise ValueError(f"Pose payload must be mapping or shape (7,), got {values.shape}.")
    return Pose7(position=values[:3], orientation_wxyz=values[3:])


def _extract_gripper(payload: Mapping[str, Any]) -> float | None:
    if "gripper" in payload and payload["gripper"] is not None:
        return float(payload["gripper"])
    command = payload.get(GRIPPER_COMMAND_KEY)
    if isinstance(command, Mapping) and command.get("value") is not None:
        return float(command["value"])
    return None


def _copy_pose_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else deepcopy(value)
            for key, value in payload.items()
        }
    if isinstance(payload, np.ndarray):
        return payload.copy()
    return deepcopy(payload)


def _first_present_key(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in payload:
            return key
    return None


def _vector(values: Any, *, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}.")
    return array.copy()
