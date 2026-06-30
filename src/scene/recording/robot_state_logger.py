from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


ROBOT_STATE_LOG_SCHEMA_VERSION = "better.robot_state_log.v1"


class RobotStateSnapshotter:
    """Capture simulator robot state in a schema suitable for replay/debugging."""

    def __init__(
        self,
        robot,
        *,
        frame_resolver: Any | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
        include_runtime_tool_base_pose: bool = True,
    ) -> None:
        self.robot = robot
        self.frame_resolver = frame_resolver
        self.runtime_metadata = dict(runtime_metadata or {})
        self.include_runtime_tool_base_pose = bool(include_runtime_tool_base_pose)

    def capture(
        self,
        *,
        step_index: int,
        timestamp: float | None = None,
        command: Mapping[str, Any] | None = None,
        command_report: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        robot_state = self.robot.capture_runtime_state()
        _add_group_aliases(robot_state)

        tool_base_pose = self._tool_base_pose_world()
        if tool_base_pose is not None:
            robot_state["tool_base_pose_world"] = tool_base_pose
            task_tcp_pose = self._task_tcp_pose_world(tool_base_pose)
            if task_tcp_pose is not None:
                robot_state["task_tcp_pose_world"] = task_tcp_pose

        snapshot: dict[str, Any] = {
            "schema_version": ROBOT_STATE_LOG_SCHEMA_VERSION,
            "event": "robot_state",
            "step_index": int(step_index),
            "wall_time": float(time.time()),
            "robot": robot_state,
        }
        if timestamp is not None:
            snapshot["timestamp"] = float(timestamp)
        if self.runtime_metadata:
            snapshot["runtime"] = _jsonable(self.runtime_metadata)
        if command is not None:
            snapshot["command"] = _jsonable(command)
        if command_report is not None:
            snapshot["command_report"] = _jsonable(command_report)
        if extra is not None:
            snapshot["extra"] = _jsonable(extra)
        return snapshot

    def _tool_base_pose_world(self) -> dict[str, Any] | None:
        if not self.include_runtime_tool_base_pose:
            return None

        runtime_path = _runtime_tool_base_prim_path(self.robot)
        if runtime_path is None:
            return None

        from isaacsim.core.prims import SingleXFormPrim  # type: ignore[import]
        from isaacsim.core.utils.stage import get_current_stage  # type: ignore[import]

        stage = get_current_stage()
        if not stage.GetPrimAtPath(runtime_path).IsValid():
            return None

        prim = SingleXFormPrim(runtime_path)
        position, orientation = prim.get_world_pose()
        return {
            "frame_role": "tool_base_frame",
            "prim_path": runtime_path,
            "position": _float_sequence(position, length=3),
            "orientation_wxyz": _float_sequence(orientation, length=4),
        }

    def _task_tcp_pose_world(self, tool_base_pose: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.frame_resolver is None:
            return None

        import numpy as np

        position, orientation = self.frame_resolver.tool_base_pose_to_task_tcp_pose(
            np.asarray(tool_base_pose["position"], dtype=float).reshape(3),
            np.asarray(tool_base_pose["orientation_wxyz"], dtype=float).reshape(4),
        )
        return {
            "frame_role": "task_tcp_frame",
            "source": "tool_base_pose_world",
            "position": _float_sequence(position, length=3),
            "orientation_wxyz": _float_sequence(orientation, length=4),
        }


class JsonlRobotStateLogger:
    """Append robot state snapshots to a JSONL file."""

    def __init__(
        self,
        path: str | Path,
        snapshotter: RobotStateSnapshotter,
        *,
        metadata: Mapping[str, Any] | None = None,
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.snapshotter = snapshotter
        self.flush_every = max(1, int(flush_every))
        self._record_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "schema_version": ROBOT_STATE_LOG_SCHEMA_VERSION,
                "event": "metadata",
                "created_wall_time": float(time.time()),
                "metadata": _jsonable(dict(metadata or {})),
            }
        )

    def log(
        self,
        *,
        step_index: int,
        timestamp: float | None = None,
        command: Mapping[str, Any] | None = None,
        command_report: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshotter.capture(
            step_index=step_index,
            timestamp=timestamp,
            command=command,
            command_report=command_report,
            extra=extra,
        )
        self._write(snapshot)
        self._record_count += 1
        if self._record_count % self.flush_every == 0:
            self._file.flush()
        return snapshot

    def close(self) -> None:
        self._file.flush()
        self._file.close()

    def __enter__(self) -> "JsonlRobotStateLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._file.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _runtime_tool_base_prim_path(robot) -> str | None:
    configured_path = robot.spec.prim_paths.get("eef_base_frame") or robot.spec.prim_paths.get(
        "eef_frame"
    )
    if configured_path is None:
        return None

    configured_path = str(configured_path)
    configured_root = str(robot.spec.prim_paths.get("arm_root") or robot.spec.prim_paths.get("robot_root", ""))
    if configured_root and configured_path.startswith(configured_root):
        return robot.prim_path.rstrip("/") + configured_path[len(configured_root) :]
    if configured_path.startswith("/World/franka"):
        suffix = configured_path[len("/World/franka") :]
        if robot.prim_path.endswith("/arm") and suffix.startswith("/arm"):
            suffix = suffix[len("/arm") :]
        return robot.prim_path.rstrip("/") + suffix
    if configured_path.startswith("/World/"):
        return configured_path
    return robot.prim_path.rstrip("/") + "/" + configured_path.lstrip("/")


def _add_group_aliases(robot_state: dict[str, Any]) -> None:
    joint_groups = robot_state.get("joint_groups", {})
    if not isinstance(joint_groups, Mapping):
        return

    arm = joint_groups.get("arm")
    if isinstance(arm, Mapping):
        robot_state.setdefault("arm_joint_names", list(arm.get("joint_names", [])))
        robot_state.setdefault("arm_joint_positions", dict(arm.get("joint_positions", {})))
        robot_state.setdefault("arm_joint_velocities", dict(arm.get("joint_velocities", {})))

    gripper = joint_groups.get("gripper")
    if isinstance(gripper, Mapping):
        robot_state.setdefault("gripper_joint_names", list(gripper.get("joint_names", [])))
        robot_state.setdefault(
            "gripper_joint_positions",
            dict(gripper.get("joint_positions", {})),
        )
        robot_state.setdefault(
            "gripper_joint_velocities",
            dict(gripper.get("joint_velocities", {})),
        )


def _float_sequence(value: Any, *, length: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    result = [float(item) for item in value]
    if len(result) != length:
        raise ValueError(f"Expected sequence of length {length}, got {len(result)}.")
    return result


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
