from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from src.scene.episode_runtime import (
    capture_episode_object_poses,
    runtime_object_poses_to_dict,
)
from src.task.specs.episode import resolved_episode_to_dict
from src.task.specs.schema import ResolvedEpisodeSpec

from .robot_state_logger import RobotStateSnapshotter


EPISODE_STATE_LOG_SCHEMA_VERSION = "better.episode_state_log.v1"


class EpisodeStateSnapshotter:
    """Capture object and robot runtime state for a resolved task episode."""

    def __init__(
        self,
        *,
        stage: Any,
        episode: ResolvedEpisodeSpec,
        robot_snapshotter: RobotStateSnapshotter | None = None,
        runtime_metadata: Mapping[str, Any] | None = None,
        xform_prim_factory: Any | None = None,
        scale_reader: Any | None = None,
    ) -> None:
        self.stage = stage
        self.episode = episode
        self.robot_snapshotter = robot_snapshotter
        self.runtime_metadata = dict(runtime_metadata or {})
        self.xform_prim_factory = xform_prim_factory
        self.scale_reader = scale_reader

    def capture(
        self,
        *,
        step_index: int,
        timestamp: float | None = None,
        command: Mapping[str, Any] | None = None,
        command_report: Mapping[str, Any] | None = None,
        evaluation: Any | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        object_poses = capture_episode_object_poses(
            self.stage,
            self.episode,
            xform_prim_factory=self.xform_prim_factory,
            scale_reader=self.scale_reader,
        )
        snapshot: dict[str, Any] = {
            "schema_version": EPISODE_STATE_LOG_SCHEMA_VERSION,
            "event": "episode_state",
            "step_index": int(step_index),
            "wall_time": float(time.time()),
            "episode": _episode_summary(self.episode),
            "objects": runtime_object_poses_to_dict(object_poses),
        }
        if timestamp is not None:
            snapshot["timestamp"] = float(timestamp)
        if self.runtime_metadata:
            snapshot["runtime"] = _jsonable(self.runtime_metadata)
        if self.robot_snapshotter is not None:
            robot_snapshot = self.robot_snapshotter.capture(
                step_index=step_index,
                timestamp=timestamp,
            )
            snapshot["robot"] = robot_snapshot["robot"]
            if "runtime" in robot_snapshot and "runtime" not in snapshot:
                snapshot["runtime"] = robot_snapshot["runtime"]
        if command is not None:
            snapshot["command"] = _jsonable(command)
        if command_report is not None:
            snapshot["command_report"] = _jsonable(command_report)
        if evaluation is not None:
            snapshot["evaluation"] = _jsonable(_evaluation_to_dict(evaluation))
        if extra is not None:
            snapshot["extra"] = _jsonable(extra)
        return snapshot


class JsonlEpisodeStateLogger:
    """Append resolved episode state snapshots to a JSONL file."""

    def __init__(
        self,
        path: str | Path,
        snapshotter: EpisodeStateSnapshotter,
        *,
        metadata: Mapping[str, Any] | None = None,
        include_resolved_episode: bool = True,
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.snapshotter = snapshotter
        self.flush_every = max(1, int(flush_every))
        self._record_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

        metadata_row: dict[str, Any] = {
            "schema_version": EPISODE_STATE_LOG_SCHEMA_VERSION,
            "event": "metadata",
            "created_wall_time": float(time.time()),
            "metadata": _jsonable(dict(metadata or {})),
        }
        if include_resolved_episode:
            metadata_row["resolved_episode"] = resolved_episode_to_dict(snapshotter.episode)
        self._write(metadata_row)

    def log(
        self,
        *,
        step_index: int,
        timestamp: float | None = None,
        command: Mapping[str, Any] | None = None,
        command_report: Mapping[str, Any] | None = None,
        evaluation: Any | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshotter.capture(
            step_index=step_index,
            timestamp=timestamp,
            command=command,
            command_report=command_report,
            evaluation=evaluation,
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

    def __enter__(self) -> "JsonlEpisodeStateLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._file.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


class PickleEpisodeStateLogger:
    """Write resolved episode state snapshots to one pickle payload per episode."""

    def __init__(
        self,
        path: str | Path,
        snapshotter: EpisodeStateSnapshotter,
        *,
        metadata: Mapping[str, Any] | None = None,
        include_resolved_episode: bool = True,
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.snapshotter = snapshotter
        self.flush_every = max(1, int(flush_every))
        self._record_count = 0
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

        metadata_row: dict[str, Any] = {
            "schema_version": EPISODE_STATE_LOG_SCHEMA_VERSION,
            "event": "metadata",
            "created_wall_time": float(time.time()),
            "metadata": _jsonable(dict(metadata or {})),
        }
        if include_resolved_episode:
            metadata_row["resolved_episode"] = resolved_episode_to_dict(snapshotter.episode)
        self._payload: dict[str, Any] = {
            "schema_version": EPISODE_STATE_LOG_SCHEMA_VERSION,
            "format": "pickle",
            "metadata": metadata_row,
            "states": [],
        }

    def log(
        self,
        *,
        step_index: int,
        timestamp: float | None = None,
        command: Mapping[str, Any] | None = None,
        command_report: Mapping[str, Any] | None = None,
        evaluation: Any | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = self.snapshotter.capture(
            step_index=step_index,
            timestamp=timestamp,
            command=command,
            command_report=command_report,
            evaluation=evaluation,
            extra=extra,
        )
        self._payload["states"].append(snapshot)
        self._record_count += 1
        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(_jsonable(self._payload), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.path)
        self._closed = True

    def __enter__(self) -> "PickleEpisodeStateLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _episode_summary(episode: ResolvedEpisodeSpec) -> dict[str, Any]:
    return {
        "schema_version": episode.schema_version,
        "task_id": episode.task_id,
        "template_type": episode.template_type,
        "variation_id": episode.variation_id,
        "episode_seed": int(episode.episode_seed),
        "object_count": len(episode.objects),
    }


def _evaluation_to_dict(evaluation: Any) -> Any:
    if is_dataclass(evaluation):
        return asdict(evaluation)
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    names = (
        "goal_raw",
        "goal_streak",
        "goal_passed",
        "fail_raw",
        "fail_streak",
        "fail_passed",
        "terminated",
        "termination_type",
        "reason",
    )
    if all(hasattr(evaluation, name) for name in names):
        return {name: getattr(evaluation, name) for name in names}
    return evaluation


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
