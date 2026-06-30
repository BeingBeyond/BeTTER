from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.task.specs.episode import resolved_episode_from_dict, resolved_episode_to_dict
from src.task.specs.schema import ResolvedEpisodeSpec

from .episode_runtime import (
    RuntimeObjectPose,
    runtime_object_poses_from_dict,
    runtime_object_poses_to_dict,
)


EPISODE_OBJECT_TRAJECTORY_SCHEMA_VERSION = "episode_object_trajectory.v1"


@dataclass(frozen=True)
class EpisodeTrajectoryFrame:
    frame_index: int
    sim_time: float
    object_poses: dict[str, RuntimeObjectPose]


@dataclass(frozen=True)
class EpisodeObjectTrajectory:
    episode: ResolvedEpisodeSpec
    frames: tuple[EpisodeTrajectoryFrame, ...]
    schema_version: str = EPISODE_OBJECT_TRAJECTORY_SCHEMA_VERSION


def episode_trajectory_to_dict(trajectory: EpisodeObjectTrajectory) -> dict[str, object]:
    return {
        "schema_version": trajectory.schema_version,
        "episode": resolved_episode_to_dict(trajectory.episode),
        "frames": [
            {
                "frame_index": int(frame.frame_index),
                "sim_time": float(frame.sim_time),
                "object_poses": runtime_object_poses_to_dict(frame.object_poses),
            }
            for frame in trajectory.frames
        ],
    }


def episode_trajectory_from_dict(payload: Mapping[str, object]) -> EpisodeObjectTrajectory:
    episode_payload = payload.get("episode")
    frames_payload = payload.get("frames", [])
    if not isinstance(episode_payload, Mapping):
        raise ValueError("Episode object trajectory field 'episode' must be a mapping")
    if not isinstance(frames_payload, list):
        raise ValueError("Episode object trajectory field 'frames' must be a list")

    return EpisodeObjectTrajectory(
        schema_version=str(
            payload.get("schema_version") or EPISODE_OBJECT_TRAJECTORY_SCHEMA_VERSION
        ),
        episode=resolved_episode_from_dict(episode_payload),
        frames=tuple(_frame_from_payload(frame) for frame in frames_payload),
    )


def write_episode_trajectory(trajectory: EpisodeObjectTrajectory, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(episode_trajectory_to_dict(trajectory), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_episode_trajectory(path: str | Path) -> EpisodeObjectTrajectory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Episode object trajectory file must contain a JSON object: {path}")
    return episode_trajectory_from_dict(payload)


def _frame_from_payload(payload: object) -> EpisodeTrajectoryFrame:
    if not isinstance(payload, Mapping):
        raise ValueError("Episode object trajectory frames must be mappings")
    object_poses = payload.get("object_poses")
    if not isinstance(object_poses, Mapping):
        raise ValueError("Episode object trajectory frame field 'object_poses' must be a mapping")
    return EpisodeTrajectoryFrame(
        frame_index=int(payload["frame_index"]),
        sim_time=float(payload.get("sim_time") or 0.0),
        object_poses=runtime_object_poses_from_dict(object_poses),
    )
