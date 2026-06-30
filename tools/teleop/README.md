# Teleoperation

This directory contains the public task-aware teleoperation entry point for collecting BeTTER human demonstrations with the Franka + Robotiq embodiment.

## Script

- `tools/teleop/franka_robotiq_task_teleop.py`

This script:

- loads one resolved BeTTER task episode configuration
- selects a concrete variation via `--variation-id`
- loads task objects, registered assets, and optional background scene
- re-samples object assets on each collected episode using a deterministic seed sequence
- applies the configured runtime controller profile
- evaluates task goal/fail conditions by default
- records robot state and object state to pickle files
- keeps collecting trajectories until the user exits

## Environment

The repository-level installation steps are documented in the main README. After completing that setup, run the teleoperation script from the repository root with the Python environment where `isaacsim` is installed.

## Minimal Collection Command

```bash
python tools/teleop/franka_robotiq_task_teleop.py \
  --warmup-frames 120
```

By default, this command keeps the Isaac Sim process open and records a sequence
of trajectories under `outputs/teleop/<task>/<variation>/<result>/`. A running
trajectory is staged under `in_progress/` and moved to `success/` or `failed/`
when that trajectory ends.

Example output paths:

```text
outputs/teleop/Packing_a_Fruit_Lunch/TV-01/success/episode_001.pkl
outputs/teleop/Packing_a_Fruit_Lunch/TV-01/failed/episode_002.pkl
```

Keyboard collection controls:

- `SHIFT+R`: finish the current trajectory, save it, reset the scene, and start the next one
- `ESCAPE`: stop collection and close the process

Robot controls:

- `K`: toggle gripper open/close
- `W` / `S`: move arm along the x-axis
- `A` / `D`: move arm along the y-axis
- `Q` / `E`: move arm along the z-axis
- `Z` / `X`: rotate arm around the x-axis
- `T` / `G`: rotate arm around the y-axis
- `C` / `V`: rotate arm around the z-axis

## Core arguments

- `--task-dir` selects the task directory
- `--variation-id` selects the variation to resolve
- `--asset-registry-root` points to `assets/objects/registry`
- `--episode-seed` sets the base seed; episode N uses `episode_seed + N - 1`
- `--no-evaluate-conditions` disables goal/fail condition evaluation
- `--no-auto-reset-on-success` disables automatic reset after success
- `--single-episode` restores the old one-process-one-trajectory behavior
- `--manual-reset-shortcut` changes the finish-and-reset shortcut
- `--warmup-frames` runs physics steps before keyboard control and logging begin
- `--record-dir` sets the root output directory
- `--record-prefix` sets the per-episode filename prefix

## Packing_a_Fruit_Lunch example

```bash
python tools/teleop/franka_robotiq_task_teleop.py \
  --task-dir assets/tasks/loose_packing/Packing_a_Fruit_Lunch \
  --variation-id TV-01 \
  --asset-registry-root assets/objects/registry \
  --warmup-frames 120 \
  --record-dir outputs/teleop \
  --record-prefix episode
```

Defaults include:

- task: `assets/tasks/loose_packing/Packing_a_Fruit_Lunch`
- object registry root: `assets/objects/registry`
- background scene: enabled by default from `assets/scenes/backgrounds/registry.v2.json`
- robot embodiment: `franka_robotiq`
- robot root position: `(-0.05, 0.0, -0.1)`
- physics dt: `1/120`
- control/render dt: `1/15`
- pre-recording warmup: `24` physics steps
- keyboard sensitivity scale: `2.0`
- success condition evaluation: enabled
- success requires `20` consecutive satisfied control steps by default
- automatic reset on success: enabled
- manual trajectory reset shortcut: `SHIFT+R`
- output root: `outputs/teleop`
- output prefix: `episode`
- recording: enabled unless `--no-record` is passed

`--warmup-frames` is used to let task objects and contacts settle before the
initial state is recorded. Increase it, for example to `120`, if an episode
contains objects that need a longer settle period.

Pass `--single-episode` when you intentionally want process termination after
one trajectory. Without `--single-episode`, the script keeps collecting and
increments the filename suffix: `episode_001.pkl`, `episode_002.pkl`, ...

Use `--no-background` to fall back to Isaac Sim's default ground plane for a minimal debugging scene.

## Task integration

This script is task-aware rather than robot-only:

- it loads task data through `load_task_spec(...)`
- it resolves a concrete episode through `resolve_episode(...)`
- it can evaluate success and failure conditions against the resolved task episode

## Outputs

Recorded demonstrations are written as pickle files, for example:

- `outputs/teleop/Packing_a_Fruit_Lunch/TV-01/success/episode_001.pkl`
- `outputs/teleop/Packing_a_Fruit_Lunch/TV-01/failed/episode_002.pkl`

## Related documentation

- [Main README](../../README.md)
- [Task Editor](../../src/isaac_sim/extensions/task_editor/README.md)
- [Assets Editor](../../src/isaac_sim/extensions/assets_editor/README.md)
- [Variation Editor](../../src/isaac_sim/extensions/variation_editor/README.md)
