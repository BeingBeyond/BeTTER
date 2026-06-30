# Rendering Tools

`render_episode.py` renders either a static resolved episode or a recorded state log
recorded by `tools/teleop/franka_robotiq_task_teleop.py`.

Run the commands below from the repository root with the Python environment
where `isaacsim` is installed.

The default mode is state-snapshot rendering:

- restore object state and robot state for every output frame
- zero velocities after restore
- run one `1/120` physics sync tick so restored Isaac articulation poses reach
  the camera renderer
- load the packaged background scene unless `--no-background` is passed
- place the robot reference root at `(-0.05, 0.0, -0.1)` unless
  `--robot-position X Y Z` is passed
- use `15` only as output video FPS metadata
- write MP4 videos under `videos/<camera_id>.mp4`
- skip per-frame PNG files unless `--save-images` is passed

This avoids trajectory rollout drift from gravity, contacts, or residual
velocities while still making robot articulation state visible in rendered
camera images.
MP4 writing uses `imageio` with ffmpeg/pyav when available, otherwise OpenCV.

## Render A Teleop Log

```bash
python tools/render/render_episode.py \
  --record-path outputs/teleop/Packing_a_Fruit_Lunch/TV-01/success/episode_001.pkl \
  --output-dir outputs/renders/Packing_a_Fruit_Lunch/TV-01/episode_001 \
  --cameras all
```

This renders the full trajectory by default. Use `--max-frames N` only for a
quick smoke test.

The command writes outputs under:

```text
outputs/renders/Packing_a_Fruit_Lunch/TV-01/episode_001/
  videos/front_camera.mp4
  manifest.jsonl
  metadata.json
```

Pass `--save-images` only when you also need per-frame PNG files:

```bash
python tools/render/render_episode.py \
  --record-path outputs/teleop/Packing_a_Fruit_Lunch/TV-01/success/episode_001.pkl \
  --output-dir outputs/renders/Packing_a_Fruit_Lunch/TV-01/episode_001 \
  --cameras all \
  --save-images
```

By default only `front_camera` is rendered for quick inspection. The README
command above passes `--cameras all` to render the fixed LoHoBench-style views:

```bash
python tools/render/render_episode.py \
  --record-path outputs/teleop/Packing_a_Fruit_Lunch/TV-01/success/episode_001.pkl \
  --output-dir outputs/renders/Packing_a_Fruit_Lunch/TV-01/episode_001 \
  --cameras all
```

`--cameras all` currently expands to `front_camera`, `left_shoulder_camera`,
and `right_shoulder_camera`. The wrist/Realsense view is intentionally not in
the default set because its mount prim depends on the robot USD.

Use `--no-background` to render against a local simple ground plane instead of
the packaged background scene. This fallback adds local Dome/Distant/Rect
lights by default and does not depend on Isaac/Nucleus asset discovery. Pass
`--no-default-lights` only when debugging lighting explicitly.

The default path renders with the packaged background scene. Keep
`--startup-frames` at its default unless you are debugging startup. When a
state log is provided, startup warmup restores deterministic random state rows,
renders them, reads camera buffers, and discards those frames before returning
to the first saved frame. This forces object materials, background lighting,
and camera buffers to settle before frame `0` is written to video.

Keep `--first-frame-warmup-frames` above `0` as a final guard. The first saved
state is restored again after this warmup, before frame `0` is captured.

## Render One Static Episode

```bash
python tools/render/render_episode.py \
  --output-dir outputs/renders/static_packing_fruit_lunch \
  --max-frames 1
```

## Physics Debug Mode

The default renderer uses exactly one small physics sync tick per saved state.
It is not a dynamic rollout because every frame is restored from the log before
that tick.

To intentionally test dynamic drift or contact stability, choose a larger
rendering timestep so multiple physics steps are run after each restored state:

```bash
python tools/render/render_episode.py \
  --record-path outputs/teleop/Packing_a_Fruit_Lunch/TV-01/success/episode_001.pkl \
  --physics-dt 0.008333333333333333 \
  --rendering-dt 0.06666666666666667 \
  --max-frames 120
```

For debugging camera/render synchronization itself, `--no-step-physics` keeps
the simulation paused, but Isaac articulation poses may appear visually stale in
camera output.
