# Franka + Robotiq 2F-85 Robot Asset

This directory contains the robot embodiment used by the current BeTTER
runtime and teleoperation stack.

## Runtime Asset

- `franka_robotiq_2f_85_flattened.usd`
  - Source: RoboLab `assets/robots/franka_robotiq_2f_85_flattened.usd`
  - Role: Isaac Sim runtime robot asset.
  - Notes: this is the tested runtime asset for BeTTER's Franka + Robotiq
    setup. It uses a single active Robotiq `finger_joint` and PhysX mimic
    joints for the remaining gripper joints.

## Planner Asset

- `franka_robotiq_planner.urdf`
  - Role: planner/kinematics description used to resolve the semantic
    `robotiq_tcp` frame.
  - Notes: the runtime USD exposes the Robotiq tool base as a prim. The task
    TCP is represented as a planner link and is resolved through the embodiment
    registry.

## Default Runtime Control

The corresponding embodiment registry entry uses:

- controller profile: `robolab_franka_robotiq_high_pd`
- gravity compensation policy: `disable_robot_gravity=true`
- active gripper command joint: `finger_joint`
- open target: `0.0`
- close target: `0.7853981633974483`

## Licensing

The top-level BeTTER code is released under the repository's main license.
The RoboLab-derived USD asset is not covered by the BeTTER MIT license. It is
distributed with RoboLab's `CC-BY-NC-4.0` license terms; see
`LICENSE.RoboLab-CC-BY-NC-4.0.txt` and the repository-level
`THIRD_PARTY_NOTICES.md`.
