from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LocalSe3KeyboardConfig:
    """Configuration for the local Isaac-Sim keyboard SE(3) jog device."""

    pos_sensitivity: float = 0.1
    rot_sensitivity: float = 0.4
    gripper_term: bool = True


class LocalSe3Keyboard:
    """Isaac-Lab-free keyboard SE(3) jog device.

    Key bindings mirror Isaac Lab's ``Se3Keyboard`` implementation
    (BSD-3-Clause, Isaac Lab Project Developers, 2022-2025), but this class
    only uses Isaac Sim's Carb keyboard interface and returns NumPy arrays.
    """

    def __init__(self, config: LocalSe3KeyboardConfig | None = None) -> None:
        self.config = LocalSe3KeyboardConfig() if config is None else config
        self._delta_pos = np.zeros(3, dtype=np.float64)
        self._delta_euler_xyz = np.zeros(3, dtype=np.float64)
        self._close_gripper = False
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._modified_callbacks: dict[tuple[str, frozenset[str]], Callable[[], None]] = {}

        import carb.input
        import omni.appwindow

        self._carb_input = carb.input
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self._input_key_mapping = self._create_key_bindings()

    def __str__(self) -> str:
        return "\n".join(
            [
                "Local keyboard controller for SE(3) jog:",
                "  Toggle gripper open/close: K",
                "  Reset keyboard command: L",
                "  Move x: W/S",
                "  Move y: A/D",
                "  Move z: Q/E",
                "  Rotate x: Z/X",
                "  Rotate y: T/G",
                "  Rotate z: C/V",
                "  Script callbacks may bind additional keys such as ESCAPE or SHIFT+R.",
            ]
        )

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._callbacks[str(key).upper()] = func

    def add_modified_callback(
        self,
        key: str,
        modifiers: set[str] | frozenset[str],
        func: Callable[[], None],
    ) -> None:
        normalized_key = str(key).upper()
        normalized_modifiers = frozenset(str(item).upper() for item in modifiers)
        self._modified_callbacks[(normalized_key, normalized_modifiers)] = func

    def reset(self) -> None:
        self._delta_pos = np.zeros(3, dtype=np.float64)
        self._delta_euler_xyz = np.zeros(3, dtype=np.float64)
        self._close_gripper = False

    def advance(self) -> np.ndarray:
        rotvec = _euler_xyz_to_rotvec(self._delta_euler_xyz)
        command = np.concatenate([self._delta_pos, rotvec]).astype(np.float32)
        if self.config.gripper_term:
            gripper = -1.0 if self._close_gripper else 1.0
            command = np.concatenate([command, np.asarray([gripper], dtype=np.float32)])
        return command

    def shutdown(self) -> None:
        if self._keyboard_sub is not None:
            if hasattr(self._input, "unsubscribe_from_keyboard_events"):
                self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            elif hasattr(self._input, "unsubscribe_to_keyboard_events"):
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            else:
                raise RuntimeError("Carb input does not expose a keyboard unsubscribe API.")
            self._keyboard_sub = None

    @property
    def close_gripper(self) -> bool:
        return self._close_gripper

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        key_name = _keyboard_input_name(event.input)
        modifier_names = _keyboard_modifier_names(event, self._carb_input)
        event_type = event.type
        keyboard_event_type = self._carb_input.KeyboardEventType

        if event_type == keyboard_event_type.KEY_PRESS:
            if key_name == "L":
                self.reset()
            elif key_name == "K":
                self._close_gripper = not self._close_gripper
            elif key_name in ("W", "S", "A", "D", "Q", "E"):
                self._delta_pos += self._input_key_mapping[key_name]
            elif key_name in ("Z", "X", "T", "G", "C", "V"):
                self._delta_euler_xyz += self._input_key_mapping[key_name]

            callback = self._modified_callbacks.get((key_name, frozenset(modifier_names)))
            if callback is None and not modifier_names:
                callback = self._callbacks.get(key_name)
            if callback is not None:
                callback()

        elif event_type == keyboard_event_type.KEY_RELEASE:
            if key_name in ("W", "S", "A", "D", "Q", "E"):
                self._delta_pos -= self._input_key_mapping[key_name]
            elif key_name in ("Z", "X", "T", "G", "C", "V"):
                self._delta_euler_xyz -= self._input_key_mapping[key_name]

        return True

    def _create_key_bindings(self) -> dict[str, np.ndarray]:
        pos = float(self.config.pos_sensitivity)
        rot = float(self.config.rot_sensitivity)
        return {
            "W": np.asarray([1.0, 0.0, 0.0], dtype=np.float64) * pos,
            "S": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64) * pos,
            "A": np.asarray([0.0, 1.0, 0.0], dtype=np.float64) * pos,
            "D": np.asarray([0.0, -1.0, 0.0], dtype=np.float64) * pos,
            "Q": np.asarray([0.0, 0.0, 1.0], dtype=np.float64) * pos,
            "E": np.asarray([0.0, 0.0, -1.0], dtype=np.float64) * pos,
            "Z": np.asarray([1.0, 0.0, 0.0], dtype=np.float64) * rot,
            "X": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64) * rot,
            "T": np.asarray([0.0, 1.0, 0.0], dtype=np.float64) * rot,
            "G": np.asarray([0.0, -1.0, 0.0], dtype=np.float64) * rot,
            "C": np.asarray([0.0, 0.0, 1.0], dtype=np.float64) * rot,
            "V": np.asarray([0.0, 0.0, -1.0], dtype=np.float64) * rot,
        }


def _euler_xyz_to_rotvec(euler_xyz: np.ndarray) -> np.ndarray:
    angles = np.asarray(euler_xyz, dtype=np.float64).reshape(3)
    rx = _axis_rotation_matrix(0, float(angles[0]))
    ry = _axis_rotation_matrix(1, float(angles[1]))
    rz = _axis_rotation_matrix(2, float(angles[2]))
    rotation = rx @ ry @ rz
    return _matrix_to_rotvec(rotation)


def _axis_rotation_matrix(axis: int, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    if axis == 0:
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == 2:
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported axis index: {axis}")


def _keyboard_input_name(input_value) -> str:
    name = getattr(input_value, "name", input_value)
    key_name = str(name)
    if "." in key_name:
        key_name = key_name.rsplit(".", 1)[-1]
    return key_name.upper()


def _keyboard_modifier_names(event, carb_input) -> set[str]:
    raw_modifiers = int(getattr(event, "modifiers", 0) or 0)
    names: set[str] = set()
    flag_specs = (
        ("SHIFT", "KEYBOARD_MODIFIER_FLAG_SHIFT"),
        ("CTRL", "KEYBOARD_MODIFIER_FLAG_CONTROL"),
        ("ALT", "KEYBOARD_MODIFIER_FLAG_ALT"),
        ("SUPER", "KEYBOARD_MODIFIER_FLAG_SUPER"),
    )
    for name, attr in flag_specs:
        flag = int(getattr(carb_input, attr, 0) or 0)
        if flag and raw_modifiers & flag:
            names.add(name)
    return names


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cos_angle = (float(np.trace(matrix)) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1.0e-12:
        return np.zeros(3, dtype=np.float64)

    axis = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * float(np.sin(angle))
    return axis * angle
