from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsecutiveCounter:
    frames_required: int
    streak: int = 0

    def __post_init__(self) -> None:
        self.frames_required = max(1, int(self.frames_required))

    def update(self, value: bool) -> tuple[int, bool]:
        if value:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak, self.streak >= self.frames_required
