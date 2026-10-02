"""Bounded interpretation of the SDK's typed completed-position records."""
from __future__ import annotations

import threading
from typing import Protocol


class StepProgress(Protocol):
    stage: str
    position: int | None
    total: int | None
    call_request: str | None
    call_attempt: int | None


class CompletedWork:
    """Count completed step positions, never SDK event sequence or stage-open churn."""

    def __init__(self):
        self.positions: dict[tuple[str, str | None, int | None], int] = {}
        self.units = 0
        self.lock = threading.Lock()

    def observe(self, frame: StepProgress) -> int | None:
        position, total = frame.position, frame.total
        if (type(position) is not int or type(total) is not int
                or not 0 < position <= total):
            return None
        key = (frame.stage, frame.call_request, frame.call_attempt)
        with self.lock:
            previous = self.positions.get(key, 0)
            if position <= previous or (key not in self.positions and len(self.positions) >= 256):
                return None
            self.positions[key] = position
            self.units += position - previous
            return self.units
