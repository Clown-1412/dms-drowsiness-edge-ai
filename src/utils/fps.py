import time
from collections import deque
from typing import Deque, Optional


class FPSCounter:
    """Bo dem FPS co ho tro FPS tuc thoi va FPS trung binh truot."""

    def __init__(self, average_window: int = 30):
        if average_window <= 0:
            raise ValueError("average_window phai lon hon 0")

        self.average_window = average_window
        self.prev_time: Optional[float] = None
        self.current_fps = 0.0
        self.fps_values: Deque[float] = deque(maxlen=average_window)

    def reset(self) -> None:
        """Dat lai bo dem FPS khi bat dau mot stream moi."""
        self.prev_time = None
        self.current_fps = 0.0
        self.fps_values.clear()

    def update(self, timestamp: Optional[float] = None) -> float:
        """Cap nhat FPS theo timestamp hien tai va tra ve FPS trung binh."""
        now = timestamp if timestamp is not None else time.time()

        if self.prev_time is None:
            self.prev_time = now
            return 0.0

        delta_time = now - self.prev_time
        self.prev_time = now

        if delta_time > 0:
            self.current_fps = 1.0 / delta_time
            self.fps_values.append(self.current_fps)

        return self.get_fps()

    def get_fps(self) -> float:
        """Tra ve FPS trung binh trong cua so gan nhat."""
        if not self.fps_values:
            return self.current_fps
        return sum(self.fps_values) / len(self.fps_values)

    def get_current_fps(self) -> float:
        """Tra ve FPS tuc thoi cua frame gan nhat."""
        return self.current_fps
