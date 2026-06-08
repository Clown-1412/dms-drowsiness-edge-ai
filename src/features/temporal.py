from collections import deque


class TemporalAnalyzer:
    """Tracks boolean eye-closed samples inside a time-based sliding window."""

    def __init__(self, window_sec=10):
        self.window_sec = window_sec
        self.history = deque()

    def update(self, eye_closed, timestamp):
        self.history.append((timestamp, bool(eye_closed)))
        self._drop_old_samples(timestamp)
        return self.get_perclos()

    def get_perclos(self):
        total = len(self.history)
        if total == 0:
            return 0.0

        closed_count = sum(1 for _, eye_closed in self.history if eye_closed)
        return closed_count / total

    def _drop_old_samples(self, timestamp):
        min_timestamp = timestamp - self.window_sec
        while self.history and self.history[0][0] < min_timestamp:
            self.history.popleft()
