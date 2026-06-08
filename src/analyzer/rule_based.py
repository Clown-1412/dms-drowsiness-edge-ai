class DrowsinessAnalyzer:
    def __init__(self, ear_threshold=0.20, mar_threshold=0.60):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold

    def analyze(self, ear, mar):
        if ear < self.ear_threshold:
            return "EYE_CLOSED"

        if mar > self.mar_threshold:
            return "YAWNING"

        return "NORMAL"
