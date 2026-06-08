class DrowsinessAnalyzer:
    def __init__(
        self,
        ear_threshold=0.20,
        mar_threshold=0.60,
        eye_closed_duration=1.5,
        perclos_threshold=0.30,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.eye_closed_duration = eye_closed_duration
        self.perclos_threshold = perclos_threshold
        self.eye_closed_start_time = None

    def analyze(self, ear, mar, perclos, timestamp, face_detected=True):
        if not face_detected:
            self.eye_closed_start_time = None
            return "NO_FACE"

        eye_closed = ear < self.ear_threshold
        if eye_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = timestamp

            if timestamp - self.eye_closed_start_time > self.eye_closed_duration:
                return "DROWSY"
            return "EYE_CLOSED"

        self.eye_closed_start_time = None

        if mar > self.mar_threshold:
            return "YAWNING"

        if perclos > self.perclos_threshold:
            return "DROWSY"

        return "NORMAL"
