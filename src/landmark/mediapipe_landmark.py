import cv2
import mediapipe as mp


class MediaPipeFaceLandmark:
    """Thin wrapper around MediaPipe Face Mesh for single-face landmark detection."""

    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        try:
            face_mesh = mp.solutions.face_mesh
        except AttributeError as exc:
            raise RuntimeError(
                "Installed mediapipe does not include legacy Face Mesh "
                "`mp.solutions`. Reinstall dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.face_mesh = face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = [
            (landmark.x, landmark.y, landmark.z)
            for landmark in face_landmarks.landmark
        ]

        return {
            "landmarks": landmarks,
            "image_size": (width, height),
        }

    def close(self):
        self.face_mesh.close()
