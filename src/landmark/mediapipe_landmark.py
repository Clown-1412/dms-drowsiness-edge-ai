import cv2
import mediapipe as mp


class MediaPipeFaceLandmark:
    """Thin wrapper around MediaPipe Face Mesh for single-face landmark detection."""

    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        try:
            face_mesh = mp.solutions.face_mesh
            drawing_utils = mp.solutions.drawing_utils
        except AttributeError as exc:
            raise RuntimeError(
                "Installed mediapipe does not include legacy Face Mesh "
                "`mp.solutions`. Reinstall dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.face_mesh_module = face_mesh
        self.drawing_utils = drawing_utils
        self.drawing_spec = drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
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
            "face_landmarks_raw": face_landmarks,
        }

    def draw_face_mesh(self, frame, detection):
        """Draw the MediaPipe face contours from an existing detection."""
        if detection is None:
            return frame

        face_landmarks = detection.get("face_landmarks_raw")
        if face_landmarks is None:
            return frame

        self.drawing_utils.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.face_mesh_module.FACEMESH_CONTOURS,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.drawing_spec,
        )
        return frame

    def close(self):
        self.face_mesh.close()
