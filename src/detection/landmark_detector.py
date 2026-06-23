from typing import Any, Dict, List, Optional

import mediapipe as mp
import numpy as np


# Cac index co ban cua MediaPipe Face Mesh.
LEFT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
MOUTH_INDEXES = [61, 81, 13, 311, 291, 402, 14, 178]
NOSE_INDEXES = [1, 2, 98, 327, 168]

# Cac diem thuong dung cho uoc luong head pose o cac layer sau.
HEAD_POSE_INDEXES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


LandmarkPoint = Dict[str, float]


class LandmarkDetector:
    """Landmark Detection Layer dung MediaPipe Face Mesh."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            self.face_mesh_module = mp.solutions.face_mesh
            self.drawing_utils = mp.solutions.drawing_utils
        except AttributeError as exc:
            raise RuntimeError(
                "Mediapipe khong ho tro mp.solutions.face_mesh. "
                "Hay cai lai dependency bang: python -m pip install -r requirements.txt"
            ) from exc

        self.mesh_drawing_spec = self.drawing_utils.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1,
        )
        self.contour_drawing_spec = self.drawing_utils.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1,
        )
        self.iris_drawing_spec = self.drawing_utils.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1,
        )
        self.face_mesh = self.face_mesh_module.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(
        self,
        processed_frame: Optional[np.ndarray],
        timestamp: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phat hien mat va tach cac nhom landmark can thiet.

        processed_frame phai la anh RGB da di qua FramePreprocessor.
        """
        if not self._is_valid_frame(processed_frame):
            return self._empty_result(timestamp=timestamp, fps=fps)

        height, width = processed_frame.shape[:2]

        try:
            rgb_frame = np.ascontiguousarray(processed_frame)
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
        except Exception as exc:
            print(f"[CANH BAO] Loi khi phat hien landmark: {exc}")
            return self._empty_result(timestamp=timestamp, fps=fps)

        if not results.multi_face_landmarks:
            return self._empty_result(timestamp=timestamp, fps=fps)

        face_landmarks_raw = results.multi_face_landmarks[0]
        face_landmarks = self._convert_landmarks(face_landmarks_raw, width, height)

        return {
            "face_detected": True,
            "face_landmarks": face_landmarks,
            "eye_landmarks": self.extract_eye_landmarks(face_landmarks),
            "mouth_landmarks": self.extract_mouth_landmarks(face_landmarks),
            "nose_landmarks": self.extract_nose_landmarks(face_landmarks),
            "head_points": self.extract_head_points(face_landmarks),
            "timestamp": timestamp,
            "fps": fps,
            "image_size": (width, height),
            "face_landmarks_raw": face_landmarks_raw,
        }

    def extract_eye_landmarks(
        self,
        face_landmarks: List[LandmarkPoint],
    ) -> Dict[str, List[LandmarkPoint]]:
        """Tach landmark mat trai va mat phai."""
        return {
            "left_eye": self._select_landmarks(face_landmarks, LEFT_EYE_INDEXES),
            "right_eye": self._select_landmarks(face_landmarks, RIGHT_EYE_INDEXES),
        }

    def extract_mouth_landmarks(
        self,
        face_landmarks: List[LandmarkPoint],
    ) -> List[LandmarkPoint]:
        """Tach cac landmark vung mieng."""
        return self._select_landmarks(face_landmarks, MOUTH_INDEXES)

    def extract_nose_landmarks(
        self,
        face_landmarks: List[LandmarkPoint],
    ) -> List[LandmarkPoint]:
        """Tach cac landmark vung mui."""
        return self._select_landmarks(face_landmarks, NOSE_INDEXES)

    def extract_head_points(
        self,
        face_landmarks: List[LandmarkPoint],
    ) -> Dict[str, LandmarkPoint]:
        """Tach cac diem chinh de phuc vu Head Pose Estimation sau nay."""
        return {
            name: face_landmarks[index]
            for name, index in HEAD_POSE_INDEXES.items()
            if index < len(face_landmarks)
        }

    def release(self) -> None:
        """Giai phong tai nguyen MediaPipe Face Mesh."""
        self.face_mesh.close()

    def close(self) -> None:
        """Alias de tuong thich voi cac wrapper MediaPipe cu."""
        self.release()

    def draw_face_mesh(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        draw_contours: bool = True,
        draw_tesselation: bool = True,
        draw_iris: bool = True,
    ) -> np.ndarray:
        """Ve mesh/contour MediaPipe len frame BGR de debug landmark."""
        if not detection or not detection.get("face_detected"):
            return frame

        face_landmarks_raw = detection.get("face_landmarks_raw")
        if face_landmarks_raw is None:
            return frame

        if draw_tesselation:
            self.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_raw,
                connections=self.face_mesh_module.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mesh_drawing_spec,
            )

        if draw_contours:
            self.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_raw,
                connections=self.face_mesh_module.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.contour_drawing_spec,
            )

        has_iris_landmarks = len(face_landmarks_raw.landmark) > 468
        if (
            draw_iris
            and has_iris_landmarks
            and hasattr(self.face_mesh_module, "FACEMESH_IRISES")
        ):
            self.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks_raw,
                connections=self.face_mesh_module.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.iris_drawing_spec,
            )

        return frame

    def _is_valid_frame(self, frame: Optional[np.ndarray]) -> bool:
        """Dam bao input la anh RGB hop le va khong lam crash pipeline."""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if frame.size == 0:
            return False
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        return True

    def _empty_result(
        self,
        timestamp: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Output thong nhat khi khong thay mat hoac frame khong hop le."""
        return {
            "face_detected": False,
            "face_landmarks": None,
            "eye_landmarks": None,
            "mouth_landmarks": None,
            "nose_landmarks": None,
            "head_points": None,
            "timestamp": timestamp,
            "fps": fps,
            "image_size": None,
            "face_landmarks_raw": None,
        }

    def _convert_landmarks(
        self,
        face_landmarks_raw: Any,
        width: int,
        height: int,
    ) -> List[LandmarkPoint]:
        """Chuyen landmark MediaPipe sang dict gom toa do normalized va pixel."""
        converted_landmarks = []

        for index, landmark in enumerate(face_landmarks_raw.landmark):
            converted_landmarks.append(
                {
                    "index": index,
                    "x": int(landmark.x * width),
                    "y": int(landmark.y * height),
                    "z": float(landmark.z),
                    "x_norm": float(landmark.x),
                    "y_norm": float(landmark.y),
                }
            )

        return converted_landmarks

    def _select_landmarks(
        self,
        face_landmarks: List[LandmarkPoint],
        indexes: List[int],
    ) -> List[LandmarkPoint]:
        """Lay cac landmark theo danh sach index da khai bao."""
        return [
            face_landmarks[index]
            for index in indexes
            if index < len(face_landmarks)
        ]

    def __enter__(self) -> "LandmarkDetector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
