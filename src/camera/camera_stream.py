import time
from typing import Any, Dict, Optional, Union

import cv2

from src.utils.fps import FPSCounter


CameraSource = Union[int, str]


class CameraStream:
    """Lop nhan frame tu webcam hoac file video bang OpenCV."""

    def __init__(
        self,
        camera_id: CameraSource = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[int] = None,
        average_fps_window: int = 30,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.capture: Optional[cv2.VideoCapture] = None
        self.fps_counter = FPSCounter(average_window=average_fps_window)

    def start(self) -> "CameraStream":
        """Mo nguon camera/video va cau hinh cac tham so co ban."""
        if self.is_opened():
            return self

        self.capture = cv2.VideoCapture(self.camera_id)

        if self.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        if not self.capture.isOpened():
            self.release()
            raise RuntimeError(f"Khong the mo camera/video source: {self.camera_id}")

        self.fps_counter.reset()
        return self

    def read(self) -> Dict[str, Any]:
        """
        Doc mot frame tu stream.

        Output chuan cho lop tiep theo:
        {
            "frame": frame_bgr,
            "timestamp": timestamp,
            "fps": fps,
        }
        """
        if not self.is_opened():
            raise RuntimeError("CameraStream chua duoc start() hoac camera da bi dong.")

        ret, frame = self.capture.read()
        timestamp = time.time()

        if not ret or frame is None:
            raise RuntimeError("Khong doc duoc frame tu camera/video.")

        fps = self.fps_counter.update(timestamp)

        return {
            "frame": frame,
            "timestamp": timestamp,
            "fps": fps,
        }

    def get_fps(self) -> float:
        """Lay FPS trung binh hien tai cua stream."""
        return self.fps_counter.get_fps()

    def is_opened(self) -> bool:
        """Kiem tra camera/video co dang mo hay khong."""
        return self.capture is not None and self.capture.isOpened()

    def release(self) -> None:
        """Giai phong tai nguyen camera/video."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
