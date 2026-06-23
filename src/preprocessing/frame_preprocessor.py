from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class FramePreprocessor:
    """Lop tien xu ly frame truoc khi dua sang Face/Landmark Detection."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 480),
        fps_threshold: float = 15.0,
        denoise_method: str = "gaussian",
        gaussian_kernel_size: Tuple[int, int] = (5, 5),
    ):
        self.target_size = target_size
        self.fps_threshold = fps_threshold
        self.denoise_method = denoise_method.lower()
        self.gaussian_kernel_size = gaussian_kernel_size
        self._validate_config()

    def _validate_config(self) -> None:
        """Kiem tra cau hinh tien xu ly ngay khi khoi tao."""
        width, height = self.target_size
        if width <= 0 or height <= 0:
            raise ValueError("target_size phai co width va height lon hon 0")

        kernel_width, kernel_height = self.gaussian_kernel_size
        if kernel_width <= 0 or kernel_height <= 0:
            raise ValueError("gaussian_kernel_size phai lon hon 0")
        if kernel_width % 2 == 0 or kernel_height % 2 == 0:
            raise ValueError("gaussian_kernel_size phai la so le")

        if self.denoise_method not in {"gaussian", "nlmeans", "none"}:
            raise ValueError("denoise_method chi ho tro: gaussian, nlmeans, none")

    def _validate_frame(self, frame: Optional[np.ndarray]) -> None:
        """Kiem tra frame co ton tai va co du lieu anh hop le."""
        if frame is None:
            raise ValueError("Frame khong hop le: frame is None")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame phai la numpy.ndarray")
        if frame.size == 0:
            raise ValueError("Frame khong hop le: frame rong")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Frame phai co 3 kenh mau BGR")

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame ve kich thuoc cau hinh san."""
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

    def denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Giam nhieu co ban cho frame."""
        if self.denoise_method == "none":
            return frame

        if self.denoise_method == "nlmeans":
            return cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        return cv2.GaussianBlur(frame, self.gaussian_kernel_size, 0)

    def convert_bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Chuyen frame tu BGR cua OpenCV sang RGB cho cac model CV/AI."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def check_fps(self, fps: float) -> bool:
        """
        Kiem tra FPS co dat nguong hay khong.

        Khong dung chuong trinh khi FPS thap, chi in canh bao de debug hieu nang.
        """
        is_stable = fps >= self.fps_threshold

        # FPS = 0 thuong xay ra o frame dau tien khi chua du du lieu do.
        if 0 < fps < self.fps_threshold:
            print(f"[CANH BAO] FPS thap: {fps:.2f} < {self.fps_threshold:.2f}")

        return is_stable

    def process(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        fps: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Chay toan bo buoc tien xu ly cho mot frame.

        Output chuan cho lop tiep theo:
        {
            "processed_frame": frame_rgb,
            "timestamp": timestamp,
            "fps": fps,
            "frame_size": (width, height),
            "is_fps_stable": True/False,
        }
        """
        self._validate_frame(frame)

        resized_frame = self.resize_frame(frame)
        denoised_frame = self.denoise_frame(resized_frame)
        processed_frame = self.convert_bgr_to_rgb(denoised_frame)
        is_fps_stable = self.check_fps(fps)

        frame_height, frame_width = processed_frame.shape[:2]

        return {
            "processed_frame": processed_frame,
            "timestamp": timestamp,
            "fps": fps,
            "frame_size": (frame_width, frame_height),
            "is_fps_stable": is_fps_stable,
        }
