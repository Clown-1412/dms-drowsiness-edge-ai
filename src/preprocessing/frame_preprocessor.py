from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class BoTienXuLyKhungHinh:
    """Lop tien xu ly frame truoc khi dua sang Face/Landmark Detection."""

    def __init__(
        self,
        kich_thuoc_dich: Tuple[int, int] = (640, 480),
        nguong_fps: float = 15.0,
        phuong_phap_giam_nhieu: str = "gaussian",
        kich_thuoc_kernel_gaussian: Tuple[int, int] = (5, 5),
    ):
        self.kich_thuoc_dich = kich_thuoc_dich
        self.nguong_fps = nguong_fps
        self.phuong_phap_giam_nhieu = phuong_phap_giam_nhieu.lower()
        self.kich_thuoc_kernel_gaussian = kich_thuoc_kernel_gaussian
        self._kiem_tra_cau_hinh()

    def _kiem_tra_cau_hinh(self) -> None:
        """Kiem tra cau hinh tien xu ly ngay khi khoi tao."""
        chieu_rong, chieu_cao = self.kich_thuoc_dich
        if chieu_rong <= 0 or chieu_cao <= 0:
            raise ValueError("kich_thuoc_dich phai co width va height lon hon 0")

        chieu_rong_kernel, chieu_cao_kernel = self.kich_thuoc_kernel_gaussian
        if chieu_rong_kernel <= 0 or chieu_cao_kernel <= 0:
            raise ValueError("kich_thuoc_kernel_gaussian phai lon hon 0")
        if chieu_rong_kernel % 2 == 0 or chieu_cao_kernel % 2 == 0:
            raise ValueError("kich_thuoc_kernel_gaussian phai la so le")

        if self.phuong_phap_giam_nhieu not in {"gaussian", "nlmeans", "none"}:
            raise ValueError("phuong_phap_giam_nhieu chi ho tro: gaussian, nlmeans, none")

    def _kiem_tra_khung_hinh(self, khung_hinh: Optional[np.ndarray]) -> None:
        """Kiem tra frame co ton tai va co du lieu anh hop le."""
        if khung_hinh is None:
            raise ValueError("Frame khong hop le: frame is None")
        if not isinstance(khung_hinh, np.ndarray):
            raise TypeError("Frame phai la numpy.ndarray")
        if khung_hinh.size == 0:
            raise ValueError("Frame khong hop le: frame rong")
        if len(khung_hinh.shape) != 3 or khung_hinh.shape[2] != 3:
            raise ValueError("Frame phai co 3 kenh mau BGR")

    def thay_doi_kich_thuoc_khung_hinh(self, khung_hinh: np.ndarray) -> np.ndarray:
        """Resize frame ve kich thuoc cau hinh san."""
        return cv2.resize(khung_hinh, self.kich_thuoc_dich, interpolation=cv2.INTER_AREA)

    def giam_nhieu_khung_hinh(self, khung_hinh: np.ndarray) -> np.ndarray:
        """Giam nhieu co ban cho frame."""
        if self.phuong_phap_giam_nhieu == "none":
            return khung_hinh

        if self.phuong_phap_giam_nhieu == "nlmeans":
            return cv2.fastNlMeansDenoisingColored(
                khung_hinh,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        return cv2.GaussianBlur(khung_hinh, self.kich_thuoc_kernel_gaussian, 0)

    def chuyen_bgr_sang_rgb(self, khung_hinh: np.ndarray) -> np.ndarray:
        """Chuyen frame tu BGR cua OpenCV sang RGB cho cac model CV/AI."""
        return cv2.cvtColor(khung_hinh, cv2.COLOR_BGR2RGB)

    def kiem_tra_fps(self, fps: float) -> bool:
        """
        Kiem tra FPS co dat nguong hay khong.

        Khong dung chuong trinh khi FPS thap, chi in canh bao de debug hieu nang.
        """
        fps_on_dinh = fps >= self.nguong_fps

        # FPS = 0 thuong xay ra o frame dau tien khi chua du du lieu do.
        if 0 < fps < self.nguong_fps:
            print(f"[CANH BAO] FPS thap: {fps:.2f} < {self.nguong_fps:.2f}")

        return fps_on_dinh

    def xu_ly(
        self,
        khung_hinh: np.ndarray,
        moc_thoi_gian: Optional[float] = None,
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
        self._kiem_tra_khung_hinh(khung_hinh)

        khung_hinh_da_resize = self.thay_doi_kich_thuoc_khung_hinh(khung_hinh)
        khung_hinh_da_giam_nhieu = self.giam_nhieu_khung_hinh(khung_hinh_da_resize)
        khung_hinh_da_xu_ly = self.chuyen_bgr_sang_rgb(khung_hinh_da_giam_nhieu)
        fps_on_dinh = self.kiem_tra_fps(fps)

        chieu_cao_khung_hinh, chieu_rong_khung_hinh = khung_hinh_da_xu_ly.shape[:2]

        return {
            "processed_frame": khung_hinh_da_xu_ly,
            "timestamp": moc_thoi_gian,
            "fps": fps,
            "frame_size": (chieu_rong_khung_hinh, chieu_cao_khung_hinh),
            "is_fps_stable": fps_on_dinh,
        }
