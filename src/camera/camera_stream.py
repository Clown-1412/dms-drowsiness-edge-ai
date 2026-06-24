import time
from typing import Any, Dict, Optional, Union

import cv2

from src.utils.fps import BoDemFPS


NguonCamera = Union[int, str]


class LuongCamera:
    """Lop nhan frame tu webcam hoac file video bang OpenCV."""

    def __init__(
        self,
        ma_camera: NguonCamera = 0,
        chieu_rong: Optional[int] = None,
        chieu_cao: Optional[int] = None,
        fps_muc_tieu: Optional[int] = None,
        cua_so_fps_trung_binh: int = 30,
    ):
        self.ma_camera = ma_camera
        self.chieu_rong = chieu_rong
        self.chieu_cao = chieu_cao
        self.fps_muc_tieu = fps_muc_tieu
        self.bo_doc: Optional[cv2.VideoCapture] = None
        self.bo_dem_fps = BoDemFPS(cua_so_trung_binh=cua_so_fps_trung_binh)

    def bat_dau(self) -> "LuongCamera":
        """Mo nguon camera/video va cau hinh cac tham so co ban."""
        if self.da_mo():
            return self

        if isinstance(self.ma_camera, int):
            self.bo_doc = cv2.VideoCapture(self.ma_camera, cv2.CAP_DSHOW)
        else:
            self.bo_doc = cv2.VideoCapture(self.ma_camera)

        if self.chieu_rong is not None:
            self.bo_doc.set(cv2.CAP_PROP_FRAME_WIDTH, self.chieu_rong)
        if self.chieu_cao is not None:
            self.bo_doc.set(cv2.CAP_PROP_FRAME_HEIGHT, self.chieu_cao)
        if self.fps_muc_tieu is not None:
            self.bo_doc.set(cv2.CAP_PROP_FPS, self.fps_muc_tieu)

        if not self.bo_doc.isOpened():
            self.giai_phong()
            raise RuntimeError(f"Khong the mo camera/video source: {self.ma_camera}")

        self.bo_dem_fps.dat_lai()
        return self

    def doc_khung_hinh(self) -> Dict[str, Any]:
        """
        Doc mot frame tu stream.

        Output chuan cho lop tiep theo:
        {
            "frame": frame_bgr,
            "timestamp": timestamp,
            "fps": fps,
        }
        """
        if not self.da_mo():
            raise RuntimeError("LuongCamera chua duoc bat_dau() hoac camera da bi dong.")

        thanh_cong, khung_hinh = self.bo_doc.read()
        moc_thoi_gian = time.time()

        if not thanh_cong or khung_hinh is None:
            raise RuntimeError("Khong doc duoc frame tu camera/video.")

        fps = self.bo_dem_fps.cap_nhat(moc_thoi_gian)

        return {
            "frame": khung_hinh,
            "timestamp": moc_thoi_gian,
            "fps": fps,
        }

    def lay_fps(self) -> float:
        """Lay FPS trung binh hien tai cua stream."""
        return self.bo_dem_fps.lay_fps()

    def da_mo(self) -> bool:
        """Kiem tra camera/video co dang mo hay khong."""
        return self.bo_doc is not None and self.bo_doc.isOpened()

    def giai_phong(self) -> None:
        """Giai phong tai nguyen camera/video."""
        if self.bo_doc is not None:
            self.bo_doc.release()
            self.bo_doc = None

    def __enter__(self) -> "LuongCamera":
        return self.bat_dau()

    def __exit__(self, loai_loi, gia_tri_loi, vet_loi) -> None:
        self.giai_phong()
