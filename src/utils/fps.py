import time
from collections import deque
from typing import Deque, Optional


class BoDemFPS:
    """Bo dem FPS co ho tro FPS tuc thoi va FPS trung binh truot."""

    def __init__(self, cua_so_trung_binh: int = 30):
        if cua_so_trung_binh <= 0:
            raise ValueError("cua_so_trung_binh phai lon hon 0")

        self.cua_so_trung_binh = cua_so_trung_binh
        self.thoi_gian_truoc: Optional[float] = None
        self.fps_hien_tai = 0.0
        self.danh_sach_fps: Deque[float] = deque(maxlen=cua_so_trung_binh)

    def dat_lai(self) -> None:
        """Dat lai bo dem FPS khi bat dau mot stream moi."""
        self.thoi_gian_truoc = None
        self.fps_hien_tai = 0.0
        self.danh_sach_fps.clear()

    def cap_nhat(self, moc_thoi_gian: Optional[float] = None) -> float:
        """Cap nhat FPS theo timestamp hien tai va tra ve FPS trung binh."""
        hien_tai = moc_thoi_gian if moc_thoi_gian is not None else time.time()

        if self.thoi_gian_truoc is None:
            self.thoi_gian_truoc = hien_tai
            return 0.0

        do_lech_thoi_gian = hien_tai - self.thoi_gian_truoc
        self.thoi_gian_truoc = hien_tai

        if do_lech_thoi_gian > 0:
            self.fps_hien_tai = 1.0 / do_lech_thoi_gian
            self.danh_sach_fps.append(self.fps_hien_tai)

        return self.lay_fps()

    def lay_fps(self) -> float:
        """Tra ve FPS trung binh trong cua so gan nhat."""
        if not self.danh_sach_fps:
            return self.fps_hien_tai
        return sum(self.danh_sach_fps) / len(self.danh_sach_fps)

    def lay_fps_hien_tai(self) -> float:
        """Tra ve FPS tuc thoi cua frame gan nhat."""
        return self.fps_hien_tai
