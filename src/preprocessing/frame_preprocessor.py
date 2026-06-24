from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class BoTienXuLyKhungHinh:
    """
    Lop tien xu ly frame truoc khi dua sang Face/Landmark Detection.

    Pipeline hien tai:
    1. Kiem tra frame dau vao co hop le khong.
    2. Resize frame ve kich thuoc chuan de cac layer sau xu ly on dinh.
    3. Giam nhieu theo cau hinh: gaussian, nlmeans, hoac none.
    4. Doi mau BGR -> RGB vi OpenCV doc BGR, con MediaPipe/model AI thuong dung RGB.
    5. Kiem tra FPS de canh bao hieu nang khi demo thoi gian thuc.
    6. Tra ve dictionary chuan cho layer phat hien landmark.
    """

    def __init__(
        self,
        kich_thuoc_dich: Tuple[int, int] = (640, 480),
        nguong_fps: float = 15.0,
        phuong_phap_giam_nhieu: str = "gaussian",
        kich_thuoc_kernel_gaussian: Tuple[int, int] = (5, 5),
    ):
        """
        Luu cau hinh tien xu ly va validate ngay khi tao object.

        kich_thuoc_dich: (width, height) cua frame sau resize.
        nguong_fps: FPS toi thieu de xem stream dang on dinh.
        phuong_phap_giam_nhieu: "gaussian", "nlmeans", hoac "none".
        kich_thuoc_kernel_gaussian: kernel blur, bat buoc la so le duong.
        """
        self.kich_thuoc_dich = kich_thuoc_dich
        self.nguong_fps = nguong_fps
        self.phuong_phap_giam_nhieu = phuong_phap_giam_nhieu.lower()
        self.kich_thuoc_kernel_gaussian = kich_thuoc_kernel_gaussian
        self._kiem_tra_cau_hinh()

    def _kiem_tra_cau_hinh(self) -> None:
        """
        Kiem tra cau hinh tien xu ly ngay khi khoi tao.

        Muc tieu la fail fast neu cau hinh sai, tranh loi kho debug trong vong lap
        camera thoi gian thuc.
        """
        # Resize can width/height duong de cv2.resize khong loi.
        chieu_rong, chieu_cao = self.kich_thuoc_dich
        if chieu_rong <= 0 or chieu_cao <= 0:
            raise ValueError("kich_thuoc_dich phai co width va height lon hon 0")

        # GaussianBlur yeu cau kernel duong va la so le tren ca hai chieu.
        chieu_rong_kernel, chieu_cao_kernel = self.kich_thuoc_kernel_gaussian
        if chieu_rong_kernel <= 0 or chieu_cao_kernel <= 0:
            raise ValueError("kich_thuoc_kernel_gaussian phai lon hon 0")
        if chieu_rong_kernel % 2 == 0 or chieu_cao_kernel % 2 == 0:
            raise ValueError("kich_thuoc_kernel_gaussian phai la so le")

        # Chi cho phep cac che do dang duoc implement trong giam_nhieu_khung_hinh().
        if self.phuong_phap_giam_nhieu not in {"gaussian", "nlmeans", "none"}:
            raise ValueError("phuong_phap_giam_nhieu chi ho tro: gaussian, nlmeans, none")

    def _kiem_tra_khung_hinh(self, khung_hinh: Optional[np.ndarray]) -> None:
        """
        Kiem tra frame co ton tai va co du lieu anh hop le.

        Dau vao hop le cua pipeline la anh BGR 3 kenh do cv2.VideoCapture tra ve.
        """
        if khung_hinh is None:
            raise ValueError("Frame khong hop le: frame is None")
        if not isinstance(khung_hinh, np.ndarray):
            raise TypeError("Frame phai la numpy.ndarray")
        if khung_hinh.size == 0:
            raise ValueError("Frame khong hop le: frame rong")
        if len(khung_hinh.shape) != 3 or khung_hinh.shape[2] != 3:
            raise ValueError("Frame phai co 3 kenh mau BGR")

    def thay_doi_kich_thuoc_khung_hinh(self, khung_hinh: np.ndarray) -> np.ndarray:
        """
        Resize frame ve kich thuoc cau hinh san.

        INTER_AREA phu hop khi can thu nho/dua frame webcam ve kich thuoc chuan,
        giup giam chi phi xu ly cho cac buoc phia sau.
        """
        return cv2.resize(khung_hinh, self.kich_thuoc_dich, interpolation=cv2.INTER_AREA)

    def giam_nhieu_khung_hinh(self, khung_hinh: np.ndarray) -> np.ndarray:
        """
        Giam nhieu co ban cho frame sau resize.

        gaussian: nhanh, phu hop demo realtime.
        nlmeans: loc nhieu manh hon nhung ton CPU hon.
        none: giu nguyen frame khi can benchmark hoac debug anh goc.
        """
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
        """
        Chuyen frame tu BGR cua OpenCV sang RGB cho cac model CV/AI.

        Buoc nay quan trong vi sai thu tu kenh mau co the lam model/MediaPipe nhan
        anh sai mau va giam chat luong phat hien.
        """
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
        # 1) Validate frame tho tu camera/file video truoc khi xu ly anh.
        self._kiem_tra_khung_hinh(khung_hinh)

        # 2) Chuan hoa kich thuoc frame de layer detection nhan input on dinh.
        khung_hinh_da_resize = self.thay_doi_kich_thuoc_khung_hinh(khung_hinh)

        # 3) Giam nhieu theo cau hinh hien tai.
        khung_hinh_da_giam_nhieu = self.giam_nhieu_khung_hinh(khung_hinh_da_resize)

        # 4) Chuyen BGR -> RGB cho MediaPipe/model AI.
        khung_hinh_da_xu_ly = self.chuyen_bgr_sang_rgb(khung_hinh_da_giam_nhieu)

        # 5) Kiem tra FPS de tra metadata va in canh bao neu stream qua cham.
        fps_on_dinh = self.kiem_tra_fps(fps)

        chieu_cao_khung_hinh, chieu_rong_khung_hinh = khung_hinh_da_xu_ly.shape[:2]

        # 6) Gom output theo schema chung ma pipeline landmark dang su dung.
        return {
            "processed_frame": khung_hinh_da_xu_ly,
            "timestamp": moc_thoi_gian,
            "fps": fps,
            "frame_size": (chieu_rong_khung_hinh, chieu_cao_khung_hinh),
            "is_fps_stable": fps_on_dinh,
        }
