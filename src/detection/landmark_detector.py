from typing import Any, Dict, List, Optional

import mediapipe as mp
import numpy as np


# Cac index co ban cua MediaPipe Face Mesh.
CHI_SO_MAT_TRAI = [33, 160, 158, 133, 153, 144]
CHI_SO_MAT_PHAI = [362, 385, 387, 263, 373, 380]
CHI_SO_MIENG = [61, 81, 13, 311, 291, 402, 14, 178]
CHI_SO_MUI = [1, 4, 98, 327, 168]

# Cac diem thuong dung cho uoc luong head pose o cac layer sau.
CHI_SO_TU_THE_DAU = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


DiemMoc = Dict[str, float]


class BoPhatHienDiemMat:
    """Landmark Detection Layer dung MediaPipe Face Mesh."""

    def __init__(
        self,
        che_do_anh_tinh: bool = False,
        so_mat_toi_da: int = 1,
        lam_min_diem_moc: bool = True,
        do_tin_cay_phat_hien_toi_thieu: float = 0.5,
        do_tin_cay_theo_doi_toi_thieu: float = 0.5,
    ):
        try:
            self.module_luoi_mat = mp.solutions.face_mesh
            self.tien_ich_ve = mp.solutions.drawing_utils
        except AttributeError as loi:
            raise RuntimeError(
                "Mediapipe khong ho tro mp.solutions.face_mesh. "
                "Hay cai lai dependency bang: python -m pip install -r requirements.txt"
            ) from loi

        self.cau_hinh_ve_luoi = self.tien_ich_ve.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1,
        )
        self.cau_hinh_ve_duong_bao = self.tien_ich_ve.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1,
        )
        self.cau_hinh_ve_mong_mat = self.tien_ich_ve.DrawingSpec(
            color=(255, 255, 255),
            thickness=1,
            circle_radius=1,
        )
        self.luoi_mat = self.module_luoi_mat.FaceMesh(
            static_image_mode=che_do_anh_tinh,
            max_num_faces=so_mat_toi_da,
            refine_landmarks=lam_min_diem_moc,
            min_detection_confidence=do_tin_cay_phat_hien_toi_thieu,
            min_tracking_confidence=do_tin_cay_theo_doi_toi_thieu,
        )

    def phat_hien(
        self,
        khung_hinh_da_xu_ly: Optional[np.ndarray],
        moc_thoi_gian: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phat hien mat va tach cac nhom landmark can thiet.

        processed_frame phai la anh RGB da di qua BoTienXuLyKhungHinh.
        """
        if not self._khung_hinh_hop_le(khung_hinh_da_xu_ly):
            return self._ket_qua_rong(moc_thoi_gian=moc_thoi_gian, fps=fps)

        chieu_cao, chieu_rong = khung_hinh_da_xu_ly.shape[:2]

        try:
            khung_hinh_rgb = np.ascontiguousarray(khung_hinh_da_xu_ly)
            khung_hinh_rgb.flags.writeable = False
            ket_qua = self.luoi_mat.process(khung_hinh_rgb)
        except Exception as loi:
            print(f"[CANH BAO] Loi khi phat hien landmark: {loi}")
            return self._ket_qua_rong(moc_thoi_gian=moc_thoi_gian, fps=fps)

        if not ket_qua.multi_face_landmarks:
            return self._ket_qua_rong(moc_thoi_gian=moc_thoi_gian, fps=fps)

        diem_mat_goc = ket_qua.multi_face_landmarks[0]
        diem_mat = self._chuyen_doi_diem_moc(diem_mat_goc, chieu_rong, chieu_cao)

        return {
            "face_detected": True,
            "face_landmarks": diem_mat,
            "eye_landmarks": self.trich_xuat_diem_mat(diem_mat),
            "mouth_landmarks": self.trich_xuat_diem_mieng(diem_mat),
            "nose_landmarks": self.trich_xuat_diem_mui(diem_mat),
            "head_points": self.trich_xuat_diem_tu_the_dau(diem_mat),
            "timestamp": moc_thoi_gian,
            "fps": fps,
            "image_size": (chieu_rong, chieu_cao),
            "face_landmarks_raw": diem_mat_goc,
        }

    def trich_xuat_diem_mat(
        self,
        diem_mat: List[DiemMoc],
    ) -> Dict[str, List[DiemMoc]]:
        """Tach landmark mat trai va mat phai."""
        return {
            "left_eye": self._chon_diem_moc(diem_mat, CHI_SO_MAT_TRAI),
            "right_eye": self._chon_diem_moc(diem_mat, CHI_SO_MAT_PHAI),
        }

    def trich_xuat_diem_mieng(
        self,
        diem_mat: List[DiemMoc],
    ) -> List[DiemMoc]:
        """Tach cac landmark vung mieng."""
        return self._chon_diem_moc(diem_mat, CHI_SO_MIENG)

    def trich_xuat_diem_mui(
        self,
        diem_mat: List[DiemMoc],
    ) -> List[DiemMoc]:
        """Tach cac landmark vung mui."""
        return self._chon_diem_moc(diem_mat, CHI_SO_MUI)

    def trich_xuat_diem_tu_the_dau(
        self,
        diem_mat: List[DiemMoc],
    ) -> Dict[str, DiemMoc]:
        """Tach cac diem chinh de phuc vu Head Pose Estimation sau nay."""
        return {
            ten: diem_mat[chi_so]
            for ten, chi_so in CHI_SO_TU_THE_DAU.items()
            if chi_so < len(diem_mat)
        }

    def giai_phong(self) -> None:
        """Giai phong tai nguyen MediaPipe Face Mesh."""
        self.luoi_mat.close()

    def ve_luoi_mat(
        self,
        khung_hinh: np.ndarray,
        ket_qua_phat_hien: Dict[str, Any],
        ve_duong_bao: bool = True,
        ve_luoi_tam_giac: bool = True,
        ve_mong_mat: bool = False,
    ) -> np.ndarray:
        """Ve mesh/contour MediaPipe len frame BGR de debug landmark."""
        if not ket_qua_phat_hien or not ket_qua_phat_hien.get("face_detected"):
            return khung_hinh

        diem_mat_goc = ket_qua_phat_hien.get("face_landmarks_raw")
        if diem_mat_goc is None:
            return khung_hinh

        if ve_luoi_tam_giac:
            self.tien_ich_ve.draw_landmarks(
                image=khung_hinh,
                landmark_list=diem_mat_goc,
                connections=self.module_luoi_mat.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.cau_hinh_ve_luoi,
            )

        if ve_duong_bao:
            self.tien_ich_ve.draw_landmarks(
                image=khung_hinh,
                landmark_list=diem_mat_goc,
                connections=self.module_luoi_mat.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.cau_hinh_ve_duong_bao,
            )

        co_diem_moc_mong_mat = len(diem_mat_goc.landmark) > 468
        if (
            ve_mong_mat
            and co_diem_moc_mong_mat
            and hasattr(self.module_luoi_mat, "FACEMESH_IRISES")
        ):
            self.tien_ich_ve.draw_landmarks(
                image=khung_hinh,
                landmark_list=diem_mat_goc,
                connections=self.module_luoi_mat.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.cau_hinh_ve_mong_mat,
            )

        return khung_hinh

    def _khung_hinh_hop_le(self, khung_hinh: Optional[np.ndarray]) -> bool:
        """Dam bao input la anh RGB hop le va khong lam crash pipeline."""
        if khung_hinh is None:
            return False
        if not isinstance(khung_hinh, np.ndarray):
            return False
        if khung_hinh.size == 0:
            return False
        if len(khung_hinh.shape) != 3 or khung_hinh.shape[2] != 3:
            return False
        return True

    def _ket_qua_rong(
        self,
        moc_thoi_gian: Optional[float] = None,
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
            "timestamp": moc_thoi_gian,
            "fps": fps,
            "image_size": None,
            "face_landmarks_raw": None,
        }

    def _chuyen_doi_diem_moc(
        self,
        diem_mat_goc: Any,
        chieu_rong: int,
        chieu_cao: int,
    ) -> List[DiemMoc]:
        """Chuyen landmark MediaPipe sang dict gom toa do normalized va pixel."""
        danh_sach_diem_da_chuyen = []

        for chi_so, diem_moc in enumerate(diem_mat_goc.landmark):
            danh_sach_diem_da_chuyen.append(
                {
                    "index": chi_so,
                    "x": int(diem_moc.x * chieu_rong),
                    "y": int(diem_moc.y * chieu_cao),
                    "z": float(diem_moc.z),
                    "x_norm": float(diem_moc.x),
                    "y_norm": float(diem_moc.y),
                }
            )

        return danh_sach_diem_da_chuyen

    def _chon_diem_moc(
        self,
        diem_mat: List[DiemMoc],
        danh_sach_chi_so: List[int],
    ) -> List[DiemMoc]:
        """Lay cac landmark theo danh sach index da khai bao."""
        return [
            diem_mat[chi_so]
            for chi_so in danh_sach_chi_so
            if chi_so < len(diem_mat)
        ]

    def __enter__(self) -> "BoPhatHienDiemMat":
        return self

    def __exit__(self, loai_loi, gia_tri_loi, vet_loi) -> None:
        self.giai_phong()
