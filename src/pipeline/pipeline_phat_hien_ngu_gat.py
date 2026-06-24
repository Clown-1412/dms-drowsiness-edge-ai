from typing import Any, Dict

from src.camera.camera_stream import LuongCamera
from src.detection.landmark_detector import BoPhatHienDiemMat
from src.preprocessing.frame_preprocessor import BoTienXuLyKhungHinh


class PipelinePhatHienNguGat:
    """Dieu phoi 3 layer demo: Camera, Pre-processing, Landmark Detection."""

    def __init__(
        self,
        luong_camera: LuongCamera,
        bo_tien_xu_ly: BoTienXuLyKhungHinh,
        bo_phat_hien_diem_mat: BoPhatHienDiemMat,
    ):
        self.luong_camera = luong_camera
        self.bo_tien_xu_ly = bo_tien_xu_ly
        self.bo_phat_hien_diem_mat = bo_phat_hien_diem_mat

    def bat_dau(self) -> "PipelinePhatHienNguGat":
        """Mo camera/video truoc khi xu ly cac vong lap."""
        self.luong_camera.bat_dau()
        return self

    def xu_ly_mot_vong(self) -> Dict[str, Any]:
        """Xu ly mot frame qua toan bo pipeline va tra ve ket qua hop nhat."""
        ket_qua_camera = self.luong_camera.doc_khung_hinh()

        ket_qua_tien_xu_ly = self.bo_tien_xu_ly.xu_ly(
            khung_hinh=ket_qua_camera["frame"],
            moc_thoi_gian=ket_qua_camera["timestamp"],
            fps=ket_qua_camera["fps"],
        )

        ket_qua_phat_hien = self.bo_phat_hien_diem_mat.phat_hien(
            khung_hinh_da_xu_ly=ket_qua_tien_xu_ly["processed_frame"],
            moc_thoi_gian=ket_qua_tien_xu_ly["timestamp"],
            fps=ket_qua_tien_xu_ly["fps"],
        )

        ket_qua_vong_lap = self._hop_nhat_ket_qua(
            ket_qua_camera,
            ket_qua_tien_xu_ly,
            ket_qua_phat_hien,
        )
        return ket_qua_vong_lap

    def giai_phong(self) -> None:
        """Giai phong tai nguyen cua cac layer co tai nguyen ben ngoai."""
        if hasattr(self.bo_phat_hien_diem_mat, "giai_phong"):
            self.bo_phat_hien_diem_mat.giai_phong()
        self.luong_camera.giai_phong()

    def _hop_nhat_ket_qua(self, *cac_ket_qua: Dict[str, Any]) -> Dict[str, Any]:
        ket_qua_vong_lap: Dict[str, Any] = {}
        for ket_qua in cac_ket_qua:
            ket_qua_vong_lap.update(ket_qua)
        return ket_qua_vong_lap

    def __enter__(self) -> "PipelinePhatHienNguGat":
        return self.bat_dau()

    def __exit__(self, loai_loi, gia_tri_loi, vet_loi) -> None:
        self.giai_phong()
