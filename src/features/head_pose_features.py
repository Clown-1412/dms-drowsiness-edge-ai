from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.features.geometry import diem_hop_le


class BoTrichXuatDacTrungTuTheDau:
    """Uoc luong pitch/yaw/roll cua dau bang cv2.solvePnP."""

    CAC_TEN_DIEM = (
        "nose_tip",
        "chin",
        "left_eye_corner",
        "right_eye_corner",
        "left_mouth_corner",
        "right_mouth_corner",
    )

    def __init__(
        self,
        nguong_cui_dau: float = 20.0,
        nguong_quay_dau: float = 25.0,
    ):
        self.nguong_cui_dau = nguong_cui_dau
        self.nguong_quay_dau = nguong_quay_dau
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

    def trich_xuat(self, head_points: Any, image_size: Any) -> Dict[str, Any]:
        """Tra ve cac goc head pose va co trang thai head_down/head_turned."""
        ket_qua_mac_dinh = {
            "head_pitch": None,
            "head_yaw": None,
            "head_roll": None,
            "head_down": False,
            "head_turned": False,
            "is_valid": False,
            "error": None,
        }

        kich_thuoc_anh = self._lay_kich_thuoc_anh(image_size)
        if kich_thuoc_anh is None:
            ket_qua_mac_dinh["error"] = "INVALID_IMAGE_SIZE"
            return ket_qua_mac_dinh

        image_points = self._lay_image_points(head_points)
        if image_points is None:
            ket_qua_mac_dinh["error"] = "MISSING_HEAD_POINTS"
            return ket_qua_mac_dinh

        try:
            rotation_vector = self._solve_pnp(image_points, kich_thuoc_anh)
            if rotation_vector is None:
                ket_qua_mac_dinh["error"] = "SOLVE_PNP_FAILED"
                return ket_qua_mac_dinh

            head_pitch, head_yaw, head_roll = self._tinh_goc_euler(rotation_vector)
        except cv2.error as loi:
            ket_qua_mac_dinh["error"] = f"HEAD_POSE_CV2_ERROR: {loi}"
            return ket_qua_mac_dinh
        except (TypeError, ValueError, np.linalg.LinAlgError) as loi:
            ket_qua_mac_dinh["error"] = f"HEAD_POSE_ERROR: {loi}"
            return ket_qua_mac_dinh
        except Exception as loi:
            ket_qua_mac_dinh["error"] = f"HEAD_POSE_UNEXPECTED_ERROR: {loi}"
            return ket_qua_mac_dinh

        return {
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "head_roll": head_roll,
            "head_down": head_pitch > self.nguong_cui_dau,
            "head_turned": abs(head_yaw) > self.nguong_quay_dau,
            "is_valid": True,
            "error": None,
        }

    def _lay_image_points(self, head_points: Any) -> Optional[np.ndarray]:
        if not isinstance(head_points, dict):
            return None

        cac_diem = []
        for ten_diem in self.CAC_TEN_DIEM:
            diem = head_points.get(ten_diem)
            if not diem_hop_le(diem):
                return None
            cac_diem.append((float(diem["x"]), float(diem["y"])))

        return np.array(cac_diem, dtype=np.float64)

    def _lay_kich_thuoc_anh(self, image_size: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            return None

        try:
            chieu_rong = float(image_size[0])
            chieu_cao = float(image_size[1])
        except (TypeError, ValueError):
            return None

        if chieu_rong <= 0.0 or chieu_cao <= 0.0:
            return None

        return chieu_rong, chieu_cao

    def _solve_pnp(
        self,
        image_points: np.ndarray,
        image_size: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        chieu_rong, chieu_cao = image_size
        tieu_cu = chieu_rong
        tam_camera = (chieu_rong / 2.0, chieu_cao / 2.0)
        camera_matrix = np.array(
            [
                [tieu_cu, 0.0, tam_camera[0]],
                [0.0, tieu_cu, tam_camera[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        thanh_cong, rotation_vector, _ = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not thanh_cong:
            return None

        return rotation_vector

    def _tinh_goc_euler(self, rotation_vector: np.ndarray) -> Tuple[float, float, float]:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        bi_suy_bien = sy < 1e-6

        if not bi_suy_bien:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return (
            self._chuan_hoa_goc_quanh_0(float(np.degrees(pitch))),
            float(np.degrees(yaw)),
            self._chuan_hoa_goc_quanh_0(float(np.degrees(roll))),
        )

    def _chuan_hoa_goc_quanh_0(self, goc: float) -> float:
        """
        Quy goc Euler ve vung de doc quanh 0 do.

        solvePnP co the tra mot nghiem tuong duong quanh +/-180 do khi he truc 3D
        cua model mat bi lech voi he truc camera. Voi demo DMS, goc -170 do gan
        nhu tuong duong voi +10 do, khong phai cui dau manh.
        """
        goc = (goc + 180.0) % 360.0 - 180.0
        if goc > 90.0:
            return goc - 180.0
        if goc < -90.0:
            return goc + 180.0
        return goc
