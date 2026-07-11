from typing import Any, Dict, List

from src.features.geometry import danh_sach_diem_hop_le, khoang_cach_euclid


class BoTrichXuatDacTrungMat:
    """Tinh cac dac trung vung mat theo tung frame."""

    def __init__(self, nguong_mat_nham: float = 0.20):
        self.nguong_mat_nham = nguong_mat_nham

    def tinh_ear(self, diem_mat: List[Dict[str, Any]]) -> float:
        """
        Tinh Eye Aspect Ratio tu 6 diem: [p1, p2, p3, p4, p5, p6].

        EAR = (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))
        """
        if not danh_sach_diem_hop_le(diem_mat, 6):
            return 0.0

        khoang_doc_1 = khoang_cach_euclid(diem_mat[1], diem_mat[5])
        khoang_doc_2 = khoang_cach_euclid(diem_mat[2], diem_mat[4])
        khoang_ngang = khoang_cach_euclid(diem_mat[0], diem_mat[3])

        if khoang_ngang <= 0.0:
            return 0.0

        return (khoang_doc_1 + khoang_doc_2) / (2.0 * khoang_ngang)

    def trich_xuat(self, eye_landmarks: Any) -> Dict[str, Any]:
        """Tra ve EAR trai/phai, EAR trung binh va trang thai mat nham."""
        ket_qua_mac_dinh = {
            "left_EAR": 0.0,
            "right_EAR": 0.0,
            "avg_EAR": 0.0,
            "left_eye_closed": False,
            "right_eye_closed": False,
            "both_eyes_closed": False,
            "one_eye_closed": False,
            "eye_closed": False,
            "is_valid": False,
            "error": None,
        }

        if not isinstance(eye_landmarks, dict):
            ket_qua_mac_dinh["error"] = "INVALID_EYE_LANDMARKS"
            return ket_qua_mac_dinh

        mat_trai = eye_landmarks.get("left_eye")
        mat_phai = eye_landmarks.get("right_eye")
        if not danh_sach_diem_hop_le(mat_trai, 6) or not danh_sach_diem_hop_le(mat_phai, 6):
            ket_qua_mac_dinh["error"] = "MISSING_EYE_POINTS"
            return ket_qua_mac_dinh

        left_ear = self.tinh_ear(mat_trai)
        right_ear = self.tinh_ear(mat_phai)
        if left_ear <= 0.0 or right_ear <= 0.0:
            cac_ear = [ear for ear in (left_ear, right_ear) if ear > 0.0]
            ket_qua_mac_dinh.update(
                {
                    "left_EAR": left_ear,
                    "right_EAR": right_ear,
                    "avg_EAR": sum(cac_ear) / len(cac_ear) if cac_ear else 0.0,
                    "error": "INVALID_EAR",
                }
            )
            return ket_qua_mac_dinh

        avg_ear = (left_ear + right_ear) / 2.0
        left_eye_closed = left_ear < self.nguong_mat_nham
        right_eye_closed = right_ear < self.nguong_mat_nham
        both_eyes_closed = left_eye_closed and right_eye_closed
        one_eye_closed = left_eye_closed != right_eye_closed

        return {
            "left_EAR": left_ear,
            "right_EAR": right_ear,
            "avg_EAR": avg_ear,
            "left_eye_closed": left_eye_closed,
            "right_eye_closed": right_eye_closed,
            "both_eyes_closed": both_eyes_closed,
            "one_eye_closed": one_eye_closed,
            "eye_closed": both_eyes_closed,
            "is_valid": True,
            "error": None,
        }
