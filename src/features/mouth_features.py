import math
from typing import Any, Dict, List

from src.features.geometry import danh_sach_diem_hop_le, khoang_cach_euclid


class BoTrichXuatDacTrungMieng:
    """Tinh cac dac trung vung mieng theo tung frame."""

    def __init__(self, nguong_mieng_mo: float = 0.60):
        self.nguong_mieng_mo = nguong_mieng_mo

    def tinh_mar(self, mouth_landmarks: List[Dict[str, Any]]) -> float:
        """
        Tinh Mouth Aspect Ratio tu 8 diem vung mieng.

        MAR = (vertical_1 + vertical_2 + vertical_3) / (2 * horizontal)
        """
        if not danh_sach_diem_hop_le(mouth_landmarks, 8):
            return 0.0

        p1, p2, p3, p4, p5, p6, p7, p8 = mouth_landmarks[:8]
        horizontal = khoang_cach_euclid(p1, p5)
        if horizontal <= 0.0:
            return 0.0

        vertical_1 = khoang_cach_euclid(p2, p8)
        vertical_2 = khoang_cach_euclid(p3, p7)
        vertical_3 = khoang_cach_euclid(p4, p6)

        return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)

    def trich_xuat(self, mouth_landmarks: Any) -> Dict[str, Any]:
        """Tra ve MAR va trang thai mieng mo."""
        ket_qua_mac_dinh = {
            "MAR": 0.0,
            "mouth_open": False,
            "is_valid": False,
            "error": None,
        }

        if not danh_sach_diem_hop_le(mouth_landmarks, 8):
            ket_qua_mac_dinh["error"] = "MISSING_MOUTH_POINTS"
            return ket_qua_mac_dinh

        horizontal = khoang_cach_euclid(mouth_landmarks[0], mouth_landmarks[4])
        if horizontal <= 0.0:
            ket_qua_mac_dinh["error"] = "INVALID_MOUTH_WIDTH"
            return ket_qua_mac_dinh

        mar = self.tinh_mar(mouth_landmarks)
        if not math.isfinite(mar) or mar < 0.0:
            ket_qua_mac_dinh.update(
                {
                    "MAR": mar,
                    "error": "INVALID_MAR",
                }
            )
            return ket_qua_mac_dinh

        return {
            "MAR": mar,
            "mouth_open": mar > self.nguong_mieng_mo,
            "is_valid": True,
            "error": None,
        }
