from typing import Any, Dict, Optional

from src.features.eye import BoTrichXuatDacTrungMat
from src.features.head_pose import BoTrichXuatDacTrungTuTheDau
from src.features.mouth import BoTrichXuatDacTrungMieng


class BoTrichXuatDacTrung:
    """Layer tong hop de trich xuat feature tu output landmark detection."""

    def __init__(
        self,
        bo_trich_xuat_mat: Optional[BoTrichXuatDacTrungMat] = None,
        bo_trich_xuat_mieng: Optional[BoTrichXuatDacTrungMieng] = None,
        bo_trich_xuat_tu_the_dau: Optional[BoTrichXuatDacTrungTuTheDau] = None,
    ):
        self.bo_trich_xuat_mat = bo_trich_xuat_mat or BoTrichXuatDacTrungMat()
        self.bo_trich_xuat_mieng = bo_trich_xuat_mieng or BoTrichXuatDacTrungMieng()
        self.bo_trich_xuat_tu_the_dau = (
            bo_trich_xuat_tu_the_dau or BoTrichXuatDacTrungTuTheDau()
        )

    def trich_xuat(self, ket_qua_phat_hien: Dict[str, Any]) -> Dict[str, Any]:
        """Trich xuat eye/mouth/head pose features tu ket qua phat hien landmark."""
        if not isinstance(ket_qua_phat_hien, dict):
            ket_qua_phat_hien = {}
        moc_thoi_gian = ket_qua_phat_hien.get("timestamp")
        fps = ket_qua_phat_hien.get("fps")
        image_size = ket_qua_phat_hien.get("image_size")
        face_detected = bool(ket_qua_phat_hien.get("face_detected"))

        if not face_detected:
            return {
                "timestamp": moc_thoi_gian,
                "fps": fps,
                "face_detected": False,
                "eye_features": None,
                "mouth_features": None,
                "head_pose_features": None,
                "quality_features": {
                    "is_feature_valid": False,
                    "feature_error": "NO_FACE",
                    "image_size": image_size,
                    "fps": fps,
                },
            }

        eye_features = self.bo_trich_xuat_mat.trich_xuat(
            ket_qua_phat_hien.get("eye_landmarks")
        )
        mouth_features = self.bo_trich_xuat_mieng.trich_xuat(
            ket_qua_phat_hien.get("mouth_landmarks")
        )
        head_pose_features = self.bo_trich_xuat_tu_the_dau.trich_xuat(
            ket_qua_phat_hien.get("head_points"),
            image_size,
        )

        is_feature_valid = bool(
            eye_features.get("is_valid") or mouth_features.get("is_valid")
        )
        feature_error = self._tong_hop_loi(
            eye_features,
            mouth_features,
            head_pose_features,
            is_feature_valid,
        )

        return {
            "timestamp": moc_thoi_gian,
            "fps": fps,
            "face_detected": face_detected,
            "eye_features": eye_features,
            "mouth_features": mouth_features,
            "head_pose_features": head_pose_features,
            "quality_features": {
                "is_feature_valid": is_feature_valid,
                "feature_error": feature_error,
                "image_size": image_size,
                "fps": fps,
            },
        }

    def _tong_hop_loi(
        self,
        eye_features: Dict[str, Any],
        mouth_features: Dict[str, Any],
        head_pose_features: Dict[str, Any],
        is_feature_valid: bool,
    ) -> Optional[str]:
        cac_loi = [
            eye_features.get("error"),
            mouth_features.get("error"),
            head_pose_features.get("error"),
        ]
        cac_loi = [loi for loi in cac_loi if loi]

        if not cac_loi:
            return None
        if is_feature_valid:
            return ";".join(cac_loi)
        return cac_loi[0]
