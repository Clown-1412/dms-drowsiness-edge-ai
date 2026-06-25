import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2


# Cho phep chay file truc tiep bang: python app/demo_feature_extraction.py
GOC_DU_AN = Path(__file__).resolve().parents[1]
if str(GOC_DU_AN) not in sys.path:
    sys.path.insert(0, str(GOC_DU_AN))

from src.camera.camera_stream import LuongCamera
from src.detection.landmark_detector import BoPhatHienDiemMat
from src.features.feature_extractor import BoTrichXuatDacTrung
from src.pipeline.pipeline_phat_hien_ngu_gat import PipelinePhatHienNguGat
from src.preprocessing.frame_preprocessor import BoTienXuLyKhungHinh


NguonCamera = Union[int, str]


def phan_tich_nguon_camera(nguon: str) -> NguonCamera:
    """Chuyen source dang so thanh camera_id, con lai giu la duong dan video."""
    try:
        return int(nguon)
    except ValueError:
        return nguon


def tao_bo_doc_tham_so() -> argparse.ArgumentParser:
    bo_doc_tham_so = argparse.ArgumentParser(
        description="Demo Feature Extraction Layer trong pipeline DMS"
    )
    bo_doc_tham_so.add_argument("--source", dest="nguon", default="0", help="Camera ID hoac duong dan video")
    bo_doc_tham_so.add_argument("--width", dest="chieu_rong", type=int, default=640, help="Chieu rong frame")
    bo_doc_tham_so.add_argument("--height", dest="chieu_cao", type=int, default=480, help="Chieu cao frame")
    bo_doc_tham_so.add_argument("--target-fps", dest="fps_muc_tieu", type=int, default=30, help="FPS camera mong muon")
    bo_doc_tham_so.add_argument("--fps-threshold", dest="nguong_fps", type=float, default=15.0, help="Nguong canh bao FPS thap")
    bo_doc_tham_so.add_argument("--no-tesselation", dest="tat_luoi_tam_giac", action="store_true", help="Tat luoi tam giac Face Mesh")
    bo_doc_tham_so.add_argument("--draw-iris", dest="ve_mong_mat", action="store_true", help="Ve iris neu MediaPipe tra ve")
    return bo_doc_tham_so


def tao_pipeline(tham_so) -> PipelinePhatHienNguGat:
    kich_thuoc_dich = (tham_so.chieu_rong, tham_so.chieu_cao)
    luong_camera = LuongCamera(
        ma_camera=phan_tich_nguon_camera(tham_so.nguon),
        chieu_rong=tham_so.chieu_rong,
        chieu_cao=tham_so.chieu_cao,
        fps_muc_tieu=tham_so.fps_muc_tieu,
    )
    bo_tien_xu_ly = BoTienXuLyKhungHinh(
        kich_thuoc_dich=kich_thuoc_dich,
        nguong_fps=tham_so.nguong_fps,
        phuong_phap_giam_nhieu="gaussian",
    )
    bo_phat_hien = BoPhatHienDiemMat(
        che_do_anh_tinh=False,
        so_mat_toi_da=1,
        lam_min_diem_moc=True,
        do_tin_cay_phat_hien_toi_thieu=0.5,
        do_tin_cay_theo_doi_toi_thieu=0.5,
    )
    bo_trich_xuat_dac_trung = BoTrichXuatDacTrung()

    return PipelinePhatHienNguGat(
        luong_camera=luong_camera,
        bo_tien_xu_ly=bo_tien_xu_ly,
        bo_phat_hien_diem_mat=bo_phat_hien,
        bo_trich_xuat_dac_trung=bo_trich_xuat_dac_trung,
    )


def dinh_dang_so(gia_tri: Optional[float], so_chu_so: int = 3) -> str:
    if gia_tri is None:
        return "None"
    try:
        return f"{float(gia_tri):.{so_chu_so}f}"
    except (TypeError, ValueError):
        return "None"


def lay_feature_bool(nhom_feature: Optional[Dict[str, Any]], ten: str) -> bool:
    if not nhom_feature:
        return False
    return bool(nhom_feature.get(ten))


def ve_dong(
    khung_hinh,
    noi_dung: str,
    vi_tri_x: int,
    vi_tri_y: int,
    font_scale: float,
    mau=(255, 255, 255),
) -> None:
    cv2.putText(
        khung_hinh,
        noi_dung,
        (vi_tri_x, vi_tri_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        mau,
        1,
        cv2.LINE_AA,
    )


def ve_thong_tin_dac_trung(khung_hinh, ket_qua_dac_trung: Dict[str, Any]) -> None:
    chieu_cao, chieu_rong = khung_hinh.shape[:2]
    face_detected = bool(ket_qua_dac_trung.get("face_detected"))
    fps = ket_qua_dac_trung.get("fps") or 0.0
    eye_features = ket_qua_dac_trung.get("eye_features") or {}
    mouth_features = ket_qua_dac_trung.get("mouth_features") or {}
    head_pose_features = ket_qua_dac_trung.get("head_pose_features") or {}

    mau_trang_thai = (0, 255, 0) if face_detected else (0, 0, 255)
    cac_dong = [
        (
            f"FPS: {fps:.1f} | face_detected: {face_detected}",
            mau_trang_thai,
        )
    ]
    if not face_detected:
        cac_dong.append(("No face detected", (0, 0, 255)))
    else:
        left_ear = dinh_dang_so(eye_features.get("left_EAR"))
        right_ear = dinh_dang_so(eye_features.get("right_EAR"))
        avg_ear = dinh_dang_so(eye_features.get("avg_EAR"))
        mar = dinh_dang_so(mouth_features.get("MAR"))
        head_pitch = dinh_dang_so(head_pose_features.get("head_pitch"), 1)
        head_yaw = dinh_dang_so(head_pose_features.get("head_yaw"), 1)
        head_roll = dinh_dang_so(head_pose_features.get("head_roll"), 1)

        cac_dong.extend(
            [
                (f"L_EAR: {left_ear} | R_EAR: {right_ear}", (255, 255, 255)),
                (
                    f"avg_EAR: {avg_ear} | eye_closed: {lay_feature_bool(eye_features, 'eye_closed')}",
                    (255, 255, 255),
                ),
                (
                    f"L_closed: {lay_feature_bool(eye_features, 'left_eye_closed')} | "
                    f"R_closed: {lay_feature_bool(eye_features, 'right_eye_closed')}",
                    (255, 255, 255),
                ),
                (
                    f"both_closed: {lay_feature_bool(eye_features, 'both_eyes_closed')} | "
                    f"one_closed: {lay_feature_bool(eye_features, 'one_eye_closed')}",
                    (255, 255, 255),
                ),
                (
                    f"MAR: {mar} | mouth_open: {lay_feature_bool(mouth_features, 'mouth_open')}",
                    (255, 255, 255),
                ),
                (
                    f"pitch: {head_pitch} | yaw: {head_yaw} | roll: {head_roll}",
                    (255, 255, 255),
                ),
                (
                    f"head_down: {lay_feature_bool(head_pose_features, 'head_down')} | "
                    f"head_turned: {lay_feature_bool(head_pose_features, 'head_turned')}",
                    (255, 255, 255),
                ),
            ]
        )

    quality_features = ket_qua_dac_trung.get("quality_features") or {}
    loi = quality_features.get("feature_error")
    if loi:
        cac_dong.append((f"feature_error: {loi}", (0, 220, 255)))

    padding = 10
    line_height = 23
    font_scale = 0.50
    so_dong_hien_thi = 10 if face_detected else 3
    overlay_height = padding * 2 + line_height * so_dong_hien_thi

    if overlay_height > chieu_cao - 20:
        font_scale = 0.45
        line_height = 20
        overlay_height = padding * 2 + line_height * so_dong_hien_thi

    x0 = 10
    y0 = max(10, chieu_cao - overlay_height - 10)
    overlay_width = min(470, chieu_rong - x0 - 10)
    x1 = x0 + overlay_width
    y1 = min(chieu_cao - 10, y0 + overlay_height)

    lop_phu = khung_hinh.copy()
    cv2.rectangle(lop_phu, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(lop_phu, 0.45, khung_hinh, 0.55, 0, khung_hinh)

    y = y0 + padding + line_height - 5
    for noi_dung, mau in cac_dong:
        ve_dong(khung_hinh, noi_dung, x0 + padding, y, font_scale, mau)
        y += line_height


def hien_thi_ket_qua(
    ket_qua: Dict[str, Any],
    bo_phat_hien: BoPhatHienDiemMat,
    tham_so,
) -> bool:
    khung_hinh_hien_thi = cv2.cvtColor(ket_qua["processed_frame"], cv2.COLOR_RGB2BGR)

    if ket_qua.get("face_detected"):
        bo_phat_hien.ve_luoi_mat(
            khung_hinh_hien_thi,
            ket_qua,
            ve_duong_bao=True,
            ve_luoi_tam_giac=not tham_so.tat_luoi_tam_giac,
            ve_mong_mat=tham_so.ve_mong_mat,
        )

    ve_thong_tin_dac_trung(khung_hinh_hien_thi, ket_qua)
    cv2.imshow("DMS Feature Extraction Demo", khung_hinh_hien_thi)

    phim = cv2.waitKey(1) & 0xFF
    return phim not in (ord("q"), 27)


def chay_demo() -> None:
    tham_so = tao_bo_doc_tham_so().parse_args()
    pipeline = None

    print("Demo Feature Extraction started")
    print("Nhan q hoac ESC de thoat")

    try:
        pipeline = tao_pipeline(tham_so)
        pipeline.bat_dau()

        while True:
            ket_qua = pipeline.xu_ly_mot_vong()
            if not hien_thi_ket_qua(
                ket_qua,
                pipeline.bo_phat_hien_diem_mat,
                tham_so,
            ):
                break

    except KeyboardInterrupt:
        print("\nDa dung chuong trinh.")
    except RuntimeError as loi:
        print(f"[LOI] {loi}")
    finally:
        if pipeline is not None:
            pipeline.giai_phong()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    chay_demo()
