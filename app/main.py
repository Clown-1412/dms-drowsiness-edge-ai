import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2


# Cho phep chay truc tiep bang: python app/main.py --camera 0
# hoac: python app/main.py --video-path path/to/video.mp4
GOC_DU_AN = Path(__file__).resolve().parents[1]
if str(GOC_DU_AN) not in sys.path:
    sys.path.insert(0, str(GOC_DU_AN))

from src.camera.camera_stream import LuongCamera
from src.detection.landmark_detector import BoPhatHienDiemMat
from src.features.feature_extractor import BoTrichXuatDacTrung
from src.pipeline.pipeline_phat_hien_ngu_gat import PipelinePhatHienNguGat
from src.preprocessing.frame_preprocessor import BoTienXuLyKhungHinh
from src.utils.landmark_drawing import ve_diem_mat, ve_diem_mieng


NguonCamera = Union[int, str]
MauBGR = Tuple[int, int, int]
DongOverlay = Tuple[str, MauBGR]


def phan_tich_nguon_camera(nguon: str) -> NguonCamera:
    """Chuyen chuoi so thanh camera_id, con lai giu la video_path/source."""
    try:
        return int(nguon)
    except (TypeError, ValueError):
        return nguon


def tao_bo_doc_tham_so() -> argparse.ArgumentParser:
    bo_doc_tham_so = argparse.ArgumentParser(
        description=(
            "Demo pipeline hien tai: Video/Camera -> Pre-processing -> "
            "Landmark Detection -> Feature Extraction"
        )
    )
    bo_doc_tham_so.add_argument(
        "video_path",
        nargs="?",
        help="Duong dan video demo. Neu bo trong thi dung --camera/--source hoac webcam 0.",
    )
    bo_doc_tham_so.add_argument(
        "--camera",
        "--camera-id",
        dest="camera_id",
        type=int,
        default=None,
        help="ID camera OpenCV, vi du: --camera 0.",
    )
    bo_doc_tham_so.add_argument(
        "--video-path",
        "--video_path",
        dest="video_path_tuy_chon",
        help="Duong dan video demo, vi du: --video-path data/raw/demo.mp4.",
    )
    bo_doc_tham_so.add_argument(
        "--source",
        dest="nguon",
        default=None,
        help="Camera ID hoac duong dan video. Alias tuong thich voi demo cu.",
    )
    bo_doc_tham_so.add_argument("--width", dest="chieu_rong", type=int, default=640, help="Chieu rong frame")
    bo_doc_tham_so.add_argument("--height", dest="chieu_cao", type=int, default=480, help="Chieu cao frame")
    bo_doc_tham_so.add_argument("--target-fps", dest="fps_muc_tieu", type=int, default=30, help="FPS camera mong muon")
    bo_doc_tham_so.add_argument("--fps-threshold", dest="nguong_fps", type=float, default=15.0, help="Nguong canh bao FPS thap")
    bo_doc_tham_so.add_argument("--no-show", dest="khong_hien_thi", action="store_true", help="Khong mo cua so hien thi")
    bo_doc_tham_so.add_argument("--max-frames", dest="so_frame_toi_da", type=int, default=0, help="Dung sau N frame, 0 la chay den het video")
    bo_doc_tham_so.add_argument("--display-delay-ms", dest="do_tre_hien_thi_ms", type=int, default=None, help="Delay cho cv2.waitKey; mac dinh lay theo FPS video neu co")
    bo_doc_tham_so.add_argument("--no-mesh", dest="tat_luoi_mat", action="store_true", help="Khong ve mesh/contour Face Mesh")
    bo_doc_tham_so.add_argument("--no-tesselation", dest="tat_luoi_tam_giac", action="store_true", help="Khong ve luoi tam giac Face Mesh")
    bo_doc_tham_so.add_argument("--draw-iris", dest="ve_mong_mat", action="store_true", help="Ve iris neu MediaPipe tra ve")
    bo_doc_tham_so.add_argument("--no-eye", dest="tat_diem_mat", action="store_true", help="Khong ve eye_landmarks")
    bo_doc_tham_so.add_argument("--no-mouth", dest="tat_diem_mieng", action="store_true", help="Khong ve mouth_landmarks")
    bo_doc_tham_so.add_argument("--terminal-interval", dest="khoang_in_terminal", type=int, default=10, help="In thong tin moi N frame")
    return bo_doc_tham_so


def la_url_stream(nguon: str) -> bool:
    return "://" in nguon


def lay_video_path(tham_so) -> Optional[str]:
    return tham_so.video_path_tuy_chon or tham_so.video_path


def chuan_hoa_video_path(video_path: str) -> str:
    if la_url_stream(video_path):
        return video_path

    duong_dan = Path(video_path).expanduser()
    if not duong_dan.exists():
        raise RuntimeError(f"Khong tim thay video_path: {duong_dan}")
    return str(duong_dan.resolve())


def chuan_hoa_nguon_dau_vao(tham_so) -> NguonCamera:
    video_path = lay_video_path(tham_so)
    co_camera = tham_so.camera_id is not None
    co_source = tham_so.nguon is not None

    if video_path and co_camera:
        raise RuntimeError("Chi chon mot input: --camera hoac --video-path/video_path.")
    if video_path and co_source:
        raise RuntimeError("Chi chon mot input: --source hoac --video-path/video_path.")
    if co_camera and co_source:
        raise RuntimeError("Chi chon mot input: --camera hoac --source.")

    if video_path:
        return chuan_hoa_video_path(video_path)

    if co_camera:
        return tham_so.camera_id

    if co_source:
        nguon = phan_tich_nguon_camera(tham_so.nguon)
        if isinstance(nguon, str):
            return chuan_hoa_video_path(nguon)
        return nguon

    return 0


def tao_pipeline(tham_so, nguon: NguonCamera) -> PipelinePhatHienNguGat:
    kich_thuoc_dich = (tham_so.chieu_rong, tham_so.chieu_cao)
    luong_camera = LuongCamera(
        ma_camera=nguon,
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

    return PipelinePhatHienNguGat(
        luong_camera=luong_camera,
        bo_tien_xu_ly=bo_tien_xu_ly,
        bo_phat_hien_diem_mat=bo_phat_hien,
        bo_trich_xuat_dac_trung=BoTrichXuatDacTrung(),
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
    mau: MauBGR = (255, 255, 255),
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


def tao_cac_dong_overlay(ket_qua: Dict[str, Any]) -> List[DongOverlay]:
    face_detected = bool(ket_qua.get("face_detected"))
    fps = ket_qua.get("fps") or 0.0
    eye_features = ket_qua.get("eye_features") or {}
    mouth_features = ket_qua.get("mouth_features") or {}
    head_pose_features = ket_qua.get("head_pose_features") or {}

    cac_dong: List[DongOverlay] = [
        (
            f"FPS: {fps:.1f} | face_detected: {face_detected}",
            (0, 255, 0) if face_detected else (0, 0, 255),
        )
    ]

    if not face_detected:
        cac_dong.append(("No face detected", (0, 0, 255)))
    else:
        cac_dong.extend(
            [
                (
                    f"L_EAR: {dinh_dang_so(eye_features.get('left_EAR'))} | "
                    f"R_EAR: {dinh_dang_so(eye_features.get('right_EAR'))}",
                    (255, 255, 255),
                ),
                (
                    f"avg_EAR: {dinh_dang_so(eye_features.get('avg_EAR'))} | "
                    f"eye_closed: {lay_feature_bool(eye_features, 'eye_closed')}",
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
                    f"MAR: {dinh_dang_so(mouth_features.get('MAR'))} | "
                    f"mouth_open: {lay_feature_bool(mouth_features, 'mouth_open')}",
                    (255, 255, 255),
                ),
                (
                    f"pitch: {dinh_dang_so(head_pose_features.get('head_pitch'), 1)} | "
                    f"yaw: {dinh_dang_so(head_pose_features.get('head_yaw'), 1)} | "
                    f"roll: {dinh_dang_so(head_pose_features.get('head_roll'), 1)}",
                    (255, 255, 255),
                ),
                (
                    f"head_down: {lay_feature_bool(head_pose_features, 'head_down')} | "
                    f"head_turned: {lay_feature_bool(head_pose_features, 'head_turned')}",
                    (255, 255, 255),
                ),
            ]
        )

    quality_features = ket_qua.get("quality_features") or {}
    loi = quality_features.get("feature_error")
    if loi:
        cac_dong.append((f"feature_error: {loi}", (0, 220, 255)))

    return cac_dong


def ve_thong_tin_pipeline(khung_hinh, ket_qua: Dict[str, Any]) -> None:
    chieu_cao, chieu_rong = khung_hinh.shape[:2]
    padding = 10
    line_height = 23
    font_scale = 0.50

    cac_dong = tao_cac_dong_overlay(ket_qua)
    so_dong_toi_da = max(1, (chieu_cao - 20 - padding * 2) // line_height)
    if len(cac_dong) > so_dong_toi_da:
        cac_dong = cac_dong[:so_dong_toi_da]

    overlay_height = padding * 2 + line_height * len(cac_dong)
    do_rong_chu = [
        cv2.getTextSize(noi_dung, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for noi_dung, _ in cac_dong
    ]
    overlay_width = min(max(do_rong_chu, default=320) + padding * 2, chieu_rong - 20)

    x0 = 10
    y0 = max(10, chieu_cao - overlay_height - 10)
    x1 = x0 + overlay_width
    y1 = min(chieu_cao - 10, y0 + overlay_height)

    lop_phu = khung_hinh.copy()
    cv2.rectangle(lop_phu, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(lop_phu, 0.45, khung_hinh, 0.55, 0, khung_hinh)

    y = y0 + padding + line_height - 5
    for noi_dung, mau in cac_dong:
        ve_dong(khung_hinh, noi_dung, x0 + padding, y, font_scale, mau)
        y += line_height


def ve_landmark_debug(
    khung_hinh,
    ket_qua: Dict[str, Any],
    bo_phat_hien: BoPhatHienDiemMat,
    tham_so,
) -> None:
    if not ket_qua.get("face_detected"):
        cv2.putText(
            khung_hinh,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    if not tham_so.tat_luoi_mat:
        bo_phat_hien.ve_luoi_mat(
            khung_hinh,
            ket_qua,
            ve_duong_bao=True,
            ve_luoi_tam_giac=not tham_so.tat_luoi_tam_giac,
            ve_mong_mat=tham_so.ve_mong_mat,
        )

    if not tham_so.tat_diem_mat:
        ve_diem_mat(khung_hinh, ket_qua.get("eye_landmarks") or {})
    if not tham_so.tat_diem_mieng:
        ve_diem_mieng(khung_hinh, ket_qua.get("mouth_landmarks") or [])


def lay_do_tre_hien_thi_ms(
    pipeline: PipelinePhatHienNguGat,
    tham_so,
    nguon: NguonCamera,
) -> int:
    if tham_so.do_tre_hien_thi_ms is not None:
        return max(1, tham_so.do_tre_hien_thi_ms)

    if isinstance(nguon, str) and pipeline.luong_camera.bo_doc is not None:
        fps_video = pipeline.luong_camera.bo_doc.get(cv2.CAP_PROP_FPS)
        if fps_video and fps_video > 0:
            return max(1, int(1000 / fps_video))

    return 1


def hien_thi_ket_qua(
    ket_qua: Dict[str, Any],
    bo_phat_hien: BoPhatHienDiemMat,
    tham_so,
    do_tre_hien_thi_ms: int,
) -> bool:
    khung_hinh = cv2.cvtColor(ket_qua["processed_frame"], cv2.COLOR_RGB2BGR)
    ve_landmark_debug(khung_hinh, ket_qua, bo_phat_hien, tham_so)
    ve_thong_tin_pipeline(khung_hinh, ket_qua)

    cv2.imshow("DMS Pipeline Demo", khung_hinh)
    phim = cv2.waitKey(do_tre_hien_thi_ms) & 0xFF
    return phim not in (ord("q"), 27)


def in_ket_qua_ngan(so_frame: int, ket_qua: Dict[str, Any]) -> None:
    eye_features = ket_qua.get("eye_features") or {}
    mouth_features = ket_qua.get("mouth_features") or {}
    head_pose_features = ket_qua.get("head_pose_features") or {}

    print(
        f"frame={so_frame} | "
        f"face_detected={ket_qua.get('face_detected')} | "
        f"fps={(ket_qua.get('fps') or 0.0):.2f} | "
        f"avg_EAR={dinh_dang_so(eye_features.get('avg_EAR'))} | "
        f"eye_closed={lay_feature_bool(eye_features, 'eye_closed')} | "
        f"both_closed={lay_feature_bool(eye_features, 'both_eyes_closed')} | "
        f"one_closed={lay_feature_bool(eye_features, 'one_eye_closed')} | "
        f"MAR={dinh_dang_so(mouth_features.get('MAR'))} | "
        f"mouth_open={lay_feature_bool(mouth_features, 'mouth_open')} | "
        f"pitch={dinh_dang_so(head_pose_features.get('head_pitch'), 1)} | "
        f"yaw={dinh_dang_so(head_pose_features.get('head_yaw'), 1)}"
    )


def la_loi_ket_thuc_video(loi: RuntimeError, nguon: NguonCamera) -> bool:
    return isinstance(nguon, str) and "Khong doc duoc frame tu camera/video" in str(loi)


def chay() -> None:
    tham_so = tao_bo_doc_tham_so().parse_args()
    pipeline = None
    so_frame_da_xu_ly = 0

    try:
        nguon = chuan_hoa_nguon_dau_vao(tham_so)
        pipeline = tao_pipeline(tham_so, nguon)
        print("Demo pipeline started: Video/Camera -> Pre-processing -> Landmark Detection -> Feature Extraction")
        print(f"source={nguon}")

        pipeline.bat_dau()
        do_tre_hien_thi_ms = lay_do_tre_hien_thi_ms(pipeline, tham_so, nguon)
        if not tham_so.khong_hien_thi:
            print("Nhan q hoac ESC de thoat")

        while True:
            try:
                ket_qua = pipeline.xu_ly_mot_vong()
            except RuntimeError as loi:
                if la_loi_ket_thuc_video(loi, nguon):
                    print("Da doc het video.")
                    break
                raise

            so_frame_da_xu_ly += 1
            co_in_terminal = (
                tham_so.khong_hien_thi
                or tham_so.khoang_in_terminal <= 1
                or so_frame_da_xu_ly % tham_so.khoang_in_terminal == 0
            )
            if co_in_terminal:
                in_ket_qua_ngan(so_frame_da_xu_ly, ket_qua)

            if not tham_so.khong_hien_thi and not hien_thi_ket_qua(
                ket_qua,
                pipeline.bo_phat_hien_diem_mat,
                tham_so,
                do_tre_hien_thi_ms,
            ):
                break

            if 0 < tham_so.so_frame_toi_da <= so_frame_da_xu_ly:
                break
    except KeyboardInterrupt:
        print("\nDa dung chuong trinh.")
    except RuntimeError as loi:
        print(f"[LOI] {loi}")
    finally:
        if pipeline is not None:
            pipeline.giai_phong()
        if not tham_so.khong_hien_thi:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    chay()
