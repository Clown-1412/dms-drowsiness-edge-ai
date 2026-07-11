import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
from src.utils.display_utils import (
    dinh_dang_so,
    hien_thi_ket_qua,
    lay_do_tre_hien_thi_ms,
    lay_feature_bool,
)


NguonCamera = Union[int, str]


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
