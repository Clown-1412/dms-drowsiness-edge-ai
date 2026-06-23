import argparse
import sys
from pathlib import Path
from typing import Union

import cv2


# Cho phep chay file truc tiep bang: python app/demo_mouth_landmarks.py
GOC_DU_AN = Path(__file__).resolve().parents[1]
if str(GOC_DU_AN) not in sys.path:
    sys.path.insert(0, str(GOC_DU_AN))

from src.camera.camera_stream import LuongCamera
from src.detection.landmark_detector import BoPhatHienDiemMat
from src.preprocessing.frame_preprocessor import BoTienXuLyKhungHinh
from src.utils.landmark_drawing import ve_diem_mieng


NguonCamera = Union[int, str]


def phan_tich_nguon_camera(nguon: str) -> NguonCamera:
    """Chuyen source dang so thanh camera_id, con lai giu la duong dan video."""
    try:
        return int(nguon)
    except ValueError:
        return nguon


def tao_bo_doc_tham_so() -> argparse.ArgumentParser:
    bo_doc_tham_so = argparse.ArgumentParser(description="Demo rieng cho mouth_landmarks")
    bo_doc_tham_so.add_argument("--source", dest="nguon", default="0", help="Camera ID hoac duong dan video")
    bo_doc_tham_so.add_argument("--width", dest="chieu_rong", type=int, default=640, help="Chieu rong frame")
    bo_doc_tham_so.add_argument("--height", dest="chieu_cao", type=int, default=480, help="Chieu cao frame")
    bo_doc_tham_so.add_argument("--target-fps", dest="fps_muc_tieu", type=int, default=30, help="FPS camera mong muon")
    bo_doc_tham_so.add_argument("--no-mesh", dest="tat_luoi_mat", action="store_true", help="Tat contour/iris MediaPipe")
    return bo_doc_tham_so


def chay_demo() -> None:
    tham_so = tao_bo_doc_tham_so().parse_args()
    nguon = phan_tich_nguon_camera(tham_so.nguon)
    kich_thuoc_dich = (tham_so.chieu_rong, tham_so.chieu_cao)

    luong_camera = LuongCamera(
        ma_camera=nguon,
        chieu_rong=tham_so.chieu_rong,
        chieu_cao=tham_so.chieu_cao,
        fps_muc_tieu=tham_so.fps_muc_tieu,
    )
    bo_tien_xu_ly = BoTienXuLyKhungHinh(kich_thuoc_dich=kich_thuoc_dich)
    bo_phat_hien = BoPhatHienDiemMat(so_mat_toi_da=1, lam_min_diem_moc=True)

    print("Demo mouth_landmarks started")
    print("Nhan q hoac ESC de thoat")

    try:
        luong_camera.bat_dau()

        while True:
            ket_qua_camera = luong_camera.doc_khung_hinh()
            ket_qua_tien_xu_ly = bo_tien_xu_ly.xu_ly(
                khung_hinh=ket_qua_camera["frame"],
                moc_thoi_gian=ket_qua_camera["timestamp"],
                fps=ket_qua_camera["fps"],
            )
            khung_hinh_rgb = ket_qua_tien_xu_ly["processed_frame"]
            ket_qua_phat_hien = bo_phat_hien.phat_hien(
                khung_hinh_da_xu_ly=khung_hinh_rgb,
                moc_thoi_gian=ket_qua_tien_xu_ly["timestamp"],
                fps=ket_qua_tien_xu_ly["fps"],
            )

            khung_hinh_hien_thi = cv2.cvtColor(khung_hinh_rgb, cv2.COLOR_RGB2BGR)

            if ket_qua_phat_hien["face_detected"]:
                if not tham_so.tat_luoi_mat:
                    bo_phat_hien.ve_luoi_mat(khung_hinh_hien_thi, ket_qua_phat_hien)

                ve_diem_mieng(
                    khung_hinh_hien_thi,
                    ket_qua_phat_hien["mouth_landmarks"],
                )
                print(
                    "face_detected=True | "
                    f"mouth_landmarks={len(ket_qua_phat_hien['mouth_landmarks'])}"
                )
            else:
                print("face_detected=False")
                cv2.putText(
                    khung_hinh_hien_thi,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("DMS Mouth Landmarks Demo", khung_hinh_hien_thi)

            phim = cv2.waitKey(1) & 0xFF
            if phim == ord("q") or phim == 27:
                break

    except RuntimeError as loi:
        print(f"[LOI] {loi}")
    finally:
        bo_phat_hien.giai_phong()
        luong_camera.giai_phong()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    chay_demo()
