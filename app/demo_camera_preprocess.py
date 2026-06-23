import argparse
import sys
from pathlib import Path
from typing import Union

import cv2

# Cho phep chay file truc tiep bang: python app/demo_camera_preprocess.py
GOC_DU_AN = Path(__file__).resolve().parents[1]
if str(GOC_DU_AN) not in sys.path:
    sys.path.insert(0, str(GOC_DU_AN))

# Muc dich: Demo su dung LuongCamera va BoTienXuLyKhungHinh cung nhau.
from src.camera.camera_stream import LuongCamera
from src.preprocessing.frame_preprocessor import BoTienXuLyKhungHinh


NguonCamera = Union[int, str]

# NguonCamera co the la camera_id (int) hoac duong dan video (str).
def phan_tich_nguon_camera(nguon: str) -> NguonCamera:
    """Chuyen source dang so thanh camera_id, con lai giu la duong dan video."""
    try:
        return int(nguon)
    except ValueError:
        return nguon


def tao_bo_doc_tham_so() -> argparse.ArgumentParser:
    bo_doc_tham_so = argparse.ArgumentParser(
        description="Demo Input Camera Layer + Frame Pre-processing Layer"
    )
    bo_doc_tham_so.add_argument(
        "--source",
        dest="nguon",
        default="0",
        help="Camera ID hoac duong dan video. Mac dinh: 0",
    )
    bo_doc_tham_so.add_argument(
        "--width",
        dest="chieu_rong",
        type=int,
        default=640,
        help="Chieu rong frame",
    )
    bo_doc_tham_so.add_argument(
        "--height",
        dest="chieu_cao",
        type=int,
        default=480,
        help="Chieu cao frame",
    )
    bo_doc_tham_so.add_argument(
        "--target-fps",
        dest="fps_muc_tieu",
        type=int,
        default=30,
        help="FPS mong muon khi cau hinh camera",
    )
    bo_doc_tham_so.add_argument(
        "--fps-threshold",
        dest="nguong_fps",
        type=float,
        default=15.0,
        help="Nguong canh bao FPS thap",
    )
    bo_doc_tham_so.add_argument(
        "--show",
        dest="che_do_hien_thi",
        choices=("processed", "original"),
        default="processed",
        help="Hien thi frame da tien xu ly hoac frame goc",
    )
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
    bo_tien_xu_ly = BoTienXuLyKhungHinh(
        kich_thuoc_dich=kich_thuoc_dich,
        nguong_fps=tham_so.nguong_fps,
        phuong_phap_giam_nhieu="gaussian",
    )

    print("Demo Camera + Pre-processing started")
    print("Nhan q hoac ESC de thoat")

    try:
        luong_camera.bat_dau()

        while True:
            ket_qua_camera = luong_camera.doc_khung_hinh()
            khung_hinh = ket_qua_camera["frame"]
            moc_thoi_gian = ket_qua_camera["timestamp"]
            fps = ket_qua_camera["fps"]

            ket_qua_tien_xu_ly = bo_tien_xu_ly.xu_ly(
                khung_hinh=khung_hinh,
                moc_thoi_gian=moc_thoi_gian,
                fps=fps,
            )

            khung_hinh_rgb = ket_qua_tien_xu_ly["processed_frame"]
            kich_thuoc_khung_hinh = ket_qua_tien_xu_ly["frame_size"]

            print(
                f"timestamp={moc_thoi_gian:.3f} | "
                f"fps={fps:.2f} | "
                f"frame_size={kich_thuoc_khung_hinh}"
            )

            if tham_so.che_do_hien_thi == "processed":
                # cv2.imshow can anh BGR, nen chuyen RGB sau tien xu ly ve BGR de hien thi.
                khung_hinh_hien_thi = cv2.cvtColor(khung_hinh_rgb, cv2.COLOR_RGB2BGR)
            else:
                khung_hinh_hien_thi = khung_hinh

            cv2.imshow("DMS Camera Preprocess Demo", khung_hinh_hien_thi)

            phim = cv2.waitKey(1) & 0xFF
            if phim == ord("q") or phim == 27:
                break

    except RuntimeError as loi:
        print(f"[LOI] {loi}")
    finally:
        luong_camera.giai_phong()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    chay_demo()
