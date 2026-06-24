import argparse
from typing import Any, Dict, Union

import cv2

from src.camera.camera_stream import LuongCamera
from src.detection.landmark_detector import BoPhatHienDiemMat
from src.features.eye import ty_le_mat
from src.features.mouth import ty_le_mieng
from src.pipeline.pipeline_phat_hien_ngu_gat import PipelinePhatHienNguGat
from src.preprocessing.frame_preprocessor import BoTienXuLyKhungHinh
from src.utils.landmark_drawing import (
    ve_diem_mat,
    ve_diem_mieng,
)


NguonCamera = Union[int, str]


def phan_tich_nguon_camera(nguon: str) -> NguonCamera:
    try:
        return int(nguon)
    except ValueError:
        return nguon


def tao_bo_doc_tham_so() -> argparse.ArgumentParser:
    bo_doc_tham_so = argparse.ArgumentParser(
        description="Demo 3 layer: Camera, Pre-processing, Face/Landmark Detection"
    )
    bo_doc_tham_so.add_argument("--source", dest="nguon", default="0", help="Camera ID hoac duong dan video")
    bo_doc_tham_so.add_argument("--width", dest="chieu_rong", type=int, default=640, help="Chieu rong frame")
    bo_doc_tham_so.add_argument("--height", dest="chieu_cao", type=int, default=480, help="Chieu cao frame")
    bo_doc_tham_so.add_argument("--target-fps", dest="fps_muc_tieu", type=int, default=30, help="FPS camera mong muon")
    bo_doc_tham_so.add_argument("--fps-threshold", dest="nguong_fps", type=float, default=15.0, help="Nguong canh bao FPS thap")
    bo_doc_tham_so.add_argument("--no-show", dest="khong_hien_thi", action="store_true", help="Khong mo cua so hien thi")
    bo_doc_tham_so.add_argument("--max-frames", dest="so_frame_toi_da", type=int, default=0, help="Dung sau N frame, 0 la chay lien tuc")
    bo_doc_tham_so.add_argument("--no-mesh", dest="tat_luoi_mat", action="store_true", help="Khong ve mesh/contour Face Mesh")
    bo_doc_tham_so.add_argument("--no-tesselation", dest="tat_luoi_tam_giac", action="store_true", help="Khong ve luoi tam giac Face Mesh")
    bo_doc_tham_so.add_argument("--draw-iris", dest="ve_mong_mat", action="store_true", help="Ve iris neu MediaPipe tra ve")
    bo_doc_tham_so.add_argument("--no-eye", dest="tat_diem_mat", action="store_true", help="Khong ve eye_landmarks")
    bo_doc_tham_so.add_argument("--no-mouth", dest="tat_diem_mieng", action="store_true", help="Khong ve mouth_landmarks")
    bo_doc_tham_so.add_argument("--terminal-interval", dest="khoang_in_terminal", type=int, default=10, help="In chi so feature moi N frame")
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
    )


def dem_diem_mat(ket_qua: Dict[str, Any]) -> int:
    diem_mat = ket_qua.get("eye_landmarks") or {}
    return len(diem_mat.get("left_eye", [])) + len(diem_mat.get("right_eye", []))


def dem_diem(diem_moc) -> int:
    return len(diem_moc or [])


def sang_toa_do_xy(cac_diem) -> list:
    return [
        (float(diem["x"]), float(diem["y"]))
        for diem in (cac_diem or [])
        if "x" in diem and "y" in diem
    ]


def tinh_chi_so_feature(ket_qua: Dict[str, Any]) -> Dict[str, float]:
    if not ket_qua.get("face_detected"):
        return {
            "ear_left": 0.0,
            "ear_right": 0.0,
            "ear": 0.0,
            "mar": 0.0,
        }

    diem_mat = ket_qua.get("eye_landmarks") or {}
    diem_mat_trai = sang_toa_do_xy(diem_mat.get("left_eye"))
    diem_mat_phai = sang_toa_do_xy(diem_mat.get("right_eye"))
    diem_mieng = sang_toa_do_xy(ket_qua.get("mouth_landmarks"))

    ear_trai = ty_le_mat(diem_mat_trai) if len(diem_mat_trai) >= 6 else 0.0
    ear_phai = ty_le_mat(diem_mat_phai) if len(diem_mat_phai) >= 6 else 0.0
    cac_ear = [ear for ear in (ear_trai, ear_phai) if ear > 0.0]
    ear = sum(cac_ear) / len(cac_ear) if cac_ear else 0.0
    mar = ty_le_mieng(diem_mieng) if len(diem_mieng) >= 7 else 0.0

    return {
        "ear_left": ear_trai,
        "ear_right": ear_phai,
        "ear": ear,
        "mar": mar,
    }


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
            (20, 140),
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
        ve_diem_mat(khung_hinh, ket_qua["eye_landmarks"])
    if not tham_so.tat_diem_mieng:
        ve_diem_mieng(khung_hinh, ket_qua["mouth_landmarks"])


def ve_thong_tin_debug(
    khung_hinh,
    ket_qua: Dict[str, Any],
    chi_so_feature: Dict[str, float],
) -> None:
    face_detected = bool(ket_qua.get("face_detected"))
    fps = ket_qua.get("fps") or 0.0
    frame_size = ket_qua.get("frame_size") or ket_qua.get("image_size")
    mau = (0, 255, 0) if face_detected else (0, 0, 255)

    dong_1 = f"face_detected={face_detected} | fps={fps:.1f}"
    dong_2 = f"frame_size={frame_size}"
    dong_3 = (
        f"EAR={chi_so_feature['ear']:.3f} "
        f"(L={chi_so_feature['ear_left']:.3f}, R={chi_so_feature['ear_right']:.3f}) | "
        f"MAR={chi_so_feature['mar']:.3f}"
    )

    cv2.rectangle(khung_hinh, (10, 10), (630, 105), (0, 0, 0), -1)
    cv2.putText(khung_hinh, dong_1, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, mau, 2, cv2.LINE_AA)
    cv2.putText(khung_hinh, dong_2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(khung_hinh, dong_3, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def hien_thi_ket_qua(
    ket_qua: Dict[str, Any],
    chi_so_feature: Dict[str, float],
    bo_phat_hien: BoPhatHienDiemMat,
    tham_so,
) -> bool:
    khung_hinh = cv2.cvtColor(ket_qua["processed_frame"], cv2.COLOR_RGB2BGR)
    ve_landmark_debug(khung_hinh, ket_qua, bo_phat_hien, tham_so)
    ve_thong_tin_debug(khung_hinh, ket_qua, chi_so_feature)

    cv2.imshow("DMS 3-Layer Pipeline Demo", khung_hinh)
    phim = cv2.waitKey(1) & 0xFF
    return phim not in (ord("q"), 27)


def in_ket_qua_ngan(ket_qua: Dict[str, Any], chi_so_feature: Dict[str, float]) -> None:
    print(
        f"face_detected={ket_qua.get('face_detected')} | "
        f"fps={(ket_qua.get('fps') or 0.0):.2f} | "
        f"frame_size={ket_qua.get('frame_size')} | "
        f"landmarks(face/eye/mouth)="
        f"{dem_diem(ket_qua.get('face_landmarks'))}/"
        f"{dem_diem_mat(ket_qua)}/"
        f"{dem_diem(ket_qua.get('mouth_landmarks'))} | "
        f"EAR={chi_so_feature['ear']:.3f} | "
        f"EAR_L={chi_so_feature['ear_left']:.3f} | "
        f"EAR_R={chi_so_feature['ear_right']:.3f} | "
        f"MAR={chi_so_feature['mar']:.3f}"
    )


def chay() -> None:
    tham_so = tao_bo_doc_tham_so().parse_args()
    pipeline = None
    so_frame_da_xu_ly = 0

    try:
        pipeline = tao_pipeline(tham_so)
        print("Demo 3 layer started: Camera -> Pre-processing -> Face/Landmark Detection")
        if not tham_so.khong_hien_thi:
            print("Nhan q hoac ESC de thoat")

        pipeline.bat_dau()
        while True:
            ket_qua = pipeline.xu_ly_mot_vong()
            chi_so_feature = tinh_chi_so_feature(ket_qua)
            so_frame_da_xu_ly += 1

            co_in_terminal = (
                tham_so.khong_hien_thi
                or tham_so.khoang_in_terminal <= 1
                or so_frame_da_xu_ly % tham_so.khoang_in_terminal == 0
            )
            if co_in_terminal:
                in_ket_qua_ngan(ket_qua, chi_so_feature)

            if tham_so.khong_hien_thi:
                pass
            elif not hien_thi_ket_qua(
                ket_qua,
                chi_so_feature,
                pipeline.bo_phat_hien_diem_mat,
                tham_so,
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
