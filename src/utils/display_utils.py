from typing import Any, Dict, List, Optional, Tuple

import cv2

from src.utils.landmark_drawing import ve_diem_mat, ve_diem_mieng


MauBGR = Tuple[int, int, int]
DongOverlay = Tuple[str, MauBGR]


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
    bo_phat_hien: Any,
    tham_so: Any,
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
        diem_mieng = (
            ket_qua.get("mouth_outline_landmarks")
            or ket_qua.get("mouth_landmarks")
            or []
        )
        ve_diem_mieng(khung_hinh, diem_mieng)


def dao_toggle(tham_so: Any, ten_thuoc_tinh: str) -> bool:
    gia_tri_moi = not bool(getattr(tham_so, ten_thuoc_tinh, False))
    setattr(tham_so, ten_thuoc_tinh, gia_tri_moi)
    return gia_tri_moi


def in_trang_thai_an_hien(ten_hien_thi: str, dang_tat: bool) -> None:
    trang_thai = "tat" if dang_tat else "bat"
    print(f"{ten_hien_thi}: {trang_thai}")


def xu_ly_phim_toggle(phim: int, tham_so: Any) -> bool:
    if phim in (255, -1):
        return True
    if phim in (ord("q"), 27):
        return False

    phim_ky_tu = chr(phim).lower()
    if phim_ky_tu == "m":
        in_trang_thai_an_hien("Face mesh", dao_toggle(tham_so, "tat_luoi_mat"))
    elif phim_ky_tu == "e":
        in_trang_thai_an_hien("Eye points", dao_toggle(tham_so, "tat_diem_mat"))
    elif phim_ky_tu == "o":
        in_trang_thai_an_hien("Mouth points", dao_toggle(tham_so, "tat_diem_mieng"))
    elif phim_ky_tu == "b":
        in_trang_thai_an_hien("Bang chi so", dao_toggle(tham_so, "tat_bang_chi_so"))
    elif phim_ky_tu == "t":
        in_trang_thai_an_hien("Tesselation", dao_toggle(tham_so, "tat_luoi_tam_giac"))
    elif phim_ky_tu == "i":
        dang_bat = dao_toggle(tham_so, "ve_mong_mat")
        trang_thai = "bat" if dang_bat else "tat"
        print(f"Iris: {trang_thai}")

    return True


def lay_do_tre_hien_thi_ms(
    pipeline: Any,
    tham_so: Any,
    nguon: Any,
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
    bo_phat_hien: Any,
    tham_so: Any,
    do_tre_hien_thi_ms: int,
) -> bool:
    khung_hinh = cv2.cvtColor(ket_qua["processed_frame"], cv2.COLOR_RGB2BGR)
    ve_landmark_debug(khung_hinh, ket_qua, bo_phat_hien, tham_so)
    if not getattr(tham_so, "tat_bang_chi_so", False):
        ve_thong_tin_pipeline(khung_hinh, ket_qua)

    cv2.imshow("DMS Pipeline Demo", khung_hinh)
    phim = cv2.waitKey(do_tre_hien_thi_ms) & 0xFF
    return xu_ly_phim_toggle(phim, tham_so)
