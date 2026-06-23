from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


DiemMoc = Dict[str, float]
Mau = Tuple[int, int, int]


def _sang_xy(diem: DiemMoc) -> Tuple[int, int]:
    """Chuyen landmark dict ve toa do pixel cho OpenCV."""
    return int(diem["x"]), int(diem["y"])


def ve_nhom_diem_moc(
    khung_hinh,
    cac_diem: Iterable[DiemMoc],
    mau: Mau,
    dong_kin: bool = True,
    ban_kinh: int = 3,
    do_day_duong: int = 1,
) -> None:
    """Ve landmark bang duong noi va cac diem tron, khong ve chu."""
    danh_sach_diem = list(cac_diem)
    if not danh_sach_diem:
        return

    cac_diem_xy = [_sang_xy(diem) for diem in danh_sach_diem]
    if len(cac_diem_xy) >= 2:
        cv2.polylines(
            khung_hinh,
            [np.array(cac_diem_xy, dtype=np.int32)],
            isClosed=dong_kin,
            color=mau,
            thickness=do_day_duong,
            lineType=cv2.LINE_AA,
        )

    for diem in danh_sach_diem:
        x, y = _sang_xy(diem)
        cv2.circle(khung_hinh, (x, y), ban_kinh, mau, -1, lineType=cv2.LINE_AA)


def ve_diem_mat(
    khung_hinh,
    diem_mat: Dict[str, List[DiemMoc]],
) -> None:
    """Ve rieng mat trai va mat phai."""
    ve_nhom_diem_moc(
        khung_hinh,
        diem_mat.get("left_eye", []),
        mau=(0, 255, 0),
        dong_kin=True,
        ban_kinh=3,
        do_day_duong=1,
    )
    ve_nhom_diem_moc(
        khung_hinh,
        diem_mat.get("right_eye", []),
        mau=(0, 180, 255),
        dong_kin=True,
        ban_kinh=3,
        do_day_duong=1,
    )


def ve_diem_mieng(
    khung_hinh,
    diem_mieng: List[DiemMoc],
) -> None:
    """Ve outline vung mieng."""
    ve_nhom_diem_moc(
        khung_hinh,
        diem_mieng,
        mau=(255, 0, 255),
        dong_kin=True,
        ban_kinh=4,
        do_day_duong=1,
    )


def ve_diem_mui(
    khung_hinh,
    diem_mui: List[DiemMoc],
) -> None:
    """Ve cac diem vung mui."""
    ve_nhom_diem_moc(
        khung_hinh,
        diem_mui,
        mau=(0, 255, 255),
        dong_kin=False,
        ban_kinh=3,
        do_day_duong=1,
    )


def ve_diem_tu_the_dau(
    khung_hinh,
    diem_tu_the_dau: Dict[str, DiemMoc],
) -> None:
    """Ve va noi cac diem quan trong de debug head pose."""
    if not diem_tu_the_dau:
        return

    ve_nhom_diem_moc(
        khung_hinh,
        diem_tu_the_dau.values(),
        mau=(255, 255, 0),
        dong_kin=False,
        ban_kinh=4,
        do_day_duong=1,
    )
