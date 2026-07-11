from collections.abc import Iterable
from typing import Any

import numpy as np


DiemMoc = dict[str, Any]


def diem_hop_le(diem: Any) -> bool:
    """Kiem tra landmark co toa do pixel x, y hop le khong."""
    if not isinstance(diem, dict):
        return False
    if "x" not in diem or "y" not in diem:
        return False

    try:
        float(diem["x"])
        float(diem["y"])
    except (TypeError, ValueError):
        return False

    return True


def danh_sach_diem_hop_le(
    danh_sach_diem: Any,
    so_diem_toi_thieu: int,
) -> bool:
    """Kiem tra list landmark co du so diem hop le toi thieu hay khong."""
    if not isinstance(danh_sach_diem, Iterable):
        return False

    cac_diem = list(danh_sach_diem)
    if len(cac_diem) < so_diem_toi_thieu:
        return False

    return all(diem_hop_le(diem) for diem in cac_diem[:so_diem_toi_thieu])


def khoang_cach_euclid(diem_1: Any, diem_2: Any) -> float:
    """Tinh khoang cach Euclid 2D theo pixel giua hai landmark dict."""
    if not diem_hop_le(diem_1) or not diem_hop_le(diem_2):
        return 0.0

    toa_do_1 = np.array([float(diem_1["x"]), float(diem_1["y"])], dtype=np.float32)
    toa_do_2 = np.array([float(diem_2["x"]), float(diem_2["y"])], dtype=np.float32)
    return float(np.linalg.norm(toa_do_1 - toa_do_2))
