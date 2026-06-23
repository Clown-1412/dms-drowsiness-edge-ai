import numpy as np


CHI_SO_MIENG = [61, 81, 13, 311, 291, 402, 14, 178]


def ty_le_mieng(cac_diem):
    diem = np.array(cac_diem, dtype=np.float32)

    chieu_cao = np.linalg.norm(diem[2] - diem[6])
    chieu_rong = np.linalg.norm(diem[0] - diem[4])

    if chieu_rong == 0:
        return 0.0

    return chieu_cao / chieu_rong


def lay_diem_mieng(cac_diem_moc, kich_thuoc_anh):
    chieu_rong, chieu_cao = kich_thuoc_anh
    return [
        _sang_pixel(cac_diem_moc[chi_so], chieu_rong, chieu_cao)
        for chi_so in CHI_SO_MIENG
    ]


def _sang_pixel(diem_moc, chieu_rong, chieu_cao):
    return int(diem_moc[0] * chieu_rong), int(diem_moc[1] * chieu_cao)
