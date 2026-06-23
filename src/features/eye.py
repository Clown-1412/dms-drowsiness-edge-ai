import numpy as np


CHI_SO_MAT_TRAI = [33, 160, 158, 133, 153, 144]
CHI_SO_MAT_PHAI = [362, 385, 387, 263, 373, 380]


def khoang_cach_euclid(diem_1, diem_2):
    diem_1_np = np.array(diem_1, dtype=np.float32)
    diem_2_np = np.array(diem_2, dtype=np.float32)
    return float(np.linalg.norm(diem_1_np - diem_2_np))


def ty_le_mat(cac_diem):
    diem = np.array(cac_diem, dtype=np.float32)

    khoang_doc_1 = khoang_cach_euclid(diem[1], diem[5])
    khoang_doc_2 = khoang_cach_euclid(diem[2], diem[4])
    khoang_ngang = khoang_cach_euclid(diem[0], diem[3])

    if khoang_ngang == 0:
        return 0.0

    return (khoang_doc_1 + khoang_doc_2) / (2.0 * khoang_ngang)


def lay_diem_mat(cac_diem_moc, kich_thuoc_anh):
    chieu_rong, chieu_cao = kich_thuoc_anh
    diem_mat_trai = [
        _sang_pixel(cac_diem_moc[chi_so], chieu_rong, chieu_cao)
        for chi_so in CHI_SO_MAT_TRAI
    ]
    diem_mat_phai = [
        _sang_pixel(cac_diem_moc[chi_so], chieu_rong, chieu_cao)
        for chi_so in CHI_SO_MAT_PHAI
    ]
    return diem_mat_trai, diem_mat_phai


def _sang_pixel(diem_moc, chieu_rong, chieu_cao):
    return int(diem_moc[0] * chieu_rong), int(diem_moc[1] * chieu_cao)
