from collections import deque


class BoPhanTichThoiGian:
    """Tracks boolean eye-closed samples inside a time-based sliding window."""

    def __init__(self, cua_so_giay=10):
        self.cua_so_giay = cua_so_giay
        self.lich_su = deque()

    def cap_nhat(self, mat_nham, moc_thoi_gian):
        self.lich_su.append((moc_thoi_gian, bool(mat_nham)))
        self._bo_mau_cu(moc_thoi_gian)
        return self.lay_perclos()

    def lay_perclos(self):
        tong_so_mau = len(self.lich_su)
        if tong_so_mau == 0:
            return 0.0

        so_mau_mat_nham = sum(1 for _, mat_nham in self.lich_su if mat_nham)
        return so_mau_mat_nham / tong_so_mau

    def _bo_mau_cu(self, moc_thoi_gian):
        moc_thoi_gian_toi_thieu = moc_thoi_gian - self.cua_so_giay
        while self.lich_su and self.lich_su[0][0] < moc_thoi_gian_toi_thieu:
            self.lich_su.popleft()
