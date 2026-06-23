class BoPhanTichNguGat:
    def __init__(
        self,
        nguong_ear=0.20,
        nguong_mar=0.60,
        thoi_luong_nham_mat=1.5,
        nguong_perclos=0.30,
    ):
        self.nguong_ear = nguong_ear
        self.nguong_mar = nguong_mar
        self.thoi_luong_nham_mat = thoi_luong_nham_mat
        self.nguong_perclos = nguong_perclos
        self.thoi_diem_bat_dau_nham_mat = None

    def phan_tich(self, ear, mar, perclos, moc_thoi_gian, phat_hien_mat=True):
        if not phat_hien_mat:
            self.thoi_diem_bat_dau_nham_mat = None
            return "NO_FACE"

        mat_nham = ear < self.nguong_ear
        if mat_nham:
            if self.thoi_diem_bat_dau_nham_mat is None:
                self.thoi_diem_bat_dau_nham_mat = moc_thoi_gian

            if moc_thoi_gian - self.thoi_diem_bat_dau_nham_mat > self.thoi_luong_nham_mat:
                return "DROWSY"
            return "EYE_CLOSED"

        self.thoi_diem_bat_dau_nham_mat = None

        if mar > self.nguong_mar:
            return "YAWNING"

        if perclos > self.nguong_perclos:
            return "DROWSY"

        return "NORMAL"
