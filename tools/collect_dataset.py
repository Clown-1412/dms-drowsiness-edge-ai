import cv2
from pathlib import Path
import time

thu_muc_dau_ra = Path("data/raw/normal")
thu_muc_dau_ra.mkdir(parents=True, exist_ok=True)

bo_doc = cv2.VideoCapture(0)
chi_so = int(time.time())

while True:
    thanh_cong, khung_hinh = bo_doc.read()
    if not thanh_cong:
        break

    cv2.imshow("Collect Dataset", khung_hinh)

    phim = cv2.waitKey(1) & 0xFF

    if phim == ord("s"):
        duong_dan = thu_muc_dau_ra / f"frame_{chi_so}.jpg"
        cv2.imwrite(str(duong_dan), khung_hinh)
        print("Saved:", duong_dan)
        chi_so += 1

    if phim == ord("q"):
        break

bo_doc.release()
cv2.destroyAllWindows()
