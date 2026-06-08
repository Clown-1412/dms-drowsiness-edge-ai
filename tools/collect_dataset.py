import cv2
from pathlib import Path
import time

output_dir = Path("data/raw/normal")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
idx = int(time.time())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collect Dataset", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        path = output_dir / f"frame_{idx}.jpg"
        cv2.imwrite(str(path), frame)
        print("Saved:", path)
        idx += 1

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
