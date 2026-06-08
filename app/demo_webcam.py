import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import cv2
from src.utils.fps import FPSCounter

def main():
    cap = cv2.VideoCapture(0)
    fps_counter = FPSCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps = fps_counter.update()

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("DMS Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
