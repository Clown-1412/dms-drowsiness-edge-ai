import argparse
import sys
from pathlib import Path
from typing import Union

import cv2


# Cho phep chay file truc tiep bang: python app/demo_eye_landmarks.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_stream import CameraStream
from src.detection.landmark_detector import LandmarkDetector
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.utils.landmark_drawing import draw_eye_landmarks


CameraSource = Union[int, str]


def parse_camera_source(source: str) -> CameraSource:
    """Chuyen source dang so thanh camera_id, con lai giu la duong dan video."""
    try:
        return int(source)
    except ValueError:
        return source


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo rieng cho eye_landmarks")
    parser.add_argument("--source", default="0", help="Camera ID hoac duong dan video")
    parser.add_argument("--width", type=int, default=640, help="Chieu rong frame")
    parser.add_argument("--height", type=int, default=480, help="Chieu cao frame")
    parser.add_argument("--target-fps", type=int, default=30, help="FPS camera mong muon")
    parser.add_argument("--no-mesh", action="store_true", help="Tat contour/iris MediaPipe")
    return parser


def get_eye_count(detection) -> int:
    eye_landmarks = detection["eye_landmarks"]
    return len(eye_landmarks["left_eye"]) + len(eye_landmarks["right_eye"])


def main() -> None:
    args = build_arg_parser().parse_args()
    source = parse_camera_source(args.source)
    target_size = (args.width, args.height)

    camera = CameraStream(
        camera_id=source,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
    )
    preprocessor = FramePreprocessor(target_size=target_size)
    detector = LandmarkDetector(max_num_faces=1, refine_landmarks=True)

    print("Demo eye_landmarks started")
    print("Nhan q hoac ESC de thoat")

    try:
        camera.start()

        while True:
            camera_output = camera.read()
            preprocess_output = preprocessor.process(
                frame=camera_output["frame"],
                timestamp=camera_output["timestamp"],
                fps=camera_output["fps"],
            )
            processed_rgb = preprocess_output["processed_frame"]
            detection = detector.detect(
                processed_frame=processed_rgb,
                timestamp=preprocess_output["timestamp"],
                fps=preprocess_output["fps"],
            )

            display_frame = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)

            if detection["face_detected"]:
                if not args.no_mesh:
                    detector.draw_face_mesh(display_frame, detection)

                draw_eye_landmarks(
                    display_frame,
                    detection["eye_landmarks"],
                )
                print(
                    "face_detected=True | "
                    f"eye_landmarks={get_eye_count(detection)}"
                )
            else:
                print("face_detected=False")
                cv2.putText(
                    display_frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("DMS Eye Landmarks Demo", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except RuntimeError as exc:
        print(f"[LOI] {exc}")
    finally:
        detector.release()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
