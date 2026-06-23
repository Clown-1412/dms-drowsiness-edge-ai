import argparse
import sys
from pathlib import Path
from typing import Union

import cv2


# Cho phep chay file truc tiep bang: python app/demo_landmark_detection.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.camera_stream import CameraStream
from src.detection.landmark_detector import LandmarkDetector
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.utils.landmark_drawing import (
    draw_eye_landmarks,
    draw_mouth_landmarks,
)


CameraSource = Union[int, str]


def parse_camera_source(source: str) -> CameraSource:
    """Chuyen source dang so thanh camera_id, con lai giu la duong dan video."""
    try:
        return int(source)
    except ValueError:
        return source


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo Input Camera + Pre-processing + Landmark Detection"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera ID hoac duong dan video. Mac dinh: 0",
    )
    parser.add_argument("--width", type=int, default=640, help="Chieu rong frame")
    parser.add_argument("--height", type=int, default=480, help="Chieu cao frame")
    parser.add_argument(
        "--target-fps",
        type=int,
        default=30,
        help="FPS mong muon khi cau hinh camera",
    )
    parser.add_argument(
        "--fps-threshold",
        type=float,
        default=15.0,
        help="Nguong canh bao FPS thap",
    )
    parser.add_argument(
        "--no-tesselation",
        action="store_true",
        help="Tat luoi tam giac Face Mesh, chi giu contour",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Tat contour/iris MediaPipe, chi ve eye/mouth landmarks",
    )
    return parser


def draw_detection(frame, detection, detector, args) -> None:
    """Ve mesh va cac nhom landmark chinh tren frame hien thi."""
    if not args.no_mesh:
        detector.draw_face_mesh(
            frame,
            detection,
            draw_contours=True,
            draw_tesselation=not args.no_tesselation,
            draw_iris=False,
        )

    # Ve cac dac trung sau Face Mesh de line trang khong de len mat/mieng/mui.
    draw_eye_landmarks(
        frame,
        detection["eye_landmarks"],
    )
    draw_mouth_landmarks(
        frame,
        detection["mouth_landmarks"],
    )


def get_eye_landmark_count(detection) -> int:
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
    preprocessor = FramePreprocessor(
        target_size=target_size,
        fps_threshold=args.fps_threshold,
        denoise_method="gaussian",
    )
    detector = LandmarkDetector(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Demo Landmark Detection started")
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

            # cv2.imshow can BGR, nen chuyen RGB sau pre-processing ve BGR.
            display_frame = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)

            if detection["face_detected"]:
                draw_detection(display_frame, detection, detector, args)
                face_count = len(detection["face_landmarks"])
                eye_count = get_eye_landmark_count(detection)
                mouth_count = len(detection["mouth_landmarks"])

                print(
                    "face_detected=True | "
                    f"face_landmarks={face_count} | "
                    f"eye_landmarks={eye_count} | "
                    f"mouth_landmarks={mouth_count}"
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

            cv2.imshow("DMS Landmark Detection Demo", display_frame)

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
