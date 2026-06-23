import argparse
import sys
from pathlib import Path
from typing import Union

import cv2

# Cho phep chay file truc tiep bang: python app/demo_camera_preprocess.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Muc dich: Demo su dung CameraStream va FramePreprocessor cung nhau, giup kiem tra va hieu chinh tham so de dat duoc FPS mong muon.
from src.camera.camera_stream import CameraStream
from src.preprocessing.frame_preprocessor import FramePreprocessor


CameraSource = Union[int, str]

# CameraSource co the la camera_id (int) hoac duong dan video (str). Ham parse_camera_source se thu nghiem chuyen doi source sang int, neu that bai se tra ve source goc (duong dan video).
def parse_camera_source(source: str) -> CameraSource:
    """Chuyen source dang so thanh camera_id, con lai giu la duong dan video."""
    try:
        return int(source)
    except ValueError:
        return source


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo Input Camera Layer + Frame Pre-processing Layer"
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
        "--show",
        choices=("processed", "original"),
        default="processed",
        help="Hien thi frame da tien xu ly hoac frame goc",
    )
    return parser


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

    print("Demo Camera + Pre-processing started")
    print("Nhan q hoac ESC de thoat")

    try:
        camera.start()

        while True:
            camera_output = camera.read()
            frame = camera_output["frame"]
            timestamp = camera_output["timestamp"]
            fps = camera_output["fps"]

            preprocess_output = preprocessor.process(
                frame=frame,
                timestamp=timestamp,
                fps=fps,
            )

            processed_rgb = preprocess_output["processed_frame"]
            frame_size = preprocess_output["frame_size"]

            print(
                f"timestamp={timestamp:.3f} | "
                f"fps={fps:.2f} | "
                f"frame_size={frame_size}"
            )

            if args.show == "processed":
                # cv2.imshow can anh BGR, nen chuyen RGB sau tien xu ly ve BGR de hien thi.
                display_frame = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame

            cv2.imshow("DMS Camera Preprocess Demo", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except RuntimeError as exc:
        print(f"[LOI] {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()