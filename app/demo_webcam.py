import time

import cv2
from src.analyzer.rule_based import DrowsinessAnalyzer
from src.features.eye import eye_aspect_ratio, get_eye_points
from src.features.mouth import get_mouth_points, mouth_aspect_ratio
from src.features.temporal import TemporalAnalyzer
from src.landmark.mediapipe_landmark import MediaPipeFaceLandmark
from src.utils.drawing import draw_points, draw_status, draw_text_info
from src.utils.fps import FPSCounter


DRAW_FULL_MESH = True


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open webcam")

    landmark_detector = MediaPipeFaceLandmark(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    analyzer = DrowsinessAnalyzer()
    temporal_analyzer = TemporalAnalyzer(window_sec=10)
    fps_counter = FPSCounter()

    print("DMS webcam demo started")
    print("Press q or ESC to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = time.time()
            fps = fps_counter.update()
            detection = landmark_detector.detect(frame)

            ear_left = None
            ear_right = None
            ear_avg = None
            mar = None
            perclos = temporal_analyzer.get_perclos()

            if detection is None:
                status = analyzer.analyze(
                    ear=1.0,
                    mar=0.0,
                    perclos=perclos,
                    timestamp=timestamp,
                    face_detected=False,
                )
            else:
                if DRAW_FULL_MESH:
                    landmark_detector.draw_face_mesh(frame, detection)

                landmarks = detection["landmarks"]
                image_size = detection["image_size"]

                left_eye_points, right_eye_points = get_eye_points(
                    landmarks,
                    image_size,
                )
                mouth_points = get_mouth_points(landmarks, image_size)

                ear_left = eye_aspect_ratio(left_eye_points)
                ear_right = eye_aspect_ratio(right_eye_points)
                ear_avg = (ear_left + ear_right) / 2.0
                mar = mouth_aspect_ratio(mouth_points)
                eye_closed = ear_avg < analyzer.ear_threshold
                perclos = temporal_analyzer.update(eye_closed, timestamp)
                status = analyzer.analyze(
                    ear=ear_avg,
                    mar=mar,
                    perclos=perclos,
                    timestamp=timestamp,
                    face_detected=True,
                )

                draw_points(frame, left_eye_points, (0, 255, 0))
                draw_points(frame, right_eye_points, (0, 255, 0))
                draw_points(frame, mouth_points, (255, 0, 255))

            draw_text_info(
                frame,
                {
                    "FPS": fps,
                    "EAR Left": ear_left,
                    "EAR Right": ear_right,
                    "EAR Avg": ear_avg,
                    "MAR": mar,
                    "PERCLOS": perclos,
                },
            )
            draw_status(frame, status)

            cv2.imshow("DMS Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        landmark_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
