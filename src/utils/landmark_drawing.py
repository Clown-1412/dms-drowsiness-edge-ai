from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


LandmarkPoint = Dict[str, float]
Color = Tuple[int, int, int]


def _to_xy(point: LandmarkPoint) -> Tuple[int, int]:
    """Chuyen landmark dict ve toa do pixel cho OpenCV."""
    return int(point["x"]), int(point["y"])


def draw_landmark_group(
    frame,
    points: Iterable[LandmarkPoint],
    color: Color,
    closed: bool = True,
    radius: int = 3,
    line_thickness: int = 1,
) -> None:
    """Ve landmark bang duong noi va cac diem tron, khong ve chu."""
    point_list = list(points)
    if not point_list:
        return

    xy_points = [_to_xy(point) for point in point_list]
    if len(xy_points) >= 2:
        cv2.polylines(
            frame,
            [np.array(xy_points, dtype=np.int32)],
            isClosed=closed,
            color=color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    for point in point_list:
        x, y = _to_xy(point)
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)


def draw_eye_landmarks(
    frame,
    eye_landmarks: Dict[str, List[LandmarkPoint]],
) -> None:
    """Ve rieng mat trai va mat phai."""
    draw_landmark_group(
        frame,
        eye_landmarks.get("left_eye", []),
        color=(0, 255, 0),
        closed=True,
        radius=2,
        line_thickness=1,
    )
    draw_landmark_group(
        frame,
        eye_landmarks.get("right_eye", []),
        color=(0, 180, 255),
        closed=True,
        radius=2,
        line_thickness=1,
    )


def draw_mouth_landmarks(
    frame,
    mouth_landmarks: List[LandmarkPoint],
) -> None:
    """Ve outline vung mieng."""
    draw_landmark_group(
        frame,
        mouth_landmarks,
        color=(255, 0, 255),
        closed=True,
        radius=3,
        line_thickness=2,
    )


def draw_nose_landmarks(
    frame,
    nose_landmarks: List[LandmarkPoint],
) -> None:
    """Ve cac diem vung mui."""
    draw_landmark_group(
        frame,
        nose_landmarks,
        color=(0, 255, 255),
        closed=False,
        radius=3,
        line_thickness=1,
    )


def draw_head_points(
    frame,
    head_points: Dict[str, LandmarkPoint],
) -> None:
    """Ve va noi cac diem quan trong de debug head pose."""
    if not head_points:
        return

    draw_landmark_group(
        frame,
        head_points.values(),
        color=(255, 255, 0),
        closed=False,
        radius=4,
        line_thickness=1,
    )
