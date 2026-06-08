import numpy as np


LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1, p2):
    point_1 = np.array(p1, dtype=np.float32)
    point_2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(point_1 - point_2))


def eye_aspect_ratio(points):
    p = np.array(points, dtype=np.float32)

    d1 = euclidean_distance(p[1], p[5])
    d2 = euclidean_distance(p[2], p[4])
    d3 = euclidean_distance(p[0], p[3])

    if d3 == 0:
        return 0.0

    return (d1 + d2) / (2.0 * d3)


def get_eye_points(landmarks, image_size):
    width, height = image_size
    left_eye_points = [
        _to_pixel(landmarks[index], width, height)
        for index in LEFT_EYE_INDICES
    ]
    right_eye_points = [
        _to_pixel(landmarks[index], width, height)
        for index in RIGHT_EYE_INDICES
    ]
    return left_eye_points, right_eye_points


def _to_pixel(landmark, width, height):
    return int(landmark[0] * width), int(landmark[1] * height)
