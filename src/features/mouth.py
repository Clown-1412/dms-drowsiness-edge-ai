import numpy as np


MOUTH_INDICES = [61, 81, 13, 311, 291, 402, 14, 178]


def mouth_aspect_ratio(points):
    p = np.array(points, dtype=np.float32)

    h = np.linalg.norm(p[2] - p[6])
    w = np.linalg.norm(p[0] - p[4])

    if w == 0:
        return 0.0

    return h / w


def get_mouth_points(landmarks, image_size):
    width, height = image_size
    return [_to_pixel(landmarks[index], width, height) for index in MOUTH_INDICES]


def _to_pixel(landmark, width, height):
    return int(landmark[0] * width), int(landmark[1] * height)
