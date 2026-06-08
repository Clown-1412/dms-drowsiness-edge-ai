import numpy as np

def mouth_aspect_ratio(points):
    p = np.array(points, dtype=np.float32)

    h = np.linalg.norm(p[2] - p[6])
    w = np.linalg.norm(p[0] - p[4])

    if w == 0:
        return 0.0

    return h / w
