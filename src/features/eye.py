import numpy as np

def eye_aspect_ratio(points):
    p = np.array(points, dtype=np.float32)

    d1 = np.linalg.norm(p[1] - p[5])
    d2 = np.linalg.norm(p[2] - p[4])
    d3 = np.linalg.norm(p[0] - p[3])

    if d3 == 0:
        return 0.0

    return (d1 + d2) / (2.0 * d3)
