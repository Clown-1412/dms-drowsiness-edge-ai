import cv2


FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_points(frame, points, color=(0, 255, 0)):
    for point in points:
        cv2.circle(frame, tuple(point), 2, color, -1)
    return frame


def draw_text_info(frame, info_dict):
    labels = [
        ("FPS", "{:.1f}"),
        ("EAR Left", "{:.3f}"),
        ("EAR Right", "{:.3f}"),
        ("EAR Avg", "{:.3f}"),
        ("MAR", "{:.3f}"),
        ("PERCLOS", "{:.3f}"),
    ]

    x = 20
    y = 35
    line_height = 28
    for label, formatter in labels:
        value = info_dict.get(label)
        if value is None:
            text = f"{label}: -"
        else:
            text = f"{label}: {formatter.format(value)}"

        cv2.putText(frame, text, (x, y), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += line_height

    return frame


def draw_status(frame, status):
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 210),
        FONT,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
