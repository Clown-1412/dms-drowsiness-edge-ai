import cv2

def draw_status(frame, status):
    cv2.putText(frame, f"Status: {status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame
