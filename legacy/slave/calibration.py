import cv2
import numpy as np

SCREEN_W, SCREEN_H = 1920, 1080
MARKER_SIZE = 150

def show_calibration(screen_id):
    base_id = screen_id * 4
    canvas = np.zeros((SCREEN_H, SCREEN_W), dtype=np.uint8)

    corners = [
        (0, 0),
        (SCREEN_W - MARKER_SIZE, 0),
        (SCREEN_W - MARKER_SIZE, SCREEN_H - MARKER_SIZE),
        (0, SCREEN_H - MARKER_SIZE),
    ]

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, (x, y) in enumerate(corners):
        marker = np.zeros((MARKER_SIZE, MARKER_SIZE), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, base_id + i, MARKER_SIZE, marker)
        canvas[y:y+MARKER_SIZE, x:x+MARKER_SIZE] = marker

    window_name = f"Screen {screen_id} - Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_calibration(0)