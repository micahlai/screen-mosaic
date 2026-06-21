"""Red-sticker hand tracker.

Finds the centroid of the largest red-coloured blob in the camera frame using
HSV thresholding. No ML model needed — just put a red sticker on your hand.

Same run() interface as hands.py so the server can swap them interchangeably.
"""

import time
import numpy as np
import cv2

# HSV ranges for red (wraps around 0/180 in OpenCV's 0-180 hue space)
_RED_LOWER1 = np.array([0,   120,  70],  dtype=np.uint8)
_RED_UPPER1 = np.array([10,  255, 255],  dtype=np.uint8)
_RED_LOWER2 = np.array([170, 120,  70],  dtype=np.uint8)
_RED_UPPER2 = np.array([180, 255, 255],  dtype=np.uint8)

# Minimum blob area in pixels to count as a detection (filters noise)
MIN_AREA = 300


def _find_red_centroid(frame):
    """Return (cx, cy, contour, mask) of the largest red blob, or (None, None, None, None)."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, _RED_LOWER1, _RED_UPPER1),
                          cv2.inRange(hsv, _RED_LOWER2, _RED_UPPER2))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, mask
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < MIN_AREA:
        return None, None, None, mask
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, None, None, mask
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]), best, mask


def _draw_debug(frame, cx, cy, contour, mask):
    """Annotate frame with detection overlay for /hands/debug."""
    dbg = frame.copy()

    # Show the red mask as a tinted overlay
    tint = np.zeros_like(dbg)
    tint[:, :, 2] = mask   # red channel
    cv2.addWeighted(tint, 0.25, dbg, 1.0, 0, dbg)

    H, W = dbg.shape[:2]

    if contour is not None and cx is not None:
        # Draw contour outline
        cv2.drawContours(dbg, [contour], -1, (0, 0, 255), 2)
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 180, 255), 1)
        # Crosshair at centroid
        cx_i, cy_i = int(cx), int(cy)
        cv2.line(dbg, (cx_i - 20, cy_i), (cx_i + 20, cy_i), (255, 255, 255), 2)
        cv2.line(dbg, (cx_i, cy_i - 20), (cx_i, cy_i + 20), (255, 255, 255), 2)
        cv2.circle(dbg, (cx_i, cy_i), 8, (0, 0, 255), -1)
        # Coordinates label
        label = f"({cx/W:.2f}, {cy/H:.2f})  area={int(cv2.contourArea(contour))}"
        cv2.putText(dbg, label, (cx_i + 14, cy_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        status = "RED DETECTED"
        status_col = (0, 220, 0)
    else:
        status = "no red detected"
        status_col = (60, 60, 200)

    cv2.putText(dbg, status, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_col, 2)
    return dbg


def run(should_run, on_hand, on_debug=None,
        camera_index=0, fps=20, cam_width=640, **_ignored):
    """Camera loop — same signature as hands.run() (extra kwargs ignored).

    Calls on_hand((cx, cy, vx, vy)) in normalized [0,1] coords each frame,
    or on_hand(None) when no red blob is visible.
    """
    cap   = None
    prev  = None
    interval = 1.0 / max(1, fps)

    while True:
        if not should_run():
            if cap is not None:
                cap.release(); cap = None
            on_hand(None)
            time.sleep(0.2)
            continue

        if cap is None:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                cap.release(); cap = None
                time.sleep(0.6)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(cam_width * 9 / 16))
            prev = None

        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(interval)
            continue

        H, W = frame.shape[:2]
        cx, cy, contour, mask = _find_red_centroid(frame)

        if cx is not None:
            cx_n = cx / W
            cy_n = cy / H
            vx_n = (cx_n - prev[0]) if prev is not None else 0.0
            vy_n = (cy_n - prev[1]) if prev is not None else 0.0
            prev = (cx_n, cy_n)
            on_hand((cx_n, cy_n, vx_n, vy_n))
        else:
            prev = None
            on_hand(None)

        if on_debug is not None:
            on_debug(_draw_debug(frame, cx, cy, contour, mask))

        time.sleep(interval)
