"""
Generate a synthetic test image: three displays, each with four AprilTag
markers (one per corner), placed and slightly rotated on a photo-like canvas.
Used to exercise the detection pipeline without a real photo.

Usage: python make_test_image.py [out.png]
"""

import sys

import cv2
import numpy as np

import detector

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def marker_img(marker_id, size=120):
    img = cv2.aruco.generateImageMarker(DICT, marker_id, size)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def place_display(canvas, marker_ids, origin, w, h, angle_deg, ms=120):
    """Place 4 markers at the corners of a w*h rect, rotated about its center."""
    ox, oy = origin
    cx, cy = ox + w / 2, oy + h / 2
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    # corner anchor points (top-left of each marker), clockwise from TL
    anchors = [
        (ox, oy),
        (ox + w - ms, oy),
        (ox + w - ms, oy + h - ms),
        (ox, oy + h - ms),
    ]
    for mid, (ax, ay) in zip(marker_ids, anchors):
        m = marker_img(mid, ms)
        pts = np.array([[ax, ay], [ax + ms, ay], [ax + ms, ay + ms], [ax, ay + ms]],
                       dtype=np.float32)
        rotated = (rot @ (pts - [cx, cy]).T).T + [cx, cy]
        src = np.array([[0, 0], [ms, 0], [ms, ms], [0, ms]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, rotated.astype(np.float32))
        warped = cv2.warpPerspective(m, H, (canvas.shape[1], canvas.shape[0]),
                                     borderValue=(255, 255, 255))
        mask = cv2.warpPerspective(np.ones_like(m) * 255, H,
                                   (canvas.shape[1], canvas.shape[0]))
        canvas[mask[:, :, 0] > 128] = warped[mask[:, :, 0] > 128]


def main(out="test_image.png"):
    canvas = np.full((1400, 1800, 3), 40, dtype=np.uint8)
    place_display(canvas, [0, 1, 2, 3], (120, 120), 500, 360, angle_deg=4)
    place_display(canvas, [4, 5, 6, 7], (760, 150), 520, 380, angle_deg=-3)
    place_display(canvas, [8, 9, 10, 11], (400, 700), 700, 480, angle_deg=2)
    cv2.imwrite(out, canvas)
    print(f"wrote {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_image.png")
