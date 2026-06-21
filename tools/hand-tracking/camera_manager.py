"""
camera_manager.py - Handles webcam capture using OpenCV.
"""

import time

import cv2

from settings import CameraSettings


class CameraManager:
    """Thin wrapper around cv2.VideoCapture with convenience helpers."""

    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.cap = cv2.VideoCapture(settings.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        self.cap.set(cv2.CAP_PROP_FPS, settings.target_fps)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {settings.index}. "
                "Check that a webcam is connected, drivers are installed, "
                "and no other application is currently using it."
            )

    def read(self):
        """Returns (success, frame_bgr, timestamp_seconds)."""
        ok, frame = self.cap.read()
        ts = time.time()
        if not ok or frame is None:
            return False, None, ts

        if self.settings.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame, ts

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
