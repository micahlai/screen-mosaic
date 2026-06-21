"""
hand_tracker.py - Wraps the MediaPipe Tasks "HandLandmarker" model to extract
21 hand landmarks, handedness, and a per-hand bounding box from a video frame.

Note: MediaPipe removed the old `mediapipe.solutions.hands` API from PyPI
releases (0.10.21+) in favor of the Tasks API used here. The required model
bundle (a few MB) is downloaded automatically the first time you run the
app and cached locally under `models/hand_landmarker.task`.
"""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from settings import TrackerSettings

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# Standard 21-point hand skeleton connections, exposed by the Tasks API.
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]

# Key landmark indices (MediaPipe Hands topology)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [2, 5, 9, 13, 17]   # base knuckle joints, used for openness/curl estimates


def ensure_model() -> Path:
    """Downloads the hand landmarker model bundle on first run, if missing."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmark model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not automatically download the hand landmark model "
            f"({exc}).\nPlease download it manually from:\n  {MODEL_URL}\n"
            f"and save it to:\n  {MODEL_PATH}"
        ) from exc
    print("Model downloaded.")
    return MODEL_PATH


@dataclass
class DetectedHand:
    """Raw per-frame detection result for a single hand."""
    label: str                        # "Left" or "Right" (selfie-view, as reported by the model)
    score: float                      # handedness confidence, 0..1
    landmarks: np.ndarray             # shape (21, 3), normalized x, y, z
    landmarks_px: np.ndarray          # shape (21, 2), pixel coordinates
    bbox: Tuple[int, int, int, int]   # x_min, y_min, x_max, y_max in pixels


class HandTracker:
    """Wraps mediapipe.tasks.python.vision.HandLandmarker for streaming video."""

    def __init__(self, settings: TrackerSettings):
        self.settings = settings
        model_path = ensure_model()

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=settings.max_num_hands,
            min_hand_detection_confidence=settings.min_detection_confidence,
            min_hand_presence_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray) -> List[DetectedHand]:
        """Runs detection on a BGR frame and returns a list of DetectedHand."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detected: List[DetectedHand] = []
        if not result.hand_landmarks:
            return detected

        for i, hand_landmarks in enumerate(result.hand_landmarks):
            pts = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32
            )
            pts_px = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1)

            x_min, y_min = pts_px.min(axis=0)
            x_max, y_max = pts_px.max(axis=0)
            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))

            if i < len(result.handedness) and result.handedness[i]:
                category = result.handedness[i][0]
                label = category.category_name
                score = category.score
            else:
                label = "Unknown"
                score = 0.0

            detected.append(
                DetectedHand(
                    label=label,
                    score=score,
                    landmarks=pts,
                    landmarks_px=pts_px,
                    bbox=bbox,
                )
            )

        return detected

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
