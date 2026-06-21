"""
hand_state.py - Maintains temporal state per tracked hand: smoothed landmark
positions, velocity, acceleration, fingertip motion trails, and derived
geometric measures (orientation, openness, pinch distance).
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from hand_tracker import DetectedHand, FINGERTIPS, FINGER_MCPS, WRIST
from settings import MotionSettings


@dataclass
class HandState:
    """Smoothed, temporally-aware state for one hand, keyed by 'Left'/'Right'."""

    label: str
    settings: MotionSettings

    landmarks_px: Optional[np.ndarray] = None      # smoothed (21, 2) pixel coords
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    speed: float = 0.0
    accel_mag: float = 0.0

    orientation_deg: float = 0.0
    openness: float = 0.0           # 0 (closed fist) .. 1 (fully open palm)
    pinch_distance: float = 0.0     # normalized by hand size; smaller = closer pinch

    last_update_time: float = field(default_factory=time.time)
    history: Deque[Tuple[np.ndarray, float]] = field(default_factory=deque)
    trails: Dict[int, Deque[np.ndarray]] = field(default_factory=dict)

    alive: bool = True
    missing_frames: int = 0

    def __post_init__(self):
        self.history = deque(maxlen=self.settings.history_length)
        self.trails = {tip: deque(maxlen=self.settings.trail_length) for tip in FINGERTIPS}

    def update(self, detection: DetectedHand, now: float):
        """Feed in a new detection and recompute smoothed / derived values."""
        new_px = detection.landmarks_px.astype(np.float32)

        if self.landmarks_px is None:
            smoothed = new_px
        else:
            a = self.settings.smoothing_alpha
            # Exponential moving average between previous smoothed pose and new raw pose.
            smoothed = a * self.landmarks_px + (1.0 - a) * new_px

        new_center = smoothed.mean(axis=0)

        dt = max(now - self.last_update_time, 1e-3)
        prev_center = self.center if self.history else new_center
        new_velocity = (new_center - prev_center) / dt

        prev_velocity = self.velocity
        new_accel = (new_velocity - prev_velocity) / dt

        # Commit new state
        self.landmarks_px = smoothed
        self.bbox = detection.bbox
        self.center = new_center
        self.velocity = new_velocity
        self.acceleration = new_accel
        self.speed = float(np.linalg.norm(new_velocity))
        self.accel_mag = float(np.linalg.norm(new_accel))
        self.missing_frames = 0
        self.alive = True

        self.history.append((new_center.copy(), now))
        self.last_update_time = now

        for tip in FINGERTIPS:
            self.trails[tip].append(smoothed[tip].copy())

        self._update_geometry(detection.landmarks)

    def _update_geometry(self, landmarks_norm: np.ndarray):
        """Derives orientation, openness, and pinch distance from normalized landmarks."""
        wrist = landmarks_norm[WRIST][:2]
        middle_mcp = landmarks_norm[9][:2]

        direction = middle_mcp - wrist
        self.orientation_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))

        hand_size = float(np.linalg.norm(middle_mcp - wrist) + 1e-6)

        tip_dists = [np.linalg.norm(landmarks_norm[tip][:2] - wrist) for tip in FINGERTIPS]
        mcp_dists = [np.linalg.norm(landmarks_norm[m][:2] - wrist) for m in FINGER_MCPS]
        ratios = [t / (m + 1e-6) for t, m in zip(tip_dists, mcp_dists)]
        avg_ratio = float(np.mean(ratios))
        # Empirically, a closed fist sits near ratio ~0.9-1.0, a wide open palm ~1.8-2.2.
        self.openness = float(np.clip((avg_ratio - 0.9) / 1.1, 0.0, 1.0))

        thumb_tip = landmarks_norm[4][:2]
        index_tip = landmarks_norm[8][:2]
        self.pinch_distance = float(np.linalg.norm(thumb_tip - index_tip) / hand_size)

    def mark_missing(self, max_missing_frames: int):
        self.missing_frames += 1
        if self.missing_frames > max_missing_frames:
            self.alive = False

    def fingertip_positions(self) -> Dict[int, np.ndarray]:
        if self.landmarks_px is None:
            return {}
        return {tip: self.landmarks_px[tip] for tip in FINGERTIPS}


class HandStateManager:
    """Keeps one HandState per label ('Left'/'Right') alive across frames."""

    def __init__(self, settings: MotionSettings):
        self.settings = settings
        self.states: Dict[str, HandState] = {}

    def update(self, detections: List[DetectedHand]) -> Dict[str, HandState]:
        now = time.time()
        seen_labels = set()

        for det in detections:
            label = det.label
            seen_labels.add(label)
            if label not in self.states:
                self.states[label] = HandState(label=label, settings=self.settings)
            self.states[label].update(det, now)

        for label, state in list(self.states.items()):
            if label not in seen_labels:
                state.mark_missing(self.settings.max_missing_frames)
                if not state.alive:
                    del self.states[label]

        return self.states

    def hand_distance(self) -> Optional[float]:
        """Distance between the two tracked hand centers, if both are present."""
        if len(self.states) < 2:
            return None
        centers = [s.center for s in self.states.values()]
        return float(np.linalg.norm(centers[0] - centers[1]))
