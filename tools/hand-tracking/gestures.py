"""
gestures.py - Rule-based gesture recognition built from HandState geometry
and raw normalized landmarks (finger extension ratios).
"""

from enum import Enum
from typing import Dict

import numpy as np

from hand_state import HandState
from settings import GestureSettings


class Gesture(str, Enum):
    OPEN_PALM = "Open Palm"
    FIST = "Fist"
    PINCH = "Pinch"
    POINTING = "Pointing"
    PEACE = "Peace Sign"
    UNKNOWN = "..."


def _finger_extension_ratios(landmarks_norm: np.ndarray) -> Dict[str, float]:
    """Per-finger extension ratio: tip-to-wrist distance over knuckle-to-wrist distance.

    Values noticeably above 1.0 indicate an extended finger; values near/under 1.0
    indicate a curled finger.
    """
    wrist = landmarks_norm[0][:2]
    fingers = {
        "thumb": (4, 2),
        "index": (8, 5),
        "middle": (12, 9),
        "ring": (16, 13),
        "pinky": (20, 17),
    }
    ratios = {}
    for name, (tip_idx, mcp_idx) in fingers.items():
        tip_d = np.linalg.norm(landmarks_norm[tip_idx][:2] - wrist)
        mcp_d = np.linalg.norm(landmarks_norm[mcp_idx][:2] - wrist) + 1e-6
        ratios[name] = float(tip_d / mcp_d)
    return ratios


class GestureRecognizer:
    """Classifies a single static gesture per hand per frame."""

    def __init__(self, settings: GestureSettings):
        self.settings = settings

    def recognize(self, state: HandState, landmarks_norm: np.ndarray) -> Gesture:
        s = self.settings
        ratios = _finger_extension_ratios(landmarks_norm)
        extended = {name: r > 1.15 for name, r in ratios.items()}
        n_extended = sum(extended.values())

        # Pinch takes priority: thumb and index tip very close together.
        if state.pinch_distance < s.pinch_threshold:
            return Gesture.PINCH

        if state.openness < s.fist_openness_threshold and n_extended <= 1:
            return Gesture.FIST

        if state.openness > s.open_openness_threshold and n_extended >= 4:
            return Gesture.OPEN_PALM

        if extended["index"] and not extended["middle"] and not extended["ring"] and not extended["pinky"]:
            return Gesture.POINTING

        if extended["index"] and extended["middle"] and not extended["ring"] and not extended["pinky"]:
            return Gesture.PEACE

        return Gesture.UNKNOWN
