"""
Digital Aquarium visualization.

Calls the Pika API once to generate a looping aquarium video, saves it to
aquarium/aquarium.mp4, then plays it back frame-by-frame as the mosaic content.

Usage (generate + play):
    python -m mosiac.visualizations.aquarium

The server picks this up automatically via the @register decorator.
Set PIKA_API_KEY in your environment before running.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np

from . import Visualization, register

AQUARIUM_DIR = Path(__file__).parent.parent.parent / "aquarium"
AQUARIUM_VIDEO = AQUARIUM_DIR / "aquarium.mp4"

PIKA_PROMPT = (
    "a beautiful underwater aquarium, colorful tropical fish swimming in "
    "different directions, coral reef, rays of sunlight through water, "
    "bioluminescent jellyfish drifting, dark deep ocean blue background, "
    "cinematic slow motion, seamlessly looping"
)


def generate_aquarium_video(width: int = 1920, height: int = 1080) -> Path:
    """
    Generate the aquarium video via Pika and save it to aquarium/aquarium.mp4.
    Returns the path to the saved file.
    Skips generation if the file already exists.
    """
    AQUARIUM_DIR.mkdir(exist_ok=True)

    if AQUARIUM_VIDEO.exists():
        print(f"[aquarium] Using existing video: {AQUARIUM_VIDEO}")
        return AQUARIUM_VIDEO

    # aspect ratio closest to the screen layout
    aspect = "16:9" if width >= height else "9:16"

    print(f"[aquarium] Generating aquarium video via Pika ({aspect})...")
    print(f"[aquarium] Prompt: {PIKA_PROMPT}")

    from mosiac.pika import generate
    mp4_bytes = generate(PIKA_PROMPT, duration=5, aspect_ratio=aspect, loop=True)

    AQUARIUM_VIDEO.write_bytes(mp4_bytes)
    print(f"[aquarium] Saved to {AQUARIUM_VIDEO} ({len(mp4_bytes) // 1024} KB)")
    return AQUARIUM_VIDEO


@register("aquarium", "Digital Aquarium")
class Aquarium(Visualization):
    """
    Plays the Pika-generated aquarium video on loop.
    On first use, generates the video (takes ~1-2 minutes).
    """

    def __init__(self, width, height):
        super().__init__(width, height)
        self._cap = None
        self._frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._ready = False
        self._lock = threading.Lock()

        # Generate/load the video in a background thread so the server
        # doesn't block on startup.
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            video_path = generate_aquarium_video(self.w, self.h)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[aquarium] Could not open video: {video_path}")
                return
            with self._lock:
                self._cap = cap
                self._ready = True
            print("[aquarium] Video loaded, starting playback.")
        except Exception as e:
            print(f"[aquarium] Error loading video: {e}")

    def step(self):
        self.t += 1
        with self._lock:
            if not self._ready or self._cap is None:
                return
            ok, frame = self._cap.read()
            if not ok:
                # Loop: rewind to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if ok:
                # Resize to render resolution
                if frame.shape[1] != self.w or frame.shape[0] != self.h:
                    frame = cv2.resize(frame, (self.w, self.h))
                self._frame = frame

    def render(self) -> np.ndarray:
        with self._lock:
            if not self._ready:
                # Show a "generating..." placeholder
                canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                msg = "Generating aquarium video..."
                cv2.putText(canvas, msg, (self.w // 2 - 300, self.h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 180, 255), 3)
                return canvas
            return self._frame.copy()


if __name__ == "__main__":
    print("Generating aquarium video (this may take 1-2 minutes)...")
    generate_aquarium_video()
    print("Done. Run the server to display it across the screens.")
