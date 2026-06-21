"""
settings.py - Centralized, tweakable configuration for the
Hand-Tracked Interactive Visualizer.

Change values here to adjust behavior without touching any logic code.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 480
    flip_horizontal: bool = True          # mirror the feed, feels natural
    target_fps: int = 60                  # requested capture FPS (camera dependent)


@dataclass
class TrackerSettings:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6


@dataclass
class MotionSettings:
    # Exponential smoothing factor for landmark positions (0 = no smoothing, 1 = frozen)
    smoothing_alpha: float = 0.35
    # How many past frames of center position to keep
    history_length: int = 8
    # Fingertip motion trail length, in frames
    trail_length: int = 25
    # Frames a hand can go undetected before being dropped
    max_missing_frames: int = 10


@dataclass
class GestureSettings:
    pinch_threshold: float = 0.45        # thumb-index distance / hand size
    fist_openness_threshold: float = 0.22
    open_openness_threshold: float = 0.62


@dataclass
class ParticleSettings:
    max_particles: int = 400
    spawn_rate: float = 0.6              # scales how many particles spawn per unit speed
    particle_lifetime: Tuple[int, int] = (20, 50)   # frames, (min, max)
    gravity: float = 0.0
    attraction_strength: float = 0.6     # pull strength during Pinch gesture
    repulsion_strength: float = 0.5      # push strength during Open Palm gesture
    force_radius: float = 200.0          # px radius of attraction/repulsion effect
    drag: float = 0.985


@dataclass
class RenderSettings:
    window_width: int = 960
    window_height: int = 720
    background_color: Tuple[int, int, int] = (8, 10, 18)
    grid_color: Tuple[int, int, int] = (20, 24, 36)
    bbox_color: Tuple[int, int, int] = (80, 220, 255)
    skeleton_color: Tuple[int, int, int] = (120, 200, 255)
    fingertip_color: Tuple[int, int, int] = (255, 180, 60)
    center_color: Tuple[int, int, int] = (255, 80, 140)
    text_color: Tuple[int, int, int] = (230, 230, 240)
    font_name: str = "consolas"
    font_size: int = 18
    hud_font_size: int = 16
    show_fps: bool = True


@dataclass
class AppSettings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    motion: MotionSettings = field(default_factory=MotionSettings)
    gesture: GestureSettings = field(default_factory=GestureSettings)
    particle: ParticleSettings = field(default_factory=ParticleSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    show_camera_window: bool = True
    max_render_fps: int = 60


SETTINGS = AppSettings()
