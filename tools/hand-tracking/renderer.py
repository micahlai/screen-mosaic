"""
renderer.py - Draws the custom, dark-background hand visualization window
using pygame: bounding boxes, skeletons, fingertips, center points, motion
trails, particles, gesture labels, and an on-screen HUD of motion metrics.
"""

from typing import Dict, Optional, Tuple

import pygame

from gestures import Gesture
from hand_state import HandState
from hand_tracker import FINGERTIPS, HAND_CONNECTIONS
from particles import ParticleSystem
from settings import RenderSettings

FINGERTIP_COLORS = {
    4: (255, 120, 120),    # thumb
    8: (120, 255, 160),    # index
    12: (120, 200, 255),   # middle
    16: (255, 220, 120),   # ring
    20: (220, 140, 255),   # pinky
}


class Renderer:
    """Owns the pygame window and draws the full visualization each frame."""

    def __init__(self, settings: RenderSettings, source_size: Tuple[int, int]):
        self.settings = settings
        self.source_w, self.source_h = source_size

        pygame.init()
        pygame.display.set_caption("Hand Visualizer")
        self.screen = pygame.display.set_mode((settings.window_width, settings.window_height))
        self.clock = pygame.time.Clock()

        try:
            self.font = pygame.font.SysFont(settings.font_name, settings.font_size)
            self.hud_font = pygame.font.SysFont(settings.font_name, settings.hud_font_size)
        except Exception:
            self.font = pygame.font.Font(None, settings.font_size)
            self.hud_font = pygame.font.Font(None, settings.hud_font_size)

        self.scale_x = settings.window_width / self.source_w
        self.scale_y = settings.window_height / self.source_h

    def _to_screen(self, pt) -> Tuple[int, int]:
        return int(pt[0] * self.scale_x), int(pt[1] * self.scale_y)

    def _draw_grid(self):
        s = self.settings
        step = 40
        for x in range(0, s.window_width, step):
            pygame.draw.line(self.screen, s.grid_color, (x, 0), (x, s.window_height), 1)
        for y in range(0, s.window_height, step):
            pygame.draw.line(self.screen, s.grid_color, (0, y), (s.window_width, y), 1)

    def _draw_trails(self, state: HandState):
        for tip, trail in state.trails.items():
            color = FINGERTIP_COLORS.get(tip, self.settings.fingertip_color)
            n = len(trail)
            for i, pos in enumerate(trail):
                fade = (i + 1) / max(n, 1)
                radius = max(1, int(3 * fade))
                faded_color = tuple(int(c * fade * 0.7) for c in color)
                pygame.draw.circle(self.screen, faded_color, self._to_screen(pos), radius)

    def _draw_skeleton(self, state: HandState):
        s = self.settings
        if state.landmarks_px is None:
            return
        pts = [self._to_screen(p) for p in state.landmarks_px]

        for a, b in HAND_CONNECTIONS:
            pygame.draw.line(self.screen, s.skeleton_color, pts[a], pts[b], 2)

        for i, pt in enumerate(pts):
            if i in FINGERTIPS:
                continue
            pygame.draw.circle(self.screen, s.skeleton_color, pt, 3)

        for tip in FINGERTIPS:
            color = FINGERTIP_COLORS.get(tip, s.fingertip_color)
            pygame.draw.circle(self.screen, color, pts[tip], 8)
            pygame.draw.circle(self.screen, (255, 255, 255), pts[tip], 8, 1)

    def _draw_bbox(self, state: HandState):
        s = self.settings
        x_min, y_min, x_max, y_max = state.bbox
        p1 = self._to_screen((x_min, y_min))
        p2 = self._to_screen((x_max, y_max))
        rect = pygame.Rect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1])
        pygame.draw.rect(self.screen, s.bbox_color, rect, 2)

    def _draw_center(self, state: HandState):
        s = self.settings
        center = self._to_screen(state.center)
        pygame.draw.circle(self.screen, s.center_color, center, 6)
        pygame.draw.circle(self.screen, (255, 255, 255), center, 6, 1)

        if state.speed > 1.0:
            end = (
                int(center[0] + state.velocity[0] * 0.15 * self.scale_x),
                int(center[1] + state.velocity[1] * 0.15 * self.scale_y),
            )
            pygame.draw.line(self.screen, s.center_color, center, end, 2)

    def _draw_labels(self, state: HandState, gesture: Gesture):
        x_min, y_min, _, _ = state.bbox
        pos = self._to_screen((x_min, y_min))

        gesture_surf = self.font.render(gesture.value, True, (255, 255, 120))
        self.screen.blit(gesture_surf, (pos[0], max(pos[1] - 44, 0)))

        label_surf = self.font.render(f"{state.label} Hand", True, self.settings.bbox_color)
        self.screen.blit(label_surf, (pos[0], max(pos[1] - 22, 0)))

    def _draw_particles(self, particle_system: ParticleSystem):
        for pos, size, life, speed in particle_system.get_render_data():
            t = min(speed / 25.0, 1.0)
            color = (
                int(60 + 195 * t),
                int(80 + 100 * (1 - t)),
                int(255 - 120 * t),
            )
            faded = tuple(int(c * life) for c in color)
            pygame.draw.circle(self.screen, faded, self._to_screen(pos), max(1, int(size * life)))

    def _draw_hud(self, hand_states: Dict[str, HandState], hand_distance: Optional[float], fps: float):
        s = self.settings
        lines = []
        if s.show_fps:
            lines.append(f"FPS: {fps:.1f}")

        for label, state in hand_states.items():
            lines.append(
                f"{label}: speed={state.speed:6.1f}px/s  accel={state.accel_mag:6.1f}px/s^2  "
                f"openness={state.openness:.2f}  pinch={state.pinch_distance:.3f}  "
                f"angle={state.orientation_deg:6.1f}deg"
            )

        if hand_distance is not None:
            lines.append(f"Hand distance: {hand_distance:.1f}px")

        for i, line in enumerate(lines):
            surf = self.hud_font.render(line, True, s.text_color)
            self.screen.blit(surf, (10, 10 + i * (s.hud_font_size + 4)))

    def render(
        self,
        hand_states: Dict[str, HandState],
        gestures: Dict[str, Gesture],
        particle_system: ParticleSystem,
        hand_distance: Optional[float],
        fps: float,
    ):
        self.screen.fill(self.settings.background_color)
        self._draw_grid()
        self._draw_particles(particle_system)

        for label, state in hand_states.items():
            if state.landmarks_px is None:
                continue
            self._draw_trails(state)
            self._draw_bbox(state)
            self._draw_skeleton(state)
            self._draw_center(state)
            self._draw_labels(state, gestures.get(label, Gesture.UNKNOWN))

        self._draw_hud(hand_states, hand_distance, fps)
        pygame.display.flip()

    def tick(self, max_fps: int) -> float:
        self.clock.tick(max_fps)
        return self.clock.get_fps()

    def poll_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def close(self):
        pygame.quit()
