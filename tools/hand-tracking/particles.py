"""
particles.py - Lightweight particle system driven by fingertip motion.

Fingertips emit particles while moving; particle color/size scale with
velocity; Pinch gestures attract nearby particles; Open Palm gestures
repel them; particles fade out over their lifetime.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from gestures import Gesture
from hand_state import HandState
from settings import ParticleSettings


@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    age: int = 0
    lifetime: int = 30

    def is_alive(self) -> bool:
        return self.age < self.lifetime

    def life_fraction(self) -> float:
        """1.0 = just born, 0.0 = about to expire (used for fade-out)."""
        return max(0.0, 1.0 - (self.age / max(self.lifetime, 1)))


class ParticleSystem:
    """Spawns, simulates, and exposes particles for rendering."""

    def __init__(self, settings: ParticleSettings):
        self.settings = settings
        self.particles: List[Particle] = []

    def spawn_from_fingertip(self, pos: np.ndarray, velocity: np.ndarray):
        s = self.settings
        if len(self.particles) >= s.max_particles:
            return

        speed = float(np.linalg.norm(velocity))
        if speed < 5.0:
            return  # don't spam particles while the hand is basically still

        n_spawn = int(np.clip(speed * s.spawn_rate * 0.02, 1, 6))
        for _ in range(n_spawn):
            if len(self.particles) >= s.max_particles:
                break
            jitter = (np.random.rand(2) - 0.5) * 6.0
            spawn_vel = velocity * 0.12 + jitter
            lifetime = random.randint(*s.particle_lifetime)
            self.particles.append(Particle(pos=pos.copy(), vel=spawn_vel, lifetime=lifetime))

    def apply_hand_forces(self, hand_states: Dict[str, HandState], gestures: Dict[str, Gesture]):
        s = self.settings
        for label, state in hand_states.items():
            gesture = gestures.get(label, Gesture.UNKNOWN)
            if gesture == Gesture.PINCH:
                self._apply_radial_force(state.center, strength=-s.attraction_strength, radius=s.force_radius)
            elif gesture == Gesture.OPEN_PALM:
                self._apply_radial_force(state.center, strength=s.repulsion_strength, radius=s.force_radius)

    def _apply_radial_force(self, center: np.ndarray, strength: float, radius: float):
        for p in self.particles:
            offset = p.pos - center
            dist = float(np.linalg.norm(offset)) + 1e-3
            if dist < radius:
                direction = offset / dist
                falloff = 1.0 - (dist / radius)
                p.vel += direction * strength * falloff * 4.0

    def update(self):
        s = self.settings
        for p in self.particles:
            p.vel *= s.drag
            p.vel[1] += s.gravity
            p.pos += p.vel
            p.age += 1
        self.particles = [p for p in self.particles if p.is_alive()]

    def get_render_data(self) -> List[Tuple[np.ndarray, float, float, float]]:
        """Returns (position, size, life_fraction, speed) tuples for rendering."""
        data = []
        for p in self.particles:
            speed = float(np.linalg.norm(p.vel))
            size = float(np.clip(2.0 + speed * 0.4, 2.0, 9.0))
            data.append((p.pos, size, p.life_fraction(), speed))
        return data
