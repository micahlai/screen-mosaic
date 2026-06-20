"""Particle-flow visualization (GPU splat + glow blur)."""

import numpy as np
import cv2

from . import Visualization, register, torch, _DEVICE, _blur

_GAIN_GLOW, _GAIN_BODY, _GAIN_CORE = 26.0, 30.0, 42.0


@register("particles", "Particle Flow")
class ParticleFlow(Visualization):
    def __init__(self, width, height, num_particles=900):
        super().__init__(width, height)
        sc = self.scale
        n = num_particles
        self.n = n
        self.x = np.random.uniform(0, self.w, n).astype(np.float32)
        self.y = np.random.uniform(0, self.h, n).astype(np.float32)
        self.sizes = (np.random.uniform(2, 8, n) * sc).astype(np.float32)
        self.brightness = np.random.uniform(0.4, 1.0, n).astype(np.float32)
        self.speed = (np.random.uniform(0.3, 1.2, n) * sc).astype(np.float32)
        self.drift = (np.random.uniform(-0.3, 0.3, n) * sc).astype(np.float32)
        self.phase = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
        self._flat = None

    def step(self):
        t = self.t
        self.y -= self.speed
        self.x += np.sin(t * 0.8 + self.phase) * self.drift + self.drift * 0.3
        self.y[self.y < 0] = self.h
        self.x[self.x < 0] += self.w
        self.x[self.x >= self.w] -= self.w
        self.t += 0.05

    def render(self):
        if torch is None:
            return self._render_cpu()
        dev = _DEVICE
        H, W = self.h, self.w
        sc = self.scale
        x = torch.as_tensor(self.x, device=dev)
        y = torch.as_tensor(self.y, device=dev)
        b = torch.as_tensor(self.brightness, device=dev)
        s = torch.as_tensor(self.sizes, device=dev)
        xi = x.round().long().clamp_(0, W - 1)
        yi = y.round().long().clamp_(0, H - 1)
        idx = yi * W + xi
        if self._flat is None:
            self._flat = torch.zeros(H * W, device=dev)

        def splat(weight):
            flat = self._flat.zero_()
            flat.scatter_add_(0, idx, weight)
            return flat.view(H, W)

        glow = _blur(splat(b * s * s), sigma=4.0 * sc)
        body = _blur(splat(b * s),     sigma=1.5 * sc)
        core = _blur(splat(b),         sigma=0.6 * sc)
        c_glow = torch.tensor([20.0, 80.0, 60.0], device=dev).view(3, 1, 1)
        c_body = torch.tensor([80.0, 220.0, 200.0], device=dev).view(3, 1, 1)
        c_core = torch.tensor([180.0, 255.0, 240.0], device=dev).view(3, 1, 1)
        canvas = (glow * _GAIN_GLOW) * c_glow \
            + (body * _GAIN_BODY) * c_body \
            + (core * _GAIN_CORE) * c_core
        canvas = canvas.clamp(0, 255).to(torch.uint8)
        return canvas.permute(1, 2, 0).contiguous().cpu().numpy()

    def _render_cpu(self):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.n):
            cx, cy = int(self.x[i]), int(self.y[i])
            r = int(self.sizes[i]); bb = self.brightness[i]
            cv2.circle(canvas, (cx, cy), r,
                       (int(200 * bb), int(220 * bb), int(80 * bb)), -1, cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=4, sigmaY=4)
        return canvas
