"""Smoke / fire visualization — a stable-fluids simulation on the GPU.

Semi-Lagrangian advection (grid_sample) + Jacobi pressure projection + buoyancy,
run on a small fluid grid and upscaled to the render resolution for display.
"""

import math
import numpy as np
import cv2

from . import Visualization, register, torch, _DEVICE


@register("smoke", "Smoke")
class SmokeSim(Visualization):
    SIM_LONG = 180          # fluid grid long side (sim res; upscaled for display)

    def __init__(self, width, height):
        super().__init__(width, height)
        if self.w >= self.h:
            self.gw = self.SIM_LONG
            self.gh = max(2, round(self.SIM_LONG * self.h / self.w))
        else:
            self.gh = self.SIM_LONG
            self.gw = max(2, round(self.SIM_LONG * self.w / self.h))

        if torch is None:
            self.d = np.zeros((self.gh, self.gw), np.float32)
            return

        dev = _DEVICE
        self.dev = dev
        z = lambda: torch.zeros((1, 1, self.gh, self.gw), device=dev)
        self.vx, self.vy, self.d = z(), z(), z()
        ys, xs = torch.meshgrid(
            torch.arange(self.gh, device=dev, dtype=torch.float32),
            torch.arange(self.gw, device=dev, dtype=torch.float32), indexing="ij")
        self.xs, self.ys = xs, ys
        self._kdx = torch.tensor([[[[0, 0, 0], [-.5, 0, .5], [0, 0, 0]]]], device=dev)
        self._kdy = torch.tensor([[[[0, -.5, 0], [0, 0, 0], [0, .5, 0]]]], device=dev)
        self._knb = torch.tensor([[[[0, 1., 0], [1, 0, 1], [0, 1., 0]]]], device=dev)

    # finite differences
    def _ddx(self, f): return torch.nn.functional.conv2d(f, self._kdx, padding=1)
    def _ddy(self, f): return torch.nn.functional.conv2d(f, self._kdy, padding=1)
    def _nb(self, f):  return torch.nn.functional.conv2d(f, self._knb, padding=1)

    def _advect(self, f):
        F = torch.nn.functional
        bx = (self.xs - self.vx[0, 0]).clamp(0, self.gw - 1)
        by = (self.ys - self.vy[0, 0]).clamp(0, self.gh - 1)
        gx = bx / (self.gw - 1) * 2 - 1
        gy = by / (self.gh - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        return F.grid_sample(f, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)

    def _project(self, iters=24):
        div = self._ddx(self.vx) + self._ddy(self.vy)
        p = torch.zeros_like(div)
        for _ in range(iters):
            p = (self._nb(p) - div) * 0.25
        self.vx = self.vx - self._ddx(p)
        self.vy = self.vy - self._ddy(p)

    def step(self):
        if torch is None:
            return self._step_cpu()
        # rising plume injected at the bottom, wandering left/right over time
        cx = int(self.gw * 0.5 + math.sin(self.t * 0.7) * self.gw * 0.12)
        r = max(2, self.gw // 12)
        y1 = self.gh - 2; y0 = max(0, y1 - 3)
        x0, x1 = max(0, cx - r), min(self.gw, cx + r)
        self.d[..., y0:y1, x0:x1] += 0.5
        self.vy[..., y0:y1, x0:x1] += -2.6                       # push upward (-y)
        self.vx[..., y0:y1, x0:x1] += math.sin(self.t * 1.3) * 0.9
        self.vy = self.vy - 0.06 * self.d                        # buoyancy
        self.vx = self._advect(self.vx)
        self.vy = self._advect(self.vy)
        self._project()
        self.d = self._advect(self.d)
        self.d = (self.d * 0.985).clamp(0, 4)
        self.t += 0.05

    def render(self):
        if torch is None:
            return self._render_cpu()
        F = torch.nn.functional
        up = F.interpolate(self.d.clamp(0, 1.5), size=(self.h, self.w),
                           mode="bilinear", align_corners=False)[0, 0]
        v = up
        # warm fire->smoke palette (BGR)
        r = (v * 1.8).clamp(0, 1)
        g = (v * 1.8 - 0.5).clamp(0, 1)
        b = (v * 1.8 - 1.15).clamp(0, 1)
        frame = torch.stack([b, g, r], dim=0) * 255
        return frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

    # crude CPU fallback (no torch): upward-scrolled, blurred noise
    def _step_cpu(self):
        self.d = np.roll(self.d, -1, axis=0)
        cx = int(self.gw * 0.5 + math.sin(self.t * 0.7) * self.gw * 0.12)
        r = max(2, self.gw // 12)
        self.d[-3:, max(0, cx - r):cx + r] += np.random.uniform(0.3, 0.7)
        self.d = cv2.GaussianBlur(self.d, (0, 0), 1.2) * 0.97
        self.t += 0.05

    def _render_cpu(self):
        up = cv2.resize(np.clip(self.d, 0, 1.5), (self.w, self.h))
        v = up
        b = np.clip(v * 1.8 - 1.15, 0, 1)
        g = np.clip(v * 1.8 - 0.5, 0, 1)
        r = np.clip(v * 1.8, 0, 1)
        return (np.stack([b, g, r], -1) * 255).astype(np.uint8)
