"""Ambient smoke visualization — full-screen drifting haze with swirls.

Instead of a single emitter, the whole field is filled with mist and stirred by
a *curl-noise* velocity field: velocity = curl(psi) where psi is an evolving
low-frequency noise potential. The curl is divergence-free, so the flow forms
natural swirling eddies. The haze is advected through it and gently replenished
toward slowly-morphing noise so it stays misty and never clumps away.
"""

import numpy as np
import cv2

from . import Visualization, register, torch, _DEVICE


@register("smoke", "Smoke")
class SmokeSim(Visualization):
    SIM_LONG = 200          # fluid grid long side (sim res; upscaled for display)
    FLOW = 9.0              # how fast the swirls advect the haze

    def __init__(self, width, height):
        super().__init__(width, height)
        if self.w >= self.h:
            self.gw = self.SIM_LONG
            self.gh = max(2, round(self.SIM_LONG * self.h / self.w))
        else:
            self.gh = self.SIM_LONG
            self.gw = max(2, round(self.SIM_LONG * self.w / self.h))

        self._gpu = torch is not None      # fixed at construction time
        if not self._gpu:
            self.d = np.random.uniform(0.2, 0.6, (self.gh, self.gw)).astype(np.float32)
            return

        dev = _DEVICE
        self.dev = dev
        # coarse noise fields (octaves of the flow potential + the haze seed)
        self.c1 = self._coarse(5)       # big slow swirls
        self.c2 = self._coarse(11)      # smaller eddies
        self.dseed = self._coarse(8)    # haze base (low freq)
        self.dseed2 = self._coarse(26)  # haze detail (gets stretched into swirls)
        ys, xs = torch.meshgrid(
            torch.arange(self.gh, device=dev, dtype=torch.float32),
            torch.arange(self.gw, device=dev, dtype=torch.float32), indexing="ij")
        self.xs, self.ys = xs, ys
        self._kdx = torch.tensor([[[[0, 0, 0], [-.5, 0, .5], [0, 0, 0]]]], device=dev)
        self._kdy = torch.tensor([[[[0, -.5, 0], [0, 0, 0], [0, .5, 0]]]], device=dev)
        # start already misty so the whole screen has smoke from frame 0
        self.d = self._haze_target()

    def _haze_target(self):
        # low-frequency base + finer detail; the detail is what the swirls show
        f = self._smooth(self.dseed) * 1.5 + self._smooth(self.dseed2) * 1.1
        return torch.sigmoid(f - 0.2)

    def _coarse(self, long_cells):
        if self.gw >= self.gh:
            cw = long_cells; ch = max(2, round(long_cells * self.gh / self.gw))
        else:
            ch = long_cells; cw = max(2, round(long_cells * self.gw / self.gh))
        return torch.randn((1, 1, ch, cw), device=self.dev)

    def _smooth(self, coarse):
        F = torch.nn.functional
        return F.interpolate(coarse, size=(self.gh, self.gw),
                             mode="bicubic", align_corners=False)

    def _ddx(self, f): return torch.nn.functional.conv2d(f, self._kdx, padding=1)
    def _ddy(self, f): return torch.nn.functional.conv2d(f, self._kdy, padding=1)

    def _advect(self, f):
        F = torch.nn.functional
        bx = (self.xs - self.vx[0, 0]).clamp(0, self.gw - 1)
        by = (self.ys - self.vy[0, 0]).clamp(0, self.gh - 1)
        gx = bx / (self.gw - 1) * 2 - 1
        gy = by / (self.gh - 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        return F.grid_sample(f, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)

    def step(self):
        if not self._gpu:
            return self._step_cpu()
        rnd = torch.randn_like
        # evolve the potential octaves as a bounded random walk (AR(1)) so the
        # swirls drift and morph smoothly over time
        self.c1 = self.c1 * 0.99 + 0.05 * rnd(self.c1)
        self.c2 = self.c2 * 0.99 + 0.06 * rnd(self.c2)
        psi = self._smooth(self.c1) + 0.5 * self._smooth(self.c2)
        # divergence-free swirling velocity = curl(psi)
        self.vx = self._ddy(psi) * self.FLOW
        self.vy = -self._ddx(psi) * self.FLOW
        # advect the haze and only gently replenish so swirls persist/streak
        self.dseed = self.dseed * 0.998 + 0.02 * rnd(self.dseed)
        self.dseed2 = self.dseed2 * 0.99 + 0.04 * rnd(self.dseed2)
        self.d = self._advect(self.d)
        self.d = (self.d * 0.96 + 0.04 * self._haze_target()).clamp(0, 1)
        self.t += 0.05

    def render(self):
        if not self._gpu:
            return self._render_cpu()
        F = torch.nn.functional
        up = F.interpolate(self.d, size=(self.h, self.w),
                           mode="bilinear", align_corners=False)[0, 0]
        v = (up * 1.35 - 0.12).clamp(0, 1)          # lift contrast so swirls read
        # cool, slightly-blue misty smoke (BGR)
        frame = torch.stack([v * 215, v * 205, v * 198], dim=0)
        return frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

    # --- CPU fallback (no torch): blurred drifting noise ---
    def _step_cpu(self):
        dx = int(2 * np.sin(self.t * 0.3)); dy = -1
        self.d = np.roll(np.roll(self.d, dy, 0), dx, 1)
        self.d = cv2.GaussianBlur(self.d, (0, 0), 1.5)
        self.d += np.random.uniform(-0.02, 0.02, self.d.shape).astype(np.float32)
        self.d = np.clip(self.d * 0.99 + 0.01 * 0.5, 0, 1)
        self.t += 0.05

    def _render_cpu(self):
        up = cv2.resize(self.d, (self.w, self.h))
        v = np.clip(up * 1.35 - 0.12, 0, 1)
        return (np.stack([v * 215, v * 205, v * 198], -1)).astype(np.uint8)
