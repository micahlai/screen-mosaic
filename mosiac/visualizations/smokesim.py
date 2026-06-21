"""Smoke — a true incompressible fluid (stable-fluids) with emergent eddies.

A persistent velocity field is advected through itself, damped, and made
divergence-free each step by a Jacobi pressure projection. Incompressibility is
what produces real eddies: when the cursor pushes fluid out of a region, the
projection forces surrounding fluid to circulate in to fill the void — a vortex
forms naturally (no fake swirl term). Gentle curl-noise forcing keeps the whole
screen drifting when idle. The misty density is advected through that flow and
coloured with the selected gradient (with the noise overlay applied first).
"""

import numpy as np
import cv2

from . import Visualization, register, torch, _DEVICE
from . import gradients


@register("smoke", "Smoke")
class SmokeSim(Visualization):
    SIM_LONG = 200          # fluid grid long side (sim res; upscaled for display)
    PROJECT_ITERS = 40      # Jacobi pressure iterations (more = cleaner eddies)
    DAMP = 0.99             # velocity damping (viscosity-ish)
    AMBIENT = 1.2           # gentle curl-noise forcing so it drifts when idle
    NOISE_LONG = 720        # overlay-noise resolution (independent of the fluid grid)
    NOISE_STRENGTH = 0.45
    # cursor interaction (the preview window drives this) — push only; the eddies
    # are emergent from the projection, not an explicit swirl term.
    POINTER_RADIUS = 0.10
    POINTER_PUSH = 0.9

    def __init__(self, width, height):
        super().__init__(width, height)
        self._ptr = None
        if self.w >= self.h:
            self.gw = self.SIM_LONG
            self.gh = max(2, round(self.SIM_LONG * self.h / self.w))
        else:
            self.gh = self.SIM_LONG
            self.gw = max(2, round(self.SIM_LONG * self.w / self.h))
        if self.w >= self.h:
            self.nw = self.NOISE_LONG
            self.nh = max(2, round(self.NOISE_LONG * self.h / self.w))
        else:
            self.nh = self.NOISE_LONG
            self.nw = max(2, round(self.NOISE_LONG * self.w / self.h))

        self._gpu = torch is not None
        if not self._gpu:
            self.d = np.random.uniform(0.2, 0.6, (self.gh, self.gw)).astype(np.float32)
            self.no_cpu = np.random.rand(self.nh, self.nw).astype(np.float32)
            return

        dev = _DEVICE
        self.dev = dev
        self.lut = None
        self._grad_ver = -1
        z = lambda: torch.zeros((1, 1, self.gh, self.gw), device=dev)
        self.vx, self.vy = z(), z()                  # persistent velocity field
        self.c1 = self._coarse(5)                    # ambient forcing potential
        self.c2 = self._coarse(11)
        self.dseed = self._coarse(8)
        self.dseed2 = self._coarse(26)
        octaves = (max(8, self.NOISE_LONG // 24), max(16, self.NOISE_LONG // 7),
                   self.NOISE_LONG)
        self.no = [self._coarse(c) for c in octaves]
        ys, xs = torch.meshgrid(
            torch.arange(self.gh, device=dev, dtype=torch.float32),
            torch.arange(self.gw, device=dev, dtype=torch.float32), indexing="ij")
        self.xs, self.ys = xs, ys
        self._kdx = torch.tensor([[[[0, 0, 0], [-.5, 0, .5], [0, 0, 0]]]], device=dev)
        self._kdy = torch.tensor([[[[0, -.5, 0], [0, 0, 0], [0, .5, 0]]]], device=dev)
        self._knb = torch.tensor([[[[0, 1., 0], [1, 0, 1], [0, 1., 0]]]], device=dev)
        self.d = self._haze_target()

    # --- helpers -----------------------------------------------------------
    def _haze_target(self):
        f = self._smooth(self.dseed) * 1.5 + self._smooth(self.dseed2) * 1.1
        return torch.sigmoid(f - 0.2)

    def _coarse(self, long_cells):
        if self.gw >= self.gh:
            cw = long_cells; ch = max(2, round(long_cells * self.gh / self.gw))
        else:
            ch = long_cells; cw = max(2, round(long_cells * self.gw / self.gh))
        return torch.randn((1, 1, ch, cw), device=self.dev)

    def _smooth(self, coarse):
        return torch.nn.functional.interpolate(
            coarse, size=(self.gh, self.gw), mode="bicubic", align_corners=False)

    def _ddx(self, f): return torch.nn.functional.conv2d(f, self._kdx, padding=1)
    def _ddy(self, f): return torch.nn.functional.conv2d(f, self._kdy, padding=1)
    def _nb(self, f):  return torch.nn.functional.conv2d(f, self._knb, padding=1)

    def set_pointer(self, ptr):
        self._ptr = ptr

    def _backtrace_grid(self):
        bx = (self.xs - self.vx[0, 0]).clamp(0, self.gw - 1)
        by = (self.ys - self.vy[0, 0]).clamp(0, self.gh - 1)
        gx = bx / (self.gw - 1) * 2 - 1
        gy = by / (self.gh - 1) * 2 - 1
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)

    @staticmethod
    def _sample(f, grid):
        return torch.nn.functional.grid_sample(
            f, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _project(self):
        """Subtract the pressure gradient so velocity is divergence-free."""
        div = self._ddx(self.vx) + self._ddy(self.vy)
        p = torch.zeros_like(div)
        for _ in range(self.PROJECT_ITERS):
            p = (self._nb(p) - div) * 0.25
        self.vx = self.vx - self._ddx(p)
        self.vy = self.vy - self._ddy(p)

    # --- simulation --------------------------------------------------------
    def step(self):
        if not self._gpu:
            return self._step_cpu()
        rnd = torch.randn_like
        # gentle ambient forcing (curl of evolving noise) so it drifts when idle
        self.c1 = self.c1 * 0.99 + 0.05 * rnd(self.c1)
        self.c2 = self.c2 * 0.99 + 0.06 * rnd(self.c2)
        psi = self._smooth(self.c1) + 0.5 * self._smooth(self.c2)
        self.vx = self.vx + self._ddy(psi) * self.AMBIENT
        self.vy = self.vy - self._ddx(psi) * self.AMBIENT
        # cursor pushes fluid (a force; the resulting void is filled by the
        # projection below -> real eddies)
        if self._ptr is not None:
            self._apply_pointer()
        # self-advect velocity, damp, then project to divergence-free
        grid = self._backtrace_grid()
        self.vx = self._sample(self.vx, grid) * self.DAMP
        self.vy = self._sample(self.vy, grid) * self.DAMP
        self._project()
        # advect the misty density through the (now incompressible) flow
        grid = self._backtrace_grid()
        self.d = self._sample(self.d, grid)
        self.dseed = self.dseed * 0.999 + 0.015 * rnd(self.dseed)
        self.dseed2 = self.dseed2 * 0.99 + 0.03 * rnd(self.dseed2)
        self.d = (self.d * 0.985 + 0.015 * self._haze_target()).clamp(0, 1)
        for i in range(len(self.no)):
            self.no[i] = self.no[i] * 0.97 + 0.06 * rnd(self.no[i])
        self.t += 0.05

    def _apply_pointer(self):
        nx, ny, nvx, nvy = self._ptr
        px, py = nx * self.gw, ny * self.gh
        dx = self.xs - px
        dy = self.ys - py
        sigma = max(self.gw, self.gh) * self.POINTER_RADIUS
        fall = torch.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))[None, None]
        self.vx = self.vx + (nvx * self.gw) * self.POINTER_PUSH * fall
        self.vy = self.vy + (nvy * self.gh) * self.POINTER_PUSH * fall

    # --- rendering (gradient LUT + noise overlay; shared look) -------------
    @staticmethod
    def _overlay(base, top):
        return torch.where(base < 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))

    def _noise_map(self):
        F = torch.nn.functional
        acc, wsum = None, 0.0
        for i, oc in enumerate(self.no):
            up = F.interpolate(oc, size=(self.h, self.w),
                               mode="bilinear", align_corners=False)[0, 0]
            w = 0.6 ** i
            acc = up * w if acc is None else acc + up * w
            wsum += w
        return (0.5 + self.NOISE_STRENGTH * (acc / wsum)).clamp(0, 1)

    def render(self):
        if not self._gpu:
            return self._render_cpu()
        F = torch.nn.functional
        if self.lut is None or self._grad_ver != gradients.version():
            self.lut = torch.tensor(gradients.current_lut(), device=self.dev)
            self._grad_ver = gradients.version()
        up = F.interpolate(self.d, size=(self.h, self.w),
                           mode="bilinear", align_corners=False)[0, 0]
        up = self._overlay(up, self._noise_map())
        v = (up * 1.35 - 0.12).clamp(0, 1)
        idx = (v * (self.lut.shape[0] - 1)).round().long()
        return self.lut[idx].clamp(0, 255).to(torch.uint8).contiguous().cpu().numpy()

    # --- CPU fallback ------------------------------------------------------
    def _step_cpu(self):
        dx = int(2 * np.sin(self.t * 0.3))
        self.d = np.roll(np.roll(self.d, -1, 0), dx, 1)
        self.d = cv2.GaussianBlur(self.d, (0, 0), 1.5)
        self.d = np.clip(self.d * 0.99 + 0.01 * 0.5, 0, 1)
        self.no_cpu = np.clip(self.no_cpu * 0.9 + 0.1 * np.random.rand(*self.no_cpu.shape), 0, 1)
        self.t += 0.05

    def _render_cpu(self):
        lut = gradients.current_lut()
        up = cv2.resize(self.d, (self.w, self.h))
        noise = cv2.resize(self.no_cpu, (self.w, self.h))
        noise = np.clip(0.5 + self.NOISE_STRENGTH * (noise - 0.5) * 2, 0, 1)
        up = np.where(up < 0.5, 2 * up * noise, 1 - 2 * (1 - up) * (1 - noise))
        v = np.clip(up * 1.35 - 0.12, 0, 1)
        idx = (v * (lut.shape[0] - 1)).round().astype(np.int32)
        return lut[idx].astype(np.uint8)
