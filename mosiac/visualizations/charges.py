"""Magnetic charge particle simulation — fully GPU-accelerated render.

N particles attract each other and chase an auto-wandering cursor.
Physics and trail canvas both live on the GPU; only the final uint8 frame
is transferred to CPU for JPEG encoding.

Colour: slow → blue-purple (hue 250), fast → pink-magenta (hue 320).
"""

import math
import numpy as np
import cv2

from . import Visualization, register, torch, _DEVICE, _blur


@register("charges", "Charges")
class ChargesSim(Visualization):
    C1    = -200     # inter-particle constant (negative = attract)
    C2    = 45000    # cursor-attraction constant
    C3    = 0.97     # drag per frame
    TRAIL = 0.84     # canvas fade factor each frame

    def __init__(self, width, height):
        super().__init__(width, height)
        sc = self.scale
        self._gpu = torch is not None
        self.N = 1000 if self._gpu else 400
        N = self.N

        if self._gpu:
            self.dev = _DEVICE

        angles = np.random.uniform(0, 2 * math.pi, N).astype(np.float32)
        radii  = (60 + np.random.uniform(0, 120, N)).astype(np.float32) * sc
        px  = (self.w / 2 + np.cos(angles) * radii).astype(np.float32)
        py  = (self.h / 2 + np.sin(angles) * radii).astype(np.float32)
        pvx = (np.random.uniform(-2, 2, N) * sc).astype(np.float32)
        pvy = (np.random.uniform(-2, 2, N) * sc).astype(np.float32)

        if self._gpu:
            self.px  = torch.as_tensor(px,  device=self.dev)
            self.py  = torch.as_tensor(py,  device=self.dev)
            self.pvx = torch.as_tensor(pvx, device=self.dev)
            self.pvy = torch.as_tensor(pvy, device=self.dev)
            bg = torch.tensor([15.0, 6.0, 6.0], device=self.dev)
            # trail canvas stays on GPU the whole time
            self._canvas_t = bg.expand(self.h, self.w, 3).clone()
        else:
            self.px, self.py, self.pvx, self.pvy = px, py, pvx, pvy
            self._canvas = np.full((self.h, self.w, 3), [15.0, 6.0, 6.0], dtype=np.float32)

    # ------------------------------------------------------------------ cursor
    def _cursor(self):
        t = self.t
        x = self.w / 2 + self.w * 0.35 * math.sin(t * 0.7)
        y = self.h / 2 + self.h * 0.30 * math.sin(t * 1.1)
        return x, y

    # ------------------------------------------------------------------ step
    def step(self):
        sc    = self.scale
        MIN_D = 6.0 * sc
        dt    = 1 / 60
        cx, cy = self._cursor()
        self._cx, self._cy = cx, cy
        if self._gpu:
            self._step_gpu(cx, cy, MIN_D, dt, sc)
        else:
            self._step_cpu(cx, cy, MIN_D, dt, sc)
        self.t += 0.05

    def _step_gpu(self, cx, cy, MIN_D, dt, sc):
        px, py, pvx, pvy = self.px, self.py, self.pvx, self.pvy
        cdx = cx - px; cdy = cy - py
        cd  = torch.clamp(torch.hypot(cdx, cdy), min=MIN_D)
        cf  = self.C2 * sc * sc / (cd * cd)
        fx  = cdx / cd * cf
        fy  = cdy / cd * cf
        dx  = px[:, None] - px[None, :]
        dy  = py[:, None] - py[None, :]
        d   = torch.clamp(torch.hypot(dx, dy), min=MIN_D)
        d.fill_diagonal_(1e9)
        f   = self.C1 * sc * sc / (d * d)
        fx += (dx / d * f).sum(dim=1)
        fy += (dy / d * f).sum(dim=1)
        pvx = (pvx + fx * dt) * self.C3
        pvy = (pvy + fy * dt) * self.C3
        px  = px + pvx
        py  = py + pvy
        m   = 20 * sc
        px  = torch.where(px < -m, torch.full_like(px, self.w + m), px)
        px  = torch.where(px > self.w + m, torch.full_like(px, -m), px)
        py  = torch.where(py < -m, torch.full_like(py, self.h + m), py)
        py  = torch.where(py > self.h + m, torch.full_like(py, -m), py)
        self.px, self.py, self.pvx, self.pvy = px, py, pvx, pvy

    def _step_cpu(self, cx, cy, MIN_D, dt, sc):
        px, py, pvx, pvy = self.px, self.py, self.pvx, self.pvy
        cdx = cx - px; cdy = cy - py
        cd  = np.maximum(np.hypot(cdx, cdy), MIN_D)
        cf  = self.C2 * sc * sc / (cd * cd)
        fx  = cdx / cd * cf
        fy  = cdy / cd * cf
        dx  = px[:, None] - px[None, :]
        dy  = py[:, None] - py[None, :]
        d   = np.maximum(np.hypot(dx, dy), MIN_D)
        np.fill_diagonal(d, 1e9)
        f   = self.C1 * sc * sc / (d * d)
        fx += (dx / d * f).sum(axis=1)
        fy += (dy / d * f).sum(axis=1)
        pvx = (pvx + fx * dt) * self.C3
        pvy = (pvy + fy * dt) * self.C3
        px  = px + pvx; py = py + pvy
        m   = 20 * sc
        px  = np.where(px < -m, self.w + m, np.where(px > self.w + m, -m, px))
        py  = np.where(py < -m, self.h + m, np.where(py > self.h + m, -m, py))
        self.px, self.py, self.pvx, self.pvy = (
            px.astype(np.float32), py.astype(np.float32),
            pvx.astype(np.float32), pvy.astype(np.float32))

    # ------------------------------------------------------------------ render
    def render(self):
        return self._render_gpu() if self._gpu else self._render_cpu()

    def _render_gpu(self):
        dev = self.dev
        sc  = self.scale
        spd = torch.hypot(self.pvx, self.pvy)

        # Speed → HSL → RGB channels (vectorised on GPU)
        t   = torch.clamp(spd / (6.0 * sc), 0.0, 1.0)
        h   = 250.0 + t * 70.0
        s_  = (80.0 - t * 20.0) / 100.0
        l_  = (38.0 + t * 48.0) / 100.0
        c_  = (1.0 - torch.abs(2.0 * l_ - 1.0)) * s_
        hh  = h / 60.0
        xv  = c_ * (1.0 - torch.abs(hh % 2 - 1.0))
        m_  = l_ - c_ / 2.0
        in4 = (hh >= 4.0) & (hh < 5.0)
        r_f = torch.where(in4, xv, c_) + m_
        g_f = m_
        b_f = torch.where(in4, c_, xv) + m_

        # Scatter particle weights onto flat canvas
        xi  = self.px.round().long().clamp(0, self.w - 1)
        yi  = self.py.round().long().clamp(0, self.h - 1)
        idx = (yi * self.w + xi)

        glow_w = torch.clamp(6.0 + spd / sc * 1.4, 1.0, 20.0) * sc
        glow_w = glow_w * glow_w   # area weighting

        flat0 = torch.zeros(self.h * self.w, device=dev)

        def splat(ch, w):
            f = flat0.clone()
            f.scatter_add_(0, idx, ch * w)
            return f.view(self.h, self.w)

        # Glow pass (wide blur)
        sig_g = sc * 3.0
        Rg = _blur(splat(r_f, glow_w), sig_g)
        Gg = _blur(splat(g_f, glow_w), sig_g)
        Bg = _blur(splat(b_f, glow_w), sig_g)

        # Core pass (tight blob)
        core_w = sc * sc * 120.0
        cw_t   = torch.full_like(r_f, core_w)
        sig_c  = max(0.5, sc * 0.6)
        Rc = _blur(splat(r_f, cw_t), sig_c)
        Gc = _blur(splat(g_f, cw_t), sig_c)
        Bc = _blur(splat(b_f, cw_t), sig_c)

        GAIN_G, GAIN_C = 20.0, 40.0
        B = (Bg * GAIN_G + Bc * GAIN_C).clamp(0, 255)
        G = (Gg * GAIN_G + Gc * GAIN_C).clamp(0, 255)
        R = (Rg * GAIN_G + Rc * GAIN_C).clamp(0, 255)

        new_frame = torch.stack([B, G, R], dim=-1)   # H×W×3 BGR float

        bg_color = torch.tensor([15.0, 6.0, 6.0], device=dev)
        self._canvas_t = self._canvas_t * self.TRAIL + bg_color * (1 - self.TRAIL)
        self._canvas_t = torch.clamp(self._canvas_t + new_frame, 0, 255)

        return self._canvas_t.to(torch.uint8).contiguous().cpu().numpy()

    def _render_cpu(self):
        sc  = self.scale
        bg  = np.array([15.0, 6.0, 6.0])
        self._canvas = self._canvas * self.TRAIL + bg * (1 - self.TRAIL)

        spd    = np.hypot(self.pvx, self.pvy)
        t_     = np.minimum(spd / (6.0 * sc), 1.0)
        h_     = 250 + t_ * 70
        s_     = (80 - t_ * 20) / 100.0
        l_     = (38 + t_ * 48) / 100.0
        c_     = (1 - np.abs(2 * l_ - 1)) * s_
        hh_    = h_ / 60.0
        xv_    = c_ * (1 - np.abs(hh_ % 2 - 1))
        m_     = l_ - c_ / 2
        in4    = (hh_ >= 4) & (hh_ < 5)
        r_f    = np.where(in4, xv_, c_) + m_
        g_f    = m_
        b_f    = np.where(in4, c_, xv_) + m_

        glow_r = np.clip((6 + spd / sc * 1.4) * sc, sc, 20 * sc).astype(int)
        tmp    = np.zeros((self.h, self.w, 3), dtype=np.float32)
        xi     = np.clip(self.px.astype(int), 0, self.w - 1)
        yi     = np.clip(self.py.astype(int), 0, self.h - 1)

        for i in range(self.N):
            x, y, r = int(xi[i]), int(yi[i]), int(glow_r[i])
            col = (float(b_f[i]) * 255, float(g_f[i]) * 255, float(r_f[i]) * 255)
            cv2.circle(tmp, (x, y), r,   [c * 0.22 for c in col], -1, cv2.LINE_AA)
            cv2.circle(tmp, (x, y), max(1, r // 5), [c * 0.9 for c in col], -1, cv2.LINE_AA)

        ks = max(1, int(sc * 2)) | 1
        tmp = cv2.GaussianBlur(tmp, (ks, ks), sc * 1.5)
        self._canvas = np.clip(self._canvas + tmp, 0, 255)
        return self._canvas.astype(np.uint8)
