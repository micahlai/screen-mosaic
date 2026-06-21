"""Magnetic charge particle simulation.

Faithful Python port of ghost_trail_v2.html:
- Radial gradient per particle (glow + core dot) with 'lighter' additive blending
- Canvas fades at 0.18 per frame (trail persistence = 0.82)
- Speed colour: hue 250→320, sat 80→60, lit 38→86
- Physics: inter-particle attraction (C1=-200) + cursor attraction (C2=45000)
"""

import math
import numpy as np
import cv2
import colorsys

from . import Visualization, register, torch, _DEVICE, _blur


# ---------------------------------------------------------------------------
# Colour helper — exactly matching speedColor() in the HTML
# ---------------------------------------------------------------------------

def _speed_rgb_np(spd_arr, sc):
    """spd_arr (render px/frame) → (N,3) float32 RGB in [0,1]"""
    t   = np.minimum(spd_arr / (6.0 * sc), 1.0)
    hue = 250.0 + t * 70.0
    sat = (80.0  - t * 20.0) / 100.0
    lit = (38.0  + t * 48.0) / 100.0
    c   = (1.0 - np.abs(2.0 * lit - 1.0)) * sat
    hh  = hue / 60.0
    x_  = c * (1.0 - np.abs(hh % 2 - 1.0))
    m   = lit - c / 2.0
    in4 = (hh >= 4) & (hh < 5)   # hue 240-300
    r   = np.where(in4, x_, c) + m
    g   = m
    b   = np.where(in4, c, x_) + m
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _speed_rgb_t(spd_t, sc):
    """torch tensor version, returns (N,3) float tensor (R,G,B in [0,1])"""
    t   = torch.clamp(spd_t / (6.0 * sc), 0.0, 1.0)
    hue = 250.0 + t * 70.0
    sat = (80.0  - t * 20.0) / 100.0
    lit = (38.0  + t * 48.0) / 100.0
    c   = (1.0 - torch.abs(2.0 * lit - 1.0)) * sat
    hh  = hue / 60.0
    x_  = c * (1.0 - torch.abs(hh % 2 - 1.0))
    m   = lit - c / 2.0
    in4 = (hh >= 4.0) & (hh < 5.0)
    r   = torch.where(in4, x_, c) + m
    g   = m
    b   = torch.where(in4, c, x_) + m
    return torch.stack([r, g, b], dim=-1)


@register("charges", "Charges")
class ChargesSim(Visualization):
    C1   = -200    # inter-particle constant (negative = attract)
    C2   = 45000   # cursor attraction constant
    C3   = 0.97    # drag per frame
    DT_S = 1 / 60  # physics timestep

    def __init__(self, width, height):
        super().__init__(width, height)
        sc = self.scale
        self._gpu = torch is not None
        self.N = 1000 if self._gpu else 400

        N = self.N
        angles = np.random.uniform(0, 2 * math.pi, N).astype(np.float32)
        radii  = (60 + np.random.uniform(0, 120, N)).astype(np.float32) * sc
        px  = (self.w / 2 + np.cos(angles) * radii).astype(np.float32)
        py  = (self.h / 2 + np.sin(angles) * radii).astype(np.float32)
        pvx = (np.random.uniform(-0.5, 0.5, N) * sc).astype(np.float32)
        pvy = (np.random.uniform(-0.5, 0.5, N) * sc).astype(np.float32)

        if self._gpu:
            dev = _DEVICE
            self.dev = dev
            self.px  = torch.as_tensor(px,  device=dev)
            self.py  = torch.as_tensor(py,  device=dev)
            self.pvx = torch.as_tensor(pvx, device=dev)
            self.pvy = torch.as_tensor(pvy, device=dev)
            # canvas lives on GPU: start at bg colour #06060f → BGR(15,6,6)
            self._canvas_t = torch.zeros(self.h, self.w, 3, device=dev)
            self._canvas_t[:, :, 0] = 15.0
            self._canvas_t[:, :, 1] = 6.0
            self._canvas_t[:, :, 2] = 6.0
        else:
            self.px, self.py, self.pvx, self.pvy = px, py, pvx, pvy
            self._canvas = np.full((self.h, self.w, 3), [15.0, 6.0, 6.0], dtype=np.float32)

    # ------------------------------------------------------------------ cursor
    def _cursor(self):
        t = self.t
        cx = self.w / 2 + self.w * 0.35 * math.sin(t * 0.7)
        cy = self.h / 2 + self.h * 0.30 * math.sin(t * 1.1)
        return cx, cy

    # ------------------------------------------------------------------ step
    def step(self):
        sc    = self.scale
        MIN_D = 6.0 * sc
        dt    = 1.0
        cx, cy = self._cursor()
        drag = self.C3 ** dt

        if self._gpu:
            px, py, pvx, pvy = self.px, self.py, self.pvx, self.pvy
            # cursor attraction
            cdx = cx - px; cdy = cy - py
            cd  = torch.clamp(torch.hypot(cdx, cdy), min=MIN_D)
            cf  = self.C2 / (cd * cd)
            fx  = cdx / cd * cf
            fy  = cdy / cd * cf
            # particle–particle (N×N, upper triangle via broadcast)
            dx  = px[:, None] - px[None, :]
            dy  = py[:, None] - py[None, :]
            d   = torch.clamp(torch.hypot(dx, dy), min=MIN_D)
            d.fill_diagonal_(1e9)
            f   = self.C1 / (d * d)
            fx += (dx / d * f).sum(dim=1)
            fy += (dy / d * f).sum(dim=1)
            pvx = (pvx + fx * self.DT_S * dt) * drag
            pvy = (pvy + fy * self.DT_S * dt) * drag
            px  = px + pvx * dt
            py  = py + pvy * dt
            m   = 20.0 * sc
            px  = torch.where(px < -m, torch.full_like(px, self.w + m), px)
            px  = torch.where(px > self.w + m, torch.full_like(px, -m), px)
            py  = torch.where(py < -m, torch.full_like(py, self.h + m), py)
            py  = torch.where(py > self.h + m, torch.full_like(py, -m), py)
            self.px, self.py, self.pvx, self.pvy = px, py, pvx, pvy
        else:
            px, py, pvx, pvy = self.px, self.py, self.pvx, self.pvy
            cdx = cx - px; cdy = cy - py
            cd  = np.maximum(np.hypot(cdx, cdy), MIN_D)
            cf  = self.C2 / (cd * cd)
            fx  = cdx / cd * cf
            fy  = cdy / cd * cf
            dx  = px[:, None] - px[None, :]
            dy  = py[:, None] - py[None, :]
            d   = np.maximum(np.hypot(dx, dy), MIN_D)
            np.fill_diagonal(d, 1e9)
            f   = self.C1 / (d * d)
            fx += (dx / d * f).sum(axis=1)
            fy += (dy / d * f).sum(axis=1)
            pvx = (pvx + fx * self.DT_S * dt) * drag
            pvy = (pvy + fy * self.DT_S * dt) * drag
            px  = px + pvx * dt; py = py + pvy * dt
            m   = 20.0 * sc
            px  = np.where(px < -m, self.w + m, np.where(px > self.w + m, -m, px))
            py  = np.where(py < -m, self.h + m, np.where(py > self.h + m, -m, py))
            self.px, self.py = px.astype(np.float32), py.astype(np.float32)
            self.pvx, self.pvy = pvx.astype(np.float32), pvy.astype(np.float32)

        self.t += 0.05

    # ------------------------------------------------------------------ render
    def render(self):
        return self._render_gpu() if self._gpu else self._render_cpu()

    # ------ GPU path: scatter_add_ + Gaussian blur ------
    def _render_gpu(self):
        dev = self.dev
        sc  = self.scale
        spd = torch.hypot(self.pvx, self.pvy)

        # Colours (N,3) float in [0,1], RGB order
        rgb = _speed_rgb_t(spd, sc)  # R, G, B
        # Convert to BGR tensor channels for OpenCV-order output
        B_ch = rgb[:, 2]; G_ch = rgb[:, 1]; R_ch = rgb[:, 0]

        xi  = self.px.round().long().clamp(0, self.w - 1)
        yi  = self.py.round().long().clamp(0, self.h - 1)
        idx = yi * self.w + xi

        # glowR = 6 + spd_logical * 1.4  (HTML formula, scale back to logical)
        spd_logical = spd / sc
        glow_r_logical = 6.0 + spd_logical * 1.4   # 6-~20 logical px
        # Weight per particle for glow splat ~ area (glowR²) × 0.85 centre alpha
        # Gaussian sigma ≈ glow_r × 0.4 ≈ matches HTML gradient 40% point
        # We use a fixed sigma averaged over the cluster; gain tuned to match
        glow_w = glow_r_logical * glow_r_logical * 0.85

        flat0 = torch.zeros(self.h * self.w, device=dev)

        def splat(ch, w):
            f = flat0.clone()
            f.scatter_add_(0, idx, ch * w)
            return f.view(self.h, self.w)

        # Glow pass — sigma ≈ avg glow_r in render px × 0.4
        avg_glow_r = (glow_r_logical.mean().item()) * sc
        sig_glow   = max(1.0, avg_glow_r * 0.5)
        Bg = _blur(splat(B_ch, glow_w), sig_glow)
        Gg = _blur(splat(G_ch, glow_w), sig_glow)
        Rg = _blur(splat(R_ch, glow_w), sig_glow)

        # Core dot — radius 1.5 logical px, alpha = min(0.5+spd_l*0.08, 1)
        core_alpha = torch.clamp(0.5 + spd_logical * 0.08, 0.0, 1.0)
        core_w     = core_alpha * (1.5 * sc) ** 2
        sig_core   = max(0.5, 1.5 * sc * 0.5)
        Bc = _blur(splat(B_ch, core_w), sig_core)
        Gc = _blur(splat(G_ch, core_w), sig_core)
        Rc = _blur(splat(R_ch, core_w), sig_core)

        # Scale to 0-255.  Gain tuned so isolated particles match the HTML brightness.
        GLOW_GAIN = 28.0
        CORE_GAIN = 80.0
        B = (Bg * GLOW_GAIN + Bc * CORE_GAIN).clamp(0, 255)
        G = (Gg * GLOW_GAIN + Gc * CORE_GAIN).clamp(0, 255)
        R = (Rg * GLOW_GAIN + Rc * CORE_GAIN).clamp(0, 255)

        new_frame = torch.stack([B, G, R], dim=-1)   # H×W×3 BGR float

        # Fade canvas toward bg colour (HTML: globalAlpha=0.18 fill with #06060f)
        bg = torch.tensor([15.0, 6.0, 6.0], device=dev)
        self._canvas_t = self._canvas_t * 0.82 + bg * 0.18
        # Additive blend (lighter composite)
        self._canvas_t = torch.clamp(self._canvas_t + new_frame, 0.0, 255.0)

        return self._canvas_t.to(torch.uint8).contiguous().cpu().numpy()

    # ------ CPU path ------
    def _render_cpu(self):
        sc  = self.scale
        bg  = np.array([15.0, 6.0, 6.0])
        self._canvas = self._canvas * 0.82 + bg * 0.18

        pvx_np, pvy_np = self.pvx, self.pvy
        spd    = np.hypot(pvx_np, pvy_np)
        rgb    = _speed_rgb_np(spd, sc)    # (N,3) R,G,B in [0,1]
        spd_l  = spd / sc
        glow_r = (6.0 + spd_l * 1.4) * sc

        # Two temp layers: glow (blurred) and core (tight)
        glow_tmp = np.zeros((self.h, self.w, 3), dtype=np.float32)
        core_tmp = np.zeros_like(glow_tmp)

        xi = np.clip(self.px.astype(int), 0, self.w - 1)
        yi = np.clip(self.py.astype(int), 0, self.h - 1)

        for i in range(self.N):
            x_, y_ = int(xi[i]), int(yi[i])
            r      = int(glow_r[i])
            bgr    = (rgb[i, 2] * 255, rgb[i, 1] * 255, rgb[i, 0] * 255)
            # Outer glow (alpha 0.85 at centre)
            cv2.circle(glow_tmp, (x_, y_), max(1, r),
                       [c * 0.85 for c in bgr], -1, cv2.LINE_AA)
            # Core dot (radius 1.5 logical)
            a_core = min(0.5 + spd_l[i] * 0.08, 1.0)
            cv2.circle(core_tmp, (x_, y_), max(1, int(1.5 * sc)),
                       [c * a_core for c in bgr], -1, cv2.LINE_AA)

        ks = max(3, int(sc * 3)) | 1
        glow_blur = cv2.GaussianBlur(glow_tmp, (ks, ks), sc * 1.5)

        self._canvas = np.clip(self._canvas + glow_blur + core_tmp, 0, 255)
        return self._canvas.astype(np.uint8)
