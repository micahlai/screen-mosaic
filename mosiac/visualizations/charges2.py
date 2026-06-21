"""Magnetic charge particle simulation (copy 2 — edit freely, independent of charges/charges1).

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
try:                       # `python -m mosiac` (package context)
    from .. import consts
except ImportError:        # `python mosiac` (mosiac dir on sys.path -> top-level)
    import consts


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


@register("charges2", "Charges 2")
class ChargesSim2(Visualization):
    USES_HANDS         = True    # hand position drives the cursor the particles chase
    HAND_TRACKER       = "red"   # red-sticker CV tracking, same as the boids
    NEEDS_PHONE_CAMERA = True    # phone must stream frames whenever this viz is active

    C1   = 12000    # inter-particle constant (negative = attract)
    C2   = -150000   # cursor attraction constant
    C3   = 0.98    # drag per frame
    DT_S = 1 / 120  # physics timestep

    def __init__(self, width, height):
        super().__init__(width, height)
        sc = self.scale
        self._gpu = torch is not None
        self.N = 3000 if self._gpu else 400

        # hand cursor (red tracker). Falls back to auto-motion when no hand is seen.
        self._ptr = None
        self.has_hand = False
        self._show_ring = True                  # gray cursor ring (toggled from phone UI)
        self._cx = self.w / 2.0
        self._cy = self.h / 2.0

        N = self.N
        # spawn uniformly across the whole allotted area (was a ring around centre)
        px  = np.random.uniform(0, self.w, N).astype(np.float32)
        py  = np.random.uniform(0, self.h, N).astype(np.float32)
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

    # ------------------------------------------------------------------ param
    def set_param(self, key, val):
        super().set_param(key, val)
        if key == "ring":            # gray cursor ring on/off (phone toggle)
            self._show_ring = (val is True or
                               str(val).lower() in ("true", "1", "on", "yes"))

    def set_pointer(self, ptr):
        """Hand force (nx, ny, nvx, nvy) in [0,1] field coords, or None."""
        self._ptr = ptr
        self.has_hand = ptr is not None

    # ------------------------------------------------------------------ cursor
    def _cursor(self):
        # hand drives the cursor; without a hand, gentle auto-motion keeps it alive
        if self.has_hand and self._ptr is not None:
            return self._ptr[0] * self.w, self._ptr[1] * self.h
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
        self._cx, self._cy = cx, cy        # remembered for the cursor ring overlay
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
        img = self._render_gpu() if self._gpu else self._render_cpu()
        if self.has_hand and self._show_ring:
            self._draw_cursor_ring(img, int(self._cx), int(self._cy))
        return img

    def _draw_cursor_ring(self, img, cx, cy):
        """Translucent gray ring at the hand position (size from FISH_HAND_MARKER_FRAC),
        blended in a sprite-sized ROI so it costs almost nothing."""
        r   = max(3, int(consts.FISH_HAND_MARKER_FRAC * max(self.w, self.h)))
        th  = max(2, r // 6)
        pad = r + th + 2
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1, y1 = min(self.w, cx + pad), min(self.h, cy + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        ov  = roi.copy()
        cv2.circle(ov, (cx - x0, cy - y0), r, (235, 245, 255), th, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.35, roi, 0.65, 0, roi)

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
