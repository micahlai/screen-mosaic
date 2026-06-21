"""Fish boid simulation — faithful Python port of boids.html.

Drawing is a direct translation of drawFish() / drawShark() from the HTML:
same coordinate layout, same colour formula, same glow/shadow technique.
"""

import math
import time
import colorsys
import numpy as np
import cv2

from . import Visualization, register, boids_update, blend_roi
try:                       # `python -m mosiac` (package context)
    from .. import consts
except ImportError:        # `python mosiac` (mosiac dir on sys.path -> top-level)
    import consts


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _lw(lx, ly, x, y, ca, sa, sz, sc):
    """HTML local-space (lx,ly) → render-pixel (wx,wy).
    sz = per-fish size multiplier, sc = render scale."""
    return (int(x + (lx * ca - ly * sa) * sz * sc),
            int(y + (lx * sa + ly * ca) * sz * sc))


def _bezier(p0, p1, p2, n=10):
    """Quadratic Bézier points in local space, matching quadraticCurveTo."""
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        pts.append((u*u*p0[0] + 2*t*u*p1[0] + t*t*p2[0],
                    u*u*p0[1] + 2*t*u*p1[1] + t*t*p2[1]))
    return pts


def _hsl_bgr(h_deg, s_pct, l_pct):
    """HSL → OpenCV BGR uint8.  colorsys uses HLS order."""
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l_pct / 100.0, s_pct / 100.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Fish sprite  (matches drawFish() in boids.html)
# ---------------------------------------------------------------------------

def draw_fish(img, glow_layer, x, y, angle, hue, spd, sz, sc, max_speed):
    """Draw one fish onto img (detail) and glow_layer (strokes to be blurred)."""
    t_spd = min(spd / (max_speed * sc), 1.0)
    lit   = 52 + t_spd * 30
    col   = _hsl_bgr(hue, 75, lit)     # strokeStyle / solid colour
    fill  = _hsl_bgr(hue, 70, lit)     # fill colour (blended at 0.18)

    ca, sa = math.cos(angle), math.sin(angle)
    s = sz * sc   # combined scale in render pixels

    # body centre at local (1, 0)
    ecx = int(x + ca * s)
    ecy = int(y + sa * s)
    rx  = max(1, int(10 * s))
    ry  = max(1, int(5.5 * s))
    ang_deg = math.degrees(angle)

    # --- translucent body fill (globalAlpha=0.18), drawn in a sprite-sized ROI ---
    blend_roi(img, ecx, ecy, max(rx, ry) + 2,
              lambda m, ox, oy: cv2.ellipse(m, (ecx - ox, ecy - oy), (rx, ry),
                                            ang_deg, 0, 360, fill, -1, cv2.LINE_AA),
              0.18, 1.0)

    # --- stroked outline + tail + fin onto glow_layer (simulates shadowBlur=4) ---
    lw = max(1, int(1.2 * sc))
    cv2.ellipse(glow_layer, (ecx, ecy), (rx, ry), ang_deg, 0, 360, col, lw, cv2.LINE_AA)

    # forked tail: (-9,0)→(-17,-7) and (-9,0)→(-17,7)
    tail_root = _lw(-9,  0, x, y, ca, sa, sz, sc)
    cv2.line(glow_layer, tail_root, _lw(-17, -7, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_AA)
    cv2.line(glow_layer, tail_root, _lw(-17,  7, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_AA)

    # dorsal fin: (-2,-5.5)→(2,-9)→(5,-5.5)
    fin = np.array([_lw(-2, -5.5, x, y, ca, sa, sz, sc),
                    _lw( 2, -9,   x, y, ca, sa, sz, sc),
                    _lw( 5, -5.5, x, y, ca, sa, sz, sc)], dtype=np.int32)
    cv2.polylines(glow_layer, [fin], False, col, max(1, int(sc)), cv2.LINE_AA)

    # --- eye (drawn on img directly) ---
    ep = _lw(5.5, -1.6, x, y, ca, sa, sz, sc)
    er = max(1, int(1.5 * s))
    cv2.circle(img, ep, er, col, -1, cv2.LINE_AA)
    hp = _lw(6.0, -2.1, x, y, ca, sa, sz, sc)
    hr = max(1, int(0.55 * s))
    cv2.circle(img, hp, hr, (200, 200, 200), -1, cv2.LINE_AA)

    # --- smile bezier: (2.5,1)→(4.2,3.2)→(6.5,1) ---
    smile_local = _bezier((2.5, 1), (4.2, 3.2), (6.5, 1), n=8)
    smile_world = np.array([_lw(px_ * sz, py_ * sz, x, y, ca, sa, 1, sc)
                             for px_, py_ in smile_local], dtype=np.int32)
    cv2.polylines(img, [smile_world], False, col, max(1, int(sc)), cv2.LINE_AA)



# ---------------------------------------------------------------------------
# Background  (matches drawBackground() in boids.html)
# ---------------------------------------------------------------------------

def _make_ocean_bg(h, w):
    """Pre-render ocean background: dark-blue gradient + caustic grid."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    # Linear gradient #030f1c → #041525 → #020b16
    for y in range(h):
        t = y / h
        if t < 0.5:
            t2 = t / 0.5
            r = int(0x03 + (0x04 - 0x03) * t2)
            g = int(0x0f + (0x15 - 0x0f) * t2)
            b = int(0x1c + (0x25 - 0x1c) * t2)
        else:
            t2 = (t - 0.5) / 0.5
            r = int(0x04 + (0x02 - 0x04) * t2)
            g = int(0x15 + (0x0b - 0x15) * t2)
            b = int(0x25 + (0x16 - 0x25) * t2)
        bg[y] = (b, g, r)
    # Subtle caustic grid lines (rgba(40,100,160,0.04))
    grid_col = (int(160 * 0.04), int(100 * 0.04), int(40 * 0.04))
    step = max(1, int(60 * h / 1080))  # scale grid to render res
    for gx in range(0, w, step):
        cv2.line(bg, (gx, 0), (gx, h), grid_col, 1)
    for gy in range(0, h, step):
        cv2.line(bg, (0, gy), (w, gy), grid_col, 1)
    return bg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

@register("fish-boids", "Fish Boids")
class FishBoids(Visualization):

    viz_params = {
        "mode": {
            "label": "Mode",
            "options": [
                {"value": "normal", "label": "Normal"},
                {"value": "game",   "label": "Game"},
            ],
            "default": "normal",
        }
    }

    USES_HANDS          = True    # hand position drives the predator the fish flee
    HAND_TRACKER        = "red"   # use red-sticker CV tracking, not YOLO
    NEEDS_PHONE_CAMERA  = True    # phone must stream frames whenever this viz is active
    N            = 220
    VISUAL_RANGE = 40.0
    SEP_RANGE    = 30.0
    MAX_SPEED    = 8.5
    MIN_SPEED    = 1.8
    W_COH        = 0.0012
    W_ALI        = 0.018
    W_SEP        = 0.38
    W_WANDER     = 0.10
    FLEE_RANGE   = 190.0
    W_FLEE       = 18.5
    EAT_RADIUS   = 34.0
    EDGE_MARGIN  = 60.0          # steer away from walls within this (logical px)
    W_EDGE       = 0.6           # edge-avoidance strength

    def __init__(self, width, height):
        super().__init__(width, height)
        sc  = self.scale
        N   = self.N

        cols = round(math.sqrt(N * self.w / self.h))
        rows = math.ceil(N / cols)
        cw, ch = self.w / cols, self.h / rows

        rng = np.random.default_rng()
        bx  = np.zeros(N, dtype=np.float32)
        by  = np.zeros(N, dtype=np.float32)
        bvx = np.zeros(N, dtype=np.float32)
        bvy = np.zeros(N, dtype=np.float32)

        for i in range(N):
            bx[i] = (i % cols + 0.5 + (rng.random() - 0.5) * 0.85) * cw
            by[i] = (i // cols + 0.5 + (rng.random() - 0.5) * 0.85) * ch
            d = rng.random() * 2 * math.pi
            s = (self.MIN_SPEED + rng.random() * 1.4) * sc
            bvx[i] = math.cos(d) * s
            bvy[i] = math.sin(d) * s

        self.bx    = bx
        self.by    = by
        self.bvx   = bvx
        self.bvy   = bvy
        self.wander = rng.random(N).astype(np.float32) * 2 * math.pi
        # per-fish random hue (148-218) and size (0.78-1.22)
        self.hues  = (148 + rng.random(N) * 70).astype(np.float32)
        self.sizes = (0.78 + rng.random(N) * 0.44).astype(np.float32)
        self.alive = np.ones(N, dtype=bool)

        # the hand (YOLO tracker) is the predator the fish flee from
        self._ptr = None
        self.has_hand = False
        self.sx = -1e9; self.sy = -1e9          # predator position (off until a hand)

        self._game_state = "waiting"
        self._start_t    = 0.0
        self._elapsed    = 0.0
        self._eaten      = 0
        self._bursts     = []

        self._bg      = _make_ocean_bg(self.h, self.w)
        # motion-fade colour #020d18 in BGR
        self._fade_col = (0x18, 0x0d, 0x02)

    # ------------------------------------------------------------------ param
    def set_param(self, key, val):
        super().set_param(key, val)
        if key == "mode":
            self._reset_game()

    def set_pointer(self, ptr):
        """Hand force (nx, ny, nvx, nvy) in [0,1] field coords, or None."""
        self._ptr = ptr

    def _reset_game(self):
        N    = self.N; sc = self.scale
        self.alive[:] = True
        self._game_state = "waiting"; self._elapsed = 0.0; self._eaten = 0
        self._bursts.clear()
        rng  = np.random.default_rng()
        cols = round(math.sqrt(N * self.w / self.h))
        rows = math.ceil(N / cols)
        cw, ch = self.w / cols, self.h / rows
        for i in range(N):
            self.bx[i] = (i % cols + 0.5 + (rng.random()-0.5)*0.85) * cw
            self.by[i] = (i // cols + 0.5 + (rng.random()-0.5)*0.85) * ch

    # ------------------------------------------------------------------ step
    def step(self):
        sc = self.scale
        mode = self._params.get("mode", "normal")

        # the hand (YOLO tracker) is the predator; fish flee its field position
        if self._ptr is not None:
            self.sx = self._ptr[0] * self.w
            self.sy = self._ptr[1] * self.h
            self.has_hand = True
        else:
            self.has_hand = False
            self.sx = self.sy = -1e9          # no predator -> no flee / no eat
        sx, sy = self.sx, self.sy

        if mode == "game" and self._game_state == "waiting" and self.has_hand:
            self._game_state = "playing"; self._start_t = time.time()
        if mode == "game" and self._game_state == "playing":
            self._elapsed = time.time() - self._start_t

        # eat collision (the hand eats fish it touches)
        if mode == "game" and self._game_state == "playing" and self.has_hand:
            er2 = (self.EAT_RADIUS * sc) ** 2
            dxe = self.bx - sx; dye = self.by - sy
            eaten_now = self.alive & (dxe*dxe + dye*dye < er2)
            if eaten_now.any():
                for i in np.where(eaten_now)[0]:
                    self.alive[i] = False; self._eaten += 1
                    self._spawn_burst(self.bx[i], self.by[i], self.hues[i])
                if self._eaten >= self.N:
                    self._game_state = "done"

        # fish boids (vectorized; flee target = the hand)
        self.bx, self.by, self.bvx, self.bvy = boids_update(
            self.bx, self.by, self.bvx, self.bvy, self.wander, self.alive, sx, sy,
            visual_range=self.VISUAL_RANGE, sep_range=self.SEP_RANGE,
            flee_range=self.FLEE_RANGE, min_speed=self.MIN_SPEED,
            max_speed=self.MAX_SPEED, w_coh=self.W_COH, w_ali=self.W_ALI,
            w_sep=self.W_SEP, w_wander=self.W_WANDER, w_flee=self.W_FLEE,
            width=self.w, height=self.h, sc=sc,
            edge_margin=self.EDGE_MARGIN, w_edge=self.W_EDGE)

        # burst particles
        for b in self._bursts:
            b['x'] += b['vx']; b['y'] += b['vy']
            b['vx'] *= 0.91;   b['vy'] *= 0.91
            b['life'] -= 0.038
        self._bursts = [b for b in self._bursts if b['life'] > 0]

        self.t += 0.016

    def _spawn_burst(self, x, y, hue):
        sc = self.scale
        for _ in range(12):
            a = np.random.random() * 2 * math.pi
            s = (1.8 + np.random.random() * 3) * sc
            self._bursts.append({'x': x, 'y': y,
                                  'vx': math.cos(a)*s, 'vy': math.sin(a)*s,
                                  'life': 1.0, 'hue': hue})

    # ------------------------------------------------------------------ render
    def render(self):
        sc   = self.scale
        mode = self._params.get("mode", "normal")

        # Background
        img = self._bg.copy()

        # Motion fade: globalAlpha=0.18, fill #020d18
        fade = np.full_like(img, self._fade_col)
        cv2.addWeighted(fade, 0.18, img, 0.82, 0, img)

        # Glow layer for fish strokes (simulates shadowBlur=4)
        glow_layer = np.zeros_like(img)

        # Fish
        for i in range(self.N):
            if not self.alive[i]: continue
            draw_fish(img, glow_layer,
                      int(self.bx[i]), int(self.by[i]),
                      math.atan2(self.bvy[i], self.bvx[i]),
                      self.hues[i],
                      math.hypot(self.bvx[i], self.bvy[i]),
                      self.sizes[i], sc,
                      self.MAX_SPEED)

        # Blur glow and composite (matches shadowBlur=4 → sigma=4*sc)
        sig = max(1.0, 4.0 * sc)
        ks  = max(3, int(sig * 2)) | 1
        glow_blur = cv2.GaussianBlur(glow_layer, (ks, ks), sig)
        cv2.addWeighted(img, 1.0, glow_blur, 0.8, 0, img)

        # Burst particles
        for b in self._bursts:
            bx_, by_ = int(b['x']), int(b['y'])
            r_  = max(1, int(3.5 * b['life'] * sc))
            a_  = b['life'] ** 2
            col = _hsl_bgr(b['hue'], 80, 75)
            blend_roi(img, bx_, by_, r_ + 1,
                      lambda m, ox, oy: cv2.circle(m, (bx_ - ox, by_ - oy), r_, col, -1, cv2.LINE_AA),
                      a_, 1.0)

        # Hand marker (the "shark") — a translucent ring at the tracked position
        if self.has_hand and self._game_state != "done":
            self._draw_hand_circle(img, int(self.sx), int(self.sy), sc)

        # HUD
        if mode == "game":
            self._draw_hud(img, sc)

        return img

    def _draw_hand_circle(self, img, cx, cy, sc):
        """Translucent ring at the hand position (size from consts.FISH_HAND_MARKER_FRAC)."""
        r = max(3, int(consts.FISH_HAND_MARKER_FRAC * max(self.w, self.h)))
        blend_roi(img, cx, cy, r + max(2, r // 6) + 2,
                  lambda m, ox, oy: cv2.circle(m, (cx - ox, cy - oy), r,
                                               (235, 245, 255), max(2, r // 6), cv2.LINE_AA),
                  0.1, 1.0)

    def _draw_hud(self, img, sc):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        state = self._game_state
        W, H  = self.w, self.h

        def put(txt, x, y, size, col, bold=False):
            th = 2 if bold else 1
            cv2.putText(img, txt, (x, y), font, size * sc, col, max(1, int(th * sc)), cv2.LINE_AA)

        if state == "waiting":
            txt = "Move your hand - the fish flee it. Catch them all!"
            tw  = cv2.getTextSize(txt, font, 0.5 * sc, max(1, int(sc)))[0][0]
            put(txt, (W - tw) // 2, int(H * 0.94), 0.5, (200, 235, 255))
        elif state == "playing":
            elapsed = self._elapsed
            put(f"{elapsed:.2f}s", W // 2 - int(40 * sc), int(42 * sc), 0.7, (235, 245, 200), bold=True)
            remaining = int(self.alive.sum())
            put(f"{remaining} remaining", W - int(120 * sc), int(34 * sc), 0.4, (200, 215, 170))
        elif state == "done":
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (24, 13, 2), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
            t1 = "All fish eaten!"
            tw1 = cv2.getTextSize(t1, font, 1.2 * sc, max(2, int(sc * 2)))[0][0]
            cv2.putText(img, t1, ((W - tw1)//2, H//2 - int(50*sc)), font,
                        1.2 * sc, (0xe0, 0xc8, 0x8c), max(2, int(sc * 2)), cv2.LINE_AA)
            t2 = f"Time: {self._elapsed:.2f}s"
            tw2 = cv2.getTextSize(t2, font, 0.9 * sc, max(2, int(sc * 1.5)))[0][0]
            cv2.putText(img, t2, ((W - tw2)//2, H//2 + int(12*sc)), font,
                        0.9 * sc, (245, 235, 220), max(1, int(sc * 1.5)), cv2.LINE_AA)
            t3 = "Click to play again"
            tw3 = cv2.getTextSize(t3, font, 0.43 * sc, max(1, int(sc)))[0][0]
            cv2.putText(img, t3, ((W - tw3)//2, H//2 + int(58*sc)), font,
                        0.43 * sc, (200, 210, 160), max(1, int(sc)), cv2.LINE_AA)
