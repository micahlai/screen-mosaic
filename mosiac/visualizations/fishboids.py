"""Fish boid simulation with shark predator.

Boid rules: coherence, alignment, separation, flee from shark.
Shark tracks an auto-wandering cursor with turning physics.

Modes (selectable via phone UI dropdown):
  normal — shark chases, fish flee, no eating
  game   — shark eats fish on collision; timer overlay; end screen when all eaten
"""

import math
import time
import numpy as np
import cv2

from . import Visualization, register


# ---------------------------------------------------------------------------
# Sprite helpers
# ---------------------------------------------------------------------------

def _rot(pts, ca, sa, cx, cy):
    """Rotate a list of (fx, fy) body-space coords → int32 screen array."""
    out = []
    for fx, fy in pts:
        out.append((int(cx + fx * ca - fy * sa),
                    int(cy + fx * sa + fy * ca)))
    return np.array(out, dtype=np.int32)


def _fish_sprite(img, x, y, ang, bl, bw, body_col, accent_col, sc):
    """Draw a teardrop fish body + forked tail + dorsal fin + eye."""
    ca, sa = math.cos(ang), math.sin(ang)

    # --- body polygon (teardrop, body-space: +x = forward) ---
    body_pts = [
        ( 0.55,  0.00),   # nose
        ( 0.35,  0.50),
        ( 0.05,  0.95),
        (-0.20,  0.85),
        (-0.45,  0.55),
        (-0.60,  0.00),   # tail junction
        (-0.45, -0.55),
        (-0.20, -0.85),
        ( 0.05, -0.95),
        ( 0.35, -0.50),
    ]
    pts = _rot([(fx * bl, fy * bw) for fx, fy in body_pts], ca, sa, x, y)
    cv2.fillPoly(img, [pts], body_col, cv2.LINE_AA)

    # --- tail fork (two filled triangles) ---
    tbk = bl * 0.60   # how far back the tail base sits
    tl  = bl * 0.45   # tail tip length beyond base
    ts  = bw * 0.80   # spread of each fork tip
    tail_base = (-tbk, 0.0)
    for sign in (1, -1):
        tri = _rot([
            tail_base,
            (-tbk - tl,  sign * ts),
            (-tbk - tl * 0.35, sign * ts * 0.35),
        ], ca, sa, x, y)
        cv2.fillPoly(img, [tri], accent_col, cv2.LINE_AA)

    # --- dorsal fin ---
    fin_pts = _rot([
        ( 0.00,  bw),
        (-0.20,  bw),
        (-0.08,  bw + bl * 0.40),
    ], ca, sa, x, y)
    cv2.fillPoly(img, [fin_pts], accent_col, cv2.LINE_AA)

    # --- outline ---
    cv2.polylines(img, [pts], True, accent_col, max(1, int(sc * 0.4)), cv2.LINE_AA)

    # --- eye ---
    ex = int(x + ca * bl * 0.32 - sa * bw * 0.30)
    ey = int(y + sa * bl * 0.32 + ca * bw * 0.30)
    er = max(1, int(bw * 0.38))
    cv2.circle(img, (ex, ey), er,         (240, 240, 240), -1, cv2.LINE_AA)
    cv2.circle(img, (ex, ey), max(1, er - int(sc * 0.3)), (20, 20, 20), -1, cv2.LINE_AA)


def _shark_sprite(img, x, y, ang, bl, bw, sc):
    ca, sa = math.cos(ang), math.sin(ang)

    body_col   = (145, 135, 125)
    belly_col  = (190, 180, 165)
    dark_col   = ( 55,  50,  45)

    # --- countershaded body: draw belly first, then top half ---
    belly_pts = [
        ( 0.60,  0.00),
        ( 0.40, -0.25),
        ( 0.15, -0.75),
        (-0.15, -0.82),
        (-0.45, -0.60),
        (-0.68, -0.22),
        (-0.75,  0.00),
    ]
    top_pts = [
        ( 0.60,  0.00),
        ( 0.40,  0.28),
        ( 0.10,  0.82),
        (-0.15,  0.88),
        (-0.45,  0.65),
        (-0.68,  0.24),
        (-0.75,  0.00),
    ]
    belly  = _rot([(fx * bl, fy * bw) for fx, fy in belly_pts], ca, sa, x, y)
    top    = _rot([(fx * bl, fy * bw) for fx, fy in top_pts],   ca, sa, x, y)
    full   = _rot([(fx * bl, fy * bw) for fx, fy in top_pts + belly_pts[::-1]], ca, sa, x, y)
    cv2.fillPoly(img, [full],  body_col,  cv2.LINE_AA)
    cv2.fillPoly(img, [belly], belly_col, cv2.LINE_AA)

    # --- dorsal fin ---
    fin = _rot([
        ( 0.05,  bw),
        (-0.25,  bw),
        (-0.10,  bw + bl * 0.55),
    ], ca, sa, x, y)
    cv2.fillPoly(img, [fin], dark_col, cv2.LINE_AA)

    # --- pectoral fin ---
    pec = _rot([
        ( 0.10,  bw * 0.7),
        (-0.05,  bw * 0.7),
        (-0.10,  bw * 1.8),
        ( 0.05,  bw * 1.6),
    ], ca, sa, x, y)
    cv2.fillPoly(img, [pec], dark_col, cv2.LINE_AA)

    # --- crescent tail ---
    tbk = bl * 0.75
    tl  = bl * 0.50
    ts  = bw * 1.10
    for sign in (1, -1):
        tri = _rot([
            (-tbk, 0.0),
            (-tbk - tl,  sign * ts),
            (-tbk - tl * 0.40, sign * ts * 0.38),
        ], ca, sa, x, y)
        cv2.fillPoly(img, [tri], dark_col, cv2.LINE_AA)

    # --- outline ---
    cv2.polylines(img, [full], True, dark_col, max(1, int(sc * 0.5)), cv2.LINE_AA)

    # --- eye ---
    ex = int(x + ca * bl * 0.38 - sa * bw * 0.32)
    ey = int(y + sa * bl * 0.38 + ca * bw * 0.32)
    er = max(1, int(bw * 0.42))
    cv2.circle(img, (ex, ey), er,          (240, 240, 240), -1, cv2.LINE_AA)
    cv2.circle(img, (ex, ey), max(1, er-1), (5, 5, 5),      -1, cv2.LINE_AA)

    # --- menacing glow aura ---
    glow_img = np.zeros_like(img)
    cv2.fillPoly(glow_img, [full], (80, 70, 60), cv2.LINE_AA)
    ks = max(3, int(sc * 4)) | 1
    glow_img = cv2.GaussianBlur(glow_img, (ks, ks), sc * 2)
    cv2.addWeighted(img, 1.0, glow_img, 0.6, 0, img)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_OCEAN_TOP = np.array([40,  15,  8],  dtype=np.float32)   # dark navy (BGR)
_OCEAN_BOT = np.array([90,  55, 20],  dtype=np.float32)   # deep teal


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

    N           = 220
    VISUAL_RANGE = 75.0
    SEP_RANGE   = 22.0
    MAX_SPEED   = 6.5
    MIN_SPEED   = 1.8
    W_COH       = 0.004
    W_ALI       = 0.048
    W_SEP       = 0.12
    W_WANDER    = 0.038
    FLEE_RANGE  = 170.0
    W_FLEE      = 7.5
    EAT_RADIUS  = 34.0
    SHARK_ACCEL    = 2.2
    SHARK_MAX_SPD  = 28.0
    SHARK_DRAG     = 0.88
    SHARK_TURN     = 0.22

    def __init__(self, width, height):
        super().__init__(width, height)
        sc   = self.scale
        N    = self.N

        cols = round(math.sqrt(N * self.w / self.h))
        rows = math.ceil(N / cols)
        bx, by, bvx, bvy, bsz = [], [], [], [], []
        rng = np.random.default_rng()
        for i in range(N):
            cx_ = (i % cols + 0.5 + rng.uniform(-0.4, 0.4)) * (self.w / cols)
            cy_ = (i // cols + 0.5 + rng.uniform(-0.4, 0.4)) * (self.h / rows)
            bx.append(cx_); by.append(cy_)
            ang = rng.uniform(0, 2 * math.pi)
            sp  = rng.uniform(self.MIN_SPEED, self.MAX_SPEED) * sc
            bvx.append(math.cos(ang) * sp)
            bvy.append(math.sin(ang) * sp)
            bsz.append(rng.uniform(0.78, 1.22))

        self.bx  = np.array(bx,  dtype=np.float32)
        self.by  = np.array(by,  dtype=np.float32)
        self.bvx = np.array(bvx, dtype=np.float32)
        self.bvy = np.array(bvy, dtype=np.float32)
        self.bsz = np.array(bsz, dtype=np.float32)
        self.alive = np.ones(N, dtype=bool)

        self.sx     = float(self.w) / 2
        self.sy     = float(self.h) / 2
        self.svx    = 0.0; self.svy = 0.0
        self.sangle = 0.0
        self._tw    = 0.0

        self._mode       = "normal"
        self._game_state = "waiting"
        self._start_t    = 0.0
        self._elapsed    = 0.0
        self._eaten      = 0

        self._bg = self._make_bg()

    # ------------------------------------------------------------------ param
    def set_param(self, key, val):
        super().set_param(key, val)
        if key == "mode":
            self._mode = val
            self._reset_game()

    def _reset_game(self):
        N    = self.N
        self.alive = np.ones(N, dtype=bool)
        self._game_state = "waiting"
        self._elapsed    = 0.0
        self._eaten      = 0
        rng  = np.random.default_rng()
        cols = round(math.sqrt(N * self.w / self.h))
        for i in range(N):
            self.bx[i] = (i % cols + 0.5 + rng.uniform(-0.4, 0.4)) * (self.w / cols)
            self.by[i] = (i // cols + 0.5 + rng.uniform(-0.4, 0.4)) * (self.h / cols)

    def _make_bg(self):
        ys  = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
        rgb = (_OCEAN_TOP[None, :] * (1 - ys) + _OCEAN_BOT[None, :] * ys).astype(np.uint8)
        return np.broadcast_to(rgb[:, None, :], (self.h, self.w, 3)).copy()

    # ------------------------------------------------------------------ cursor
    def _cursor(self):
        t  = self._tw
        cx = self.w / 2 + self.w * 0.38 * math.sin(t * 0.53)
        cy = self.h / 2 + self.h * 0.32 * math.sin(t * 0.79)
        return cx, cy

    # ------------------------------------------------------------------ step
    def step(self):
        sc   = self.scale
        self._tw += 0.04
        cx, cy = self._cursor()

        mode = self._params.get("mode", self._mode)
        if mode == "game":
            if self._game_state == "waiting":
                self._game_state = "playing"
                self._start_t = time.time()
            elif self._game_state == "playing":
                self._elapsed = time.time() - self._start_t

        alive = self.alive
        bx, by = self.bx, self.by
        bvx, bvy = self.bvx, self.bvy

        VR2  = (self.VISUAL_RANGE * sc) ** 2
        SR2  = (self.SEP_RANGE    * sc) ** 2
        FR2  = (self.FLEE_RANGE   * sc) ** 2
        ER2  = (self.EAT_RADIUS   * sc) ** 2
        maxV = self.MAX_SPEED * sc
        minV = self.MIN_SPEED * sc

        # shark update
        sx, sy = self.sx, self.sy
        target_ang = math.atan2(cy - sy, cx - sx)
        diff = target_ang - self.sangle
        while diff >  math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        self.sangle += math.copysign(min(abs(diff), self.SHARK_TURN * sc / 8), diff)
        dist     = math.hypot(cx - sx, cy - sy)
        throttle = min(dist / (20 * sc), 1.0)
        self.svx += math.cos(self.sangle) * self.SHARK_ACCEL * sc / 8 * throttle
        self.svy += math.sin(self.sangle) * self.SHARK_ACCEL * sc / 8 * throttle
        spd = math.hypot(self.svx, self.svy)
        if spd > self.SHARK_MAX_SPD * sc / 8:
            f = self.SHARK_MAX_SPD * sc / 8 / spd
            self.svx *= f; self.svy *= f
        self.svx *= self.SHARK_DRAG; self.svy *= self.SHARK_DRAG
        self.sx = max(0, min(self.w, sx + self.svx))
        self.sy = max(0, min(self.h, sy + self.svy))
        sx, sy  = self.sx, self.sy

        fx = np.zeros(self.N, dtype=np.float32)
        fy = np.zeros(self.N, dtype=np.float32)

        for i in range(self.N):
            if not alive[i]:
                continue
            dx = bx - bx[i]; dy = by - by[i]
            d2 = dx * dx + dy * dy
            mask = alive & (d2 > 0) & (d2 < VR2)
            if mask.any():
                fx[i] += (bx[mask].mean() - bx[i]) * self.W_COH
                fy[i] += (by[mask].mean() - by[i]) * self.W_COH
                fx[i] += bvx[mask].mean() * self.W_ALI
                fy[i] += bvy[mask].mean() * self.W_ALI
            smask = alive & (d2 > 0) & (d2 < SR2)
            if smask.any():
                d_s = np.sqrt(d2[smask])
                fx[i] -= (dx[smask] / d_s).sum() * self.W_SEP
                fy[i] -= (dy[smask] / d_s).sum() * self.W_SEP
            sdx = bx[i] - sx; sdy = by[i] - sy
            sd2 = sdx * sdx + sdy * sdy
            if sd2 < FR2 and sd2 > 0:
                sd  = math.sqrt(sd2)
                p   = self.W_FLEE * ((1 - sd / (self.FLEE_RANGE * sc)) ** 1.5) / sd
                fx[i] += sdx * p; fy[i] += sdy * p
            fx[i] += np.random.uniform(-1, 1) * self.W_WANDER * sc
            fy[i] += np.random.uniform(-1, 1) * self.W_WANDER * sc

        bvx += fx; bvy += fy
        spd_arr = np.hypot(bvx, bvy)
        tf = alive & (spd_arr > maxV)
        ts = alive & (spd_arr < minV) & (spd_arr > 0)
        bvx[tf] = bvx[tf] / spd_arr[tf] * maxV
        bvy[tf] = bvy[tf] / spd_arr[tf] * maxV
        bvx[ts] = bvx[ts] / spd_arr[ts] * minV
        bvy[ts] = bvy[ts] / spd_arr[ts] * minV
        bx += bvx; by += bvy
        bx[bx < 0] += self.w; bx[bx > self.w] -= self.w
        by[by < 0] += self.h; by[by > self.h] -= self.h
        self.bx, self.by, self.bvx, self.bvy = bx, by, bvx, bvy

        if mode == "game" and self._game_state == "playing":
            dx_s = bx - sx; dy_s = by - sy
            caught = alive & (dx_s * dx_s + dy_s * dy_s < ER2)
            if caught.any():
                self.alive[caught] = False
                self._eaten += int(caught.sum())
            if self._eaten >= self.N:
                self._game_state = "done"

        self.t += 0.016

    # ------------------------------------------------------------------ render
    def render(self):
        sc    = self.scale
        img   = self._bg.copy()
        mode  = self._params.get("mode", self._mode)
        alive = self.alive

        bl = 14.0 * sc   # fish body half-length
        bw =  6.0 * sc   # fish body half-width

        spd_arr = np.hypot(self.bvx, self.bvy)
        maxV    = self.MAX_SPEED * sc

        for i in range(self.N):
            if not alive[i]:
                continue
            x, y  = int(self.bx[i]), int(self.by[i])
            ang   = math.atan2(self.bvy[i], self.bvx[i])
            sz    = self.bsz[i]
            # velocity-tinted colour: calm=teal-blue, fleeing=warm gold
            t_spd = min(spd_arr[i] / maxV, 1.0)
            body_col = (
                int(130 - t_spd * 60),   # B
                int(165 - t_spd * 20),   # G
                int( 70 + t_spd * 140),  # R
            )
            accent_col = (
                int( 60 - t_spd * 20),
                int( 90 - t_spd * 30),
                int( 35 + t_spd * 80),
            )
            _fish_sprite(img, x, y, ang,
                         bl * sz, bw * sz,
                         body_col, accent_col, sc)

        # shark
        sbl = 30.0 * sc
        sbw = 10.0 * sc
        _shark_sprite(img, int(self.sx), int(self.sy), self.sangle, sbl, sbw, sc)

        if mode == "game":
            self._draw_hud(img, sc)
        return img

    def _draw_hud(self, img, sc):
        font  = cv2.FONT_HERSHEY_DUPLEX
        fscl  = sc * 0.55
        thick = max(1, int(sc * 0.8))
        state = self._game_state

        def shadow_text(txt, pos, scale, col, th):
            cv2.putText(img, txt, (pos[0]+max(1,int(sc*0.15)), pos[1]+max(1,int(sc*0.15))),
                        font, scale, (0,0,0), th+1, cv2.LINE_AA)
            cv2.putText(img, txt, pos, font, scale, col, th, cv2.LINE_AA)

        if state == "waiting":
            txt = "Move shark to start!"
            tw, th = cv2.getTextSize(txt, font, fscl, thick)[0]
            shadow_text(txt, ((self.w-tw)//2, (self.h+th)//2), fscl, (220,220,100), thick)
        elif state == "playing":
            shadow_text(f"{self._elapsed:.1f}s",
                        (int(20*sc), int(50*sc)), fscl, (220,220,220), thick)
            shadow_text(f"Fish: {int(self.alive.sum())}/{self.N}",
                        (int(20*sc), int(100*sc)), fscl, (220,220,220), thick)
        elif state == "done":
            for j, (txt, col) in enumerate([
                ("All eaten!",          (80, 220, 255)),
                (f"Time: {self._elapsed:.2f}s", (200, 220, 255)),
            ]):
                tw, th = cv2.getTextSize(txt, font, fscl*1.4, thick)[0]
                y = self.h//2 - int(fscl*40) + j*int(fscl*75)
                shadow_text(txt, ((self.w-tw)//2, y), fscl*1.4, col, thick+1)
