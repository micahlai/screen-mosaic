"""Bird boid simulation with hawk predator in a sky scene.

Identical boid physics to fish-boids; visual theme swapped to birds on a sky
gradient with wispy clouds. Birds have animated flapping wings. Predator is
a larger hawk silhouette.

Modes:
  normal — hawk circles, birds flock and flee
  game   — hawk catches birds; timer; done screen
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
    out = []
    for fx, fy in pts:
        out.append((int(cx + fx * ca - fy * sa),
                    int(cy + fx * sa + fy * ca)))
    return np.array(out, dtype=np.int32)


def _bird_sprite(img, x, y, ang, wing_len, flap, color, thick):
    """Animated V-wing bird silhouette.
    flap: -1..1, drives the dihedral angle of the wings.
    """
    ca, sa = math.cos(ang), math.sin(ang)
    # each wing sweeps back ~120° from heading, with flap modifying the droop
    base_sweep = math.pi * 0.62
    droop      = flap * math.pi * 0.18   # positive = wings down

    # Draw two curved wings as 3-point polylines (shoulder → elbow → tip)
    for sign in (1, -1):
        sweep = base_sweep + droop * sign
        # shoulder: close to body
        shx = int(x + ca * wing_len * 0.10 - sa * wing_len * 0.10 * sign)
        shy = int(y + sa * wing_len * 0.10 + ca * wing_len * 0.10 * sign)
        # elbow: halfway along wing, slightly kinked
        el_ang = ang + math.pi + sign * sweep * 0.55
        elx = int(x + math.cos(el_ang) * wing_len * 0.52)
        ely = int(y + math.sin(el_ang) * wing_len * 0.52)
        # tip
        tip_ang = ang + math.pi + sign * sweep
        tx  = int(x + math.cos(tip_ang) * wing_len)
        ty  = int(y + math.sin(tip_ang) * wing_len)
        pts = np.array([(shx, shy), (elx, ely), (tx, ty)], dtype=np.int32)
        cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)

    # tiny body oval
    body_len = max(1, int(wing_len * 0.18))
    body_w   = max(1, int(wing_len * 0.10))
    cv2.ellipse(img, (x, y), (body_len, body_w),
                math.degrees(ang), 0, 360, color, -1, cv2.LINE_AA)


def _hawk_sprite(img, x, y, ang, wing_len, flap, sc):
    """Larger, darker hawk with more swept-back wings and a fan tail."""
    ca, sa = math.cos(ang), math.sin(ang)

    dark   = (25,  25,  70)
    mid    = (55,  50, 110)
    shadow = (10,  10,  35)

    # glow aura first
    glow_img = np.zeros_like(img)
    for sign in (1, -1):
        sweep   = math.pi * 0.58 + flap * math.pi * 0.15 * sign
        tip_ang = ang + math.pi + sign * sweep
        tx  = int(x + math.cos(tip_ang) * wing_len)
        ty  = int(y + math.sin(tip_ang) * wing_len)
        cv2.line(glow_img, (x, y), (tx, ty), (100, 80, 160), max(3, int(sc * 3)), cv2.LINE_AA)
    ks = max(3, int(sc * 5)) | 1
    glow_img = cv2.GaussianBlur(glow_img, (ks, ks), sc * 2.5)
    cv2.addWeighted(img, 1.0, glow_img, 0.55, 0, img)

    # wings: filled wedge silhouettes
    for sign in (1, -1):
        sweep    = math.pi * 0.58 + flap * math.pi * 0.15 * sign
        tip_ang  = ang + math.pi + sign * sweep
        el_ang   = ang + math.pi + sign * sweep * 0.52
        tx  = int(x + math.cos(tip_ang) * wing_len)
        ty  = int(y + math.sin(tip_ang) * wing_len)
        elx = int(x + math.cos(el_ang) * wing_len * 0.50)
        ely = int(y + math.sin(el_ang) * wing_len * 0.50)
        # inner wing
        inner_ang = ang + math.pi + sign * sweep * 0.22
        ix  = int(x + math.cos(inner_ang) * wing_len * 0.28)
        iy  = int(y + math.sin(inner_ang) * wing_len * 0.28)
        wing_poly = np.array([(x, y), (ix, iy), (elx, ely), (tx, ty)], dtype=np.int32)
        cv2.fillPoly(img, [wing_poly], mid, cv2.LINE_AA)
        cv2.polylines(img, [wing_poly], False, shadow, max(1, int(sc * 0.6)), cv2.LINE_AA)

    # body
    body_pts = _rot([
        ( 0.28,  0.00),
        ( 0.14,  0.40),
        (-0.20,  0.45),
        (-0.50,  0.00),
        (-0.20, -0.45),
        ( 0.14, -0.40),
    ], ca, sa, x, y)
    body_len = wing_len * 0.30
    cv2.fillPoly(img, [_rot([(fx*body_len, fy*body_len)
                             for fx, fy in [( 0.28,0),( 0.14,0.40),(-0.20,0.45),
                                             (-0.50,0),(-0.20,-0.45),( 0.14,-0.40)]],
                             ca, sa, x, y)],
                 dark, cv2.LINE_AA)

    # fan tail
    tail_base_x = int(x - ca * body_len * 0.50)
    tail_base_y = int(y - sa * body_len * 0.50)
    for i, frac in enumerate([-0.6, -0.3, 0.0, 0.3, 0.6]):
        tail_ang = ang + math.pi + frac * math.pi * 0.25
        tl = body_len * 0.6
        tip = (int(tail_base_x + math.cos(tail_ang)*tl),
               int(tail_base_y + math.sin(tail_ang)*tl))
        cv2.line(img, (tail_base_x, tail_base_y), tip,
                 shadow if abs(frac) < 0.4 else dark,
                 max(1, int(sc * 0.7)), cv2.LINE_AA)

    # eye
    ex = int(x + ca * body_len * 0.22 - sa * body_len * 0.18)
    ey = int(y + sa * body_len * 0.22 + ca * body_len * 0.18)
    er = max(1, int(body_len * 0.18))
    cv2.circle(img, (ex, ey), er,         (230, 230, 230), -1)
    cv2.circle(img, (ex, ey), max(1,er-1), (0, 0, 0),       -1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_SKY_TOP = np.array([200, 140,  80], dtype=np.float32)   # deep azure (BGR)
_SKY_BOT = np.array([255, 220, 170], dtype=np.float32)   # pale horizon


@register("bird-boids", "Bird Boids")
class BirdBoids(Visualization):

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
    HAWK_ACCEL  = 2.2
    HAWK_MAX_SPD = 28.0
    HAWK_DRAG   = 0.88
    HAWK_TURN   = 0.22

    def __init__(self, width, height):
        super().__init__(width, height)
        sc   = self.scale
        N    = self.N

        cols = round(math.sqrt(N * self.w / self.h))
        rows = math.ceil(N / cols)
        bx, by, bvx, bvy, bsz, bphase = [], [], [], [], [], []
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
            bphase.append(rng.uniform(0, 2 * math.pi))   # individual flap offset

        self.bx     = np.array(bx,     dtype=np.float32)
        self.by     = np.array(by,     dtype=np.float32)
        self.bvx    = np.array(bvx,    dtype=np.float32)
        self.bvy    = np.array(bvy,    dtype=np.float32)
        self.bsz    = np.array(bsz,    dtype=np.float32)
        self.bphase = np.array(bphase, dtype=np.float32)
        self.alive  = np.ones(N, dtype=bool)

        self.hx      = float(self.w) / 2
        self.hy      = float(self.h) / 2
        self.hvx     = 0.0; self.hvy = 0.0
        self.hangle  = 0.0
        self.hflap_t = 0.0
        self._tw     = 0.0

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
        N = self.N
        self.alive = np.ones(N, dtype=bool)
        self._game_state = "waiting"
        self._elapsed = 0.0; self._eaten = 0
        rng  = np.random.default_rng()
        cols = round(math.sqrt(N * self.w / self.h))
        for i in range(N):
            self.bx[i] = (i % cols + 0.5 + rng.uniform(-0.4, 0.4)) * (self.w / cols)
            self.by[i] = (i // cols + 0.5 + rng.uniform(-0.4, 0.4)) * (self.h / cols)

    def _make_bg(self):
        ys  = np.linspace(0, 1, self.h, dtype=np.float32)[:, None]
        rgb = (_SKY_TOP[None, :] * (1 - ys) + _SKY_BOT[None, :] * ys).astype(np.uint8)
        bg  = np.broadcast_to(rgb[:, None, :], (self.h, self.w, 3)).copy()
        # wispy cloud layers
        rng = np.random.default_rng(42)
        for _ in range(14):
            cx_  = rng.integers(0, self.w)
            cy_  = rng.integers(int(self.h * 0.03), int(self.h * 0.50))
            rw   = rng.integers(int(self.w * 0.05), int(self.w * 0.16))
            rh   = max(1, int(rw * 0.16))
            alpha = rng.uniform(0.10, 0.25)
            cloud = bg.copy()
            cv2.ellipse(cloud, (cx_, cy_), (rw, rh), 0, 0, 360,
                        (255, 255, 255), -1, cv2.LINE_AA)
            ks = max(3, int(rw * 0.30)) | 1
            cloud = cv2.GaussianBlur(cloud, (ks, ks), rw * 0.15)
            cv2.addWeighted(cloud, alpha, bg, 1 - alpha, 0, bg)
        return bg

    # ------------------------------------------------------------------ cursor
    def _cursor(self):
        t  = self._tw
        cx = self.w / 2 + self.w * 0.38 * math.sin(t * 0.53)
        cy = self.h / 2 + self.h * 0.32 * math.sin(t * 0.79)
        return cx, cy

    # ------------------------------------------------------------------ step
    def step(self):
        sc = self.scale
        self._tw    += 0.04
        self.hflap_t += 0.18   # hawk flap speed
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

        hx, hy = self.hx, self.hy
        target_ang = math.atan2(cy - hy, cx - hx)
        diff = target_ang - self.hangle
        while diff >  math.pi: diff -= 2 * math.pi
        while diff < -math.pi: diff += 2 * math.pi
        self.hangle += math.copysign(min(abs(diff), self.HAWK_TURN * sc / 8), diff)
        dist     = math.hypot(cx - hx, cy - hy)
        throttle = min(dist / (20 * sc), 1.0)
        self.hvx += math.cos(self.hangle) * self.HAWK_ACCEL * sc / 8 * throttle
        self.hvy += math.sin(self.hangle) * self.HAWK_ACCEL * sc / 8 * throttle
        spd = math.hypot(self.hvx, self.hvy)
        if spd > self.HAWK_MAX_SPD * sc / 8:
            f = self.HAWK_MAX_SPD * sc / 8 / spd
            self.hvx *= f; self.hvy *= f
        self.hvx *= self.HAWK_DRAG; self.hvy *= self.HAWK_DRAG
        self.hx  = max(0, min(self.w, hx + self.hvx))
        self.hy  = max(0, min(self.h, hy + self.hvy))
        hx, hy   = self.hx, self.hy

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
            sdx = bx[i] - hx; sdy = by[i] - hy
            sd2 = sdx * sdx + sdy * sdy
            if sd2 < FR2 and sd2 > 0:
                sd = math.sqrt(sd2)
                p  = self.W_FLEE * ((1 - sd / (self.FLEE_RANGE * sc)) ** 1.5) / sd
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
            dx_h = bx - hx; dy_h = by - hy
            caught = alive & (dx_h * dx_h + dy_h * dy_h < ER2)
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

        spd_arr = np.hypot(self.bvx, self.bvy)
        maxV    = self.MAX_SPEED * sc
        wing    = 16.0 * sc
        thick   = max(1, int(sc * 1.3))

        for i in range(self.N):
            if not alive[i]:
                continue
            x, y  = int(self.bx[i]), int(self.by[i])
            ang   = math.atan2(self.bvy[i], self.bvx[i])
            sz    = self.bsz[i]
            spd_i = spd_arr[i]
            # flap faster when fleeing
            flap_freq = 5.0 + min(spd_i / maxV, 1.0) * 4.0
            flap = math.sin(self.t * flap_freq + self.bphase[i])
            # colour: dark slate normally, blue-tinted when calm
            t_spd = min(spd_i / maxV, 1.0)
            bval  = int(30 + t_spd * 20)
            color = (bval + 10, bval, bval - 5)
            _bird_sprite(img, x, y, ang,
                         wing * sz, flap, color,
                         max(1, int(thick * sz)))

        # hawk
        h_flap = math.sin(self.hflap_t)
        _hawk_sprite(img, int(self.hx), int(self.hy),
                     self.hangle, wing * 2.2, h_flap, sc)

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
                        font, scale, (200,200,200), th+1, cv2.LINE_AA)
            cv2.putText(img, txt, pos, font, scale, col, th, cv2.LINE_AA)

        if state == "waiting":
            txt = "Hawk hunts — ready!"
            tw, th = cv2.getTextSize(txt, font, fscl, thick)[0]
            shadow_text(txt, ((self.w-tw)//2, (self.h+th)//2), fscl, (40,40,40), thick)
        elif state == "playing":
            shadow_text(f"{self._elapsed:.1f}s",
                        (int(20*sc), int(50*sc)), fscl, (40,40,40), thick)
            shadow_text(f"Birds: {int(self.alive.sum())}/{self.N}",
                        (int(20*sc), int(100*sc)), fscl, (40,40,40), thick)
        elif state == "done":
            for j, (txt, col) in enumerate([
                ("All caught!",         (40, 40, 160)),
                (f"Time: {self._elapsed:.2f}s", (60, 60, 180)),
            ]):
                tw, th = cv2.getTextSize(txt, font, fscl*1.4, thick)[0]
                y = self.h//2 - int(fscl*40) + j*int(fscl*75)
                shadow_text(txt, ((self.w-tw)//2, y), fscl*1.4, col, thick+1)
