"""Bird boid simulation — same drawing style as fishboids.html.

Birds use the same ellipse-body + stroke-glow approach as the fish, adapted:
  body: slimmer ellipse, swept-back wings, tail fan, beak spike
  hue range: 35-85 (warm amber → yellow-green, earthy flock colours)
  hawk: same structure as shark but with wide wings and hooked beak
Background: sky gradient + cloud wisps
"""

import math
import time
import colorsys
import numpy as np
import cv2

from . import Visualization, register, boids_update, blend_roi, blur_down


# ---------------------------------------------------------------------------
# Helpers shared with fishboids
# ---------------------------------------------------------------------------

def _lw(lx, ly, x, y, ca, sa, sz, sc):
    return (int(x + (lx * ca - ly * sa) * sz * sc),
            int(y + (lx * sa + ly * ca) * sz * sc))


def _bezier(p0, p1, p2, n=10):
    pts = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        pts.append((u*u*p0[0]+2*t*u*p1[0]+t*t*p2[0],
                    u*u*p0[1]+2*t*u*p1[1]+t*t*p2[1]))
    return pts


def _hsl_bgr(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Bird sprite  (same structure as drawFish; adapted anatomy)
# ---------------------------------------------------------------------------
#
# Local coords (fish-style: +x = forward, +y = down):
#   body:  ellipse(0.5, 0, 9, 3.5)          — slender bird body
#   wing:  lines (-1,0)→(-11,-9), (-1,0)→(-11,9)   swept-back wings
#   tail:  fan of 3 short lines from (-9, 0)
#   beak:  (9,0)→(14,0)                      spike at the front
#   eye:   arc(5.5,-1.0, r=1.2)

def draw_bird(img, glow_layer, x, y, angle, hue, spd, sz, sc, max_speed, flap):
    """flap ∈ [-1,1]: wing droop offset for animation."""
    t_spd = min(spd / (max_speed * sc), 1.0)
    lit   = 52 + t_spd * 25
    col   = _hsl_bgr(hue, 72, lit)
    fill  = _hsl_bgr(hue, 65, lit)

    ca, sa = math.cos(angle), math.sin(angle)
    s = sz * sc

    # body
    ecx = int(x + ca * 0.5 * s); ecy = int(y + sa * 0.5 * s)
    rx  = max(1, int(9 * s));    ry  = max(1, int(3.5 * s))
    ang_deg = math.degrees(angle)
    lw  = max(1, int(1.2 * sc))

    blend_roi(img, ecx, ecy, max(rx, ry) + 2,
              lambda m, ox, oy: cv2.ellipse(m, (ecx - ox, ecy - oy), (rx, ry),
                                            ang_deg, 0, 360, fill, -1, cv2.LINE_8),
              0.20, 1.0)
    cv2.ellipse(glow_layer, (ecx, ecy), (rx, ry), ang_deg, 0, 360, col, lw, cv2.LINE_8)

    # swept-back wings — droop modulated by flap
    wing_y = 9 + flap * 3.5   # ±3.5 px droop
    wing_root = _lw(-1,  0,     x, y, ca, sa, sz, sc)
    cv2.line(glow_layer, wing_root, _lw(-11, -wing_y, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_8)
    cv2.line(glow_layer, wing_root, _lw(-11,  wing_y, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_8)
    # secondary feather lines for thickness
    cv2.line(glow_layer, wing_root, _lw(-7, -wing_y*0.6, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_8)
    cv2.line(glow_layer, wing_root, _lw(-7,  wing_y*0.6, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_8)

    # tail fan (three lines)
    tail_base = _lw(-9, 0, x, y, ca, sa, sz, sc)
    for ty in (-4, 0, 4):
        cv2.line(glow_layer, tail_base, _lw(-14, ty, x, y, ca, sa, sz, sc), col, lw, cv2.LINE_8)

    # beak spike
    beak_base = _lw(9, 0, x, y, ca, sa, sz, sc)
    beak_tip  = _lw(14, 0.5, x, y, ca, sa, sz, sc)
    cv2.line(glow_layer, beak_base, beak_tip, col, max(1, int(sc)), cv2.LINE_8)

    # eye
    ep = _lw(5.5, -1.0, x, y, ca, sa, sz, sc)
    er = max(1, int(1.2 * s))
    cv2.circle(img, ep, er, col, -1, cv2.LINE_8)
    hp = _lw(6.0, -1.4, x, y, ca, sa, sz, sc)
    cv2.circle(img, hp, max(1, int(0.45 * s)), (210, 210, 210), -1, cv2.LINE_8)


# ---------------------------------------------------------------------------
# Hawk sprite  (same structure as drawShark; bird-of-prey adapted)
# ---------------------------------------------------------------------------
#
# Local coords:
#   body:   ellipse(0, 0, 22, 8)
#   wings:  broad swept triangle each side
#   tail:   fan of 5 lines from (-22, 0)
#   beak:   hooked (15,0)→(22,3)→(20,5)
#   eye:    arc(11,-2, r=3)

def draw_hawk(img, x, y, angle, sc):
    ca, sa = math.cos(angle), math.sin(angle)
    col   = (45, 60, 95)      # dark brown in BGR
    mid   = (70, 90, 130)
    light = (120, 150, 190)

    def lw_(lx, ly):
        return (int(x + (lx*ca - ly*sa)*sc), int(y + (lx*sa + ly*ca)*sc))

    rx = max(1, int(22*sc)); ry = max(1, int(8*sc))
    ks = max(3, int(sc * 8)) | 1
    # glow aura: wide wing silhouettes blurred, within an ROI (incl. blur margin)
    wings = [np.array([lw_(0, 0), lw_(-4, sign*8), lw_(-18, sign*24), lw_(-24, sign*18)],
                      dtype=np.int32) for sign in (1, -1)]
    allp = np.vstack(wings)
    H, W = img.shape[:2]
    gx0, gy0 = max(0, int(allp[:, 0].min()) - ks), max(0, int(allp[:, 1].min()) - ks)
    gx1, gy1 = min(W, int(allp[:, 0].max()) + ks), min(H, int(allp[:, 1].max()) + ks)
    if gx1 > gx0 and gy1 > gy0:
        roi = img[gy0:gy1, gx0:gx1]
        glow = np.zeros_like(roi)
        for pts in wings:
            cv2.fillPoly(glow, [pts - [gx0, gy0]], mid, cv2.LINE_8)
        glow = blur_down(glow, sc * 4)
        img[gy0:gy1, gx0:gx1] = cv2.addWeighted(roi, 1.0, glow, 0.50, 0)

    # wing silhouettes
    for sign in (1, -1):
        inner = np.array([lw_(2, 0),
                          lw_(-2, sign*ry),
                          lw_(-10, sign*20*sc/sc),
                          lw_(-4, sign*ry)], dtype=np.int32)
        outer = np.array([lw_(-2, sign*ry),
                          lw_(-10, sign*20),
                          lw_(-20, sign*22),
                          lw_(-24, sign*18),
                          lw_(-18, sign*24)], dtype=np.int32)
        cv2.fillPoly(img, [inner], mid,  cv2.LINE_8)
        cv2.fillPoly(img, [outer], col,  cv2.LINE_8)

    # body fill (translucent)
    blend_roi(img, x, y, max(rx, ry) + 2,
              lambda m, ox, oy: cv2.ellipse(m, (x - ox, y - oy), (rx, ry),
                                            math.degrees(angle), 0, 360, light, -1, cv2.LINE_8),
              0.20, 1.0)
    cv2.ellipse(img, (x, y), (rx, ry), math.degrees(angle), 0, 360, col,
                max(1, int(1.5*sc)), cv2.LINE_8)

    # tail fan
    for ty in (-8, -4, 0, 4, 8):
        cv2.line(img, lw_(-22, 0), lw_(-34, ty), col, max(1, int(sc)), cv2.LINE_8)

    # hooked beak
    bk = np.array([lw_(15, 0), lw_(22, 3), lw_(20, 5)], dtype=np.int32)
    cv2.polylines(img, [bk], False, col, max(1, int(sc)), cv2.LINE_8)

    # eye
    ep = lw_(11, -2)
    cv2.circle(img, ep, max(2, int(3*sc)), col,   -1, cv2.LINE_8)
    cv2.circle(img, ep, max(1, int(sc)),   light, -1, cv2.LINE_8)


# ---------------------------------------------------------------------------
# Background  (sky gradient + clouds)
# ---------------------------------------------------------------------------

def _make_sky_bg(h, w):
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    # #1a3a6e (deep azure) → #b8d4f0 (pale horizon)
    for y in range(h):
        t = y / h
        r = int(0x1a + (0xb8 - 0x1a) * t)
        g = int(0x3a + (0xd4 - 0x3a) * t)
        b = int(0x6e + (0xf0 - 0x6e) * t)
        bg[y] = (b, g, r)
    # wispy clouds (soft blurred ellipses)
    rng = np.random.default_rng(7)
    for _ in range(14):
        cx_ = rng.integers(0, w)
        cy_ = rng.integers(int(h * 0.04), int(h * 0.55))
        rw  = rng.integers(int(w * 0.04), int(w * 0.15))
        rh  = max(1, int(rw * 0.15))
        alpha = rng.uniform(0.10, 0.28)
        cloud = bg.copy()
        cv2.ellipse(cloud, (cx_, cy_), (rw, rh), 0, 0, 360, (255,255,255), -1, cv2.LINE_AA)
        ks = max(3, int(rw * 0.32)) | 1
        cloud = cv2.GaussianBlur(cloud, (ks, ks), rw * 0.16)
        cv2.addWeighted(cloud, alpha, bg, 1 - alpha, 0, bg)
    return bg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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

    RENDER_LONG  = 1440          # vector art -> render at 1440; slaves downsample
    N            = 220
    VISUAL_RANGE = 75.0
    SEP_RANGE    = 22.0
    MAX_SPEED    = 6.5
    MIN_SPEED    = 1.8
    W_COH        = 0.0040
    W_ALI        = 0.048
    W_SEP        = 0.12
    W_WANDER     = 0.038
    FLEE_RANGE   = 170.0
    W_FLEE       = 7.5
    EAT_RADIUS   = 34.0
    HAWK_ACCEL   = 2.2
    HAWK_MAX_SPD = 28.0
    HAWK_DRAG    = 0.88
    HAWK_TURN    = 0.22

    def __init__(self, width, height):
        super().__init__(width, height)
        sc  = self.scale
        N   = self.N

        cols = round(math.sqrt(N * self.w / self.h))
        rows = math.ceil(N / cols)
        cw, ch = self.w / cols, self.h / rows

        rng = np.random.default_rng()
        bx  = np.zeros(N, np.float32); by  = np.zeros(N, np.float32)
        bvx = np.zeros(N, np.float32); bvy = np.zeros(N, np.float32)
        for i in range(N):
            bx[i]  = (i%cols + 0.5 + (rng.random()-0.5)*0.85)*cw
            by[i]  = (i//cols + 0.5 + (rng.random()-0.5)*0.85)*ch
            d = rng.random()*2*math.pi
            s = (self.MIN_SPEED + rng.random()*1.4)*sc
            bvx[i] = math.cos(d)*s; bvy[i] = math.sin(d)*s

        self.bx = bx; self.by = by; self.bvx = bvx; self.bvy = bvy
        self.wander = rng.random(N).astype(np.float32)*2*math.pi
        # Warm earthy hues: 35-85 (amber/gold/yellow-green)
        self.hues   = (35 + rng.random(N)*50).astype(np.float32)
        self.sizes  = (0.78 + rng.random(N)*0.44).astype(np.float32)
        self.phases = (rng.random(N)*2*math.pi).astype(np.float32)  # per-bird flap phase
        self.alive  = np.ones(N, dtype=bool)

        self.hx = float(self.w)/2; self.hy = float(self.h)/2
        self.hvx = 0.0; self.hvy = 0.0; self.hangle = 0.0
        self.hawk_on = False
        self._tw = 0.0

        self._game_state = "waiting"; self._start_t = 0.0
        self._elapsed = 0.0; self._eaten = 0
        self._bursts  = []

        self._bg       = _make_sky_bg(self.h, self.w)
        self._fade_col = (0xe0, 0xd0, 0xb8)   # pale sky blue for fade (BGR)
        # Bake the constant motion-fade into the background once, and preallocate
        # the glow buffer, so render() churns one full-frame array per frame (the
        # returned output) instead of three — far less GC pressure / no hitches.
        _fade = np.full_like(self._bg, self._fade_col)
        self._base = cv2.addWeighted(_fade, 0.14, self._bg, 0.86, 0)
        self._glow = np.zeros_like(self._bg)

    # ------------------------------------------------------------------ param
    def set_param(self, key, val):
        super().set_param(key, val)
        if key == "mode":
            self._reset_game()

    def _reset_game(self):
        N = self.N; sc = self.scale
        self.alive[:] = True
        self._game_state = "waiting"; self._elapsed = 0.0; self._eaten = 0
        self._bursts.clear(); self.hawk_on = False
        self.hx = float(self.w)/2; self.hy = float(self.h)/2
        self.hvx = 0.0; self.hvy = 0.0
        rng = np.random.default_rng()
        cols = round(math.sqrt(N*self.w/self.h)); rows = math.ceil(N/cols)
        cw, ch = self.w/cols, self.h/rows
        for i in range(N):
            self.bx[i] = (i%cols+0.5+(rng.random()-0.5)*0.85)*cw
            self.by[i] = (i//cols+0.5+(rng.random()-0.5)*0.85)*ch

    def _cursor(self):
        t = self._tw
        return (self.w/2 + self.w*0.38*math.sin(t*0.53),
                self.h/2 + self.h*0.32*math.sin(t*0.79))

    # ------------------------------------------------------------------ step
    def step(self):
        sc = self.scale
        self._tw += 0.04
        cx, cy = self._cursor()

        mode = self._params.get("mode", "normal")
        if not self.hawk_on:
            self.hx = cx; self.hy = cy; self.hawk_on = True
        if mode == "game" and self._game_state == "waiting":
            self._game_state = "playing"; self._start_t = time.time()
        if mode == "game" and self._game_state == "playing":
            self._elapsed = time.time() - self._start_t

        # hawk update
        hx, hy = self.hx, self.hy
        dx_, dy_ = cx - hx, cy - hy
        dist = math.hypot(dx_, dy_)
        if dist > 2:
            ta   = math.atan2(dy_, dx_)
            diff = ta - self.hangle
            while diff >  math.pi: diff -= 2*math.pi
            while diff < -math.pi: diff += 2*math.pi
            self.hangle += math.copysign(min(abs(diff), self.HAWK_TURN), diff)
            thr = min(dist/20, 1)
            self.hvx += math.cos(self.hangle)*self.HAWK_ACCEL*thr
            self.hvy += math.sin(self.hangle)*self.HAWK_ACCEL*thr
        self.hvx *= self.HAWK_DRAG; self.hvy *= self.HAWK_DRAG
        spd_ = math.hypot(self.hvx, self.hvy)
        if spd_ > self.HAWK_MAX_SPD:
            f = self.HAWK_MAX_SPD/spd_; self.hvx *= f; self.hvy *= f
        self.hx += self.hvx*sc; self.hy += self.hvy*sc
        hx, hy = self.hx, self.hy

        # eat
        if mode == "game" and self._game_state == "playing":
            er2 = (self.EAT_RADIUS*sc)**2
            dxe = self.bx-hx; dye = self.by-hy
            caught = self.alive & (dxe*dxe+dye*dye < er2)
            if caught.any():
                for i in np.where(caught)[0]:
                    self.alive[i] = False; self._eaten += 1
                    self._spawn_burst(self.bx[i], self.by[i], self.hues[i])
                if self._eaten >= self.N:
                    self._game_state = "done"

        # boids (vectorized; flee target = hawk)
        self.bx, self.by, self.bvx, self.bvy = boids_update(
            self.bx, self.by, self.bvx, self.bvy, self.wander, self.alive, hx, hy,
            visual_range=self.VISUAL_RANGE, sep_range=self.SEP_RANGE,
            flee_range=self.FLEE_RANGE, min_speed=self.MIN_SPEED,
            max_speed=self.MAX_SPEED, w_coh=self.W_COH, w_ali=self.W_ALI,
            w_sep=self.W_SEP, w_wander=self.W_WANDER, w_flee=self.W_FLEE,
            width=self.w, height=self.h, sc=sc)

        for b in self._bursts:
            b['x'] += b['vx']; b['y'] += b['vy']
            b['vx'] *= 0.91;   b['vy'] *= 0.91; b['life'] -= 0.038
        self._bursts = [b for b in self._bursts if b['life'] > 0]
        self.t += 0.016

    def _spawn_burst(self, x, y, hue):
        sc = self.scale
        for _ in range(12):
            a = np.random.random()*2*math.pi; s = (1.8+np.random.random()*3)*sc
            self._bursts.append({'x':x,'y':y,'vx':math.cos(a)*s,'vy':math.sin(a)*s,
                                  'life':1.0,'hue':hue})

    # ------------------------------------------------------------------ render
    def render(self):
        sc   = self.scale
        mode = self._params.get("mode", "normal")
        img  = self._base.copy()        # fade baked in; fresh buffer (caster reads it)

        glow_layer = self._glow         # reused (not returned) -> zero in place
        glow_layer.fill(0)

        for i in range(self.N):
            if not self.alive[i]: continue
            spd_i = math.hypot(self.bvx[i], self.bvy[i])
            flap  = math.sin(self.t * (5.0 + spd_i/(self.MAX_SPEED*sc)*4.0) + self.phases[i])
            draw_bird(img, glow_layer,
                      int(self.bx[i]), int(self.by[i]),
                      math.atan2(self.bvy[i], self.bvx[i]),
                      self.hues[i], spd_i, self.sizes[i], sc,
                      self.MAX_SPEED, flap)

        sig = max(1.0, 4.0*sc); ks = max(3, int(sig*2))|1
        cv2.addWeighted(img, 1.0, blur_down(glow_layer, sig), 1.05, 0, img)

        for b in self._bursts:
            r_ = max(1, int(3.5*b['life']*sc)); a_ = b['life']**2
            bx_, by_ = int(b['x']), int(b['y']); col = _hsl_bgr(b['hue'], 75, 70)
            blend_roi(img, bx_, by_, r_ + 1,
                      lambda m, ox, oy: cv2.circle(m, (bx_ - ox, by_ - oy), r_, col, -1, cv2.LINE_AA),
                      a_, 1.0)

        if self.hawk_on and self._game_state != "done":
            draw_hawk(img, int(self.hx), int(self.hy), self.hangle, sc)

        if mode == "game":
            self._draw_hud(img, sc)
        return img

    def _draw_hud(self, img, sc):
        font = cv2.FONT_HERSHEY_SIMPLEX
        W, H = self.w, self.h
        state = self._game_state

        def put(txt, x, y, size, col, bold=False):
            th = 2 if bold else 1
            cv2.putText(img, txt, (x,y), font, size*sc, col, max(1,int(th*sc)), cv2.LINE_AA)

        if state == "waiting":
            txt = "Hawk hunts — lead it into the flock!"
            tw  = cv2.getTextSize(txt, font, 0.5*sc, max(1,int(sc)))[0][0]
            put(txt, (W-tw)//2, int(H*0.94), 0.5, (40,40,40))
        elif state == "playing":
            put(f"{self._elapsed:.2f}s", W//2-int(40*sc), int(42*sc), 0.7, (40,40,40), bold=True)
            put(f"{int(self.alive.sum())} remaining", W-int(120*sc), int(34*sc), 0.4, (60,60,60))
        elif state == "done":
            overlay = img.copy(); cv2.rectangle(overlay,(0,0),(W,H),(230,220,200),-1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            for j,(txt,col) in enumerate([("All birds caught!",(30,30,100)),
                                           (f"Time: {self._elapsed:.2f}s",(40,40,130)),
                                           ("Click to play again",(80,80,160))]):
                sz = [1.1,0.85,0.42][j]; th = [2,1,1][j]
                tw = cv2.getTextSize(txt,font,sz*sc,max(1,int(th*sc)))[0][0]
                yy = H//2+[-50*sc,12*sc,58*sc][j]
                cv2.putText(img,txt,((W-tw)//2,int(yy)),font,sz*sc,col,max(1,int(th*sc)),cv2.LINE_AA)
