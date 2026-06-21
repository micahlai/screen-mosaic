"""Bird boid simulation.

Same red-sticker hand tracking as the fish (the hand is the predator the birds
flee). Birds are solid black silhouettes (no animation). The twist: near a
screen's *bottom edge* (a line segment from the calibration data, not necessarily
parallel to the render bottom) a bird has a chance to perch and stay there until
it's scared by the hand or the flock pulls hard enough to yank it off.
Background: sky gradient + cloud wisps.
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


def _hsl_bgr(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Bird sprite — a solid black silhouette (pointed head, swept wings, forked tail)
# ---------------------------------------------------------------------------
# Local coords: forward = +x, up = -y. Outline walked clockwise.
_BIRD_SHAPE = np.array([
    (10, 0), (5, -1.2), (1, -1.5), (-7, -12), (-5, -2.5),
    (-12, -3), (-15, 0), (-12, 3), (-5, 2.5), (-7, 12), (1, 1.5), (5, 1.2),
], dtype=np.float32)


def draw_bird(img, x, y, angle, sz, sc):
    """Solid black bird silhouette rotated to `angle` (no animation)."""
    ca, sa = math.cos(angle), math.sin(angle)
    s = sz * sc
    lx = _BIRD_SHAPE[:, 0] * s
    ly = _BIRD_SHAPE[:, 1] * s
    xs = x + lx * ca - ly * sa
    ys = y + lx * sa + ly * ca
    pts = np.stack([xs, ys], axis=-1).astype(np.int32)
    cv2.fillPoly(img, [pts], (18, 18, 18), cv2.LINE_AA)


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

    USES_HANDS         = True    # hand position drives the predator the birds flee
    HAND_TRACKER       = "red"   # red-sticker CV tracking, same as the fish
    NEEDS_PHONE_CAMERA = True    # phone must stream frames whenever this viz is active

    RENDER_LONG  = 1440          # vector art -> render at 1440; slaves downsample
    N            = 220
    VISUAL_RANGE = 75.0
    SEP_RANGE    = 22.0
    MAX_SPEED    = 6.5
    MIN_SPEED    = 1.8
    W_COH        = 0.0012
    W_ALI        = 0.018
    W_SEP        = 0.38
    W_WANDER     = 0.10
    FLEE_RANGE   = 170.0
    W_FLEE       = 7.5
    EAT_RADIUS   = 34.0

    # perching on a screen's bottom edge
    PERCH_DIST        = 26.0     # logical px from the edge to consider perching
    PERCH_CHANCE      = 0.05     # per-frame probability to perch when in range
    PERCH_BREAK_SPEED = 3.2      # boid pull (logical px/frame) that yanks a bird off
    SCARE_RANGE       = 150.0    # hand within this (logical px) scares perched birds off

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
        self.sizes  = (0.78 + rng.random(N)*0.44).astype(np.float32)
        self.hues   = (35 + rng.random(N)*50).astype(np.float32)   # burst-particle colours
        self.alive  = np.ones(N, dtype=bool)

        # hand predator (red tracker) — same as the fish
        self._ptr = None
        self.has_hand = False
        self._show_ring = True
        self.sx = -1e9; self.sy = -1e9

        # perch state + the bottom-edge segments (filled once via set_screens)
        self._bottom_edges = []                       # [((x1,y1),(x2,y2)), ...] field px
        self.perched = np.zeros(N, dtype=bool)
        self.perch_x = np.zeros(N, np.float32)
        self.perch_y = np.zeros(N, np.float32)
        self.perch_a = np.zeros(N, np.float32)

        self._game_state = "waiting"; self._start_t = 0.0
        self._elapsed = 0.0; self._eaten = 0
        self._bursts  = []

        self._bg       = _make_sky_bg(self.h, self.w)
        self._fade_col = (0xe0, 0xd0, 0xb8)   # pale sky blue tint (BGR)
        # Bake the constant tint into the background once; render() copies this.
        _fade = np.full_like(self._bg, self._fade_col)
        self._base = cv2.addWeighted(_fade, 0.14, self._bg, 0.86, 0)

    # --------------------------------------------------------------- screens
    def set_screens(self, segments):
        """segments: list of ((u1,v1),(u2,v2)) bottom edges normalized to the UV
        bbox. Stored once at load and converted to this field's pixel space."""
        self._bottom_edges = [((u1*self.w, v1*self.h), (u2*self.w, v2*self.h))
                              for (u1, v1), (u2, v2) in segments]

    def _nearest_edge(self, px, py):
        """Per-point nearest bottom edge → (dist, cx, cy, ang) arrays, or None."""
        if not self._bottom_edges:
            return None
        best_d = np.full(px.shape, np.inf, np.float32)
        best_x = np.zeros_like(px); best_y = np.zeros_like(py); best_a = np.zeros_like(px)
        for (ax, ay), (bx2, by2) in self._bottom_edges:
            abx = bx2 - ax; aby = by2 - ay
            L2  = abx*abx + aby*aby + 1e-9
            t   = np.clip(((px - ax)*abx + (py - ay)*aby) / L2, 0.0, 1.0)
            cx  = ax + t*abx; cy = ay + t*aby
            d   = np.hypot(px - cx, py - cy)
            m   = d < best_d
            best_d = np.where(m, d, best_d)
            best_x = np.where(m, cx, best_x)
            best_y = np.where(m, cy, best_y)
            best_a = np.where(m, math.atan2(aby, abx), best_a)
        return best_d, best_x, best_y, best_a

    # ------------------------------------------------------------------ param
    def set_param(self, key, val):
        super().set_param(key, val)
        if key == "mode":
            self._reset_game()
        elif key == "ring":          # gray hand-marker ring on/off (phone toggle)
            self._show_ring = (val is True or
                               str(val).lower() in ("true", "1", "on", "yes"))

    def set_pointer(self, ptr):
        """Hand force (nx, ny, nvx, nvy) in [0,1] field coords, or None."""
        self._ptr = ptr

    def _reset_game(self):
        N = self.N
        self.alive[:] = True
        self.perched[:] = False
        self._game_state = "waiting"; self._elapsed = 0.0; self._eaten = 0
        self._bursts.clear()
        rng = np.random.default_rng()
        cols = round(math.sqrt(N*self.w/self.h)); rows = math.ceil(N/cols)
        cw, ch = self.w/cols, self.h/rows
        for i in range(N):
            self.bx[i] = (i%cols+0.5+(rng.random()-0.5)*0.85)*cw
            self.by[i] = (i//cols+0.5+(rng.random()-0.5)*0.85)*ch

    # ------------------------------------------------------------------ step
    def step(self):
        sc = self.scale
        mode = self._params.get("mode", "normal")

        # hand predator (same as the fish): birds flee its field position
        if self._ptr is not None:
            self.sx = self._ptr[0]*self.w; self.sy = self._ptr[1]*self.h
            self.has_hand = True
        else:
            self.has_hand = False
            self.sx = self.sy = -1e9
        sx, sy = self.sx, self.sy

        if mode == "game" and self._game_state == "waiting" and self.has_hand:
            self._game_state = "playing"; self._start_t = time.time()
        if mode == "game" and self._game_state == "playing":
            self._elapsed = time.time() - self._start_t

        # eat collision (the hand catches birds it touches — perched or flying)
        if mode == "game" and self._game_state == "playing" and self.has_hand:
            er2 = (self.EAT_RADIUS*sc)**2
            dxe = self.bx-sx; dye = self.by-sy
            caught = self.alive & (dxe*dxe+dye*dye < er2)
            if caught.any():
                for i in np.where(caught)[0]:
                    self.alive[i] = False; self.perched[i] = False; self._eaten += 1
                    self._spawn_burst(self.bx[i], self.by[i], self.hues[i])
                if self._eaten >= self.N:
                    self._game_state = "done"

        # candidate boids step for every bird (flee target = the hand)
        nbx, nby, nbvx, nbvy = boids_update(
            self.bx, self.by, self.bvx, self.bvy, self.wander, self.alive, sx, sy,
            visual_range=self.VISUAL_RANGE, sep_range=self.SEP_RANGE,
            flee_range=self.FLEE_RANGE, min_speed=self.MIN_SPEED,
            max_speed=self.MAX_SPEED, w_coh=self.W_COH, w_ali=self.W_ALI,
            w_sep=self.W_SEP, w_wander=self.W_WANDER, w_flee=self.W_FLEE,
            width=self.w, height=self.h, sc=sc)

        if self._bottom_edges:
            # break perched birds that are scared by the hand or pulled hard by the flock
            if self.has_hand:
                sd = np.hypot(self.perch_x - sx, self.perch_y - sy)
                scared = self.perched & (sd < self.SCARE_RANGE*sc)
            else:
                scared = np.zeros(self.N, dtype=bool)
            cand_spd = np.hypot(nbvx, nbvy)
            pulled = self.perched & (cand_spd > self.PERCH_BREAK_SPEED*sc)
            self.perched &= ~(scared | pulled)        # those break free this frame

            # newly perch: flying birds that drift near an edge, by chance
            dist, ex, ey, ea = self._nearest_edge(nbx, nby)
            near = (~self.perched) & self.alive & (dist < self.PERCH_DIST*sc)
            newp = near & (np.random.random(self.N) < self.PERCH_CHANCE)
            self.perch_x = np.where(newp, ex, self.perch_x)
            self.perch_y = np.where(newp, ey, self.perch_y)
            self.perch_a = np.where(newp, ea, self.perch_a)
            self.perched |= newp

            pin = self.perched
            self.bx  = np.where(pin, self.perch_x, nbx).astype(np.float32)
            self.by  = np.where(pin, self.perch_y, nby).astype(np.float32)
            self.bvx = np.where(pin, 0.0, nbvx).astype(np.float32)
            self.bvy = np.where(pin, 0.0, nbvy).astype(np.float32)
        else:
            self.bx, self.by, self.bvx, self.bvy = nbx, nby, nbvx, nbvy

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
        img  = self._base.copy()        # tinted sky; fresh buffer (caster reads it)

        for i in range(self.N):
            if not self.alive[i]: continue
            if self.perched[i]:
                ang = self.perch_a[i]
            else:
                ang = math.atan2(self.bvy[i], self.bvx[i])
            draw_bird(img, int(self.bx[i]), int(self.by[i]), ang, self.sizes[i], sc)

        for b in self._bursts:
            r_ = max(1, int(3.5*b['life']*sc)); a_ = b['life']**2
            bx_, by_ = int(b['x']), int(b['y']); col = _hsl_bgr(b['hue'], 75, 70)
            blend_roi(img, bx_, by_, r_ + 1,
                      lambda m, ox, oy: cv2.circle(m, (bx_ - ox, by_ - oy), r_, col, -1, cv2.LINE_AA),
                      a_, 1.0)

        # hand marker (the predator) — translucent gray ring at the tracked position
        if self.has_hand and self._game_state != "done" and self._show_ring:
            self._draw_hand_circle(img, int(self.sx), int(self.sy), sc)

        if mode == "game":
            self._draw_hud(img, sc)
        return img

    def _draw_hand_circle(self, img, cx, cy, sc):
        r = max(3, int(consts.FISH_HAND_MARKER_FRAC * max(self.w, self.h)))
        blend_roi(img, cx, cy, r + max(2, r // 6) + 2,
                  lambda m, ox, oy: cv2.circle(m, (cx - ox, cy - oy), r,
                                               (235, 245, 255), max(2, r // 6), cv2.LINE_AA),
                  0.1, 1.0)

    def _draw_hud(self, img, sc):
        font = cv2.FONT_HERSHEY_SIMPLEX
        W, H = self.w, self.h
        state = self._game_state

        def put(txt, x, y, size, col, bold=False):
            th = 2 if bold else 1
            cv2.putText(img, txt, (x,y), font, size*sc, col, max(1,int(th*sc)), cv2.LINE_AA)

        if state == "waiting":
            txt = "Move your hand — the birds flee it. Catch them all!"
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
