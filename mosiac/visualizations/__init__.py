"""
Visualization framework.

This package's __init__ is the central module: it owns the registry, the
`Visualization` base class, the shared GPU helpers, and the dispatch calls used
by the server (`available()`, `create()`, `gpu_device()`). Each visualization
lives in its own file in this directory (e.g. ``smokesim.py``, ``particleflow.py``)
and registers itself with the ``@register`` decorator.

Add a new visualization:
    1. create ``mosiac/visualizations/myviz.py`` with a @register'd subclass,
    2. import it at the bottom of this file.
It then shows up automatically in the phone app's visualization dropdown.

Heavy rasterization runs on the GPU via PyTorch (CUDA / Apple-Silicon MPS);
without torch the visualizations fall back to slower CPU paths.
"""

import numpy as np  # noqa: F401  (re-exported for visualization modules)
import cv2          # noqa: F401

# --- adjust this ---------------------------------------------------------
# Render-resolution multiplier (see README). 4.0 -> ~4K field at ~17 fps on MPS.
RESOLUTION_SCALE = 2.0
# Hard cap on the rendered long side. The slaves re-warp/downsample the stream
# to their own screens, so rendering much past 4K wastes time on the JPEG encode
# and tanks the frame-rate (8K ≈ 1.4 fps) while adding ~no visible sharpness.
# Raise this only if you accept lower fps for a very wide multi-screen wall.
MAX_RENDER_LONG = 3840
# -------------------------------------------------------------------------

try:
    import torch
    _DEVICE = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available()
               else "cpu")
except Exception:
    torch = None
    _DEVICE = "cpu"


def gpu_device() -> str:
    return _DEVICE if torch is not None else "cpu (numpy)"


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------

_REGISTRY = {}


def register(name, label=None):
    def deco(cls):
        cls.viz_name = name
        cls.viz_label = label or name.replace("_", " ").title()
        _REGISTRY[name] = cls
        return cls
    return deco


def available():
    """List of {name, label, needs_phone_camera} for every registered visualization."""
    return [{"name": n, "label": c.viz_label,
             "needs_phone_camera": bool(getattr(c, "NEEDS_PHONE_CAMERA", False))}
            for n, c in _REGISTRY.items()]


def create(name, width, height):
    if name not in _REGISTRY:
        raise KeyError(f"unknown visualization: {name}")
    return _REGISTRY[name](width, height)


def supports_pointer(name):
    """True if the visualization reacts to a pointer force (set_pointer)."""
    return hasattr(_REGISTRY.get(name), "set_pointer")


def uses_hands(name):
    """True if the visualization is driven by a hand tracker."""
    return bool(getattr(_REGISTRY.get(name), "USES_HANDS", False))


def hand_tracker(name):
    """Return the tracker type for this viz: 'yolo' (default) or 'red'."""
    return getattr(_REGISTRY.get(name), "HAND_TRACKER", "yolo")


def all_viz_params():
    """Returns {viz_name: param_defs} for vizs that expose parameters."""
    return {n: c.viz_params for n, c in _REGISTRY.items() if c.viz_params}


class Visualization:
    """Base class. Subclasses implement step() and render() -> HxWx3 uint8 BGR.
    The render canvas is (width, height) * RESOLUTION_SCALE.

    Subclasses can expose user-settable parameters by overriding viz_params:

        viz_params = {
            "mode": {
                "label": "Mode",
                "options": [{"value": "normal", "label": "Normal"}, ...],
                "default": "normal",
            }
        }

    The phone UI discovers these via GET /viz/params and sets them via
    POST /viz/param.
    """

    viz_params: dict = {}   # override in subclass to expose phone-UI dropdowns
    RENDER_LONG = None      # per-viz render long-side cap (None -> MAX_RENDER_LONG).
                            # CPU-rendered vector viz (boids) set this low: the slaves
                            # re-warp/downsample anyway, so 4K just wastes draw time.

    def __init__(self, width, height):
        # effective scale = RESOLUTION_SCALE, clamped so the long side <= the cap
        cap = self.RENDER_LONG or MAX_RENDER_LONG
        eff = RESOLUTION_SCALE
        long = max(width, height) * eff
        if long > cap:
            eff *= cap / long
        self.scale = eff                       # sizes/speeds use the effective scale
        self.w = max(1, int(round(width * eff)))
        self.h = max(1, int(round(height * eff)))
        self.t = 0.0
        self._params = {k: v["default"] for k, v in self.__class__.viz_params.items()}

    def set_param(self, key, val):
        """Called by the server when the phone UI changes a dropdown value."""
        defs = self.__class__.viz_params
        if key in defs:
            valid = [o["value"] for o in defs[key]["options"]]
            if val in valid:
                self._params[key] = val

    def get_param(self, key, default=None):
        return self._params.get(key, default)

    def step(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared GPU helpers (used by the visualization modules)
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma, device):
    radius = max(1, int(round(sigma * 3)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / k.sum(), radius


def _gauss(img, sigma):
    """Separable Gaussian blur of a (H, W) tensor (direct convolution)."""
    k, r = _gaussian_kernel(sigma, img.device)
    x = img.view(1, 1, *img.shape)
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x.view(*img.shape)


def _blur(img, sigma):
    """Gaussian blur; big blurs run on a downsampled grid (scale-independent)."""
    if sigma <= 6:
        return _gauss(img, sigma)
    F = torch.nn.functional
    down = min(8, max(2, int(round(sigma / 3))))
    small = F.avg_pool2d(img.view(1, 1, *img.shape), down)
    small = _gauss(small.view(*small.shape[2:]), sigma / down)
    up = F.interpolate(small.view(1, 1, *small.shape), size=img.shape,
                       mode="bilinear", align_corners=False)
    return up.view(*img.shape)


# ---------------------------------------------------------------------------
# Shared CPU helpers for the boids sims (vectorized physics + ROI sprite blend)
# ---------------------------------------------------------------------------

def boids_update(bx, by, bvx, bvy, wander, alive, pred_x, pred_y, *,
                 visual_range, sep_range, flee_range, min_speed, max_speed,
                 w_coh, w_ali, w_sep, w_wander, w_flee, width, height, sc,
                 edge_margin=0.0, w_edge=0.0):
    """Vectorized boids step (cohesion/alignment/separation/wander/flee). Replaces
    the O(N^2) Python double loop with numpy broadcasting. Updates `wander` in
    place; returns new (bx, by, bvx, bvy).

    edge_margin/w_edge (optional): steer boids away from the walls within
    `edge_margin` (logical px) of an edge, strongest at the wall."""
    N = bx.shape[0]
    VR2 = (visual_range * sc) ** 2
    SR2 = (sep_range * sc) ** 2
    FR2 = (flee_range * sc) ** 2
    SR_px = sep_range * sc
    FR_px = flee_range * sc
    minV, maxV = min_speed * sc, max_speed * sc

    dx = bx[None, :] - bx[:, None]                  # dx[i,j] = bx[j]-bx[i]
    dy = by[None, :] - by[:, None]
    d2 = dx * dx + dy * dy
    av = alive[None, :] & alive[:, None]
    neigh = av & (d2 <= VR2) & (d2 > 0.0)
    nn = neigh.sum(axis=1)
    has = nn > 0
    inv = np.where(has, 1.0 / np.maximum(nn, 1), 0.0).astype(np.float32)

    fx = np.zeros(N, np.float32); fy = np.zeros(N, np.float32)
    cohX = (bx[None, :] * neigh).sum(1); cohY = (by[None, :] * neigh).sum(1)
    aliVx = (bvx[None, :] * neigh).sum(1); aliVy = (bvy[None, :] * neigh).sum(1)
    fx += (cohX * inv - bx) * w_coh * has;  fy += (cohY * inv - by) * w_coh * has
    fx += (aliVx * inv - bvx) * w_ali * has; fy += (aliVy * inv - bvy) * w_ali * has

    sep = av & (d2 <= SR2) & (d2 > 0.0)
    d = np.sqrt(np.where(d2 > 0, d2, 1.0))
    sfac = np.where(sep, (SR_px - d) / SR_px / d, 0.0)
    fx += -(dx * sfac).sum(1) * w_sep
    fy += -(dy * sfac).sum(1) * w_sep

    wander += (np.random.random(N).astype(np.float32) - 0.5) * 0.06
    fx += np.cos(wander) * w_wander; fy += np.sin(wander) * w_wander

    sxv = bx - pred_x; syv = by - pred_y; sd2 = sxv * sxv + syv * syv
    flee = (sd2 < FR2) & (sd2 > 0.0)
    sd = np.sqrt(np.where(sd2 > 0, sd2, 1.0))
    p = np.where(flee, w_flee * np.clip(1 - sd / FR_px, 0, 1) ** 1.5 / sd, 0.0)
    fx += sxv * p; fy += syv * p

    if edge_margin > 0.0 and w_edge > 0.0:          # steer away from the walls
        m = edge_margin * sc
        fx += np.where(bx < m,          w_edge * (1 - bx / m),            0.0)
        fx += np.where(bx > width - m, -w_edge * (1 - (width - bx) / m),  0.0)
        fy += np.where(by < m,          w_edge * (1 - by / m),            0.0)
        fy += np.where(by > height - m, -w_edge * (1 - (height - by) / m), 0.0)

    bvx = bvx + fx; bvy = bvy + fy
    sp = np.hypot(bvx, bvy)
    k = np.where(sp > maxV, maxV / np.maximum(sp, 1e-9),
                 np.where((sp < minV) & (sp > 0), minV / np.maximum(sp, 1e-9), 1.0))
    bvx = (bvx * k).astype(np.float32); bvy = (bvy * k).astype(np.float32)

    bx = np.where(alive, bx + bvx, bx); by = np.where(alive, by + bvy, by)
    bx = np.where(bx < -20 * sc, bx + width + 40 * sc, bx)
    bx = np.where(bx > width + 20 * sc, bx - (width + 40 * sc), bx)
    by = np.where(by < -20 * sc, by + height + 40 * sc, by)
    by = np.where(by > height + 20 * sc, by - (height + 40 * sc), by)
    return bx.astype(np.float32), by.astype(np.float32), bvx, bvy


def blend_roi(img, cx, cy, pad, draw, alpha, base_w=1.0):
    """Blend a sprite-sized layer instead of a full-frame one:
    img[roi] = base_w*img[roi] + alpha*draw(mask). `draw(mask, ox, oy)` draws into
    the small local mask (subtract ox/oy from coords). Turns O(frame) per sprite
    into O(sprite)."""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    x1, y1 = min(w, int(cx + pad) + 1), min(h, int(cy + pad) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    mask = np.zeros_like(roi)
    draw(mask, x0, y0)
    img[y0:y1, x0:x1] = cv2.addWeighted(mask, alpha, roi, base_w, 0)


def blur_down(layer, sigma, down=4):
    """Cheap large-radius blur for the soft glow layers: downsample, blur small,
    upsample. The glow is low-frequency so this is ~identical at a fraction of the
    cost of a full-resolution Gaussian."""
    h, w = layer.shape[:2]
    if min(h, w) < down * 2 or sigma < 1.5:
        return cv2.GaussianBlur(layer, (0, 0), max(0.6, sigma))
    small = cv2.resize(layer, (w // down, h // down), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.6, sigma / down))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# --- gradient maps (used by the smoke viz) ---
from . import gradients      # noqa: E402,F401

# --- register built-in visualizations (importing each runs its @register) ---
from . import particleflow   # noqa: E402,F401
from . import smokesim       # noqa: E402,F401
from . import charges        # noqa: E402,F401
from . import fishboids      # noqa: E402,F401
from . import birdboids      # noqa: E402,F401
