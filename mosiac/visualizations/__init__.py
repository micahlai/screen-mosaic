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
RESOLUTION_SCALE = 8.0
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
    """List of {name, label} for every registered visualization."""
    return [{"name": n, "label": c.viz_label} for n, c in _REGISTRY.items()]


def create(name, width, height):
    if name not in _REGISTRY:
        raise KeyError(f"unknown visualization: {name}")
    return _REGISTRY[name](width, height)


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

    def __init__(self, width, height):
        # effective scale = RESOLUTION_SCALE, clamped so the long side <= the cap
        eff = RESOLUTION_SCALE
        long = max(width, height) * eff
        if long > MAX_RENDER_LONG:
            eff *= MAX_RENDER_LONG / long
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


# --- gradient maps (used by the smoke viz) ---
from . import gradients      # noqa: E402,F401

# --- register built-in visualizations (importing each runs its @register) ---
from . import particleflow   # noqa: E402,F401
from . import smokesim       # noqa: E402,F401
from . import charges        # noqa: E402,F401
from . import fishboids      # noqa: E402,F401
from . import birdboids      # noqa: E402,F401
