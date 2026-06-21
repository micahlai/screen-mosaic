"""Gradient (color-ramp) maps for the smoke visualization.

Each gradient is a JSON file in the ``gradients/`` subdirectory with positional
stops::

    { "stops": [ {"pos": 0.0, "color": "00082a"}, {"pos": 0.1667, ...}, ... ] }

``pos`` is the density (0..1) the color maps to; positions need not be evenly
spaced. The smoke sim looks up the currently-selected gradient's LUT each frame.
"""

import json
import os

import numpy as np

GRAD_DIR = os.path.join(os.path.dirname(__file__), "gradients")


def available():
    """Sorted list of gradient names (JSON filenames without extension)."""
    if not os.path.isdir(GRAD_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(GRAD_DIR) if f.endswith(".json"))


def _hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))   # B, G, R


def _load_stops(name):
    with open(os.path.join(GRAD_DIR, name + ".json")) as f:
        data = json.load(f)
    stops = data["stops"] if isinstance(data, dict) else data
    out = [(float(s["pos"]), _hex_to_bgr(s["color"])) for s in stops]
    out.sort(key=lambda s: s[0])
    return out


def build_lut(name, n=256):
    """256-entry BGR lookup table (float 0..255) for the named gradient."""
    stops = _load_stops(name)
    pos = np.array([p for p, _ in stops], dtype=np.float32)
    cols = np.array([c for _, c in stops], dtype=np.float32)     # (k, 3) BGR
    xs = np.linspace(0.0, 1.0, n).astype(np.float32)
    lut = np.empty((n, 3), dtype=np.float32)
    for ch in range(3):                                          # np.interp clamps edges
        lut[:, ch] = np.interp(xs, pos, cols[:, ch])
    return lut


# --- current selection (the smoke sim reads this) ---------------------------

_current_name = None
_version = 0
_lut = None


def set_current(name):
    """Select the active gradient by name; bumps the version so sims rebuild."""
    global _current_name, _version, _lut
    _lut = build_lut(name)
    _current_name = name
    _version += 1
    return _current_name


def _ensure():
    if _lut is None:
        names = available()
        set_current(names[0] if names else "gradient1")


def current_lut():
    _ensure()
    return _lut


def current_name():
    _ensure()
    return _current_name


def version():
    _ensure()
    return _version
