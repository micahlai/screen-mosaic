"""
Ghost trail v2 — directional ripple prototype.

Spawns wavy, velocity-stretched ripple rings at the cursor position.
Ring size, wave amplitude, and spawn density all scale with cursor speed.

Usage:
    python ghost_trail_v2.py

Next step: replace mouse events with MediaPipe hand-landmark coordinates.
"""

import pathlib
import webbrowser

html = pathlib.Path(__file__).with_name("ghost_trail_v2.html").resolve()
if not html.exists():
    raise FileNotFoundError(f"ghost_trail_v2.html not found ({html})")

print("Opening ghost_trail_v2.html …")
print("  · slow movement  → small, tight ripples")
print("  · fast flick     → large stretched rings + sparks")
webbrowser.open(html.as_uri())
