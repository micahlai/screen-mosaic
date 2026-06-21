"""
Ghost trail prototype — mouse input.

Opens ghost_trail.html in the default browser.
Later: replace mouse events with camera hand-detection and
inject coordinates into the main mosiac project.

Usage:
    python ghost_trail.py
"""

import pathlib
import webbrowser

html = pathlib.Path(__file__).with_name("ghost_trail.html").resolve()
if not html.exists():
    raise FileNotFoundError(f"ghost_trail.html not found next to this script ({html})")

print(f"Opening {html.name} in browser…")
webbrowser.open(html.as_uri())
print("Move your mouse around. Close the tab when done.")
print()
print("Next step: replace mouse events with MediaPipe hand-landmark")
print("coordinates and pipe them into the mosiac UV content layer.")
