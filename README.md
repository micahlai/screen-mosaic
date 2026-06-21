# Mosiac

Turn several ordinary displays into one coordinated canvas using a single phone
photo. Each screen shows four ArUco markers in its corners; you photograph them
all from one spot; the host then warps content per-screen so that — viewed from
where the photo was taken — every screen lines up into one continuous image.

## Run the host

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python mosiac
```

This starts the backend and **two web apps** on `:5001` (LAN URLs are printed):

- **Screen slave** — open `http://<host-ip>:5001/display` on each screen. Each
  browser auto-claims the next display slot (1st → `display_1` with marker IDs
  0–3, 2nd → `display_2` with 4–7, …) and shows its four corner markers.
- **Phone** — open `http://<host-ip>:5001/phone`. Take a photo of all screens,
  then map content onto them.

## Phases (switched from the phone)

1. **Calibration** — every screen shows ArUco markers flush in its far corners.
   Tapping *Take Photo* on the phone reveals the markers; the photo is detected,
   each screen's corners are recovered (using the marker corner that touches the
   real screen corner), and any screen that didn't fully make the photo is
   highlighted on that screen with a message.
2. **Mapping** (default) — each screen renders mapped content, projectively
   warped by its photographed corners so skewed screens look straight from the
   camera. Content options:
   - **UV map** — x→red, y→green gradient (default).
   - **Uploaded image** — *Fill* (stretch to the screens' bounding box) or
     *Fit* (preserve aspect ratio).
   - **Visualization** — a live animation rendered server-side (GPU) at a
     resolution matching the screens' bounding-box orientation, streamed (MJPEG)
     and warped per screen. Built in: **Particle Flow** and **Smoke** (a
     stable-fluids fire/smoke sim).

The phone's *Content* dropdown offers UV map / Upload image / Visualization;
picking Visualization reveals a second dropdown populated from whatever is
registered in `visualization.py`.

3. **Live calibration** — a camera continuously watches the screens and updates
   each screen's warp live (default **12 fps**). Start it from the phone's
   **🔴 Live calibration** button, then pick the **camera source**:
   - *Phone camera* — the phone streams its own camera frames to the host.
   - *Server device camera* — the host opens a local camera (`cv2.VideoCapture`).

   Each screen keeps four *smaller* markers on screen (over the content, no ID
   labels) so the camera can track them. If a screen isn't fully visible in a
   frame, its warp holds at the last good value instead of blanking.

   Browsers only allow camera access over a **secure context**, so the host
   serves **HTTPS** with a self-signed cert (accept the one-time warning on each
   device). Toggle with `USE_HTTPS` in `consts.py`.

Tunables in `mosiac/consts.py`: `PORT`, `USE_HTTPS`, `MARKER_PX`,
`LIVE_MARKER_PX`, `LIVE_FPS`, `CAMERA_INDEX`.

### Adding a visualization

Drop a new file in `mosiac/visualizations/` and import it from that package's
`__init__.py`:

```python
# mosiac/visualizations/rings.py
from . import Visualization, register, torch, _DEVICE

@register("rings", "Rings")
class Rings(Visualization):
    def step(self): ...
    def render(self): return frame   # H x W x 3 uint8 BGR
```

It appears in the phone dropdown automatically (`GET /visualizations`) — no
server or frontend changes needed. Preview locally with
`python -m mosiac.visualizations rings`.

The UV domain is the bounding box of all detected screen corners (plus a small
margin), so the gradient/image/particles span only the region the screens cover.

## Layout

| Path | Purpose |
|------|---------|
| `mosiac/` | The host. `python mosiac` runs `__main__` → `server.py`. |
| `mosiac/server.py` | Flask host: both web apps, calibration, mapping, content. |
| `mosiac/detector.py` | ArUco/AprilTag detection → grouped, ordered, normalized. |
| `mosiac/visualizations/` | Visualization package: framework in `__init__.py`, one file per viz (`particleflow.py`, `smokesim.py`). |
| `tools/` | Standalone analysis utilities (`python -m tools.cli IMAGE`, etc.). |
| `legacy/` | Earlier desktop prototype (`master/`, `slave/`, `shared/`). |

## Tools

```bash
python -m tools.make_test_image            # writes a synthetic test image
python -m tools.cli IMAGE --annotated out.png   # detect + visualize one image
python -m tools.app                        # standalone image-analysis web UI
```

## Notes

Coordinates are always the photo's own space (origin top-left, x right, y down);
no real-world depth/scale/pose is estimated. Marker→display grouping lives in
`detector.DEFAULT_DISPLAY_MAPPING`.
