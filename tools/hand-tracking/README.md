# Hand-Tracked Interactive Visualizer

Real-time webcam hand tracking (up to two hands) with a custom-rendered,
dark-themed visualization: skeletons, bounding boxes, motion trails,
gesture labels, motion metrics, and a reactive particle system.

> Built against current MediaPipe (0.10.21+), which uses the new **Tasks API**
> (`mediapipe.tasks.python.vision.HandLandmarker`) — Google removed the older
> `mediapipe.solutions.hands` API from recent PyPI releases. If you've seen
> older hand-tracking tutorials using `mp.solutions.hands`, that's why this
> code looks a little different.

## Architecture

```
Camera Manager (camera_manager.py)
    -> MediaPipe Tracker (hand_tracker.py)
    -> Hand State Processor (hand_state.py)
    -> Gesture Recognizer (gestures.py)
    -> Particle System (particles.py)
    -> Renderer (renderer.py)
```

`main.py` wires the pipeline together each frame. `settings.py` holds every
tunable parameter (camera resolution, smoothing, thresholds, colors, particle
behavior, etc.) so you can tweak behavior without touching logic code.

| File | Responsibility |
|---|---|
| `camera_manager.py` | Opens the webcam, reads/mirrors frames |
| `hand_tracker.py` | Runs MediaPipe Hands, extracts 21 landmarks + bbox + handedness |
| `hand_state.py` | Smooths landmarks, computes velocity/acceleration/orientation/openness/pinch distance, keeps fingertip trails |
| `gestures.py` | Classifies Open Palm / Fist / Pinch / Pointing / Peace Sign |
| `particles.py` | Spawns/simulates particles from fingertip motion; pinch attracts, open palm repels |
| `renderer.py` | Draws the pygame visualization window |
| `main.py` | Orchestrates the pipeline, also shows an annotated raw camera window |

## Setup

Requires **Python 3.11+** and a webcam.

```bash
cd hand_visualizer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on MediaPipe compatibility:** MediaPipe's prebuilt wheels lag behind
> the newest CPython releases. If `pip install mediapipe` fails on your
> Python version, install Python 3.11 or 3.12 specifically for this project
> (e.g. via `pyenv` or a dedicated venv), or check
> https://pypi.org/project/mediapipe/ for the latest supported versions.

## Run

```bash
python main.py
```

> **First run:** the app automatically downloads the MediaPipe hand-landmark
> model bundle (a few MB) to `models/hand_landmarker.task` the first time you
> launch it. This requires an internet connection once; after that it's
> cached locally and runs fully offline. If your environment has no network
> access, manually download the file from:
> `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
> and place it at `hand_visualizer/models/hand_landmarker.task`.

Two windows open:

1. **Camera Feed** — the raw webcam feed annotated with bounding boxes,
   landmarks, connections, and Left/Right labels.
2. **Hand Visualizer** — the custom dark-background visualization with
   skeletons, fingertip trails, particles, gesture labels, and a live HUD
   showing speed, acceleration, openness, pinch distance, orientation angle,
   and inter-hand distance.

Press **ESC** or **Q**, or close either window, to quit.

## Tuning performance

If FPS drops below ~30 on your machine, try in `settings.py`:

- Lower `CameraSettings.width` / `height` (e.g. 480x360)
- Set `AppSettings.show_camera_window = False` to skip the extra OpenCV window
- Reduce `ParticleSettings.max_particles`
- Lower `AppSettings.max_render_fps` slightly if your CPU is the bottleneck

## Gestures recognized

| Gesture | Trigger |
|---|---|
| Open Palm | High openness score, 4+ fingers extended |
| Fist | Low openness score, 0-1 fingers extended |
| Pinch | Thumb and index fingertip very close together |
| Pointing | Only the index finger extended |
| Peace Sign | Index + middle fingers extended, ring/pinky curled |

## Customization quick-reference (`settings.py`)

- `CameraSettings` — device index, resolution, mirroring, target FPS
- `TrackerSettings` — max hands, detection/tracking confidence, model complexity
- `MotionSettings` — landmark smoothing strength, trail length, history depth
- `GestureSettings` — thresholds for pinch/fist/open-palm classification
- `ParticleSettings` — spawn rate, lifetime, attraction/repulsion strength, drag
- `RenderSettings` — window size, colors, fonts, HUD toggle
