"""Tunable constants for the Mosiac host."""

# --- Server ---
PORT = 5003                # HTTP port — everything works here (open this normally)
# The phone's *live camera* (getUserMedia) needs a secure context, so when
# USE_HTTPS is on we ALSO serve HTTPS on HTTPS_PORT with an ad-hoc self-signed
# cert (accept the one-time browser warning). Plain HTTP on PORT keeps working
# for the screens, photo calibration, and everything else.
USE_HTTPS = True
HTTPS_PORT = PORT + 1      # HTTPS port (only needed for the phone live camera)

# --- Screen map ---
# Fraction of the screen-corner bounding box added as margin around the UV map.
UV_MARGIN = 0.0

# --- Fish boids ---
# Radius of the hand "predator" circle the fish flee, as a fraction of the
# rendered field's long side.
FISH_HAND_MARKER_FRAC = 0.015

# --- Marker sizes shown on the screen-slave pages (CSS pixels) ---
MARKER_PX = 200            # normal calibration markers (phase "calibration")
LIVE_MARKER_PX = 90        # smaller markers kept on screen during live calibration

# --- Live calibration ---
# In "live" mode a camera feed continuously re-detects the markers and updates
# each screen's warp in real time. The phone chooses the source:
#   "phone"  — the phone streams its own camera frames to the host (needs HTTPS)
#   "server" — the host opens a local camera at CAMERA_INDEX (cv2.VideoCapture)
LIVE_FPS = 24              # frames per second the feed sends / warps update at
CAMERA_INDEX = 0           # cv2.VideoCapture index for the "server" camera source

# Live-mode detection is pinned to the dictionary the screens actually render
# (skips the multi-dictionary scan) and frames are downscaled for speed. These
# only affect live mode; the one-off phone photo still auto-detects + refines.
LIVE_DICT = "DICT_4X4_50"  # marker dictionary used on the screens
LIVE_MAX_WIDTH = 1280      # downscale live frames to at most this width before detect

# --- Hand tracking ---
# Hand-driven visualizations (e.g. smoke) use the host camera + a YOLOv8-pose
# model to track hands; the hand that's been on screen the longest pushes the
# sim. The camera is assumed to sit where the calibration photo was taken.
HAND_FPS = 12              # hand-tracking / detection rate
HAND_DEVICE = "cpu"        # YOLO device for the PyTorch fallback ("cpu"|"mps"|"cuda")
HAND_CONF = 0.3            # min keypoint confidence to count a wrist as a hand
HAND_IMGSZ = 320           # full-frame inference size — lower = faster (e.g. 256)
HAND_COREML = True         # on macOS, run the pose model on the Neural Engine (CoreML)
HAND_ROI_IMGSZ = 192       # inference size once detection is cropped to the person
HAND_CAM_WIDTH = 960       # camera capture width (lower = lower latency / less work)
HAND_DEBUG = True           # compute + stream the annotated debug view (/hands/debug)
# COCO pose only gives wrists; push the tracked point this fraction of the
# forearm length past the wrist toward the fingers (0 = wrist, ~0.4 = hand).
HAND_FINGER_EXTEND = 0.4
