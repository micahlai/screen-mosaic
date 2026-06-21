"""Tunable constants for the Mosiac host."""

# --- Server ---
PORT = 5003                # HTTP port — everything works here (open this normally)
# The phone's *live camera* (getUserMedia) needs a secure context, so when
# USE_HTTPS is on we ALSO serve HTTPS on HTTPS_PORT with an ad-hoc self-signed
# cert (accept the one-time browser warning). Plain HTTP on PORT keeps working
# for the screens, photo calibration, and everything else.
USE_HTTPS = True
HTTPS_PORT = PORT + 1      # HTTPS port (only needed for the phone live camera)

# --- Marker sizes shown on the screen-slave pages (CSS pixels) ---
MARKER_PX = 150            # normal calibration markers (phase "calibration")
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
