import cv2
import numpy as np
import requests
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.transform import warp_master_to_slave
from calibration import show_calibration

SCREEN_W, SCREEN_H = 1920, 1080
SCREEN_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

MASTER_URL = "http://127.0.0.1:5001"  # change to your partner's IP later

def get_corners():
    resp = requests.get(f"{MASTER_URL}/config/{SCREEN_ID}")
    return resp.json()["corners_uv"]

def get_frame():
    resp = requests.get(f"{MASTER_URL}/frame")
    img_array = np.frombuffer(resp.content, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# --- Phase 1: Calibration ---
print(f"Screen {SCREEN_ID}: showing calibration markers. Press any key when scanned.")
show_calibration(SCREEN_ID)

# --- Phase 2: Fetch this screen's UV corners from master ---
print(f"Screen {SCREEN_ID}: fetching corner config from master...")
corners_uv = get_corners()
print(f"Screen {SCREEN_ID}: got corners: {corners_uv}")

# --- Phase 3: Display loop ---
window_name = f"Slave {SCREEN_ID}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print(f"Screen {SCREEN_ID}: entering display loop. Press Q to quit.")
while True:
    master_frame = get_frame()
    slave_frame = warp_master_to_slave(master_frame, corners_uv, SCREEN_W, SCREEN_H)
    cv2.imshow(window_name, slave_frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()