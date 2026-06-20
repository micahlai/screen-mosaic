import cv2
import numpy as np
from flask import Flask, jsonify, Response
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from master.visualization import ParticleFlow

app = Flask(__name__)

W, H = 1920, 1080
sim = ParticleFlow(W, H, num_particles=800)
current_frame = np.zeros((H, W, 3), dtype=np.uint8)
frame_lock = threading.Lock()

# UV corners for each screen — update these once your partner gives real values
SCREEN_CONFIGS = {
    0: {"corners_uv": [[0.0,   0.0], [0.333, 0.0], [0.333, 1.0], [0.0,   1.0]]},
    1: {"corners_uv": [[0.333, 0.0], [0.666, 0.0], [0.666, 1.0], [0.333, 1.0]]},
    2: {"corners_uv": [[0.666, 0.0], [1.0,   0.0], [1.0,   1.0], [0.666, 1.0]]},
}

def simulation_loop():
    global current_frame
    while True:
        sim.step()
        frame = sim.render()
        with frame_lock:
            current_frame = frame

thread = threading.Thread(target=simulation_loop, daemon=True)
thread.start()

@app.route("/config/<int:screen_id>")
def config(screen_id):
    if screen_id not in SCREEN_CONFIGS:
        return jsonify({"error": "unknown screen"}), 404
    return jsonify(SCREEN_CONFIGS[screen_id])

@app.route("/frame")
def frame():
    with frame_lock:
        frame_copy = current_frame.copy()
    _, jpeg = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(jpeg.tobytes(), mimetype="image/jpeg")

if __name__ == "__main__":
    print("Mock master running at http://0.0.0.0:5001")
    print("Screens configured:", list(SCREEN_CONFIGS.keys()))
    app.run(host="0.0.0.0", port=5001, threaded=True)
