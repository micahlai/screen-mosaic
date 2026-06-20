import cv2
import numpy as np
from flask import Flask, jsonify, Response

app = Flask(__name__)

# Fake master image: three color zones
master = np.zeros((1080, 1920, 3), dtype=np.uint8)
master[:, :640]    = [255, 0, 0]   # left: blue
master[:, 640:1280] = [0, 255, 0]  # middle: green
master[:, 1280:]   = [0, 0, 255]   # right: red

# Fake UV corners for each screen (3 screens side by side)
SCREEN_CONFIGS = {
    0: {"corners_uv": [[0.0,   0.0], [0.333, 0.0], [0.333, 1.0], [0.0,   1.0]]},
    1: {"corners_uv": [[0.333, 0.0], [0.666, 0.0], [0.666, 1.0], [0.333, 1.0]]},
    2: {"corners_uv": [[0.666, 0.0], [1.0,   0.0], [1.0,   1.0], [0.666, 1.0]]},
}

@app.route("/config/<int:screen_id>")
def config(screen_id):
    if screen_id not in SCREEN_CONFIGS:
        return jsonify({"error": "unknown screen"}), 404
    return jsonify(SCREEN_CONFIGS[screen_id])

@app.route("/frame")
def frame():
    _, jpeg = cv2.imencode(".jpg", master)
    return Response(jpeg.tobytes(), mimetype="image/jpeg")

if __name__ == "__main__":
    print("Mock master running at http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001)
