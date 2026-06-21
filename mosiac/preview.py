"""Desktop preview of the smoke field being cast to the screen slaves.

Shows the full field (the same image the slaves warp), and lets you push/swirl
the smoke with the cursor — the cursor force is sent to the HOST's sim, so the
screens react too. Launched automatically by the host when Smoke is active, or
run manually:

    python mosiac/preview.py [http://HOST:PORT]
"""

import json
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5003"
STREAM = URL + "/content/stream"          # full-field MJPEG (what the slaves warp)
POINTER = URL + "/viz/pointer"
DISPLAY_MAX = 1100                        # window long-side size

_latest = {"frame": None}


def _reader():
    """Read the multipart-MJPEG field stream, keeping the latest frame."""
    while True:
        try:
            with urllib.request.urlopen(STREAM, timeout=5) as r:
                buf = b""
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        start = buf.find(b"--frame")
                        hdr = buf.find(b"\r\n\r\n", start) if start != -1 else -1
                        nxt = buf.find(b"--frame", hdr + 4) if hdr != -1 else -1
                        if nxt == -1:
                            break
                        jpg = buf[hdr + 4:nxt].rstrip(b"\r\n")
                        buf = buf[nxt:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            _latest["frame"] = img
        except Exception:
            time.sleep(0.5)


def main():
    threading.Thread(target=_reader, daemon=True).start()

    view = {"dw": DISPLAY_MAX, "dh": DISPLAY_MAX}
    mouse = {"x": 0.5, "y": 0.5}

    def on_mouse(event, x, y, flags, param):
        mouse["x"] = min(1.0, max(0.0, x / view["dw"]))
        mouse["y"] = min(1.0, max(0.0, y / view["dh"]))

    win = "Smoke preview — drag the cursor to push the smoke (ESC to close)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    prev = (mouse["x"], mouse["y"])
    while True:
        frame = _latest["frame"]
        if frame is not None:
            h, w = frame.shape[:2]
            sc = DISPLAY_MAX / max(w, h)
            view["dw"], view["dh"] = max(1, int(w * sc)), max(1, int(h * sc))
            cv2.imshow(win, cv2.resize(frame, (view["dw"], view["dh"])))
            # cursor velocity = movement since the last frame (normalized)
            vx, vy = mouse["x"] - prev[0], mouse["y"] - prev[1]
            prev = (mouse["x"], mouse["y"])
            try:
                data = json.dumps({"x": mouse["x"], "y": mouse["y"],
                                   "vx": vx, "vy": vy}).encode()
                req = urllib.request.Request(
                    POINTER, data=data, headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=1).read()
            except Exception:
                pass
        if cv2.waitKey(30) & 0xFF == 27:      # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
