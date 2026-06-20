"""
Calibration backend.

Run this one file to host TWO web apps from the same backend computer:

  1. Screen-slave display  ->  GET /display
     Open this on each screen you want to calibrate. The number of displays is
     NOT known up front: each browser that opens /display dynamically claims the
     next display slot. The first becomes display_1 (marker IDs 0,1,2,3), the
     second display_2 (IDs 4,5,6,7), and so on. Each screen renders four ArUco
     markers flush in its corners with debug text naming the corner + ID.

     After a phone capture, if a screen did not fully make the photo, the slave
     page highlights exactly which corner(s) were missed and shows a message.

  2. Phone capture          ->  GET /phone
     Open on a phone. Take a photo (native iOS camera) or upload one capturing
     all on-screen markers. The backend detects them, computes each screen
     corner as the *marker corner touching the real screen corner* (not the
     center), saves a debug JSON file, and the phone draws the resulting
     bounding boxes over the photo.

Only the calibration phase is implemented for now.

Run:
    python calibration_server.py
"""

from __future__ import annotations

import datetime
import io
import json
import socket
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

import detector

# Markers generated/served here (DICT_4X4_50 supports up to 12 displays).
MARKER_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKERS_PER_DISPLAY = 4

DEBUG_DIR = Path(__file__).parent / "calibration_debug"
DEBUG_DIR.mkdir(exist_ok=True)

CORNER_LABELS = {
    "top_left": "Top-Left",
    "top_right": "Top-Right",
    "bottom_right": "Bottom-Right",
    "bottom_left": "Bottom-Left",
}
SLOT_CLASS = {"top_left": "tl", "top_right": "tr",
              "bottom_right": "br", "bottom_left": "bl"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ---------------------------------------------------------------------------
# Dynamic display registry (in-memory; reset on restart or via /reset)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
# client_id (browser-persisted) -> display record
_displays: "dict[str, dict]" = {}
_seq = 0


def _register(client_id: str) -> dict:
    """Return the display record for client_id, creating the next one if new."""
    global _seq
    with _lock:
        if client_id in _displays:
            return _displays[client_id]
        _seq += 1
        index = _seq
        base = (index - 1) * MARKERS_PER_DISPLAY
        marker_ids = [base + i for i in range(MARKERS_PER_DISPLAY)]
        record = {
            "display_id": f"display_{index}",
            "index": index,
            "client_id": client_id,
            "marker_ids": marker_ids,
            "slots": [
                {"slot": s, "marker_id": m}
                for s, m in zip(detector.CORNER_SLOTS, marker_ids)
            ],
            "status": None,  # filled in after a capture
        }
        _displays[client_id] = record
        return record


def _current_mapping() -> "dict[str, list[int]]":
    with _lock:
        return {d["display_id"]: list(d["marker_ids"]) for d in _displays.values()}


def _display_by_id(display_id: str):
    with _lock:
        for d in _displays.values():
            if d["display_id"] == display_id:
                return d
    return None


# ---------------------------------------------------------------------------
# Marker image endpoint
# ---------------------------------------------------------------------------

@app.get("/marker/<int:marker_id>.png")
def marker_png(marker_id: int):
    size = 400
    img = cv2.aruco.generateImageMarker(MARKER_DICT, marker_id, size)
    border = size // 8
    img = cv2.copyMakeBorder(img, border, border, border, border,
                             cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", img)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


# ---------------------------------------------------------------------------
# 1) Screen-slave calibration display
# ---------------------------------------------------------------------------

DISPLAY_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Calibration</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden;
               font-family: -apple-system, system-ui, sans-serif; }
  .corner { position: fixed; display: flex; align-items: center; gap: 10px;
            padding: 6px; border: 4px solid transparent; border-radius: 8px; }
  .corner img { width: 160px; height: 160px; image-rendering: pixelated; display: block; }
  .dbg { font: 600 14px/1.3 monospace; color: #000; background: #ffeb3b;
         padding: 4px 8px; border-radius: 4px; white-space: nowrap; }
  .tl { top: 0; left: 0; flex-direction: row; }
  .tr { top: 0; right: 0; flex-direction: row-reverse; }
  .br { bottom: 0; right: 0; flex-direction: row-reverse; }
  .bl { bottom: 0; left: 0; flex-direction: row; }
  /* highlight a corner that didn't make the photo */
  .corner.missing { border-color: #ff3b30; animation: pulse 1s infinite; }
  .corner.missing .dbg { background: #ff3b30; color: #fff; }
  .corner.present .dbg { background: #34c759; color: #fff; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
  .banner { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
            text-align: center; font: 600 18px system-ui; color: #555; max-width: 70%; }
  .banner b { font-size: 28px; color: #111; display: block; margin-bottom: 6px; }
  .incomplete { position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
                background: #ff3b30; color: #fff; font: 600 16px system-ui;
                padding: 12px 18px; border-radius: 10px; display: none;
                text-align: center; max-width: 80%; }
</style>
</head>
<body>
  <div id="incomplete" class="incomplete">
    This screen is incomplete and did not fully make the photo
  </div>
  <div class="banner">
    <b id="title">registering…</b>
    <span id="subtitle">Open the phone app and capture this screen</span>
  </div>
  <!-- four corner slots, populated after registration -->
  <div class="corner tl" data-slot="top_left"></div>
  <div class="corner tr" data-slot="top_right"></div>
  <div class="corner br" data-slot="bottom_right"></div>
  <div class="corner bl" data-slot="bottom_left"></div>

<script>
const LABELS = {top_left:"Top-Left", top_right:"Top-Right",
                bottom_right:"Bottom-Right", bottom_left:"Bottom-Left"};
let DISPLAY_ID = null;

function clientId() {
  let id = localStorage.getItem("calib_client_id");
  if (!id) { id = "c-" + Math.random().toString(36).slice(2) + "-" + Date.now();
             localStorage.setItem("calib_client_id", id); }
  return id;
}

async function register() {
  const res = await fetch("/register", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({client_id: clientId()})
  });
  const d = await res.json();
  DISPLAY_ID = d.display_id;
  document.getElementById("title").textContent = d.display_id;
  for (const {slot, marker_id} of d.slots) {
    const el = document.querySelector(`.corner[data-slot="${slot}"]`);
    el.innerHTML = `<img src="/marker/${marker_id}.png" alt="marker ${marker_id}">` +
                   `<span class="dbg">${LABELS[slot]} · ID ${marker_id}</span>`;
  }
  poll();
  setInterval(poll, 1500);
}

async function poll() {
  if (!DISPLAY_ID) return;
  const res = await fetch(`/status/${DISPLAY_ID}`);
  const s = await res.json();
  const incomplete = document.getElementById("incomplete");
  document.querySelectorAll(".corner").forEach(c => c.classList.remove("missing","present"));
  if (!s.captured) { incomplete.style.display = "none"; return; }
  for (const [slot, info] of Object.entries(s.corners)) {
    const el = document.querySelector(`.corner[data-slot="${slot}"]`);
    el.classList.add(info.present ? "present" : "missing");
  }
  incomplete.style.display = s.complete ? "none" : "block";
}

register();
</script>
</body>
</html>
"""


@app.get("/display")
def display():
    return render_template_string(DISPLAY_PAGE)


@app.post("/register")
def register():
    body = request.get_json(silent=True) or {}
    client_id = body.get("client_id") or request.remote_addr or "anon"
    record = _register(client_id)
    return jsonify({"display_id": record["display_id"], "slots": record["slots"]})


@app.get("/status/<display_id>")
def status(display_id: str):
    d = _display_by_id(display_id)
    if d is None:
        return jsonify({"error": "unknown display"}), 404
    st = d["status"]
    if st is None:
        return jsonify({"captured": False})
    return jsonify({"captured": True, **st})


@app.post("/reset")
def reset():
    global _seq
    with _lock:
        _displays.clear()
        _seq = 0
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# 2) Phone capture app
# ---------------------------------------------------------------------------

PHONE_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Calibration Capture</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0 auto;
         padding: 20px; max-width: 560px; }
  h1 { font-size: 20px; } p.sub { color: #666; font-size: 14px; margin-top: 4px; }
  input[type=file] { display: none; }
  .btn { display: block; width: 100%; text-align: center; padding: 16px;
         border-radius: 12px; background: #007aff; color: #fff; font-weight: 600;
         font-size: 17px; border: 0; cursor: pointer; margin-top: 12px; }
  .btn.alt { background: #e5e5ea; color: #111; }
  .stage { position: relative; margin-top: 16px; display: none; }
  .stage img { width: 100%; border-radius: 12px; display: block; }
  .stage canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  pre { background: #f2f2f7; color: #111; border-radius: 12px; padding: 14px;
        overflow-x: auto; font-size: 12px; margin-top: 16px; }
  .status { margin-top: 14px; font-size: 14px; }
  .ok { color: #34c759; } .err { color: #ff3b30; }
  @media (prefers-color-scheme: dark) {
    pre { background: #1c1c1e; color: #eee; } .btn.alt { background: #2c2c2e; color: #eee; } }
</style>
</head>
<body>
  <h1>Screen Calibration Capture</h1>
  <p class="sub">Frame all screens so every corner marker is visible, then capture.</p>

  <input id="camera" type="file" accept="image/*" capture="environment">
  <input id="library" type="file" accept="image/*">
  <label class="btn" for="camera">📷 Take Photo</label>
  <label class="btn alt" for="library">Upload from Library</label>

  <div id="stage" class="stage">
    <img id="preview" alt="preview">
    <canvas id="overlay"></canvas>
  </div>
  <div id="status" class="status"></div>
  <pre id="json" style="display:none"></pre>

<script>
const preview = document.getElementById('preview');
const overlay = document.getElementById('overlay');
const stage = document.getElementById('stage');
const statusEl = document.getElementById('status');
const out = document.getElementById('json');

function handle(input) {
  input.addEventListener('change', () => {
    const file = input.files[0];
    if (file) send(file);
  });
}
handle(document.getElementById('camera'));
handle(document.getElementById('library'));

async function send(file) {
  statusEl.textContent = 'Detecting markers…'; statusEl.className = 'status';
  const url = URL.createObjectURL(file);
  await new Promise(r => { preview.onload = r; preview.src = url; });
  stage.style.display = 'block';

  const fd = new FormData(); fd.append('file', file);
  try {
    const res = await fetch('/calibrate', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed');
    draw(data);
    const n = data.result.displays.filter(d => d.complete).length;
    const bad = data.result.displays.filter(d => !d.complete).length;
    statusEl.innerHTML = `<span class="ok">✓ Saved ${data.saved_as}</span> — ` +
      `${data.result.marker_count} markers, ${n} complete` +
      (bad ? `, <span class="err">${bad} incomplete</span>` : '');
    out.style.display = 'block';
    out.textContent = JSON.stringify(data.result, null, 2);
  } catch (e) {
    statusEl.innerHTML = `<span class="err">${e.message}</span>`;
  }
}

function draw(data) {
  const W = data.result.image_size.width, H = data.result.image_size.height;
  overlay.width = W; overlay.height = H;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = Math.max(3, W / 400);
  ctx.font = `${Math.max(18, W/60)}px system-ui`;
  const ORDER = ['top_left','top_right','bottom_right','bottom_left'];

  for (const d of data.result.displays) {
    const pts = d.corner_points || {};        // pixel coords keyed by slot
    const present = ORDER.filter(s => pts[s]);
    ctx.strokeStyle = d.complete ? '#34c759' : '#ff9500';
    ctx.fillStyle = ctx.strokeStyle;
    if (d.complete) {
      ctx.beginPath();
      ORDER.forEach((s,i) => { const [x,y]=pts[s];
        i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
      ctx.closePath(); ctx.stroke();
    } else {
      // draw whatever corners we have; flag the missing ones in red
      present.forEach(s => { const [x,y]=pts[s];
        ctx.beginPath(); ctx.arc(x,y,ctx.lineWidth*2,0,7); ctx.fill(); });
      ctx.fillStyle = '#ff3b30';
      const lh = Math.max(22, W/55);
      d.missing_corners.forEach((s, i) => {
        ctx.fillText('✗ missing ' + s, 14, lh * (i + 1));  // marker absent: no pixel location
      });
    }
    // label near first available corner
    const anchor = pts[present[0]] || [20, 30];
    ctx.fillStyle = d.complete ? '#34c759' : '#ff3b30';
    ctx.fillText(d.id + (d.complete ? '' : ' (incomplete)'), anchor[0], anchor[1] - 10);
  }
}
</script>
</body>
</html>
"""


@app.get("/phone")
def phone():
    return render_template_string(PHONE_PAGE)


# ---------------------------------------------------------------------------
# Calibration capture endpoint
# ---------------------------------------------------------------------------

@app.post("/calibrate")
def calibrate():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    height, width = image.shape[:2]
    markers, dict_name = detector.detect_markers(image)
    by_id = {m["id"]: m for m in markers}

    with _lock:
        records = sorted(_displays.values(), key=lambda d: d["index"])

    displays_out = []
    for rec in records:
        slots = list(zip(detector.CORNER_SLOTS, rec["marker_ids"]))
        present = [by_id[mid] for _, mid in slots if mid in by_id]
        centroid = (np.mean([m["center"] for m in present], axis=0)
                    if present else None)

        corner_points, corners_norm, status_corners = {}, {}, {}
        for slot, mid in slots:
            ok = mid in by_id and centroid is not None
            if ok:
                # outer corner == the marker corner touching the real screen corner
                pt = detector._screen_point(by_id[mid], centroid, "outer")
                corner_points[slot] = pt
                corners_norm[slot] = [pt[0] / width, pt[1] / height]
            status_corners[slot] = {"marker_id": mid, "present": ok}

        complete = len(present) == len(slots)
        missing = [s for s, mid in slots if mid not in by_id]

        displays_out.append({
            "id": rec["display_id"],
            "complete": complete,
            "missing_corners": missing,
            "corners": corners_norm if complete else None,
            "corner_points": corner_points,   # pixel coords for phone overlay
        })

        # publish status for the slave page to poll
        rec["status"] = {
            "complete": complete,
            "corners": status_corners,
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }

    result = {
        "image_size": {"width": int(width), "height": int(height)},
        "dictionary": dict_name,
        "marker_count": len(markers),
        "displays": displays_out,
    }

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_as = f"calibration-{stamp}.json"
    (DEBUG_DIR / saved_as).write_text(json.dumps(
        {"captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "source_filename": file.filename, "corner_mode": "outer", **result},
        indent=2))

    return jsonify({"saved_as": saved_as, "result": result})


# ---------------------------------------------------------------------------
# Landing page + entry point
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    ip = _lan_ip()
    return (
        f"<h2>Calibration backend</h2>"
        f"<ul style='font:16px/1.8 system-ui'>"
        f"<li>Screen slave (open one per screen): <a href='/display'>/display</a></li>"
        f"<li>Phone capture: <a href='/phone'>/phone</a></li>"
        f"<li>Reset display registry: <code>POST /reset</code></li>"
        f"</ul>"
        f"<p style='font:14px system-ui;color:#666'>On the LAN open "
        f"<code>http://{ip}:5001/display</code> on each screen and "
        f"<code>http://{ip}:5001/phone</code> on the phone.</p>"
    )


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ip = _lan_ip()
    print("Calibration backend running. Open on the LAN:")
    print(f"  Screen slave : http://{ip}:5001/display")
    print(f"  Phone capture: http://{ip}:5001/phone")
    print(f"Debug captures saved to: {DEBUG_DIR}")
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
