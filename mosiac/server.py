"""
Calibration + mapping backend.

Run this one file to host TWO web apps from the same backend computer.

PHASES (global, switched from the phone app):
  - "calibration": each screen shows four ArUco markers flush in its far
    corners (no margin, using the entire screen). The phone captures all
    screens; missing corners are highlighted back on the slave.
  - "mapping": each screen renders a UV map (x -> red channel, y -> green
    channel) that is *projectively* warped using the screen's photographed
    corners. A tilted/skewed screen gets a correspondingly skewed UV map, so
    that — viewed from where the phone photo was taken — all screens together
    show one continuous, undistorted UV map.

ROUTES:
  GET  /display          screen-slave page (auto-claims the next display slot)
  GET  /phone            phone capture + phase control
  POST /register         claim/lookup a display slot for a browser
  GET  /status/<id>      per-display poll (phase, capture state, corners)
  GET  /phase            current phase
  POST /phase            set phase ("calibration" | "mapping")
  POST /calibrate        receive a photo, detect, store corners, save debug JSON
  POST /reset            clear the display registry + phase
  GET  /marker/<id>.png  rendered ArUco marker

  - "particles": a live particle-flow animation (visualization.ParticleFlow),
    rendered server-side at a resolution matching the screens' bounding-box
    orientation and streamed to every slave, warped per screen like the UV map.

Run the host (both web apps):
    python mosiac
"""

from __future__ import annotations

import datetime
import io
import json
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request, send_file

try:                       # `python -m mosiac` / imported as a package
    from . import detector
    from .visualization import ParticleFlow
except ImportError:        # `python mosiac` (directory on sys.path)
    import detector
    from visualization import ParticleFlow

MARKER_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKERS_PER_DISPLAY = 4

DEBUG_DIR = Path(__file__).parent / "calibration_debug"
DEBUG_DIR.mkdir(exist_ok=True)

CORNER_LABELS = {"top_left": "Top-Left", "top_right": "Top-Right",
                 "bottom_right": "Bottom-Right", "bottom_left": "Bottom-Left"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# ---------------------------------------------------------------------------
# Global state: dynamic display registry + current phase (in-memory)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_displays: "dict[str, dict]" = {}   # client_id -> display record
_seq = 0
_phase = "mapping"                  # mapping by default; slaves wait for a photo
_uv_bounds = None                   # global UV domain = bbox of all screen corners

# Fraction of the screen-corner bounding box added as margin around the UV map.
UV_MARGIN = 0.05

# Optional content mapped onto the UV space instead of the color gradient.
_content_kind = None     # None | "image" | "particles"
_content_bytes = None
_content_mime = "image/png"
_content_version = 0
_content_mode = "fill"   # "fill" = stretch to UV box | "fit" = preserve aspect

# Live particle-flow visualization, sized to the UV bounding-box orientation.
PARTICLE_MAX_SIDE = 960
_particles = None
_particle_frame = None   # latest rendered JPEG bytes
_particle_size = None    # (w, h) the sim is currently running at
_particle_started = False


def _register(client_id: str) -> dict:
    global _seq
    with _lock:
        if client_id in _displays:
            return _displays[client_id]
        _seq += 1
        base = (_seq - 1) * MARKERS_PER_DISPLAY
        marker_ids = [base + i for i in range(MARKERS_PER_DISPLAY)]
        record = {
            "display_id": f"display_{_seq}",
            "index": _seq,
            "client_id": client_id,
            "marker_ids": marker_ids,
            "slots": [{"slot": s, "marker_id": m}
                      for s, m in zip(detector.CORNER_SLOTS, marker_ids)],
            "capture": None,
        }
        _displays[client_id] = record
        return record


def _display_by_id(display_id: str):
    with _lock:
        for d in _displays.values():
            if d["display_id"] == display_id:
                return d
    return None


# ---------------------------------------------------------------------------
# Marker image endpoint  (small quiet zone so markers can sit flush in corners)
# ---------------------------------------------------------------------------

@app.get("/marker/<int:marker_id>.png")
def marker_png(marker_id: int):
    size = 400
    img = cv2.aruco.generateImageMarker(MARKER_DICT, marker_id, size)
    border = size // 12
    img = cv2.copyMakeBorder(img, border, border, border, border,
                             cv2.BORDER_CONSTANT, value=255)
    ok, buf = cv2.imencode(".png", img)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


# ---------------------------------------------------------------------------
# Screen-slave page (calibration markers + mapping UV warp)
# ---------------------------------------------------------------------------

DISPLAY_PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Calibration</title>
<style>
  html, body { margin: 0; height: 100%; background: #fff; overflow: hidden;
               font-family: -apple-system, system-ui, sans-serif; }
  /* markers flush in the far corners, no margin */
  .corner { position: fixed; display: flex; align-items: center; gap: 8px; padding: 0; }
  .corner img { width: 150px; height: 150px; image-rendering: pixelated; display: block; }
  .dbg { font: 600 14px/1.3 monospace; color: #000; background: #ffeb3b;
         padding: 4px 8px; border-radius: 4px; white-space: nowrap; }
  .tl { top: 0; left: 0; flex-direction: row; }
  .tr { top: 0; right: 0; flex-direction: row-reverse; }
  .br { bottom: 0; right: 0; flex-direction: row-reverse; }
  .bl { bottom: 0; left: 0; flex-direction: row; }
  .corner.missing img { box-shadow: inset 0 0 0 6px #ff3b30; animation: pulse 1s infinite; }
  .corner.present img { box-shadow: inset 0 0 0 6px #34c759; }
  .corner.missing .dbg { background: #ff3b30; color: #fff; }
  .corner.present .dbg { background: #34c759; color: #fff; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .banner { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
            text-align: center; font: 600 18px system-ui; color: #555; max-width: 70%; }
  .banner b { font-size: 28px; color: #111; display: block; margin-bottom: 6px; }
  .incomplete { position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
                background: #ff3b30; color: #fff; font: 600 16px system-ui;
                padding: 12px 18px; border-radius: 10px; display: none;
                text-align: center; max-width: 80%; z-index: 5; }
  /* mapping layer */
  #uv { position: fixed; inset: 0; overflow: hidden; background: #000; display: none; }
  #uvquad { position: absolute; left: 0; top: 0; transform-origin: 0 0;
            background-size: 100% 100%; image-rendering: auto; background: #000; }
  #uvimg { position: absolute; display: none; }
  #uvmsg { position: fixed; inset: 0; display: none; align-items: center;
           justify-content: center; text-align: center; background: #111;
           color: #fff; font: 600 18px system-ui; padding: 24px; }
</style>
</head>
<body>
  <div id="incomplete" class="incomplete">
    This screen is incomplete and did not fully make the photo
  </div>
  <div id="cal">
    <div class="banner"><b id="title">registering…</b>
      <span id="subtitle">Open the phone app and capture this screen</span></div>
    <div class="corner tl" data-slot="top_left"></div>
    <div class="corner tr" data-slot="top_right"></div>
    <div class="corner br" data-slot="bottom_right"></div>
    <div class="corner bl" data-slot="bottom_left"></div>
  </div>
  <div id="uv"><div id="uvquad"><img id="uvimg" alt=""></div></div>
  <div id="uvmsg">This screen was not fully captured.<br>Go back to calibration and recapture.</div>

<script>
const S = 1000;                         // virtual size of the UV source quad
const ORDER = ["top_left","top_right","bottom_right","bottom_left"];
const LABELS = {top_left:"Top-Left", top_right:"Top-Right",
                bottom_right:"Bottom-Right", bottom_left:"Bottom-Left"};
let DISPLAY_ID = null, LAST = null, GRAD = null;

function clientId() {
  let id = localStorage.getItem("calib_client_id");
  if (!id) { id = "c-" + Math.random().toString(36).slice(2) + "-" + Date.now();
             localStorage.setItem("calib_client_id", id); }
  return id;
}

// 2x2 UV gradient (R=x, G=y), stretched smoothly by the browser.
function uvDataURL() {
  const cv = document.createElement("canvas"); cv.width = 2; cv.height = 2;
  const ctx = cv.getContext("2d"); const d = ctx.createImageData(2,2);
  const px = [[0,0],[255,0],[0,255],[255,255]];  // TL,TR,BL,BR -> (R=x,G=y)
  for (let i=0;i<4;i++){ d.data[i*4]=px[i][0]; d.data[i*4+1]=px[i][1];
                         d.data[i*4+2]=0; d.data[i*4+3]=255; }
  ctx.putImageData(d,0,0); return cv.toDataURL();
}

// Solve an 8x8 linear system (Gaussian elimination with partial pivoting).
function solve8(A, b) {
  const n = 8;
  for (let c=0;c<n;c++){
    let p=c; for(let r=c+1;r<n;r++) if(Math.abs(A[r][c])>Math.abs(A[p][c])) p=r;
    [A[c],A[p]]=[A[p],A[c]]; [b[c],b[p]]=[b[p],b[c]];
    for(let r=0;r<n;r++){ if(r===c) continue;
      const f=A[r][c]/A[c][c];
      for(let k=c;k<n;k++) A[r][k]-=f*A[c][k];
      b[r]-=f*b[c];
    }
  }
  return b.map((v,i)=>v/A[i][i]);
}

// Homography mapping src[4] -> dst[4]; returns [[a,b,c],[d,e,f],[g,h,1]].
function homography(src, dst){
  const A=[], b=[];
  for(let i=0;i<4;i++){
    const [x,y]=src[i], [X,Y]=dst[i];
    A.push([x,y,1,0,0,0,-X*x,-X*y]); b.push(X);
    A.push([0,0,0,x,y,1,-Y*x,-Y*y]); b.push(Y);
  }
  const h=solve8(A,b);
  return [[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],1]];
}

async function register(){
  const res = await fetch("/register",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({client_id: clientId()})});
  const d = await res.json();
  DISPLAY_ID = d.display_id;
  document.getElementById("title").textContent = d.display_id;
  for (const {slot, marker_id} of d.slots){
    const el = document.querySelector(`.corner[data-slot="${slot}"]`);
    el.innerHTML = `<img src="/marker/${marker_id}.png" alt="marker ${marker_id}">` +
                   `<span class="dbg">${LABELS[slot]} · ID ${marker_id}</span>`;
  }
  GRAD = uvDataURL();
  document.getElementById("uvimg").addEventListener("load", renderUV);
  poll(); setInterval(poll, 1000);
  window.addEventListener("resize", renderUV);
}

async function poll(){
  if (!DISPLAY_ID) return;
  const s = await (await fetch(`/status/${DISPLAY_ID}`)).json();
  LAST = s;
  if (s.phase === "mapping") showMapping(s); else showCalibration(s);
}

function showCalibration(s){
  document.getElementById("cal").style.display = "block";
  document.getElementById("uv").style.display = "none";
  document.getElementById("uvmsg").style.display = "none";
  document.querySelectorAll(".corner").forEach(c=>c.classList.remove("missing","present"));
  const inc = document.getElementById("incomplete");
  if (!s.captured){ inc.style.display="none"; return; }
  for (const [slot, info] of Object.entries(s.corners)){
    document.querySelector(`.corner[data-slot="${slot}"]`)
            .classList.add(info.present ? "present" : "missing");
  }
  inc.style.display = s.complete ? "none" : "block";
}

function showMapping(s){
  document.getElementById("cal").style.display = "none";
  document.getElementById("incomplete").style.display = "none";
  const uv = document.getElementById("uv"), msg = document.getElementById("uvmsg");
  if (!s.captured){
    uv.style.display = "none"; msg.style.display = "flex";
    msg.innerHTML = "Waiting for phone picture…";
    return;
  }
  if (!(s.complete && s.corner_points)){
    uv.style.display = "none"; msg.style.display = "flex";
    msg.innerHTML = "This screen was not fully captured.<br>Take the photo again.";
    return;
  }
  msg.style.display = "none"; uv.style.display = "block";
  renderUV();
}

function renderUV(){
  if (!(LAST && LAST.phase==="mapping" && LAST.complete && LAST.corner_points)) return;
  const W = window.innerWidth, H = window.innerHeight;
  // UV domain = bounding box of all screens' corners (+ margin), shared by every
  // slave so the extreme top-right screen point — not the photo corner — is red.
  const b = LAST.uv_bounds ||
    {min_x:0, min_y:0, max_x:LAST.image_size.width, max_y:LAST.image_size.height};
  const bw = b.max_x - b.min_x, bh = b.max_y - b.min_y;
  const src = ORDER.map(slot => {
    const [px,py] = LAST.corner_points[slot];
    return [(px - b.min_x)/bw*S, (py - b.min_y)/bh*S];
  });
  const dst = [[0,0],[W,0],[W,H],[0,H]];      // the physical screen rectangle
  const m = homography(src, dst);             // photo(UV space) -> screen
  const q = document.getElementById("uvquad");
  q.style.width = S+"px"; q.style.height = S+"px";
  // CSS matrix3d (column-major) for the 2D homography with z=0
  q.style.transform =
    `matrix3d(${m[0][0]},${m[1][0]},0,${m[2][0]},` +
    `${m[0][1]},${m[1][1]},0,${m[2][1]},` +
    `0,0,1,0,` +
    `${m[0][2]},${m[1][2]},0,${m[2][2]})`;

  // Fill the UV box with either an uploaded image or the color gradient.
  const img = document.getElementById("uvimg");
  if (LAST.content && LAST.content.url){
    q.style.backgroundImage = "none";
    if (img.dataset.url !== LAST.content.url){
      img.dataset.url = LAST.content.url; img.src = LAST.content.url; // load -> renderUV
    }
    img.style.display = "block";
    if (LAST.content.mode === "fit" && img.naturalWidth){
      // preserve the image aspect inside the (possibly non-square) UV box
      const sc = Math.min(bw/img.naturalWidth, bh/img.naturalHeight);
      const wf = img.naturalWidth*sc/bw, hf = img.naturalHeight*sc/bh;
      img.style.left = (1-wf)/2*S+"px"; img.style.top = (1-hf)/2*S+"px";
      img.style.width = wf*S+"px";      img.style.height = hf*S+"px";
    } else {                              // "fill": stretch across the whole UV box
      img.style.left = "0px"; img.style.top = "0px";
      img.style.width = S+"px"; img.style.height = S+"px";
    }
  } else {
    img.style.display = "none";
    q.style.backgroundImage = `url(${GRAD})`;
  }
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
    rec = _register(client_id)
    return jsonify({"display_id": rec["display_id"], "slots": rec["slots"]})


@app.get("/status/<display_id>")
def status(display_id: str):
    d = _display_by_id(display_id)
    if d is None:
        return jsonify({"error": "unknown display"}), 404
    with _lock:
        phase = _phase
        bounds = _uv_bounds
        content = _content_descriptor()
    cap = d["capture"]
    if cap is None:
        return jsonify({"captured": False, "phase": phase,
                        "uv_bounds": bounds, "content": content})
    return jsonify({"captured": True, "phase": phase,
                    "uv_bounds": bounds, "content": content, **cap})


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

@app.get("/phase")
def get_phase():
    with _lock:
        return jsonify({"phase": _phase})


@app.post("/phase")
def set_phase():
    global _phase
    body = request.get_json(silent=True) or {}
    phase = body.get("phase")
    if phase not in ("calibration", "mapping"):
        return jsonify({"error": "phase must be 'calibration' or 'mapping'"}), 400
    with _lock:
        _phase = phase
    return jsonify({"phase": phase})


@app.post("/reset")
def reset():
    global _seq, _phase, _uv_bounds, _content_kind, _content_bytes, _content_version
    with _lock:
        _displays.clear()
        _seq = 0
        _phase = "mapping"
        _uv_bounds = None
        _content_kind = None
        _content_bytes = None
        _content_version = 0
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Content mapped onto the UV space: uploaded image OR live particle flow
# ---------------------------------------------------------------------------

def _content_descriptor():
    """Build the `content` object for /status. Caller holds _lock."""
    if _content_kind == "image" and _content_bytes is not None:
        return {"kind": "image", "url": f"/content/image?v={_content_version}",
                "mode": _content_mode}
    if _content_kind == "particles":
        return {"kind": "particles", "url": "/content/stream", "mode": _content_mode}
    return None


@app.post("/content")
def upload_content():
    global _content_kind, _content_bytes, _content_mime, _content_version, _content_mode
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    mode = request.form.get("mode", "fill")
    if mode not in ("fill", "fit"):
        mode = "fill"
    with _lock:
        _content_kind = "image"
        _content_bytes = file.read()
        _content_mime = file.mimetype or "image/png"
        _content_mode = mode
        _content_version += 1
        version = _content_version
    return jsonify({"ok": True, "kind": "image", "version": version, "mode": mode})


@app.post("/content/particles")
def start_particles():
    """Switch the mapped content to the live particle-flow visualization."""
    global _content_kind, _content_mode
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "fill")
    if mode not in ("fill", "fit"):
        mode = "fill"
    with _lock:
        _content_kind = "particles"
        _content_mode = mode
    _ensure_particle_thread()
    return jsonify({"ok": True, "kind": "particles", "mode": mode})


@app.post("/content/clear")
def clear_content():
    """Drop any mapped content; screens fall back to the UV color gradient."""
    global _content_kind
    with _lock:
        _content_kind = None
    return jsonify({"ok": True, "kind": None})


@app.get("/content/stream")
def content_stream():
    """MJPEG stream of the particle flow; an <img> pointed here auto-updates."""
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with _lock:
                frame = _particle_frame
            if frame is None:
                time.sleep(0.03)
                continue
            yield boundary + frame + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _desired_particle_size():
    """Resolution matching the screens' bounding-box orientation/aspect."""
    b = _uv_bounds
    if not b:
        return (PARTICLE_MAX_SIDE, PARTICLE_MAX_SIDE * 9 // 16)
    bw, bh = b["max_x"] - b["min_x"], b["max_y"] - b["min_y"]
    if bw >= bh:
        return (PARTICLE_MAX_SIDE, max(1, round(PARTICLE_MAX_SIDE * bh / bw)))
    return (max(1, round(PARTICLE_MAX_SIDE * bw / bh)), PARTICLE_MAX_SIDE)


def _particle_loop():
    global _particles, _particle_frame, _particle_size
    while True:
        if _content_kind != "particles":
            time.sleep(0.1)
            continue
        size = _desired_particle_size()
        if _particles is None or _particle_size != size:
            # scale particle count with area so density stays consistent
            n = max(200, min(1200, size[0] * size[1] // 1200))
            _particles = ParticleFlow(size[0], size[1], num_particles=n)
            _particle_size = size
        _particles.step()
        frame = _particles.render()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _particle_frame = buf.tobytes()
        time.sleep(0.005)  # render is the limiter at high resolution


def _ensure_particle_thread():
    global _particle_started
    with _lock:
        if _particle_started:
            return
        _particle_started = True
    threading.Thread(target=_particle_loop, daemon=True).start()


@app.post("/content/mode")
def set_content_mode():
    global _content_mode
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if mode not in ("fill", "fit"):
        return jsonify({"error": "mode must be 'fill' or 'fit'"}), 400
    with _lock:
        _content_mode = mode
    return jsonify({"mode": mode})


@app.get("/content/image")
def content_image():
    with _lock:
        data, mime = _content_bytes, _content_mime
    if data is None:
        return jsonify({"error": "no content image"}), 404
    return send_file(io.BytesIO(data), mimetype=mime)


# ---------------------------------------------------------------------------
# Phone capture + phase control app
# ---------------------------------------------------------------------------

PHONE_PAGE = r"""
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
  .btn.green { background: #34c759; } .btn.grey { background: #8e8e93; }
  .phase { margin-top: 18px; padding: 12px 14px; border-radius: 12px;
           background: #f2f2f7; font-size: 14px; }
  .phase b { text-transform: uppercase; letter-spacing: .04em; }
  .stage { position: relative; margin-top: 16px; display: none; }
  .stage img { width: 100%; border-radius: 12px; display: block; }
  .stage canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  pre { background: #f2f2f7; color: #111; border-radius: 12px; padding: 14px;
        overflow-x: auto; font-size: 12px; margin-top: 16px; }
  .status { margin-top: 14px; font-size: 14px; }
  .ok { color: #34c759; } .err { color: #ff3b30; }
  hr { border: none; border-top: 1px solid #ddd; margin: 22px 0; }
  @media (prefers-color-scheme: dark) {
    pre,.phase { background: #1c1c1e; color: #eee; } .btn.alt { background: #2c2c2e; color: #eee; } }
</style>
</head>
<body>
  <h1>Screen Calibration</h1>
  <p class="sub">Frame all screens so every corner marker is visible, then capture.</p>

  <input id="camera" type="file" accept="image/*" capture="environment">
  <input id="library" type="file" accept="image/*">
  <label class="btn" for="camera">📷 Take Photo</label>
  <label class="btn alt" for="library">Upload from Library</label>

  <div class="stage" id="stage">
    <img id="preview" alt="preview"><canvas id="overlay"></canvas>
  </div>
  <div id="status" class="status"></div>
  <details id="debug" style="display:none; margin-top:14px;">
    <summary style="cursor:pointer; font-weight:600; font-size:14px;">Debug information</summary>
    <pre id="json"></pre>
  </details>

  <hr>
  <div class="phase">Current phase: <b id="phaseName">…</b></div>
  <button class="btn green" id="toMapping">🗺️ Show UV Map (mapping)</button>
  <button class="btn grey" id="toCalib">🔳 Show ArUco markers (calibration)</button>

  <hr>
  <h2 style="font-size:17px; margin-bottom:0">Map an image to the screens</h2>
  <p class="sub">Warped per screen so it looks straight from the camera's position.</p>
  <input id="content" type="file" accept="image/*">
  <label class="btn" for="content">🖼️ Upload Image to Map</label>
  <button class="btn" id="particles" style="background:#af52de">✨ Show Particles</button>
  <button class="btn grey" id="clearContent">Clear (UV gradient)</button>
  <div style="margin-top:12px; font-size:15px;">
    <label><input type="radio" name="cmode" value="fill" checked> Fill (stretch)</label>
    &nbsp;&nbsp;&nbsp;
    <label><input type="radio" name="cmode" value="fit"> Fit (keep aspect)</label>
  </div>
  <div id="cstatus" class="status"></div>

<script>
const preview=document.getElementById('preview'), overlay=document.getElementById('overlay'),
      stage=document.getElementById('stage'), statusEl=document.getElementById('status'),
      out=document.getElementById('json'), debug=document.getElementById('debug'),
      phaseName=document.getElementById('phaseName');
const ORDER=['top_left','top_right','bottom_right','bottom_left'];

function handle(input){ input.addEventListener('change',()=>{ const f=input.files[0]; if(f) send(f);}); }
handle(document.getElementById('camera')); handle(document.getElementById('library'));

// Tapping a capture button opens the ArUco markers on every screen so the photo
// captures them; we revert to the UV map after a successful capture.
['camera','library'].forEach(id=>
  document.querySelector(`label[for="${id}"]`).addEventListener('click',()=>setPhase('calibration')));

async function refreshPhase(){
  const {phase}=await (await fetch('/phase')).json();
  phaseName.textContent=phase;
}
async function setPhase(p){
  await fetch('/phase',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({phase:p})});
  refreshPhase();
}
document.getElementById('toMapping').onclick=()=>setPhase('mapping');
document.getElementById('toCalib').onclick=()=>setPhase('calibration');

// --- map an image onto the UV space ---
const contentInput=document.getElementById('content'), cstatus=document.getElementById('cstatus');
const currentMode=()=>document.querySelector('input[name=cmode]:checked').value;
contentInput.addEventListener('change', async ()=>{
  const f=contentInput.files[0]; if(!f) return;
  const fd=new FormData(); fd.append('file',f); fd.append('mode',currentMode());
  cstatus.textContent='Uploading…'; cstatus.className='status';
  try{
    const res=await fetch('/content',{method:'POST',body:fd});
    const d=await res.json(); if(!res.ok) throw new Error(d.error||'Failed');
    setPhase('mapping');
    cstatus.innerHTML=`<span class="ok">✓ Mapped image (${d.mode})</span>`;
  }catch(e){ cstatus.innerHTML=`<span class="err">${e.message}</span>`; }
});
document.querySelectorAll('input[name=cmode]').forEach(r=>r.addEventListener('change', async ()=>{
  await fetch('/content/mode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:currentMode()})});
  cstatus.innerHTML=`<span class="ok">Mode: ${currentMode()}</span>`;
}));
document.getElementById('particles').addEventListener('click', async ()=>{
  cstatus.textContent='Starting particles…'; cstatus.className='status';
  try{
    const res=await fetch('/content/particles',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode:currentMode()})});
    if(!res.ok) throw new Error('Failed');
    setPhase('mapping');
    cstatus.innerHTML='<span class="ok">✨ Particles mapped to screens</span>';
  }catch(e){ cstatus.innerHTML=`<span class="err">${e.message}</span>`; }
});
document.getElementById('clearContent').addEventListener('click', async ()=>{
  await fetch('/content/clear',{method:'POST'});
  cstatus.innerHTML='<span class="ok">Cleared — showing UV gradient</span>';
});

async function send(file){
  statusEl.textContent='Detecting markers…'; statusEl.className='status';
  await new Promise(r=>{ preview.onload=r; preview.src=URL.createObjectURL(file); });
  stage.style.display='block';
  const fd=new FormData(); fd.append('file',file);
  try{
    const res=await fetch('/calibrate',{method:'POST',body:fd});
    const data=await res.json();
    if(!res.ok) throw new Error(data.error||'Failed');
    draw(data);
    const n=data.result.displays.filter(d=>d.complete).length;
    const bad=data.result.displays.filter(d=>!d.complete).length;
    statusEl.innerHTML=`<span class="ok">✓ Saved ${data.saved_as}</span> — `+
      `${data.result.marker_count} markers, ${n} complete`+
      (bad?`, <span class="err">${bad} incomplete</span>`:'');
    debug.style.display='block'; out.textContent=JSON.stringify(data.result,null,2);
    setPhase('mapping');   // back to the UV map now that we have corners
  }catch(e){ statusEl.innerHTML=`<span class="err">${e.message}</span>`; }
}

function draw(data){
  const W=data.result.image_size.width, H=data.result.image_size.height;
  overlay.width=W; overlay.height=H;
  const ctx=overlay.getContext('2d'); ctx.clearRect(0,0,W,H);
  ctx.lineWidth=Math.max(3,W/400); ctx.font=`${Math.max(18,W/60)}px system-ui`;
  for(const d of data.result.displays){
    const pts=d.corner_points||{}; const present=ORDER.filter(s=>pts[s]);
    ctx.strokeStyle=d.complete?'#34c759':'#ff9500'; ctx.fillStyle=ctx.strokeStyle;
    if(d.complete){
      ctx.beginPath();
      ORDER.forEach((s,i)=>{ const [x,y]=pts[s]; i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
      ctx.closePath(); ctx.stroke();
    } else {
      present.forEach(s=>{ const [x,y]=pts[s];
        ctx.beginPath(); ctx.arc(x,y,ctx.lineWidth*2,0,7); ctx.fill(); });
      ctx.fillStyle='#ff3b30'; const lh=Math.max(22,W/55);
      d.missing_corners.forEach((s,i)=> ctx.fillText('✗ missing '+s, 14, lh*(i+1)));
    }
    const anchor=pts[present[0]]||[20,30];
    ctx.fillStyle=d.complete?'#34c759':'#ff3b30';
    ctx.fillText(d.id+(d.complete?'':' (incomplete)'), anchor[0], anchor[1]-10);
  }
}
refreshPhase();
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
                # outer corner = the marker corner touching the real screen corner
                pt = detector._screen_point(by_id[mid], centroid, "outer")
                corner_points[slot] = pt
                corners_norm[slot] = [pt[0] / width, pt[1] / height]
            status_corners[slot] = {"marker_id": mid, "present": ok}

        complete = len(present) == len(slots)
        missing = [s for s, mid in slots if mid not in by_id]

        displays_out.append({
            "id": rec["display_id"], "complete": complete,
            "missing_corners": missing,
            "corners": corners_norm if complete else None,
            "corner_points": corner_points,
        })

        # publish for the slave page to poll (used by both phases)
        rec["capture"] = {
            "complete": complete,
            "corners": status_corners,
            "corner_points": corner_points,
            "image_size": {"width": int(width), "height": int(height)},
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }

    # UV domain = bounding box of every captured screen corner (+ margin), so the
    # UV map spans only the region the screens actually cover, not the whole photo.
    global _uv_bounds
    all_pts = [pt for d in displays_out if d["complete"]
               for pt in d["corner_points"].values()]
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        mx, my = (max_x - min_x) * UV_MARGIN, (max_y - min_y) * UV_MARGIN
        bounds = {"min_x": min_x - mx, "min_y": min_y - my,
                  "max_x": max_x + mx, "max_y": max_y + my}
    else:
        bounds = None
    with _lock:
        _uv_bounds = bounds

    result = {
        "image_size": {"width": int(width), "height": int(height)},
        "dictionary": dict_name, "marker_count": len(markers),
        "uv_bounds": bounds, "displays": displays_out,
    }

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_as = f"calibration-{stamp}.json"
    (DEBUG_DIR / saved_as).write_text(json.dumps(
        {"captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "source_filename": file.filename, "corner_mode": "outer", **result}, indent=2))

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
        f"<li>Phone capture + phase control: <a href='/phone'>/phone</a></li>"
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


def main():
    ip = _lan_ip()
    print("Mosiac host running. Open on the LAN:")
    print(f"  Screen slave : http://{ip}:5001/display")
    print(f"  Phone capture: http://{ip}:5001/phone")
    print(f"Debug captures saved to: {DEBUG_DIR}")
    # debug=False so the particle background thread isn't duplicated by the reloader
    app.run(host="0.0.0.0", port=5001, threaded=True)


if __name__ == "__main__":
    main()
