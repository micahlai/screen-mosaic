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

  - "visualization": a live animation (mosiac/visualizations: particles, smoke,
    ...) rendered server-side at a resolution matching the screens' bounding-box
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
from flask_sock import Sock

try:                       # `python -m mosiac` / imported as a package
    from . import detector
    from . import visualizations
    from . import consts
    from . import hands
except ImportError:        # `python mosiac` (directory on sys.path)
    import detector
    import visualizations
    import consts
    import hands

MARKER_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKERS_PER_DISPLAY = 4

DEBUG_DIR = Path(__file__).parent / "calibration_debug"
DEBUG_DIR.mkdir(exist_ok=True)

CORNER_LABELS = {"top_left": "Top-Left", "top_right": "Top-Right",
                 "bottom_right": "Bottom-Right", "bottom_left": "Bottom-Left"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
sock = Sock(app)

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
_content_kind = None     # None | "image" | "visualization"
_content_bytes = None
_content_mime = "image/png"
_content_version = 0
_content_mode = "fill"   # "fill" = stretch to UV box | "fit" = preserve aspect

# Live visualization (particles / smoke / ...), sized to the UV box orientation.
# We keep the latest *raw* field (numpy BGR) and warp it per-screen on demand, so
# each slave only receives its own region at its own resolution (not the full
# field). A version counter + condition let the per-screen streams send a frame
# only when a new one is rendered.
VIZ_MAX_SIDE = 960
VIZ_SCREEN_MAX = 2560    # cap on a per-screen warped output's long side
_viz_name = "particles"
_viz = None
_viz_raw = None          # latest rendered field as a numpy BGR array
_viz_ver = 0             # bumped each new rendered field
_viz_cond = threading.Condition()
_viz_size = None         # (w, h) the sim is currently running at
_viz_started = False
# Hand-driven input (YOLO tracker -> current hand -> sim force)
_pointer = None          # (u, v, vu, vv, ts) in field coords, fed to the sim
_current_hand = None     # latest current-hand info (for /hands/status)
_hands_started = False
_hands_debug = None      # latest annotated YOLO camera frame (JPEG bytes)

# Live calibration state
_live_source = "phone"   # "phone" (phone streams frames) | "server" (host camera)
_last_live_ts = 0.0      # timestamp of the last frame the phone streamed in
_camera_started = False  # server-camera thread started
_camera_ok = False       # server camera currently producing frames
_camera_err = None       # last server-camera error (for the phone to show)
# Change-notification for WebSocket pushes: bumped whenever live corners change,
# so slave sockets push a new warp only on actual updates (no polling).
_live_cond = threading.Condition()
_live_rev = 0


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
  #markers { position: fixed; inset: 0; z-index: 4; display: none; }
  #markers.nolabels .dbg { display: none; }   /* live mode: markers only, no ID tags */
  .corner img { width: var(--mk, 150px); height: var(--mk, 150px);
                image-rendering: pixelated; display: block; }
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
            background-color: #000; background-size: 100% 100%;
            background-repeat: no-repeat; image-rendering: auto; }
  #uvimg { position: absolute; display: none; }
  /* server-warped per-screen visualization: fills the screen 1:1, no client warp */
  #screenimg { position: absolute; inset: 0; width: 100%; height: 100%;
               object-fit: fill; display: none; }
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
  </div>
  <div id="uv"><div id="uvquad"><img id="uvimg" alt=""></div><img id="screenimg" alt=""></div>
  <!-- markers live in their own layer so they can sit over the content in live mode -->
  <div id="markers">
    <div class="corner tl" data-slot="top_left"></div>
    <div class="corner tr" data-slot="top_right"></div>
    <div class="corner br" data-slot="bottom_right"></div>
    <div class="corner bl" data-slot="bottom_left"></div>
  </div>
  <div id="uvmsg">This screen was not fully captured.<br>Go back to calibration and recapture.</div>

<script>
const S = 1000;                         // virtual size of the UV source quad
const MARKER_PX = {{ MARKER_PX }}, LIVE_MARKER_PX = {{ LIVE_MARKER_PX }}, LIVE_FPS = {{ LIVE_FPS }};
const ORDER = ["top_left","top_right","bottom_right","bottom_left"];
const LABELS = {top_left:"Top-Left", top_right:"Top-Right",
                bottom_right:"Bottom-Right", bottom_left:"Bottom-Left"};
let DISPLAY_ID = null, LAST = null, GRAD = null, GRAD_KEY = null;

function clientId() {
  let id = localStorage.getItem("calib_client_id");
  if (!id) { id = "c-" + Math.random().toString(36).slice(2) + "-" + Date.now();
             localStorage.setItem("calib_client_id", id); }
  return id;
}

// UV gradient (R=x, G=y) with a checkerboard in the BLUE channel. The checker
// cells are squares measured in the ORIGINAL CALIBRATION PHOTO: the UV box spans
// bw x bh photo pixels, so we size cells in photo pixels (different cell counts
// per axis) rather than in the normalized box. Rendered full-res -> no tiling.
const CHECKER_CELLS_LONG = 8;   // checker cells along the longer photo axis
function uvDataURL(bounds) {
  const N = 512;
  const bw = bounds.max_x - bounds.min_x, bh = bounds.max_y - bounds.min_y;
  const cell = Math.max(bw, bh) / CHECKER_CELLS_LONG;   // square cell size in photo px
  const cv = document.createElement("canvas"); cv.width = N; cv.height = N;
  const ctx = cv.getContext("2d"); const d = ctx.createImageData(N, N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const o = (j*N + i)*4;
      // photo-space position of this UV pixel
      const px = bounds.min_x + (i / N) * bw;
      const py = bounds.min_y + (j / N) * bh;
      const cx = Math.floor(px / cell), cy = Math.floor(py / cell);
      d.data[o]   = Math.round(i / (N-1) * 255);           // R = x
      d.data[o+1] = Math.round(j / (N-1) * 255);           // G = y
      d.data[o+2] = ((cx + cy) & 1) ? 255 : 0;             // B = square checkerboard
      d.data[o+3] = 255;
    }
  }
  ctx.putImageData(d, 0, 0); return cv.toDataURL();
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
  document.getElementById("uvimg").addEventListener("load", renderUV);
  poll();   // self-scheduling (faster cadence while live)
  window.addEventListener("resize", renderUV);
}

// In live mode the server PUSHES corner updates over a WebSocket. Rendering is
// driven ONLY by those pushes (which fire only when corners actually change), so
// the warp never flickers — between updates the last frame just stays frozen.
// The 1 s poll only handles phase transitions, not re-rendering.
let cornersWS = null, liveRendered = false;
function wsURL(path){ return (location.protocol==="https:"?"wss:":"ws:")+"//"+location.host+path; }
function openCornersWS(){
  if (cornersWS) return;
  cornersWS = new WebSocket(wsURL("/live/corners/"+DISPLAY_ID));
  cornersWS.onmessage = (ev)=>{ try{ LAST=JSON.parse(ev.data);
    if(LAST.phase==="live"){ liveRendered=true; showLive(LAST); } }catch(e){} };
  cornersWS.onclose = ()=>{ cornersWS=null; };
  cornersWS.onerror = ()=>{ try{cornersWS.close();}catch(e){} };
}
function closeCornersWS(){ if(cornersWS){ try{cornersWS.close();}catch(e){} cornersWS=null; } liveRendered=false; }

async function poll(){
  if (!DISPLAY_ID){ setTimeout(poll, 200); return; }
  try{
    const s = await (await fetch(`/status/${DISPLAY_ID}`)).json();
    LAST = s;
    if (s.phase === "live"){
      openCornersWS();
      if (!liveRendered) showLive(s);   // first paint only; WS drives the rest
    } else {
      closeCornersWS();
      if (s.phase === "mapping") showMapping(s); else showCalibration(s);
    }
  }catch(e){}
  setTimeout(poll, 1000);   // WS drives live warp updates; poll handles phase
}

function setMarkers(show, sizePx){
  document.getElementById("markers").style.display = show ? "block" : "none";
  if (show) document.documentElement.style.setProperty("--mk", sizePx + "px");
}

function showCalibration(s){
  document.getElementById("cal").style.display = "block";
  document.getElementById("uv").style.display = "none";
  document.getElementById("uvmsg").style.display = "none";
  document.getElementById("markers").classList.remove("nolabels");
  setMarkers(true, MARKER_PX);
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
  setMarkers(false);
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

// Live calibration: small markers stay on screen for the camera; the warp
// updates every poll. If this screen isn't seen this frame the server keeps the
// last corners, so we keep showing the last warp instead of blanking.
function showLive(s){
  document.getElementById("cal").style.display = "none";
  document.getElementById("incomplete").style.display = "none";
  document.getElementById("markers").classList.add("nolabels");   // no ID tags in live
  setMarkers(true, LIVE_MARKER_PX);   // markers must stay visible to the camera
  document.querySelectorAll(".corner").forEach(c=>c.classList.remove("missing","present"));
  const uv = document.getElementById("uv"), msg = document.getElementById("uvmsg");
  if (s.captured && s.complete && s.corner_points){
    msg.style.display = "none"; uv.style.display = "block";
    renderUV();
  } else {
    uv.style.display = "none"; msg.style.display = "flex";
    msg.innerHTML = "Live calibrating…<br>point the camera at this screen";
  }
}

function renderUV(){
  if (!(LAST && (LAST.phase==="mapping" || LAST.phase==="live")
        && LAST.complete && LAST.corner_points)) return;
  const screenimg = document.getElementById("screenimg");
  // Visualizations are warped per-screen on the SERVER — just display the stream
  // (the server re-warps on every frame, including live corner changes).
  if (LAST.content && LAST.content.kind === "visualization"){
    document.getElementById("uvquad").style.display = "none";
    document.getElementById("uvimg").style.display = "none";
    const url = `/content/stream/${DISPLAY_ID}?w=${Math.round(window.innerWidth)}`+
                `&h=${Math.round(window.innerHeight)}`;
    if (screenimg.dataset.url !== url){ screenimg.dataset.url = url; screenimg.src = url; }
    screenimg.style.display = "block";
    return;
  }
  // non-viz (image / gradient): client-side homography warp
  if (screenimg.dataset.url){ screenimg.dataset.url=""; screenimg.removeAttribute("src");
                              screenimg.style.display="none"; }
  document.getElementById("uvquad").style.display = "block";
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
    // (re)build the gradient+checker only when the box size changes — reassigning
    // the background every frame causes a repaint flash (flicker).
    const key = `${Math.round(bw)}x${Math.round(bh)}`;
    if (key !== GRAD_KEY){
      GRAD = uvDataURL(b); GRAD_KEY = key;
      q.style.backgroundImage = `url(${GRAD})`;
      q.style.backgroundSize = "100% 100%";   // fill the UV box exactly, no tiling
      q.style.backgroundRepeat = "no-repeat";
    }
  }
}

register();
</script>
</body>
</html>
"""


@app.get("/display")
def display():
    return render_template_string(
        DISPLAY_PAGE, MARKER_PX=consts.MARKER_PX,
        LIVE_MARKER_PX=consts.LIVE_MARKER_PX, LIVE_FPS=consts.LIVE_FPS)


@app.post("/register")
def register():
    body = request.get_json(silent=True) or {}
    client_id = body.get("client_id") or request.remote_addr or "anon"
    rec = _register(client_id)
    return jsonify({"display_id": rec["display_id"], "slots": rec["slots"]})


def _status_payload(display_id: str):
    """The status dict a slave needs to render. None if the display is unknown."""
    d = _display_by_id(display_id)
    if d is None:
        return None
    with _lock:
        phase = _phase
        bounds = _uv_bounds
        content = _content_descriptor()
    cap = d["capture"]
    if cap is None:
        return {"captured": False, "phase": phase,
                "uv_bounds": bounds, "content": content}
    return {"captured": True, "phase": phase,
            "uv_bounds": bounds, "content": content, **cap}


@app.get("/status/<display_id>")
def status(display_id: str):
    payload = _status_payload(display_id)
    if payload is None:
        return jsonify({"error": "unknown display"}), 404
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Phase control
# ---------------------------------------------------------------------------

@app.get("/phase")
def get_phase():
    with _lock:
        return jsonify({"phase": _phase})


@app.post("/phase")
def set_phase():
    global _phase, _uv_bounds
    body = request.get_json(silent=True) or {}
    phase = body.get("phase")
    if phase not in ("calibration", "mapping", "live"):
        return jsonify({"error": "phase must be 'calibration', 'mapping' or 'live'"}), 400
    with _lock:
        _phase = phase
        if phase == "live":
            # live corners come from the phone's live camera feed — a different
            # coordinate space than a prior still photo — so start fresh.
            for rec in _displays.values():
                rec["capture"] = None
            _uv_bounds = None
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
# Content mapped onto the UV space: uploaded image OR a live visualization
# ---------------------------------------------------------------------------

def _content_descriptor():
    """Build the `content` object for /status. Caller holds _lock."""
    if _content_kind == "image" and _content_bytes is not None:
        return {"kind": "image", "url": f"/content/image?v={_content_version}",
                "mode": _content_mode}
    if _content_kind == "visualization":
        return {"kind": "visualization", "name": _viz_name, "url": "/content/stream",
                "mode": _content_mode}
    return None


@app.get("/visualizations")
def list_visualizations():
    """The visualizations registered in mosiac/visualizations (drives the phone menu)."""
    return jsonify(visualizations.available())


@app.get("/gradients")
def list_gradients():
    """Gradient maps available to the smoke visualization (drives its selector)."""
    return jsonify({"gradients": visualizations.gradients.available(),
                    "current": visualizations.gradients.current_name()})


@app.post("/content/gradient")
def set_gradient():
    """Select the smoke gradient; the running sim picks it up on its next frame."""
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if name not in visualizations.gradients.available():
        return jsonify({"error": f"unknown gradient: {name}"}), 400
    visualizations.gradients.set_current(name)
    return jsonify({"gradient": name})


@app.get("/viz/params")
def get_viz_params():
    """Return param definitions for every visualization that exposes them."""
    return jsonify(visualizations.all_viz_params())


@app.post("/viz/param")
def set_viz_param():
    """Set a param on the currently running visualization instance."""
    body = request.get_json(silent=True) or {}
    key  = body.get("key")
    val  = body.get("value")
    if key is None or val is None:
        return jsonify({"error": "key and value required"}), 400
    with _lock:
        if _viz is not None:
            _viz.set_param(key, val)
    return jsonify({"ok": True, "key": key, "value": val})


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


@app.post("/content/visualization")
def start_visualization():
    """Switch the mapped content to a named live visualization (particles/smoke/...)."""
    global _content_kind, _content_mode, _viz_name, _viz
    body = request.get_json(silent=True) or {}
    name = body.get("name", "particles")
    mode = body.get("mode", "fill")
    if mode not in ("fill", "fit"):
        mode = "fill"
    if name not in {v["name"] for v in visualizations.available()}:
        return jsonify({"error": f"unknown visualization: {name}"}), 400
    with _lock:
        _content_kind = "visualization"
        _content_mode = mode
        if name != _viz_name:
            _viz_name = name
            _viz = None          # force the loop to rebuild with the new sim
    _ensure_viz_thread()
    if visualizations.uses_hands(name):
        _ensure_hands_thread()       # YOLO hand tracker drives the sim
    return jsonify({"ok": True, "kind": "visualization", "name": name, "mode": mode})


# ---------------------------------------------------------------------------
# Hand tracking (YOLO) -> current hand -> field-coordinate force on the sim
# ---------------------------------------------------------------------------

def _calibration_image_size():
    """The image size the calibration (and thus uv_bounds) is expressed in."""
    with _lock:
        for d in _displays.values():
            cap = d.get("capture")
            if cap and cap.get("complete"):
                return cap["image_size"]["width"], cap["image_size"]["height"]
    return None


def _hand_to_field(cx, cy):
    """Map a normalized camera position to normalized field (UV) coords, using
    the calibrated screen bounds (camera assumed at the calibration position)."""
    b = _uv_bounds
    cal = _calibration_image_size()
    if not b or not cal:
        return cx, cy, 1.0, 1.0            # fallback: whole camera -> whole field
    cw, ch = cal
    bx0, by0 = b["min_x"] / cw, b["min_y"] / ch
    bw = (b["max_x"] - b["min_x"]) / cw or 1e-6
    bh = (b["max_y"] - b["min_y"]) / ch or 1e-6
    return (cx - bx0) / bw, (cy - by0) / bh, bw, bh


def _on_hand(hand):
    """Callback from the hand tracker: hand = (cx, cy, vx, vy) normalized camera
    coords + velocity, or None. Convert to a field-space force in _pointer."""
    global _pointer, _current_hand
    if hand is None:
        _current_hand = None
        return
    cx, cy, vx, vy = hand
    u, v, bw, bh = _hand_to_field(cx, cy)
    vu, vv = vx / bw, vy / bh
    u = min(1.0, max(0.0, u)); v = min(1.0, max(0.0, v))
    _pointer = (u, v, vu, vv, time.time())
    _current_hand = {"u": u, "v": v, "vu": vu, "vv": vv}


def _on_hand_debug(frame):
    """Store the annotated YOLO frame (downscaled) for the /hands/debug stream."""
    global _hands_debug
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, max(1, round(640 * h / w))))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        _hands_debug = buf.tobytes()


def _ensure_hands_thread():
    global _hands_started
    with _lock:
        if _hands_started:
            return
        _hands_started = True

    def should_run():
        # run while a hand-driven viz is the content and the host camera is free
        return (_content_kind == "visualization"
                and visualizations.uses_hands(_viz_name)
                and not (_phase == "live" and _live_source == "server"))

    threading.Thread(
        target=hands.run,
        kwargs=dict(should_run=should_run, on_hand=_on_hand,
                    on_debug=_on_hand_debug if consts.HAND_DEBUG else None,
                    camera_index=consts.CAMERA_INDEX, fps=consts.HAND_FPS,
                    device=consts.HAND_DEVICE, conf=consts.HAND_CONF,
                    imgsz=consts.HAND_IMGSZ, roi_imgsz=consts.HAND_ROI_IMGSZ,
                    cam_width=consts.HAND_CAM_WIDTH, use_coreml=consts.HAND_COREML,
                    finger_extend=consts.HAND_FINGER_EXTEND),
        daemon=True).start()


@app.get("/hands/status")
def hands_status():
    return jsonify({"active": visualizations.uses_hands(_viz_name)
                    and _content_kind == "visualization",
                    "current_hand": _current_hand})


@app.get("/hands/debug")
def hands_debug():
    """MJPEG stream of the YOLO camera feed with detections drawn (debug view).
    Open this URL in a browser tab while a hand-driven viz is active."""
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame = _hands_debug
            if frame is None:
                time.sleep(0.1)
                continue
            yield boundary + frame + b"\r\n"
            time.sleep(1.0 / max(1, consts.HAND_FPS))
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/content/clear")
def clear_content():
    """Drop any mapped content; screens fall back to the UV color gradient."""
    global _content_kind
    with _lock:
        _content_kind = None
    return jsonify({"ok": True, "kind": None})


def _warp_field_to_screen(field, display_id, w, h):
    """Warp the rendered field into this screen's rectangle (its homography).
    Returns a (h, w, 3) BGR image, or None if the display isn't calibrated."""
    d = _display_by_id(display_id)
    cap = d["capture"] if d else None
    bounds = _uv_bounds
    if field is None or bounds is None or not (cap and cap["complete"]):
        return None
    fh, fw = field.shape[:2]
    bw = bounds["max_x"] - bounds["min_x"]
    bh = bounds["max_y"] - bounds["min_y"]
    src = []
    for slot in detector.CORNER_SLOTS:               # TL, TR, BR, BL
        px, py = cap["corner_points"][slot]
        u = (px - bounds["min_x"]) / bw
        v = (py - bounds["min_y"]) / bh
        src.append([u * fw, v * fh])
    src = np.array(src, dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(field, M, (w, h))


@app.get("/content/stream/<display_id>")
def content_stream_display(display_id):
    """Per-screen visualization stream: the server warps the field into THIS
    screen's rectangle and sends only that, at the screen's resolution, on each
    new rendered frame (#1, #2, #3). The slave just displays it — no client warp."""
    def _dim(name, default):
        try:
            return max(1, min(VIZ_SCREEN_MAX, int(request.args.get(name, default))))
        except (TypeError, ValueError):
            return default
    w, h = _dim("w", 1280), _dim("h", 720)

    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last_ver = -1
        while True:
            with _viz_cond:
                _viz_cond.wait_for(lambda: _viz_ver != last_ver, timeout=1.0)
                last_ver = _viz_ver
                field = _viz_raw
            warped = _warp_field_to_screen(field, display_id, w, h)
            if warped is None:
                continue
            ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield boundary + buf.tobytes() + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/content/stream")
def content_stream():
    """Full-field debug stream (not used by slaves; they use /content/stream/<id>)."""
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last_ver = -1
        while True:
            with _viz_cond:
                _viz_cond.wait_for(lambda: _viz_ver != last_ver, timeout=1.0)
                last_ver = _viz_ver
                field = _viz_raw
            if field is None:
                continue
            ok, buf = cv2.imencode(".jpg", field, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield boundary + buf.tobytes() + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _desired_viz_size():
    """Resolution matching the screens' bounding-box orientation/aspect."""
    b = _uv_bounds
    if not b:
        return (VIZ_MAX_SIDE, VIZ_MAX_SIDE * 9 // 16)
    bw, bh = b["max_x"] - b["min_x"], b["max_y"] - b["min_y"]
    if bw >= bh:
        return (VIZ_MAX_SIDE, max(1, round(VIZ_MAX_SIDE * bh / bw)))
    return (max(1, round(VIZ_MAX_SIDE * bw / bh)), VIZ_MAX_SIDE)


def _viz_loop():
    global _viz, _viz_raw, _viz_ver, _viz_size
    while True:
        if _content_kind != "visualization":
            time.sleep(0.1)
            continue
        size = _desired_viz_size()
        if _viz is None or _viz_size != size:
            _viz = visualizations.create(_viz_name, size[0], size[1])
            _viz_size = size
        # feed the current hand's force to the sim (if it accepts one, recent)
        if hasattr(_viz, "set_pointer"):
            p = _pointer
            _viz.set_pointer((p[0], p[1], p[2], p[3])
                             if (p and time.time() - p[4] < 0.3) else None)
        _viz.step()
        frame = _viz.render()                 # raw BGR field; warped per-screen later
        with _viz_cond:
            _viz_raw = frame
            _viz_ver += 1
            _viz_cond.notify_all()
        time.sleep(0.005)  # render is the limiter at high resolution


def _ensure_viz_thread():
    global _viz_started
    with _lock:
        if _viz_started:
            return
        _viz_started = True
    threading.Thread(target=_viz_loop, daemon=True).start()


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
  .lbl { display:block; font-size:12px; color:#888; margin:14px 0 4px;
         text-transform:uppercase; letter-spacing:.04em; }
  .sel { width:100%; padding:12px; font-size:16px; border-radius:10px;
         border:1px solid #ccc; background:#fff; color:#111; margin-top:8px; }
  @media (prefers-color-scheme: dark){ .sel{background:#1c1c1e;color:#eee;border-color:#333;} }
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
  <button class="btn" id="toLive" style="background:#ff3b30">🔴 Live calibration</button>
  <div id="liveCtl" style="display:none">
    <label class="lbl">Camera source</label>
    <select id="liveSource" class="sel">
      <option value="phone">Phone camera</option>
      <option value="server">Server device camera</option>
    </select>
    <video id="livevideo" playsinline autoplay muted
      style="width:100%;border-radius:12px;margin-top:10px;display:none;background:#000"></video>
  </div>
  <div id="livestatus" class="status"></div>

  <hr>
  <h2 style="font-size:17px; margin-bottom:0">Screen content</h2>
  <p class="sub">Warped per screen so it looks straight from the camera's position.</p>
  <label class="lbl">Content</label>
  <select id="contentType" class="sel">
    <option value="uv">UV map</option>
    <option value="image">Upload image</option>
    <option value="viz">Visualization</option>
  </select>
  <select id="vizName" class="sel" style="display:none"></select>
  <select id="gradName" class="sel" style="display:none"></select>
  <select id="vizParamMode" class="sel" style="display:none"></select>
  <input id="content" type="file" accept="image/*">
  <label class="btn" id="uploadBtn" for="content" style="display:none">🖼️ Choose Image</label>
  <div id="modeRow" style="margin-top:12px; font-size:15px; display:none;">
    <label><input type="radio" name="cmode" value="fill" checked> Fill (stretch)</label>
    &nbsp;&nbsp;&nbsp;
    <label><input type="radio" name="cmode" value="fit"> Fit (keep aspect)</label>
  </div>
  <div id="cstatus" class="status"></div>

<script>
const LIVE_FPS = {{ LIVE_FPS }}, LIVE_MAX_WIDTH = {{ LIVE_MAX_WIDTH }},
      HTTPS_PORT = {{ HTTPS_PORT }}, HTTPS_ENABLED = {{ USE_HTTPS }};
function wsURL(path){ return (location.protocol==="https:"?"wss:":"ws:")+"//"+location.host+path; }
let CURRENT_PHASE = 'mapping';
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
  CURRENT_PHASE=phase; phaseName.textContent=phase;
}
async function setPhase(p){
  await fetch('/phase',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({phase:p})});
  CURRENT_PHASE=p; await refreshPhase();
}
document.getElementById('toMapping').onclick=()=>{ stopLive(); setPhase('mapping'); };
document.getElementById('toCalib').onclick=()=>{ stopLive(); setPhase('calibration'); };

// --- live calibration: phone camera (streams frames) or server device camera ---
const liveStatus=document.getElementById('livestatus'),
      liveCtl=document.getElementById('liveCtl'),
      liveSource=document.getElementById('liveSource'),
      liveVideo=document.getElementById('livevideo');
let livePoll=null, liveStream=null, liveSend=null, liveWS=null;

function stopPhoneStream(){
  if(liveSend){ clearInterval(liveSend); liveSend=null; }
  if(liveWS){ try{liveWS.close();}catch(e){} liveWS=null; }
  if(liveStream){ liveStream.getTracks().forEach(t=>t.stop()); liveStream=null; }
  liveVideo.style.display='none';
}
async function startPhoneStream(){
  stopPhoneStream();
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.isSecureContext){
    const url = `https://${location.hostname}:${HTTPS_PORT}/phone`;
    liveStatus.innerHTML = HTTPS_ENABLED
      ? `<span class="err">Phone camera needs the secure page — <a href="${url}">open ${url}</a> and accept the warning (or use “Server device camera”).</span>`
      : '<span class="err">Phone camera needs HTTPS (set USE_HTTPS in consts.py), or use “Server device camera”.</span>';
    return;
  }
  try{
    liveStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
  }catch(e){ liveStatus.innerHTML=`<span class="err">Camera blocked: ${e.message}</span>`; return; }
  liveVideo.srcObject=liveStream; liveVideo.style.display='block';
  try{ await liveVideo.play(); }catch(e){}
  // stream JPEG frames over a WebSocket (downscaled to LIVE_MAX_WIDTH)
  liveWS=new WebSocket(wsURL('/live/frames'));
  const cv=document.createElement('canvas');
  liveSend=setInterval(()=>{
    if(!liveVideo.videoWidth || !liveWS || liveWS.readyState!==1) return;
    const vw=liveVideo.videoWidth, vh=liveVideo.videoHeight;
    const sc=Math.min(1, LIVE_MAX_WIDTH/vw);
    cv.width=Math.round(vw*sc); cv.height=Math.round(vh*sc);
    cv.getContext('2d').drawImage(liveVideo,0,0,cv.width,cv.height);
    cv.toBlob(b=>{ if(b && liveWS && liveWS.readyState===1) liveWS.send(b); },
              'image/jpeg', 0.6);
  }, Math.round(1000/LIVE_FPS));
}
async function applyLiveSource(){
  const src=liveSource.value;
  await fetch('/live/source',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({source:src})});
  if(src==='phone') startPhoneStream(); else stopPhoneStream();
}
function stopLive(){
  stopPhoneStream(); liveCtl.style.display='none';
  if(livePoll){ clearInterval(livePoll); livePoll=null; } liveStatus.textContent='';
}
document.getElementById('toLive').onclick=async ()=>{
  await setPhase('live');
  liveCtl.style.display='block';
  applyLiveSource();
  if(!livePoll) livePoll=setInterval(async ()=>{
    let s; try{ s=await (await fetch('/live/status')).json(); }catch(e){ return; }
    if(!s.live){ stopLive(); return; }
    if(s.source==='server' && !s.camera_ok)
      liveStatus.innerHTML=`<span class="err">${s.error||'Server camera unavailable'}</span>`;
    else if(s.camera_ok)
      liveStatus.innerHTML=`<span class="ok">🔴 Live (${s.source}) — tracking at ${s.fps} fps</span>`;
    else
      liveStatus.innerHTML='Waiting for camera…';
  }, 1000);
};
liveSource.addEventListener('change', applyLiveSource);

// --- screen content: UV map / uploaded image / visualization ---
const contentType=document.getElementById('contentType'),
      vizName=document.getElementById('vizName'),
      gradName=document.getElementById('gradName'),
      vizParamMode=document.getElementById('vizParamMode'),
      uploadBtn=document.getElementById('uploadBtn'),
      contentInput=document.getElementById('content'),
      cstatus=document.getElementById('cstatus');
const currentMode=()=>document.querySelector('input[name=cmode]:checked').value;

// all viz param definitions loaded once at startup
let _vizParamDefs = {};

// populate the visualization + gradient dropdowns from the server
(async ()=>{
  try{
    const list=await (await fetch('/visualizations')).json();
    vizName.innerHTML=list.map(v=>`<option value="${v.name}">${v.label}</option>`).join('');
  }catch(e){}
  try{
    const g=await (await fetch('/gradients')).json();
    gradName.innerHTML=g.gradients.map(n=>`<option value="${n}">${n}</option>`).join('');
    if(g.current) gradName.value=g.current;
  }catch(e){}
  try{
    _vizParamDefs=await (await fetch('/viz/params')).json();
  }catch(e){}
})();

function refreshContentUI(){
  const t=contentType.value;
  const vn=vizName.value;
  vizName.style.display   = t==='viz'  ? 'block' : 'none';
  uploadBtn.style.display = t==='image'? 'block' : 'none';
  gradName.style.display  = (t==='viz' && vn==='smoke') ? 'block' : 'none';
  document.getElementById('modeRow').style.display = t==='image' ? 'block' : 'none';
  // show mode dropdown for any viz that has a "mode" param
  const modeDef = t==='viz' && _vizParamDefs[vn] && _vizParamDefs[vn]['mode'];
  if(modeDef){
    vizParamMode.innerHTML=modeDef.options.map(o=>`<option value="${o.value}">${o.label}</option>`).join('');
    vizParamMode.value=modeDef.default;
    vizParamMode.style.display='block';
  } else {
    vizParamMode.style.display='none';
  }
}
gradName.addEventListener('change', async ()=>{
  await fetch('/content/gradient',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:gradName.value})});
  cstatus.innerHTML=`<span class="ok">Gradient: ${gradName.value}</span>`;
});
vizParamMode.addEventListener('change', async ()=>{
  await fetch('/viz/param',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:'mode', value:vizParamMode.value})});
  cstatus.innerHTML=`<span class="ok">Mode: ${vizParamMode.options[vizParamMode.selectedIndex].text}</span>`;
});
async function applyContent(){
  const t=contentType.value;
  // content shows in both mapping and live; only switch to mapping if not live
  const toMappingIfNeeded=()=>{ if(CURRENT_PHASE!=='live') setPhase('mapping'); };
  try{
    if(t==='uv'){
      await fetch('/content/clear',{method:'POST'});
      cstatus.innerHTML='<span class="ok">UV map</span>'; toMappingIfNeeded();
    } else if(t==='viz'){
      // visualizations are rendered at the screens' aspect already -> always fill
      const res=await fetch('/content/visualization',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({name:vizName.value, mode:'fill'})});
      if(!res.ok) throw new Error('Failed');
      cstatus.innerHTML=`<span class="ok">✨ ${vizName.options[vizName.selectedIndex].text}</span>`;
      toMappingIfNeeded();
    } else {
      cstatus.textContent='Choose an image…';
    }
  }catch(e){ cstatus.innerHTML=`<span class="err">${e.message}</span>`; }
}
contentType.addEventListener('change', ()=>{ refreshContentUI(); applyContent(); });
vizName.addEventListener('change', ()=>{ refreshContentUI(); applyContent(); });
refreshContentUI();

contentInput.addEventListener('change', async ()=>{
  const f=contentInput.files[0]; if(!f) return;
  const fd=new FormData(); fd.append('file',f); fd.append('mode',currentMode());
  cstatus.textContent='Uploading…'; cstatus.className='status';
  try{
    const res=await fetch('/content',{method:'POST',body:fd});
    const d=await res.json(); if(!res.ok) throw new Error(d.error||'Failed');
    if(CURRENT_PHASE!=='live') setPhase('mapping');
    cstatus.innerHTML=`<span class="ok">✓ Mapped image (${d.mode})</span>`;
  }catch(e){ cstatus.innerHTML=`<span class="err">${e.message}</span>`; }
});
document.querySelectorAll('input[name=cmode]').forEach(r=>r.addEventListener('change', async ()=>{
  await fetch('/content/mode',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({mode:currentMode()})});
  cstatus.innerHTML=`<span class="ok">Mode: ${currentMode()}</span>`;
}));

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
    return render_template_string(
        PHONE_PAGE, LIVE_FPS=consts.LIVE_FPS,
        LIVE_MAX_WIDTH=consts.LIVE_MAX_WIDTH,
        HTTPS_PORT=consts.HTTPS_PORT,
        USE_HTTPS=("true" if consts.USE_HTTPS else "false"))


# ---------------------------------------------------------------------------
# Calibration capture endpoint
# ---------------------------------------------------------------------------

def _detect_and_update(image, keep_missing, dictionary=None, refine=True):
    """Detect markers in ``image``, update each display's stored corners, and
    recompute the global UV bounds. Shared by the phone photo (/calibrate) and
    the live feed.

    keep_missing=True (live): if a display's four markers aren't all visible this
    frame, leave its stored corners untouched — the screen keeps its last warp
    instead of going blank. keep_missing=False (photo): store partial captures so
    missing corners can be highlighted.

    ``dictionary``/``refine`` are forwarded to the detector — the live path pins
    the dictionary and skips sub-pixel refinement for speed.

    Returns a per-display summary dict (the /calibrate response shape).
    """
    global _uv_bounds
    # Live frames are downscaled before detection; coordinates stay consistent
    # because everything downstream is normalized by this same image_size.
    if keep_missing and image.shape[1] > consts.LIVE_MAX_WIDTH:
        s = consts.LIVE_MAX_WIDTH / image.shape[1]
        image = cv2.resize(image, (consts.LIVE_MAX_WIDTH, int(round(image.shape[0] * s))))
    height, width = image.shape[:2]
    markers, dict_name = detector.detect_markers(image, dictionary=dictionary, refine=refine)
    by_id = {m["id"]: m for m in markers}

    with _lock:
        records = sorted(_displays.values(), key=lambda d: d["index"])

    def corners_for(slots, centroid, only_present):
        cp, status = {}, {}
        for slot, mid in slots:
            seen = mid in by_id and centroid is not None
            if seen:
                cp[slot] = detector._screen_point(by_id[mid], centroid, "outer")
            status[slot] = {"marker_id": mid, "present": seen}
        return cp, status

    displays_out = []
    changed = False
    for rec in records:
        slots = list(zip(detector.CORNER_SLOTS, rec["marker_ids"]))
        present = [by_id[mid] for _, mid in slots if mid in by_id]
        centroid = np.mean([m["center"] for m in present], axis=0) if present else None
        complete = len(present) == len(slots)
        missing = [s for s, mid in slots if mid not in by_id]

        if complete:
            cp, status = corners_for(slots, centroid, only_present=False)
            rec["capture"] = {
                "complete": True, "corners": status, "corner_points": cp,
                "image_size": {"width": int(width), "height": int(height)},
                "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            changed = True
        elif not keep_missing:
            cp, status = corners_for(slots, centroid, only_present=True)
            rec["capture"] = {
                "complete": False, "corners": status, "corner_points": cp,
                "image_size": {"width": int(width), "height": int(height)},
                "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        # else: live + not fully visible -> keep the previous capture as-is

        cap = rec["capture"]
        is_complete = bool(cap and cap["complete"])
        norm = None
        if is_complete:
            iw, ih = cap["image_size"]["width"], cap["image_size"]["height"]
            norm = {s: [p[0] / iw, p[1] / ih] for s, p in cap["corner_points"].items()}
        displays_out.append({
            "id": rec["display_id"], "complete": is_complete,
            "missing_corners": missing, "corners": norm,
            "corner_points": cap["corner_points"] if cap else {},
        })

    # UV domain = bbox of every *currently stored* complete screen (+ margin)
    all_pts = [pt for rec in records if rec["capture"] and rec["capture"]["complete"]
               for pt in rec["capture"]["corner_points"].values()]
    if all_pts:
        xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        mx, my = (max_x - min_x) * UV_MARGIN, (max_y - min_y) * UV_MARGIN
        bounds = {"min_x": min_x - mx, "min_y": min_y - my,
                  "max_x": max_x + mx, "max_y": max_y + my}
    else:
        bounds = None
    with _lock:
        _uv_bounds = bounds

    # wake any live WebSocket pushers — only when corners actually changed
    if changed:
        global _live_rev
        with _live_cond:
            _live_rev += 1
            _live_cond.notify_all()

    return {"image_size": {"width": int(width), "height": int(height)},
            "dictionary": dict_name, "marker_count": len(markers),
            "uv_bounds": bounds, "displays": displays_out}


@app.post("/calibrate")
def calibrate():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = _detect_and_update(image, keep_missing=False)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_as = f"calibration-{stamp}.json"
    (DEBUG_DIR / saved_as).write_text(json.dumps(
        {"captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
         "source_filename": file.filename, "corner_mode": "outer", **result}, indent=2))

    return jsonify({"saved_as": saved_as, "result": result})


# ---------------------------------------------------------------------------
# Live calibration: the PHONE streams camera frames here; we re-detect each one
# ---------------------------------------------------------------------------

@app.get("/live/status")
def live_status():
    with _lock:
        if _live_source == "server":
            ok, err = _camera_ok, _camera_err
        else:
            ok, err = (time.time() - _last_live_ts) < 1.5, None
        return jsonify({"live": _phase == "live", "source": _live_source,
                        "camera_ok": ok, "error": err, "fps": consts.LIVE_FPS})


@app.post("/live/source")
def set_live_source():
    """Choose where live frames come from: the phone or the host's own camera."""
    global _live_source
    body = request.get_json(silent=True) or {}
    source = body.get("source")
    if source not in ("phone", "server"):
        return jsonify({"error": "source must be 'phone' or 'server'"}), 400
    with _lock:
        _live_source = source
    if source == "server":
        _ensure_camera_thread()
    return jsonify({"source": source})


def _live_detect(image):
    """Live detection: pinned dictionary + no sub-pixel refine (fast path)."""
    return _detect_and_update(image, keep_missing=True,
                              dictionary=consts.LIVE_DICT, refine=False)


def _decode_frame(buf: bytes):
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.post("/live/frame")
def live_frame():
    """A single live camera frame from the phone (HTTP fallback to the WS)."""
    global _last_live_ts
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "no frame"}), 400
    image = _decode_frame(file.read())
    if image is None:
        return jsonify({"error": "could not decode frame"}), 400
    with _lock:
        _last_live_ts = time.time()
    result = _live_detect(image)
    return jsonify({"marker_count": result["marker_count"]})


# --- WebSockets (live mode only) --------------------------------------------

@sock.route("/live/frames")
def ws_live_frames(ws):
    """Phone streams JPEG frames here (replaces per-frame POST)."""
    global _last_live_ts
    while True:
        data = ws.receive()
        if data is None:
            break
        if isinstance(data, str):      # ignore any text (e.g. pings)
            continue
        image = _decode_frame(data)
        if image is None:
            continue
        with _lock:
            _last_live_ts = time.time()
        try:
            _live_detect(image)
        except Exception:
            pass


@sock.route("/live/corners/<display_id>")
def ws_live_corners(ws, display_id):
    """Push this display's warp to the slave whenever corners change — no poll."""
    last_rev = -1
    while True:
        payload = _status_payload(display_id)
        if payload is None:
            break
        try:
            ws.send(json.dumps(payload))
        except Exception:
            break
        # block until corners change (or 1 s keepalive / phase-change resync)
        with _live_cond:
            _live_cond.wait_for(lambda: _live_rev != last_rev, timeout=1.0)
            last_rev = _live_rev


def _camera_loop():
    """Server-camera source: open a local camera and feed frames to detection.
    Runs only while phase == live and the chosen source is 'server'."""
    global _camera_ok, _camera_err
    cap = None
    interval = 1.0 / max(1, consts.LIVE_FPS)
    while True:
        if not (_phase == "live" and _live_source == "server"):
            if cap is not None:
                cap.release(); cap = None
            with _lock:
                _camera_ok = False
            time.sleep(0.15)
            continue
        if cap is None:
            cap = cv2.VideoCapture(consts.CAMERA_INDEX)
            if not cap.isOpened():
                cap.release(); cap = None
                with _lock:
                    _camera_ok = False
                    _camera_err = ("Could not open server camera "
                                   f"(index {consts.CAMERA_INDEX}). Grant the "
                                   "terminal camera access in System Settings.")
                time.sleep(0.6)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, consts.LIVE_MAX_WIDTH)
        ok, frame = cap.read()
        if not ok or frame is None:
            with _lock:
                _camera_ok = False
            time.sleep(interval)
            continue
        with _lock:
            _camera_ok = True
            _camera_err = None
        try:
            _live_detect(frame)
        except Exception:
            pass
        time.sleep(interval)


def _ensure_camera_thread():
    global _camera_started
    with _lock:
        if _camera_started:
            return
        _camera_started = True
    threading.Thread(target=_camera_loop, daemon=True).start()


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
        f"<code>http://{ip}:{consts.PORT}/display</code> on each screen and "
        f"<code>http://{ip}:{consts.PORT}/phone</code> on the phone."
        + (f" For the phone <b>live camera</b>, open the secure page "
           f"<code>https://{ip}:{consts.HTTPS_PORT}/phone</code>."
           if consts.USE_HTTPS else "")
        + "</p>"
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
    from werkzeug.serving import make_server

    ip = _lan_ip()
    lines = ["Mosiac host running. Open on the LAN:",
             f"  Screen slave : http://{ip}:{consts.PORT}/display",
             f"  Phone        : http://{ip}:{consts.PORT}/phone"]
    if consts.USE_HTTPS:
        lines += [f"  Phone (live) : https://{ip}:{consts.HTTPS_PORT}/phone",
                  "   ^ the phone live camera needs this secure URL; accept the",
                  "     one-time self-signed-cert warning. Everything else uses HTTP."]
    lines.append(f"Debug captures saved to: {DEBUG_DIR}")
    print("\n".join(lines), flush=True)

    # HTTPS (self-signed) runs alongside HTTP — only the phone live camera needs
    # it; HTTP keeps working for the screens, photo calibration, and everything.
    if consts.USE_HTTPS:
        try:
            https = make_server("0.0.0.0", consts.HTTPS_PORT, app,
                                threaded=True, ssl_context="adhoc")
            threading.Thread(target=https.serve_forever, daemon=True).start()
        except Exception as e:
            print(f"  (HTTPS disabled: {e})")

    make_server("0.0.0.0", consts.PORT, app, threaded=True).serve_forever()


if __name__ == "__main__":
    main()
