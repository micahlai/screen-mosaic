"""
Flask web app for the multi-display marker detector.

Upload a photo of one or more displays (each showing four corner markers) and
get back the spec-shaped JSON plus an annotated visualization.

Run:
    python app.py
    # open http://127.0.0.1:5000
"""

import base64
import io
import json

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from mosiac import detector
from .cli import _spec_output, annotate

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB uploads

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Display Marker Detector</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #0d1117; color: #e6edf3; }
  header { padding: 24px 32px; border-bottom: 1px solid #30363d; }
  h1 { margin: 0; font-size: 20px; }
  p.sub { margin: 4px 0 0; color: #8b949e; font-size: 13px; }
  main { display: grid; grid-template-columns: 360px 1fr; gap: 0;
         height: calc(100vh - 73px); }
  aside { padding: 24px 32px; border-right: 1px solid #30363d; overflow-y: auto; }
  section.view { padding: 24px 32px; overflow-y: auto; }
  label { display: block; font-size: 12px; color: #8b949e; margin: 16px 0 6px;
          text-transform: uppercase; letter-spacing: .04em; }
  select, input[type=file] { width: 100%; padding: 8px; background: #161b22;
          border: 1px solid #30363d; color: #e6edf3; border-radius: 6px; }
  button { margin-top: 20px; width: 100%; padding: 10px; border: 0;
           border-radius: 6px; background: #238636; color: #fff; font-weight: 600;
           cursor: pointer; font-size: 14px; }
  button:disabled { opacity: .5; cursor: default; }
  img { max-width: 100%; border-radius: 8px; border: 1px solid #30363d; }
  pre { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; overflow-x: auto; font-size: 12px; line-height: 1.5;
        margin-top: 20px; }
  .stat { display: inline-block; background: #161b22; border: 1px solid #30363d;
          border-radius: 6px; padding: 6px 10px; margin: 4px 8px 4px 0; font-size: 12px; }
  .empty { color: #8b949e; font-size: 14px; margin-top: 40px; text-align: center; }
  .err { color: #f85149; margin-top: 16px; font-size: 13px; }
</style>
</head>
<body>
<header>
  <h1>Display Marker Detector</h1>
  <p class="sub">Detect ArUco / AprilTag markers, group into displays, return normalized screen quads.</p>
</header>
<main>
  <aside>
    <form id="form">
      <label for="file">Image</label>
      <input id="file" name="file" type="file" accept="image/*" required>
      <label for="corner_mode">Screen corner source</label>
      <select id="corner_mode" name="corner_mode">
        <option value="center">Marker center</option>
        <option value="inner">Inner marker corner</option>
        <option value="outer">Outer marker corner</option>
      </select>
      <label for="dictionary">Marker dictionary</label>
      <select id="dictionary" name="dictionary">
        <option value="">Auto-detect</option>
        {% for name in dictionaries %}
        <option value="{{ name }}">{{ name }}</option>
        {% endfor %}
      </select>
      <button id="go" type="submit">Analyze</button>
      <div id="err" class="err"></div>
    </form>
  </aside>
  <section class="view">
    <div id="stats"></div>
    <div id="result"><p class="empty">Upload an image to begin.</p></div>
    <pre id="json" style="display:none"></pre>
  </section>
</main>
<script>
const form = document.getElementById('form');
const go = document.getElementById('go');
const err = document.getElementById('err');
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  err.textContent = '';
  go.disabled = true; go.textContent = 'Analyzing…';
  try {
    const res = await fetch('/analyze', { method: 'POST', body: new FormData(form) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    render(data);
  } catch (ex) {
    err.textContent = ex.message;
  } finally {
    go.disabled = false; go.textContent = 'Analyze';
  }
});
function render(data) {
  const stats = document.getElementById('stats');
  const complete = data.full.displays.filter(d => d.complete).length;
  stats.innerHTML =
    `<span class="stat">${data.full.marker_count} markers</span>` +
    `<span class="stat">${complete} displays</span>` +
    `<span class="stat">dict: ${data.full.dictionary || 'none'}</span>` +
    `<span class="stat">${data.full.image_size.width}×${data.full.image_size.height}</span>`;
  document.getElementById('result').innerHTML =
    `<img src="data:image/png;base64,${data.annotated}">`;
  const j = document.getElementById('json');
  j.style.display = 'block';
  j.textContent = JSON.stringify(data.spec, null, 2);
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    names = [n for n, _ in detector.CANDIDATE_DICTIONARIES]
    return render_template_string(PAGE, dictionaries=names)


@app.post("/analyze")
def analyze_route():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    corner_mode = request.form.get("corner_mode", "center")
    dictionary = request.form.get("dictionary") or None

    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = detector.analyze(image, dictionary=dictionary, corner_mode=corner_mode)
    annotated = annotate(image, result)
    ok, buf = cv2.imencode(".png", annotated)
    annotated_b64 = base64.b64encode(buf).decode("ascii") if ok else ""

    return jsonify(
        {
            "spec": _spec_output(result),
            "full": {k: v for k, v in result.items() if k != "markers"} | {
                "marker_count": result["marker_count"],
            },
            "annotated": annotated_b64,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
