# Display Marker Detector

Analyze a single image containing one or more **displays**. Each display shows
four fiducial markers (ArUco or AprilTag), one at every screen corner. The app
detects all markers, groups them into displays, builds each display's screen
quadrilateral, and returns the corners in **normalized image coordinates**.

The photo itself is the only coordinate space: `origin = top-left`, `x = right`,
`y = down`. No real-world geometry (depth, scale, distance, camera pose) is
estimated.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web app

```bash
python app.py
# open http://127.0.0.1:5000
```

Upload an image, pick how each screen corner is derived (marker center / inner /
outer corner), and get the annotated visualization plus the JSON output.

## CLI

```bash
python cli.py IMAGE [--corner-mode center|inner|outer]
                    [--dictionary NAME] [--annotated out.png] [--full]
```

`--full` adds per-marker diagnostics; the default prints just the spec output.

## Try it without a photo

```bash
python make_test_image.py            # writes test_image.png (3 displays)
python cli.py test_image.png --annotated annotated.png
```

## Output schema

```json
{
  "image_size": { "width": 4032, "height": 3024 },
  "displays": [
    {
      "id": "display_1",
      "corners": {
        "top_left":     [0.12, 0.31],
        "top_right":    [0.42, 0.28],
        "bottom_right": [0.45, 0.57],
        "bottom_left":  [0.15, 0.60]
      }
    }
  ]
}
```

## How it works

1. **Detect** — `cv2.aruco.ArucoDetector` with sub-pixel corner refinement.
   Several marker dictionaries (AprilTag + ArUco families) are tried and the one
   with the most detections wins, so the marker family need not be known up
   front. Each marker returns `{id, center, corners}` in image pixels.
2. **Group** — markers are assigned to displays using a fixed ID mapping
   (`detector.DEFAULT_DISPLAY_MAPPING`). Within each display the list position
   encodes the corner, clockwise from top-left:
   `[top_left, top_right, bottom_right, bottom_left]`. A display is only emitted
   when all four of its markers are found; partial displays are flagged with
   their missing IDs in the full diagnostics.
3. **Construct** — each screen corner is taken from its marker's center
   (default) or a designated marker corner (`inner`/`outer` relative to the
   display centroid).
4. **Normalize** — every corner is divided by image width/height into `[0, 1]`.

To change which IDs map to which display, edit `DEFAULT_DISPLAY_MAPPING` in
`detector.py`.

## Files

| File                  | Purpose                                            |
|-----------------------|----------------------------------------------------|
| `detector.py`         | Core pipeline: detect → group → construct → normalize |
| `app.py`              | Flask web UI (upload + annotated result + JSON)    |
| `cli.py`              | Command-line interface and annotation drawing      |
| `make_test_image.py`  | Synthetic 3-display test image generator           |
