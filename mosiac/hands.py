"""YOLO hand tracking for hand-driven visualizations (e.g. the smoke sim).

Opens the host camera, runs a YOLOv8-pose model, and treats the wrist keypoints
as hand positions. Hands are tracked across frames; the "current hand" is the one
that has been continuously on screen the longest. Its normalized camera position
+ velocity are handed to a callback each frame.

Optimizations: the pose model is exported to CoreML so it runs on the Apple
Neural Engine (offloading the CPU); once a hand is found, detection runs only on
a crop around that person (smaller/lower-res); the camera is read with a 1-frame
buffer for low latency. The camera is assumed to be in the same physical position
as the one used to calibrate.
"""

import os
import platform
import time

import numpy as np

WRIST_KEYPOINTS = (9, 10)   # COCO pose: left/right wrist
_model = None
_is_coreml = False


def _get_model(imgsz, use_coreml):
    """Load the pose model — CoreML (Neural Engine) when possible, else PyTorch."""
    global _model, _is_coreml
    if _model is not None:
        return _model
    from ultralytics import YOLO
    base = YOLO("yolov8n-pose.pt")
    if use_coreml and platform.system() == "Darwin":
        try:
            pt = str(getattr(base, "ckpt_path", None) or "yolov8n-pose.pt")
            mlp = os.path.splitext(pt)[0] + ".mlpackage"
            if not os.path.exists(mlp):
                print("Exporting YOLO pose -> CoreML (one-time)…")
                base.export(format="coreml", imgsz=imgsz)   # no baked NMS (pose)
            _model = YOLO(mlp, task="pose")
            _is_coreml = True
            print("Hand model: CoreML (Neural Engine)")
            return _model
        except Exception as e:
            print("CoreML unavailable, using PyTorch model:", e)
    _model = base
    return _model


def _detect(model, frame, device, conf, imgsz):
    """Detect people; return [(box, [wrist (x,y), ...]), ...] in frame pixels."""
    res = model.predict(frame, imgsz=imgsz, conf=conf, max_det=6,
                        verbose=False, device=device)[0]
    boxes = (res.boxes.xyxy.cpu().numpy()
             if res.boxes is not None and res.boxes.xyxy is not None
             else np.zeros((0, 4)))
    persons = []
    if res.keypoints is not None and res.keypoints.xy is not None:
        xy = res.keypoints.xy.cpu().numpy()                  # (n, 17, 2)
        cf = res.keypoints.conf
        cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
        for p in range(xy.shape[0]):
            ws = [(float(xy[p, w, 0]), float(xy[p, w, 1])) for w in WRIST_KEYPOINTS
                  if cf[p, w] > conf and (xy[p, w] > 0).all()]
            box = boxes[p].tolist() if p < len(boxes) else None
            persons.append((box, ws))
    return persons


class _Tracker:
    """Nearest-neighbour hand tracker; current() = longest-lived live track."""

    def __init__(self, max_dist=120, max_miss=10):
        self.tracks = []
        self.max_dist = max_dist
        self.max_miss = max_miss
        self._next = 0

    def update(self, points):
        for t in self.tracks:
            t["matched"] = False
        for (x, y) in points:
            best, bestd = None, self.max_dist
            for t in self.tracks:
                if t["matched"]:
                    continue
                d = ((t["x"] - x) ** 2 + (t["y"] - y) ** 2) ** 0.5
                if d < bestd:
                    bestd, best = d, t
            if best is not None:
                best["px"], best["py"] = best["x"], best["y"]
                best["x"], best["y"] = x, y
                best["age"] += 1
                best["miss"] = 0
                best["matched"] = True
            else:
                self.tracks.append({"id": self._next, "x": x, "y": y, "px": x,
                                    "py": y, "age": 1, "miss": 0, "matched": True})
                self._next += 1
        for t in self.tracks:
            if not t["matched"]:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]

    def current(self):
        live = [t for t in self.tracks if t["miss"] == 0]
        return max(live, key=lambda t: t["age"]) if live else None


def _draw_debug(frame, boxes, wrists, cur, roi):
    import cv2
    out = frame.copy()
    if roi is not None:
        cv2.rectangle(out, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])),
                      (255, 150, 0), 1)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (80, 200, 80), 2)
    for (x, y) in wrists:
        cv2.circle(out, (int(x), int(y)), 7, (0, 220, 255), 2)
    if cur is not None:
        cv2.circle(out, (int(cur["x"]), int(cur["y"])), 16, (0, 0, 255), 3)
        cv2.putText(out, f"current hand (age {cur['age']})",
                    (int(cur["x"]) + 18, int(cur["y"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(out, f"hands: {len(wrists)}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def run(should_run, on_hand, on_debug=None, camera_index=0, fps=12, device="cpu",
        conf=0.3, imgsz=320, roi_imgsz=192, cam_width=960, use_coreml=True):
    """Camera loop. While should_run() is true, detect+track hands and call
    on_hand((cx, cy, vx, vy)) with the current hand in normalized camera coords
    (+ velocity), or on_hand(None). on_debug (if given) gets an annotated frame."""
    import cv2
    cap, model, tracker, roi = None, None, _Tracker(), None
    interval = 1.0 / max(1, fps)
    while True:
        if not should_run():
            if cap is not None:
                cap.release(); cap = None
            on_hand(None)
            time.sleep(0.2)
            continue
        if cap is None:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                cap.release(); cap = None
                time.sleep(0.6)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)               # low latency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, round(cam_width * 9 / 16))
            try:
                model = _get_model(imgsz, use_coreml)
            except Exception as e:
                print("hand model load failed:", e)
                time.sleep(2.0)
                continue
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(interval)
            continue
        H, W = frame.shape[:2]

        # ROI: once we have a person around the current hand, detect on a crop
        ox, oy, sub = 0, 0, frame
        det_imgsz = imgsz
        if roi is not None:
            x0, y0 = max(0, int(roi[0])), max(0, int(roi[1]))
            x1, y1 = min(W, int(roi[2])), min(H, int(roi[3]))
            if x1 - x0 > 32 and y1 - y0 > 32:
                sub, ox, oy = frame[y0:y1, x0:x1], x0, y0
                if not _is_coreml:        # CoreML input size is fixed at export
                    det_imgsz = roi_imgsz
        try:
            persons = _detect(model, sub, device, conf, det_imgsz)
        except Exception:
            persons = []

        wrists, pboxes = [], []
        for box, ws in persons:
            wrists.extend((x + ox, y + oy) for (x, y) in ws)
            if box:
                pboxes.append([box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy])
        if roi is not None and not wrists:
            roi = None                    # lost it in the crop -> full frame next

        tracker.update(wrists)
        cur = tracker.current()

        # next ROI = expanded person box that contains the current hand
        roi = None
        if cur is not None:
            for b in pboxes:
                if b[0] <= cur["x"] <= b[2] and b[1] <= cur["y"] <= b[3]:
                    mw, mh = (b[2] - b[0]) * 0.25, (b[3] - b[1]) * 0.25
                    roi = [b[0] - mw, b[1] - mh, b[2] + mw, b[3] + mh]
                    break

        if cur is not None:
            cx, cy = cur["x"] / W, cur["y"] / H
            vx, vy = (cur["x"] - cur["px"]) / W, (cur["y"] - cur["py"]) / H
            on_hand((cx, cy, vx, vy))
        else:
            on_hand(None)
        if on_debug is not None:
            try:
                on_debug(_draw_debug(frame, pboxes, wrists, cur, roi))
            except Exception:
                pass
        time.sleep(interval)
