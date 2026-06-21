"""
main.py - Entry point for the Hand-Tracked Interactive Visualizer.

Pipeline:
    Camera Manager -> MediaPipe Tracker -> Hand State Processor ->
    Gesture Recognizer -> Particle System -> Renderer

Run with:
    python main.py

Controls:
    ESC or Q  -> quit
    Close either window also quits
"""

import time

import cv2

from camera_manager import CameraManager
from gestures import Gesture, GestureRecognizer
from hand_state import HandStateManager
from hand_tracker import FINGERTIPS, HAND_CONNECTIONS, HandTracker
from particles import ParticleSystem
from renderer import Renderer
from settings import SETTINGS

BOX_COLOR = (255, 200, 80)
LANDMARK_COLOR = (80, 220, 255)
LINE_COLOR = (180, 180, 180)


def draw_camera_overlay(frame, detections):
    """Annotates the raw camera frame with boxes, landmarks, connections, labels."""
    for det in detections:
        x_min, y_min, x_max, y_max = det.bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), BOX_COLOR, 2)

        pts = det.landmarks_px.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), LINE_COLOR, 1)
        for i, pt in enumerate(pts):
            radius = 5 if i in FINGERTIPS else 3
            cv2.circle(frame, tuple(pt), radius, LANDMARK_COLOR, -1)

        label = f"{det.label} ({det.score:.2f})"
        cv2.putText(
            frame, label, (x_min, max(y_min - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2,
        )
    return frame


def main():
    settings = SETTINGS

    camera = CameraManager(settings.camera)
    tracker = HandTracker(settings.tracker)
    state_manager = HandStateManager(settings.motion)
    gesture_recognizer = GestureRecognizer(settings.gesture)
    particle_system = ParticleSystem(settings.particle)
    renderer = Renderer(settings.render, (settings.camera.width, settings.camera.height))

    landmarks_lookup = {}   # label -> latest normalized (21,3) landmarks, for gesture math
    fps_smooth = 30.0
    prev_time = time.time()

    print("Hand-Tracked Interactive Visualizer running. Press ESC / Q to quit.")

    try:
        while True:
            ok, frame, _ts = camera.read()
            if not ok:
                print("Camera read failed, stopping.")
                break

            detections = tracker.process(frame)
            for det in detections:
                landmarks_lookup[det.label] = det.landmarks

            hand_states = state_manager.update(detections)

            gestures = {}
            for label, state in hand_states.items():
                norm_lm = landmarks_lookup.get(label)
                if norm_lm is not None and state.landmarks_px is not None:
                    gestures[label] = gesture_recognizer.recognize(state, norm_lm)
                else:
                    gestures[label] = Gesture.UNKNOWN

            for state in hand_states.values():
                if state.landmarks_px is None:
                    continue
                for tip in FINGERTIPS:
                    particle_system.spawn_from_fingertip(state.landmarks_px[tip], state.velocity)

            particle_system.apply_hand_forces(hand_states, gestures)
            particle_system.update()

            hand_distance = state_manager.hand_distance()

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            prev_time = now

            renderer.render(hand_states, gestures, particle_system, hand_distance, fps_smooth)
            if renderer.poll_quit():
                break

            if settings.show_camera_window:
                annotated = draw_camera_overlay(frame.copy(), detections)
                cv2.putText(
                    annotated, f"FPS: {fps_smooth:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                cv2.imshow("Camera Feed", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            renderer.tick(settings.max_render_fps)

    finally:
        camera.release()
        tracker.close()
        renderer.close()
        cv2.destroyAllWindows()
        print("Shut down cleanly.")


if __name__ == "__main__":
    main()
