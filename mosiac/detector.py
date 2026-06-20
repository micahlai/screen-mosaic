"""
Marker-based multi-display detector.

Analyzes an image containing one or more displays. Each display shows four
fiducial markers (ArUco or AprilTag), one at each screen corner. The module
detects every marker, groups markers into displays using a predefined ID
mapping, orders each display's corners, and returns screen quadrilaterals in
normalized image coordinates.

Coordinate system: the photo itself is the global frame.
    origin = image top-left, x = right, y = down
    normalized_x = pixel_x / image_width
    normalized_y = pixel_y / image_height

No real-world geometry is estimated (no depth, scale, distance, or pose). The
uploaded image is the sole coordinate space.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Which marker IDs belong to which display. Within each list the position
# encodes the display corner the marker sits at, in clockwise order starting
# from the top-left:
#     index 0 -> top_left
#     index 1 -> top_right
#     index 2 -> bottom_right
#     index 3 -> bottom_left
DEFAULT_DISPLAY_MAPPING: Dict[str, List[int]] = {
    "display_1": [0, 1, 2, 3],
    "display_2": [4, 5, 6, 7],
    "display_3": [8, 9, 10, 11],
}

# The four display-corner slots, in the order the mapping lists them.
CORNER_SLOTS = ["top_left", "top_right", "bottom_right", "bottom_left"]

# Candidate marker dictionaries. We try each and keep whichever yields the
# most detections, so the caller does not need to know the marker family up
# front. AprilTag families are tried first since the prompt mentions them.
CANDIDATE_DICTIONARIES: List[Tuple[str, int]] = [
    ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
    ("DICT_APRILTAG_25h9", cv2.aruco.DICT_APRILTAG_25h9),
    ("DICT_APRILTAG_16h5", cv2.aruco.DICT_APRILTAG_16h5),
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
]

CornerMode = str  # "center" | "inner" | "outer"


# ---------------------------------------------------------------------------
# Low-level detection
# ---------------------------------------------------------------------------

def _build_detector(dictionary_id: int) -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement gives noticeably cleaner corner pixels,
    # which matters because everything downstream keys off corner accuracy.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(
    image: np.ndarray,
    dictionary: Optional[str] = None,
) -> Tuple[List[dict], str]:
    """Detect every marker in ``image``.

    Returns ``(markers, dictionary_name)`` where each marker is::

        {"id": int, "center": [u, v], "corners": [[x,y], [x,y], [x,y], [x,y]]}

    Corners follow OpenCV's order: top-left, top-right, bottom-right,
    bottom-left in the marker's own frame. Coordinates are image pixels.

    If ``dictionary`` is given (a name from ``CANDIDATE_DICTIONARIES``) only
    that family is used; otherwise every candidate family is tried and the one
    with the most detections wins.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    if dictionary is not None:
        candidates = [(n, d) for (n, d) in CANDIDATE_DICTIONARIES if n == dictionary]
        if not candidates:
            raise ValueError(f"Unknown dictionary: {dictionary}")
    else:
        candidates = CANDIDATE_DICTIONARIES

    best: Tuple[List[dict], str] = ([], "")
    for name, dict_id in candidates:
        detector = _build_detector(dict_id)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        markers = _to_marker_records(corners, ids)
        if len(markers) > len(best[0]):
            best = (markers, name)

    return best


def _to_marker_records(corners: Sequence[np.ndarray], ids: np.ndarray) -> List[dict]:
    markers: List[dict] = []
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape(4, 2).astype(float)
        center = pts.mean(axis=0)
        markers.append(
            {
                "id": int(marker_id),
                "center": [float(center[0]), float(center[1])],
                "corners": [[float(x), float(y)] for x, y in pts],
            }
        )
    return markers


# ---------------------------------------------------------------------------
# Grouping and screen-corner construction
# ---------------------------------------------------------------------------

def _screen_point(marker: dict, centroid: np.ndarray, mode: CornerMode) -> List[float]:
    """Pick the point representing this marker's screen corner."""
    if mode == "center":
        return [marker["center"][0], marker["center"][1]]

    pts = np.array(marker["corners"], dtype=float)
    dists = np.linalg.norm(pts - centroid, axis=1)
    idx = int(np.argmin(dists)) if mode == "inner" else int(np.argmax(dists))
    return [float(pts[idx][0]), float(pts[idx][1])]


def build_displays(
    markers: List[dict],
    image_width: int,
    image_height: int,
    mapping: Optional[Dict[str, List[int]]] = None,
    corner_mode: CornerMode = "center",
) -> List[dict]:
    """Group markers into displays and build normalized screen quadrilaterals.

    A display is included only if all four of its markers were detected.
    Each display's corner slot (top_left, top_right, ...) is filled from the
    marker whose ID sits at the matching position in the mapping list.

    ``corner_mode``:
        "center" - use each marker's center (default)
        "inner"  - use each marker's corner nearest the display centroid
        "outer"  - use each marker's corner farthest from the display centroid
    """
    mapping = mapping or DEFAULT_DISPLAY_MAPPING
    by_id = {m["id"]: m for m in markers}

    displays: List[dict] = []
    for display_id, marker_ids in mapping.items():
        present = [mid for mid in marker_ids if mid in by_id]
        if len(present) != len(marker_ids):
            # Skip displays we cannot fully resolve; report what was missing.
            displays.append(
                {
                    "id": display_id,
                    "complete": False,
                    "missing_marker_ids": [mid for mid in marker_ids if mid not in by_id],
                    "corners": None,
                }
            )
            continue

        group_markers = [by_id[mid] for mid in marker_ids]
        centroid = np.mean([m["center"] for m in group_markers], axis=0)

        corners = {}
        for slot, mid in zip(CORNER_SLOTS, marker_ids):
            corners[slot] = _normalize(
                _screen_point(by_id[mid], centroid, corner_mode),
                image_width,
                image_height,
            )

        displays.append({"id": display_id, "complete": True, "corners": corners})

    return displays


def _normalize(point: Sequence[float], width: int, height: int) -> List[float]:
    return [point[0] / width, point[1] / height]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyze(
    image: np.ndarray,
    mapping: Optional[Dict[str, List[int]]] = None,
    dictionary: Optional[str] = None,
    corner_mode: CornerMode = "center",
) -> dict:
    """Run the full pipeline on a BGR image array.

    Returns the spec-shaped result plus diagnostic fields (``markers``,
    ``dictionary``) that callers may use for visualization or debugging.
    """
    height, width = image.shape[:2]
    markers, dict_name = detect_markers(image, dictionary=dictionary)
    displays = build_displays(markers, width, height, mapping, corner_mode)

    return {
        "image_size": {"width": int(width), "height": int(height)},
        "dictionary": dict_name,
        "marker_count": len(markers),
        "markers": markers,
        "displays": displays,
    }
