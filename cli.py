"""
Command-line interface for the multi-display marker detector.

Usage:
    python cli.py IMAGE [--corner-mode center|inner|outer]
                        [--dictionary NAME] [--annotated OUT.png]
                        [--full]

By default prints the spec-shaped JSON (image_size + displays). Use --full to
include per-marker diagnostics.
"""

import argparse
import json
import sys

import cv2

import detector


def _spec_output(result: dict) -> dict:
    """Trim the analysis to the exact output schema from the spec."""
    return {
        "image_size": result["image_size"],
        "displays": [
            {"id": d["id"], "corners": d["corners"]}
            for d in result["displays"]
            if d.get("complete")
        ],
    }


def annotate(image, result):
    """Draw detected markers and display quadrilaterals onto a copy of image."""
    import numpy as np

    out = image.copy()
    h, w = image.shape[:2]

    for m in result["markers"]:
        pts = np.array(m["corners"], dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        cx, cy = int(m["center"][0]), int(m["center"][1])
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(out, str(m["id"]), (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for d in result["displays"]:
        if not d.get("complete"):
            continue
        quad = np.array(
            [
                [d["corners"][slot][0] * w, d["corners"][slot][1] * h]
                for slot in detector.CORNER_SLOTS
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [quad], True, (255, 128, 0), 3)
        tl = quad[0]
        cv2.putText(out, d["id"], (tl[0], tl[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), 2)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect markers and build display quads.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--corner-mode", choices=["center", "inner", "outer"],
                        default="center", help="How to derive each screen corner")
    parser.add_argument("--dictionary", default=None,
                        help="Force a marker dictionary (default: auto-detect)")
    parser.add_argument("--annotated", default=None,
                        help="Write an annotated visualization image to this path")
    parser.add_argument("--full", action="store_true",
                        help="Print full diagnostics instead of just the spec output")
    args = parser.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 1

    result = detector.analyze(
        image, dictionary=args.dictionary, corner_mode=args.corner_mode
    )

    if args.annotated:
        cv2.imwrite(args.annotated, annotate(image, result))

    print(json.dumps(result if args.full else _spec_output(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
