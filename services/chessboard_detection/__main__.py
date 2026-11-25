"""Run board detection on a screenshot and optionally save a diagnostic overlay."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import Grid, detect_grid


def _draw_grid(image: np.ndarray, grid: Grid) -> np.ndarray:
    """Draw the detected lines on a copy of the screenshot."""
    overlay = image.copy()
    points = np.rint(grid.intersections).astype(np.int32)
    for line in (*points, *points.transpose(1, 0, 2)):
        points_on_line = np.ascontiguousarray(line)
        cv2.polylines(overlay, [points_on_line], False, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(overlay, [points_on_line], False, (78, 232, 84), 1, cv2.LINE_AA)
    return overlay


def main() -> None:
    """Read a screenshot, report its grid as JSON and save an overlay when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a screenshot")
    parser.add_argument("--overlay", type=Path, help="Save a PNG with the detected grid")
    args = parser.parse_args()

    try:
        if args.overlay and args.overlay.resolve() == args.image.resolve():
            parser.error("the overlay path must differ from the input path")
        image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"could not decode {args.image}")
        grid = detect_grid(image)
        if args.overlay and grid is not None:
            success, encoded = cv2.imencode(".png", _draw_grid(image, grid))
            if not success:
                parser.error("could not encode the grid overlay")
            args.overlay.write_bytes(encoded.tobytes())
    except (OSError, cv2.error) as error:
        parser.error(str(error))

    print(json.dumps({"grid": asdict(grid) if grid is not None else None}))


if __name__ == "__main__":
    main()
