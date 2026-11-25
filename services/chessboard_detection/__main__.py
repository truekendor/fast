"""Run board detection on a screenshot and optionally save a diagnostic overlay."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import Grid, detect_grid
from .line_detector import LineGrid, detect_line_grids


def _draw_grids(image: np.ndarray, grids: list[Grid] | list[LineGrid]) -> np.ndarray:
    """Draw the detected grids on a copy of the screenshot."""
    overlay = image.copy()
    for grid in grids:
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
    parser.add_argument("--detector", choices=("colors", "lines"), default="colors")
    parser.add_argument("--overlay", type=Path, help="Save a PNG with the detected grid")
    args = parser.parse_args()

    try:
        if args.overlay and args.overlay.resolve() == args.image.resolve():
            parser.error("the overlay path must differ from the input path")
        image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"could not decode {args.image}")
        if args.detector == "lines":
            grids = detect_line_grids(image)
            result = {"grids": [asdict(grid) for grid in grids]}
        else:
            grid = detect_grid(image)
            grids = [] if grid is None else [grid]
            result = {"grid": asdict(grid) if grid is not None else None}
        if args.overlay and grids:
            success, encoded = cv2.imencode(".png", _draw_grids(image, grids))
            if not success:
                parser.error("could not encode the grid overlay")
            args.overlay.write_bytes(encoded.tobytes())
    except (OSError, cv2.error) as error:
        parser.error(str(error))

    print(json.dumps(result))


if __name__ == "__main__":
    main()
