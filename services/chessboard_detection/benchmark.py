"""Measure board detection against the bundled, labelled 2D screenshots."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import detect_grid
from .line_detector import detect_line_grids

_FIXTURES = Path(__file__).with_name("fixtures") / "screenshots"
MATCH_IOU = 0.90
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ScreenshotResult:
    """Report how many predictions and labels match, and the best overlap."""

    filename: str
    labelled_boards: int
    returned_grids: int
    matched_boards: int
    best_iou: float | None


def _read_labels(path: Path, width: int, height: int) -> list[Box]:
    """Convert normalized YOLO board boxes into screenshot pixel coordinates."""
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        category, cx, cy, w, h = map(float, line.split())
        x1, x2 = (cx - w / 2) * width, (cx + w / 2) * width
        y1, y2 = (cy - h / 2) * height, (cy + h / 2) * height
        if category != 0 or not (-1 <= x1 < x2 <= width + 1 and -1 <= y1 < y2 <= height + 1):
            raise ValueError(f"{path}:{line_number}: invalid chessboard box")
        boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
    if not boxes:
        raise ValueError(f"{path}: expected at least one labelled board")
    return boxes


def _iou(a: Box, b: Box) -> float:
    """Measure the part of the union of two boxes shared by both boxes."""
    overlap_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_width * overlap_height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _match(predictions: list[Box], labels: list[Box]) -> tuple[int, float | None]:
    """Count one-to-one matches, using the highest overlap first."""
    overlaps = sorted(
        ((_iou(prediction, label), i, j)
         for i, prediction in enumerate(predictions) for j, label in enumerate(labels)),
        reverse=True,
    )
    matched_predictions = set()
    matched_labels = set()
    for overlap, i, j in overlaps:
        if overlap < MATCH_IOU:
            break
        if i not in matched_predictions and j not in matched_labels:
            matched_predictions.add(i)
            matched_labels.add(j)
    return len(matched_labels), overlaps[0][0] if overlaps else None


def evaluate(detector: str = "colors") -> list[ScreenshotResult]:
    """Run the selected detector and compare every returned grid with the YOLO labels."""
    results = []
    for path in sorted((_FIXTURES / "images").iterdir()):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode {path}")
        height, width = image.shape[:2]
        boxes = _read_labels(_FIXTURES / "labels" / f"{path.stem}.txt", width, height)
        if detector == "lines":
            grids = detect_line_grids(image)
        else:
            grid = detect_grid(image)
            grids = [] if grid is None else [grid]
        predictions = []
        for grid in grids:
            points = np.asarray(grid.intersections)
            left, top = points.min(axis=(0, 1))
            right, bottom = points.max(axis=(0, 1))
            predictions.append((left, top, right, bottom))
        matched, best_iou = _match(predictions, boxes)
        results.append(ScreenshotResult(path.name, len(boxes), len(predictions), matched, best_iou))
    return results


def _report(name: str, results: list[ScreenshotResult]) -> None:
    """Print per-image matches and separate totals for single-board and multi-board frames."""
    print(f"\n{name}")
    for result in results:
        overlap = "no grid" if result.best_iou is None else f"best IoU {result.best_iou:.3f}"
        print(
            f"{result.filename:<27} {result.matched_boards:>2}/{result.labelled_boards:<2} boards  "
            f"{result.returned_grids:>2} returned  {overlap}"
        )
    groups = (
        ("One board", [result for result in results if result.labelled_boards == 1]),
        ("Multiple boards", [result for result in results if result.labelled_boards > 1]),
        ("Total", results),
    )
    print()
    for title, group in groups:
        found = sum(result.matched_boards for result in group)
        boards = sum(result.labelled_boards for result in group)
        returned = sum(result.returned_grids for result in group)
        frames = sum(result.matched_boards > 0 for result in group)
        print(
            f"{title}: {frames}/{len(group)} frames with a match, {found}/{boards} labelled boards, "
            f"{found}/{returned} returned grids matched"
        )


def main() -> None:
    """Compare the color and line detectors on the same labelled sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("colors", "lines", "both"), default="both")
    args = parser.parse_args()
    print(f"Match threshold: IoU >= {MATCH_IOU:.2f}")
    for name in ("colors", "lines"):
        if args.detector in (name, "both"):
            _report(name.capitalize(), evaluate(name))


if __name__ == "__main__":
    main()
