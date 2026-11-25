"""Measure board detection against the bundled, labelled 2D screenshots."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import detect_grid

_FIXTURES = Path(__file__).with_name("fixtures") / "screenshots"
MATCH_IOU = 0.90
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ScreenshotResult:
    """Report the number of labelled boards and how well the returned grid matches."""

    filename: str
    labelled_boards: int
    best_iou: float | None

    @property
    def matched_boards(self) -> int:
        """Count the match for a detector that returns at most one grid."""
        return int(self.best_iou is not None and self.best_iou >= MATCH_IOU)


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


def evaluate() -> list[ScreenshotResult]:
    """Run the detector on each bundled image and compare its grid with the YOLO labels."""
    results = []
    for path in sorted((_FIXTURES / "images").iterdir()):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode {path}")
        height, width = image.shape[:2]
        boxes = _read_labels(_FIXTURES / "labels" / f"{path.stem}.txt", width, height)
        grid = detect_grid(image)
        best_iou = None
        if grid is not None:
            points = np.asarray(grid.intersections)
            left, top = points.min(axis=(0, 1))
            right, bottom = points.max(axis=(0, 1))
            best_iou = max(_iou((left, top, right, bottom), box) for box in boxes)
        results.append(ScreenshotResult(path.name, len(boxes), best_iou))
    return results


def main() -> None:
    """Print per-image overlap and totals for single-board and multi-board screenshots."""
    results = evaluate()
    print(f"Match threshold: IoU >= {MATCH_IOU:.2f}")
    for result in results:
        overlap = "no grid" if result.best_iou is None else f"IoU {result.best_iou:.3f}"
        print(
            f"{result.filename:<27} {result.matched_boards:>2}/{result.labelled_boards:<2} boards  "
            f"{overlap}"
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
        returned = sum(result.best_iou is not None for result in group)
        frames = sum(result.matched_boards > 0 for result in group)
        print(
            f"{title}: {frames}/{len(group)} frames with a match, {found}/{boards} labelled boards, "
            f"{found}/{returned} returned grids matched"
        )


if __name__ == "__main__":
    main()
