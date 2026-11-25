"""Check line-based detection on uneven pixel grids and multiple boards."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from .benchmark import _iou, _match, _read_labels
from .line_detector import detect_line_grids


class LineDetectionTests(unittest.TestCase):
    """Check grid geometry and handling of multiple boards."""

    def test_ignores_thumbnails(self) -> None:
        """Ignore a complete 64-pixel board beside an 80-pixel board."""
        image = np.full((116, 194, 3), 55, dtype=np.uint8)
        tiles = (np.indices((8, 8)).sum(axis=0) % 2 * 100 + 100).astype(np.uint8)
        for x, pitch in ((12, 8), (100, 10)):
            side = 8 * pitch
            board = cv2.resize(tiles, (side, side), interpolation=cv2.INTER_NEAREST)
            image[12 : 12 + side, x : x + side] = board[:, :, None]

        grids = detect_line_grids(image)
        self.assertEqual(len(grids), 1)
        np.testing.assert_allclose(grids[0].intersections[0][0], (100, 12), atol=1, rtol=0)

    def test_accepts_rounded_cell_boundaries(self) -> None:
        """Allow small spacing and aspect differences while rejecting an elongated grid."""
        image = np.full((290, 635, 3), 70, dtype=np.uint8)
        y = np.array([22, 51, 81, 109, 138, 167, 197, 225, 254])
        square_x = np.array([30, 58, 87, 114, 142, 171, 198, 226, 254])
        wide_x = 302 + np.arange(9) * 39
        for x in (square_x, wide_x):
            for row in range(8):
                for col in range(8):
                    shade = 205 if (row + col) % 2 else 95
                    image[y[row] : y[row + 1], x[col] : x[col + 1]] = shade

        grids = detect_line_grids(image)
        self.assertEqual(len(grids), 1)
        points = np.asarray(grids[0].intersections)
        expected = np.stack(np.meshgrid(square_x, y), axis=-1)
        np.testing.assert_allclose(points, expected, atol=2, rtol=0)

    def test_finds_all_boards_in_montage(self) -> None:
        """Find all eight distinct labelled boards in a screen with adjacent games."""
        fixtures = Path(__file__).with_name("fixtures") / "screenshots"
        path = fixtures / "images" / "chessboard_detect_0100.png"
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        labels = _read_labels(fixtures / "labels" / f"{path.stem}.txt", width, height)
        predictions = []
        for grid in detect_line_grids(image):
            top_left, bottom_right = grid.intersections[0][0], grid.intersections[-1][-1]
            predictions.append((*top_left, *bottom_right))
        self.assertEqual(len(predictions), 8)
        self.assertEqual(_match(predictions, labels)[0], 8)

    def test_matching_does_not_credit_duplicate_predictions(self) -> None:
        """Count two identical grids on one board as a single match."""
        box = (8.0, 8.0, 80.0, 80.0)
        other = (92.0, 8.0, 164.0, 80.0)
        self.assertEqual(_iou(box, other), 0)
        self.assertEqual(_match([box, box], [box, other]), (1, 1.0))
