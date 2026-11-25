"""Check the detector on repository screenshots and larger inputs."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from .benchmark import evaluate
from .detector import detect_grid


class DetectionTests(unittest.TestCase):
    """Verify the positions and spacing of the returned grids."""

    def test_labelled_screenshots(self) -> None:
        """Keep locating the six boards that the first OpenCV prototype could find."""
        results = evaluate()
        self.assertEqual(len(results), 15)
        self.assertEqual(sum(result.labelled_boards for result in results), 41)
        baseline = {"0004", "0012", "0028", "0050", "0082", "0108"}
        matched = {Path(result.filename).stem[-4:] for result in results if result.matched_boards}
        missing = baseline - matched
        self.assertFalse(missing, f"Previously matched screenshots missed: {sorted(missing)}")

    def test_fast_screenshots(self) -> None:
        """Find the playable board, excluding its frame and coordinates."""
        screenshots = Path(__file__).resolve().parents[2] / "docs" / "screenshots"
        boards = {
            "analyse": (329, 174, 72),
            "chess960-start": (329, 185, 44),
            "chess960": (329, 174, 37),
            "database": (1301, 359, 30),
            "play": (329, 185, 79),
            "train": (329, 188, 94),
        }
        for name, (x, y, pitch) in boards.items():
            with self.subTest(screenshot=name):
                image = cv2.imread(str(screenshots / f"{name}.png"))
                self.assert_grid(image, x, y, pitch, tolerance=1)

        for name in ("position-patterns", "settings"):
            with self.subTest(screenshot=name):
                image = cv2.imread(str(screenshots / f"{name}.png"))
                self.assertIsNotNone(image)
                self.assertIsNone(detect_grid(image))

    def test_large_screenshot(self) -> None:
        """Report original pixel coordinates after downsampling with fractional cell sizes."""
        image = np.full((1101, 2201, 3), 40, dtype=np.uint8)
        x, y, pitch = 117, 291, 97
        for row in range(8):
            for col in range(8):
                left, top = x + col * pitch, y + row * pitch
                color = (73, 132, 99) if (row + col) % 2 else (170, 207, 222)
                image[top : top + pitch, left : left + pitch] = color
                if row in (2, 6):
                    cv2.circle(image, (left + 48, top + 48), 24, (4, 14, 36), -1)
        self.assert_grid(image, x, y, pitch, tolerance=2)

    def assert_grid(self, image: np.ndarray, x: int, y: int, pitch: int, tolerance: int) -> None:
        """Compare all 81 intersections with the expected original pixel coordinates."""
        self.assertIsNotNone(image)
        grid = detect_grid(image)
        self.assertIsNotNone(grid)
        expected = [[(x + pitch * col, y + pitch * row) for col in range(9)] for row in range(9)]
        np.testing.assert_allclose(grid.intersections, expected, atol=tolerance, rtol=0)
