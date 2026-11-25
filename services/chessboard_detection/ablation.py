"""Compare the line detector's validation checks on fixed screenshots."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .benchmark import Box, MATCH_IOU, _FIXTURES, _match, _read_labels
from .line_detector import (
    _MIN_BOARD_SIDE_PX,
    _Checks,
    _Fit,
    _check_grid,
    _distinct_fits,
    _gradients,
    _locate_axes,
    _propose,
)

_VARIANTS = {
    "Full detector": _Checks(),
    "Without spacing check": _Checks(spacing=False),
    "Without square check": _Checks(square=False),
    "Without crossing check": _Checks(crossings=False),
    "Without boundary check": _Checks(boundary=False),
    "Without spacing and square checks": _Checks(spacing=False, square=False),
}
_STRICT_IOU = 0.95


@dataclass(frozen=True)
class _Result:
    """Record the number of matches and predictions for one detector and screenshot."""

    variant: str
    filename: str
    labels: int
    matches: int
    strict_matches: int
    returned: int


def _boxes(fits: list[_Fit], width: int, height: int) -> list[Box]:
    """Give each distinct fit its outer board bounds in source pixels."""
    return [
        (
            max(0, fit.x),
            max(0, fit.y),
            min(width, fit.x + 8 * fit.dx),
            min(height, fit.y + 8 * fit.dy),
        )
        for fit in _distinct_fits(fits)
    ]


def _contours(gray: np.ndarray) -> list[Box]:
    """Find upright rectangular borders without looking for chessboard structure."""
    height, width = gray.shape
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < _MIN_BOARD_SIDE_PX:
            continue
        extent = cv2.contourArea(contour) / (w * h)
        if extent >= 0.75:
            fits.append(_Fit(x, y, w / 8, h / 8, 0, extent))
    return _boxes(fits, width, height)


def _detect_variants(image: np.ndarray) -> dict[str, list[Box]]:
    """Reuse the same candidate search for all line variants on this screenshot."""
    gray, gx, gy = _gradients(image)
    height, width = gray.shape
    candidates = _propose(gray)
    predictions = {}
    axes_by_spacing = {
        check_spacing: [
            (axes, np.sign(candidate.response)) for candidate in candidates
            if (axes := _locate_axes(gx, gy, candidate, check_spacing=check_spacing)) is not None
        ]
        for check_spacing in (True, False)
    }
    for name, checks in _VARIANTS.items():
        fits = [
            fit for axes, polarity in axes_by_spacing[checks.spacing]
            if (fit := _check_grid(gray, axes, polarity, checks)) is not None
        ]
        predictions[name] = _boxes(fits, width, height)

    coarse = []
    for candidate in candidates:
        x, y, pitch = candidate.x, candidate.y, candidate.pitch
        size = 8 * pitch
        if min(x, y, width - x - size, height - y - size) >= -2 and size >= _MIN_BOARD_SIDE_PX:
            coarse.append(_Fit(x, y, pitch, pitch, 0, abs(candidate.response)))
    predictions["Crossing template only"] = _boxes(coarse, width, height)
    predictions["Rectangular contours"] = _contours(gray)
    return predictions


def _run() -> list[_Result]:
    """Compare every detector on the same labelled and board-free screenshots."""
    negatives = Path(__file__).resolve().parents[2] / "docs" / "screenshots"
    paths = sorted((_FIXTURES / "images").iterdir()) + [
        negatives / f"{name}.png" for name in ("position-patterns", "settings")
    ]
    results = []
    for path in paths:
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode {path}")
        height, width = image.shape[:2]
        boxes = [] if path.parent == negatives else _read_labels(
            _FIXTURES / "labels" / f"{path.stem}.txt", width, height
        )
        for variant, predictions in _detect_variants(image).items():
            matches, _ = _match(predictions, boxes)
            strict_matches, _ = _match(predictions, boxes, _STRICT_IOU)
            results.append(
                _Result(variant, path.name, len(boxes), matches, strict_matches, len(predictions))
            )
    return results


def _print_summary(results: list[_Result]) -> None:
    """Report matches and unmatched detections separately for positive and negative frames."""
    print(f"Match threshold: IoU >= {MATCH_IOU:.2f}")
    print(
        "| Variant | Matched boards | IoU >= 0.95 | Multi-board matches |"
        " Unmatched on positives | Detections on negatives |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for variant in dict.fromkeys(result.variant for result in results):
        rows = [result for result in results if result.variant == variant]
        positive = [result for result in rows if result.labels]
        multi = [result for result in rows if result.labels > 1]
        matched = sum(result.matches for result in positive)
        labelled = sum(result.labels for result in positive)
        strict = sum(result.strict_matches for result in positive)
        multi_matched = sum(result.matches for result in multi)
        multi_labelled = sum(result.labels for result in multi)
        unmatched = sum(result.returned - result.matches for result in positive)
        negatives = sum(result.returned for result in rows if not result.labels)
        print(
            f"| {variant} | {matched}/{labelled} | {strict}/{labelled} | "
            f"{multi_matched}/{multi_labelled} | {unmatched} | {negatives} |"
        )


def main() -> None:
    """Print an ablation summary and optionally save the per-screenshot results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Write per-screenshot scores to this file")
    args = parser.parse_args()
    results = _run()
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("variant", "filename", "labels", "matches_iou90", "matches_iou95", "returned")
            )
            writer.writerows(
                (
                    result.variant, result.filename, result.labels,
                    result.matches, result.strict_matches, result.returned,
                )
                for result in results
            )
    _print_summary(results)


if __name__ == "__main__":
    main()
