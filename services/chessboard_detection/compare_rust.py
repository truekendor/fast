"""Compare the native grid detector and the matching Python reference on the same fixtures."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np

from .benchmark import Box, _FIXTURES, _match, _read_labels
from .line_detector import detect_line_grids


def _bounds(intersections: list | tuple) -> Box:
    """Return the rectangle covered by a grid."""
    points = np.asarray(intersections)
    left, top = points.min(axis=(0, 1))
    right, bottom = points.max(axis=(0, 1))
    return left, top, right, bottom


def _run_fast(executable: Path, image: Path, output: Path, *, scalar: bool) -> tuple[dict, float]:
    """Run the same native executable with the selected CPU path and time the command."""
    command = [executable, "--detect-grid", image, output]
    if scalar:
        command.append("--scalar")
    start = perf_counter()
    subprocess.run(command, check=True)
    elapsed = (perf_counter() - start) * 1000
    return json.loads(output.read_text(encoding="utf-8")), elapsed


def _compare(executable: Path, images: Path) -> list[dict]:
    """Decode both implementations' outputs and score them against the same labels."""
    rows = []
    with TemporaryDirectory() as directory:
        for path in sorted(images.iterdir()):
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            labels = _read_labels(_FIXTURES / "labels" / f"{path.stem}.txt", width, height)
            python = [_bounds(grid.intersections) for grid in detect_line_grids(image, check_square=False)]
            payload, auto_ms = _run_fast(
                executable, path, Path(directory) / f"{path.stem}-auto.json", scalar=False
            )
            scalar, scalar_ms = _run_fast(
                executable, path, Path(directory) / f"{path.stem}-scalar.json", scalar=True
            )
            if payload != scalar:
                raise ValueError(f"the automatic and scalar grids differ for {path.name}")
            rust = [_bounds(grid["intersections"]) for grid in payload["grids"]]
            row = {
                "image": path.name,
                "labels": len(labels),
                "rust_auto_ms": round(auto_ms, 3),
                "rust_scalar_ms": round(scalar_ms, 3),
            }
            for name, boxes in (("python", python), ("rust", rust)):
                row[f"{name}_iou90"] = _match(boxes, labels)[0]
                row[f"{name}_iou95"] = _match(boxes, labels, 0.95)[0]
                row[f"{name}_returned"] = len(boxes)
            rows.append(row)
    return rows


def main() -> None:
    """Run both detectors and print results, optionally saving per-image results as CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="Path to the release Fast executable")
    parser.add_argument("--csv", type=Path, help="Write the per-image results to this path")
    parser.add_argument("--occlusions", action="store_true", help="Use the three occluded screenshots")
    args = parser.parse_args()
    images = _FIXTURES / ("occlusions" if args.occlusions else "images")
    rows = _compare(args.executable.resolve(strict=True), images)
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for group, sample in (("All frames", rows), ("Multiple boards", [row for row in rows if row["labels"] > 1])):
        print(group)
        total = sum(row["labels"] for row in sample)
        for name in ("python", "rust"):
            found = sum(row[f"{name}_iou90"] for row in sample)
            strict = sum(row[f"{name}_iou95"] for row in sample)
            returned = sum(row[f"{name}_returned"] for row in sample)
            print(f"  {name}: {found}/{total} at IoU >= 0.90, {strict}/{total} at IoU >= 0.95, {returned - found} unmatched")
    for row in rows:
        print(
            f"{row['image']:<27} labels={row['labels']:>2} "
            f"Python={row['python_iou90']:>2}/{row['python_returned']:<2} "
            f"Rust={row['rust_iou90']:>2}/{row['rust_returned']:<2} (matched/returned)"
        )
    print(f"Identical Rust automatic and scalar output on {len(rows)} frames.")


if __name__ == "__main__":
    main()
