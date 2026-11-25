"""Locate axis-aligned chessboard grids from the repeated colors of their cells."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Grid:
    """Store the grid in source pixels, from the top left of the playable area."""

    intersections: tuple[tuple[tuple[float, float], ...], ...]
    matched_cells: int


def _find_square_regions(image: np.ndarray) -> np.ndarray:
    """Find near-square components of recurring colors around the pieces."""
    blue, green, red = cv2.split(image.astype(np.uint16) >> 3)
    colors = blue | (green << 5) | (red << 10)
    counts = np.bincount(colors.ravel(), minlength=1 << 15)
    regions = []
    for color in np.argsort(counts)[-16:]:
        if counts[color] < 500:
            continue
        mask = (colors == color).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        x, y, width, height, area = stats[1:].T
        size = (width + height) / 2
        valid = (
            (size >= 12)
            & (size <= min(image.shape[:2]) / 8 + 2)
            & (abs(width - height) <= np.maximum(2, size * 0.06))
            & (area >= width * height * 0.35)
        )
        count = np.count_nonzero(valid)
        if 8 <= count <= 128:
            regions.extend(zip(x[valid], y[valid], width[valid], height[valid], [color] * count))
    return np.array(regions, dtype=np.float64).reshape(-1, 5)


def _best_grid_axes(
    regions: np.ndarray, width: int, height: int
) -> tuple[tuple[float, ...], tuple[float, ...], int] | None:
    """Find the grid supported by the most alternating cells and refine its boundaries."""
    best = None
    best_score = (0, float("-inf"))
    cell_offsets = np.array([(col, row) for row in range(8) for col in range(8)])
    image_size = np.array([width, height])
    positions = regions[:, :2]
    sizes = regions[:, 2:4].mean(axis=1)
    colors = regions[:, 4].astype(int)

    for pitch in np.unique(sizes):
        tolerance = max(2, pitch * 0.06)
        nearby = abs(sizes - pitch) <= tolerance
        if np.count_nonzero(nearby) < 24:
            continue
        nearby_regions = regions[nearby]
        squares = positions[nearby]
        square_colors = colors[nearby]
        # Each visible square could be any of the 64 cells. Test origins that fit the image.
        origins = (squares[:, None, :] - cell_offsets * pitch).reshape(-1, 2)
        inside = np.all(
            (origins >= -tolerance) & (origins + 8 * pitch <= image_size + tolerance), axis=1
        )
        origins = np.unique(origins[inside], axis=0)

        for origin in origins:
            relative = (squares - origin) / pitch
            cells = np.rint(relative).astype(int)
            aligned = np.all(
                (abs(relative - cells) * pitch <= tolerance) & (cells >= 0) & (cells < 8),
                axis=1,
            )
            if np.count_nonzero(aligned) < max(24, best_score[0]):
                continue
            parity = cells.sum(axis=1) % 2
            matches = np.zeros(len(squares), dtype=bool)
            picked_colors = []
            for color_parity in (0, 1):
                on_parity = aligned & (parity == color_parity)
                values, counts = np.unique(square_colors[on_parity], return_counts=True)
                if len(values) == 0:
                    break
                color = values[np.argmax(counts)]
                picked_colors.append(color)
                matches |= on_parity & (square_colors == color)
            if len(set(picked_colors)) != 2:
                continue

            matched_cells = cells[matches]
            distinct = np.unique(matched_cells, axis=0)
            support = len(distinct)
            odd_cells = np.count_nonzero(distinct.sum(axis=1) % 2)
            if support < max(24, best_score[0]) or not 8 <= odd_cells <= support - 8:
                continue
            if np.any(np.ptp(distinct, axis=0) < 7):
                continue

            matched_regions = nearby_regions[matches]
            x_lines, x_error = _fit_lines(
                matched_cells[:, 0], matched_regions[:, 0], matched_regions[:, 2], width
            )
            y_lines, y_error = _fit_lines(
                matched_cells[:, 1], matched_regions[:, 1], matched_regions[:, 3], height
            )
            score = (support, -(x_error + y_error))
            if score > best_score:
                best = (x_lines, y_lines, support)
                best_score = score
    return best


def _fit_lines(
    cells: np.ndarray, starts: np.ndarray, sizes: np.ndarray, limit: int
) -> tuple[tuple[float, ...], float]:
    """Fit both edges of the visible cells to a single evenly spaced axis."""
    indices = np.concatenate((cells, cells + 1))
    edges = np.concatenate((starts, starts + sizes))
    design = np.column_stack((np.ones(len(indices)), indices))
    origin, pitch = np.linalg.lstsq(design, edges, rcond=None)[0]
    mean_error = float(np.mean(abs(origin + pitch * indices - edges)))
    lines = tuple(float(np.clip(origin + index * pitch, 0, limit)) for index in range(9))
    return lines, mean_error


def detect_grid(image: np.ndarray) -> Grid | None:
    """Return the best 8×8 grid in a BGR screenshot, or None if evidence is weak."""
    height, width = image.shape[:2]
    scale = min(1, 1600 / max(width, height))
    reduced_width = max(1, round(width * scale))
    reduced_height = max(1, round(height * scale))
    if scale < 1:
        image = cv2.resize(image, (reduced_width, reduced_height), interpolation=cv2.INTER_NEAREST)
    axes = _best_grid_axes(_find_square_regions(image), reduced_width, reduced_height)
    if axes is None:
        return None
    x_lines, y_lines, matched_cells = axes
    intersections = tuple(
        tuple(
            (round(x * width / reduced_width, 3), round(y * height / reduced_height, 3))
            for x in x_lines
        )
        for y in y_lines
    )
    return Grid(intersections, matched_cells)
