"""Find upright checkerboard grids from aligned edges and alternating crossings."""

from dataclasses import dataclass

import cv2
import numpy as np

_TEMPLATE_PITCH = 10
_LINE_TOLERANCE_PX = 2.0
_MIN_BOARD_SIDE_PX = 80
_GRID_INDICES = np.arange(9)
_INNER = _GRID_INDICES[1:-1]
_PARITY = (1 - 2 * ((_GRID_INDICES[:, None] + _GRID_INDICES) % 2)).astype(np.float32)
_INNER_PARITY = _PARITY[1:-1, 1:-1]
_LINE_PAIRS = np.triu_indices(7, 1)


@dataclass(frozen=True)
class LineGrid:
    """Store the grid in source pixels and the number of supporting internal crossings."""

    intersections: tuple[tuple[tuple[float, float], ...], ...]
    matched_crossings: int


@dataclass(frozen=True)
class _Candidate:
    """Store the coarse origin, common cell pitch, and strength of a grid."""

    x: float
    y: float
    pitch: float
    response: float


@dataclass(frozen=True)
class _Fit:
    """Store the fitted axes and the junction evidence that supports them."""

    x: float
    y: float
    dx: float
    dy: float
    crossings: int
    strength: float


@dataclass(frozen=True)
class _Checks:
    """Select the validation steps to run after the shared candidate search."""

    spacing: bool = True
    square: bool = True
    crossings: bool = True
    boundary: bool = True


def _crossing_template() -> np.ndarray:
    """Make a matched filter that responds to seven-by-seven alternating crossings."""
    pitch = _TEMPLATE_PITCH
    template = np.zeros((8 * pitch, 8 * pitch), np.float32)
    for row, y in enumerate(_INNER * pitch):
        for col, x in enumerate(_INNER * pitch):
            sign = _INNER_PARITY[row, col]
            template[y - 4 : y - 1, x - 4 : x - 1] = sign
            template[y + 1 : y + 4, x + 1 : x + 4] = sign
            template[y - 4 : y - 1, x + 1 : x + 4] = -sign
            template[y + 1 : y + 4, x - 4 : x - 1] = -sign
    return template / (2 * 9 * 49)


def _propose(gray: np.ndarray) -> list[_Candidate]:
    """Scan scales for square grids whose crossings land at a common pitch."""
    template = _crossing_template()
    height, width = gray.shape
    proposals = []
    pitch = (_MIN_BOARD_SIDE_PX - _LINE_TOLERANCE_PX) / 8
    peak_window = np.ones((15, 15), np.uint8)
    while pitch <= min(width, height) / 8 * 1.04:
        scale = _TEMPLATE_PITCH / pitch
        scaled = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        # A small pad permits slightly undersized hypotheses at the image boundary.
        scaled = cv2.copyMakeBorder(scaled, 3, 3, 3, 3, cv2.BORDER_REPLICATE)
        response = cv2.matchTemplate(scaled, template, cv2.TM_CCORR)
        magnitude = abs(response)
        y_peaks, x_peaks = np.where(
            (magnitude >= 7) & (magnitude == cv2.dilate(magnitude, peak_window))
        )
        # A montage can contain many boards at exactly the same scale.
        rank = np.argsort(magnitude[y_peaks, x_peaks])[-240:]
        proposals.extend(
            _Candidate(
                (x_peaks[i] - 3) / scale,
                (y_peaks[i] - 3) / scale,
                pitch,
                float(response[y_peaks[i], x_peaks[i]]),
            )
            for i in rank
        )
        pitch *= 1.045
    return sorted(proposals, key=lambda candidate: abs(candidate.response), reverse=True)


def _fit_axis(
    derivative: np.ndarray,
    start: float,
    cross_start: float,
    pitch: float,
    polarity: float,
    *,
    check_spacing: bool = True,
) -> tuple[float, float] | None:
    """Refine seven rough lines and fit positions within the pixel tolerance."""
    expected = start + _INNER * pitch
    search = max(2, round(pitch * 0.28))
    displacement = np.arange(-search, search + 1)
    positions = np.rint(expected[:, None]).astype(int) + displacement
    positions = positions.clip(0, derivative.shape[1] - 1)
    along = cross_start + _INNER * pitch
    offset = pitch * 0.22
    above = np.rint(along - offset).astype(int).clip(0, derivative.shape[0] - 1)
    below = np.rint(along + offset).astype(int).clip(0, derivative.shape[0] - 1)
    edges = derivative[below[:, None, None], positions] - derivative[above[:, None, None], positions]
    response = np.mean(edges * _INNER_PARITY[:, :, None], axis=0) * polarity
    peaks = np.argmax(response, axis=1)
    observed = positions[np.arange(7), peaks] + 0.5
    weights = response[np.arange(7), peaks]

    if not check_spacing:
        strong = weights > 2
        if np.count_nonzero(strong) < 5:
            return None
        # Estimate the bounding grid directly, without selecting evenly spaced peaks.
        refined_pitch, origin = np.polyfit(_INNER[strong], observed[strong], 1)
        return float(origin), float(refined_pitch)

    # Use all pairs as pitch hypotheses so a piece crossing a line cannot skew the fit.
    first, second = _LINE_PAIRS
    pitches = (observed[second] - observed[first]) / (second - first)
    origins = observed[first] - _INNER[first] * pitches
    errors = abs(observed - (origins[:, None] + pitches[:, None] * _INNER))
    support = (errors <= _LINE_TOLERANCE_PX) & (weights > 2)
    support[abs(pitches - pitch) > max(1.5, pitch * 0.065)] = False
    best = np.argmax(support.sum(axis=1) + support @ weights.clip(0, 50) / 500)
    inliers = support[best]
    if np.count_nonzero(inliers) < 5:
        return None
    indices = _INNER[inliers]
    measured = observed[inliers]
    refined_pitch, origin = np.polyfit(indices, measured, 1)
    measured_pitches = np.diff(measured) / np.diff(indices)
    if np.count_nonzero(abs(measured_pitches - refined_pitch) <= _LINE_TOLERANCE_PX) < 3:
        return None
    return float(origin), float(refined_pitch)


def _junction_contrast(
    gray: np.ndarray, xs: np.ndarray, ys: np.ndarray, dx: float, dy: float, polarity: float
) -> np.ndarray:
    """Measure the alternating contrast on both arms of a checkerboard junction."""
    height, width = gray.shape
    sign = polarity * _PARITY
    samples = []
    for offset in (0.14, 0.21, 0.28):
        left = np.rint(xs - dx * offset).astype(int).clip(0, width - 1)
        right = np.rint(xs + dx * offset).astype(int).clip(0, width - 1)
        top = np.rint(ys - dy * offset).astype(int).clip(0, height - 1)
        bottom = np.rint(ys + dy * offset).astype(int).clip(0, height - 1)
        tl, tr = gray[np.ix_(top, left)], gray[np.ix_(top, right)]
        bl, br = gray[np.ix_(bottom, left)], gray[np.ix_(bottom, right)]
        horizontal = np.minimum((tl - tr) * sign, (br - bl) * sign)
        vertical = np.minimum((tl - bl) * sign, (br - tr) * sign)
        samples.append(np.minimum(horizontal, vertical))
    return np.max(samples, axis=0)


def _locate_axes(
    gx: np.ndarray,
    gy: np.ndarray,
    candidate: _Candidate,
    *,
    check_spacing: bool = True,
) -> tuple[float, float, float, float] | None:
    """Fit the horizontal and vertical grid coordinates from the directional gradients."""
    x, y, pitch = candidate.x, candidate.y, candidate.pitch
    polarity = np.sign(candidate.response)
    horizontal_axis = _fit_axis(gx, x, y, pitch, polarity, check_spacing=check_spacing)
    if horizontal_axis is None:
        return None
    vertical_axis = _fit_axis(gy.T, y, x, pitch, polarity, check_spacing=check_spacing)
    if vertical_axis is None:
        return None
    x, dx = horizontal_axis
    y, dy = vertical_axis
    return x, y, dx, dy


def _check_grid(
    gray: np.ndarray,
    axes: tuple[float, float, float, float],
    polarity: float,
    checks: _Checks,
) -> _Fit | None:
    """Validate the fitted axes and the contrasting junctions within their bounds."""
    x, y, dx, dy = axes
    height, width = gray.shape
    if round(8 * min(dx, dy)) < _MIN_BOARD_SIDE_PX:
        return None
    if checks.square and abs(dx - dy) > min(_LINE_TOLERANCE_PX, max(0.75, 0.05 * max(dx, dy))):
        return None
    if min(x, y, width - x - 8 * dx, height - y - 8 * dy) < -_LINE_TOLERANCE_PX:
        return None

    contrast = _junction_contrast(
        gray, x + _GRID_INDICES * dx, y + _GRID_INDICES * dy, dx, dy, polarity
    )
    inner = contrast[1:-1, 1:-1]
    supported = inner >= 6
    crossings = int(np.count_nonzero(supported))
    supported_cols = np.count_nonzero(supported.any(axis=0))
    supported_rows = np.count_nonzero(supported.any(axis=1))
    if checks.crossings and (crossings < 26 or min(supported_cols, supported_rows) < 6):
        return None
    interior_strength = float(inner.clip(0, 40).mean())
    if not checks.boundary:
        return _Fit(x, y, dx, dy, crossings, interior_strength)
    # A full extra row of crossings is evidence that the fit is shifted by one cell.
    border = np.array(
        [contrast[0, 1:-1], contrast[-1, 1:-1], contrast[1:-1, 0], contrast[1:-1, -1]]
    )
    continuation = float(np.max(border.clip(0, 40).mean(axis=1)))
    if continuation > 0.85 * interior_strength:
        return None
    strength = interior_strength - 0.65 * continuation
    return _Fit(x, y, dx, dy, crossings, strength)


def _overlaps(a: _Fit, b: _Fit) -> bool:
    """Identify duplicate grids while keeping neighbouring boards separate."""
    width = max(0, min(a.x + 8 * a.dx, b.x + 8 * b.dx) - max(a.x, b.x))
    height = max(0, min(a.y + 8 * a.dy, b.y + 8 * b.dy) - max(a.y, b.y))
    return width * height > 0.55 * 64 * min(a.dx * a.dy, b.dx * b.dy)


def _to_grid(fit: _Fit, width: int, height: int) -> LineGrid:
    """Convert a fitted grid to the public pixel-coordinate result."""
    xs = np.clip(fit.x + _GRID_INDICES * fit.dx, 0, width).round(3).tolist()
    ys = np.clip(fit.y + _GRID_INDICES * fit.dy, 0, height).round(3).tolist()
    return LineGrid(tuple(tuple((x, y) for x in xs) for y in ys), fit.crossings)


def _gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the grayscale image and the directional gradients used by every variant."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0.65)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    return smooth, gx, gy


def _distinct_fits(fits: list[_Fit]) -> list[_Fit]:
    """Prefer strong grids within each overlap group and sort them in image order."""
    distinct = []
    for fit in sorted(fits, key=lambda f: (f.strength, f.crossings), reverse=True):
        if not any(_overlaps(fit, previous) for previous in distinct):
            distinct.append(fit)
    return sorted(distinct, key=lambda f: (f.y, f.x))


def detect_line_grids(image: np.ndarray, *, check_square: bool = True) -> list[LineGrid]:
    """Return distinct 8×8 grids in screenshot order, with support counted at crossings."""
    smooth, gx, gy = _gradients(image)
    checks = _Checks(square=check_square)
    fits = []
    for candidate in _propose(smooth):
        axes = _locate_axes(gx, gy, candidate)
        if axes is None:
            continue
        fit = _check_grid(smooth, axes, np.sign(candidate.response), checks)
        if fit is not None:
            fits.append(fit)

    height, width = smooth.shape
    return [_to_grid(fit, width, height) for fit in _distinct_fits(fits)]
