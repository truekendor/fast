//! Refines candidate lines and checks their spacing, crossings and outer edges.

use crate::{LINE_TOLERANCE, MIN_BOARD_SIDE, candidates::Candidate, raster::Raster};

// A narrow crop at the image edge can remove the same fraction of a cell at any resolution.
const MAX_EDGE_CROP_FRACTION: f64 = 0.10;
/// Requires sufficient contrast at more than half of the 49 internal crossings.
const MIN_SUPPORTED_CROSSINGS: usize = 26;

/// Holds a fitted grid and the evidence used to resolve overlapping proposals.
pub(super) struct Fit {
    pub x: f64,
    pub y: f64,
    pub dx: f64,
    pub dy: f64,
    pub crossings: usize,
    pub strength: f32,
}

/// Finds and fits seven internal edges in one direction, allowing obscured lines.
fn fit_axis(
    derivative: &Raster,
    start: f64,
    cross_start: f64,
    pitch: f64,
    polarity: f32,
) -> Option<(f64, f64)> {
    let search = (pitch * 0.28).round_ties_even().max(2.0) as isize;
    let offset = pitch * 0.22;
    let rows: [(&[f32], &[f32]); 7] = std::array::from_fn(|index| {
        let crossing = cross_start + (index + 1) as f64 * pitch;
        let row =
            |y: f64| derivative.row((y.round_ties_even() as usize).min(derivative.height - 1));
        (row(crossing - offset), row(crossing + offset))
    });

    let mut observed = [0.0; 7];
    let mut weights = [0.0; 7];

    for index in 0..7 {
        let expected = (start + (index + 1) as f64 * pitch).round_ties_even() as isize;
        weights[index] = f32::NEG_INFINITY;

        for shift in -search..=search {
            let x = (expected + shift).clamp(0, derivative.width as isize - 1) as usize;
            let mut response = 0.0;

            for (cross, &(above, below)) in rows.iter().enumerate() {
                let edge = below[x] - above[x];
                response += if (cross + index).is_multiple_of(2) {
                    edge
                } else {
                    -edge
                };
            }

            response = response / 7.0 * polarity;

            if response > weights[index] {
                weights[index] = response;
                observed[index] = x as f64 + 0.5;
            }
        }
    }

    regular_lines(&observed, &weights, pitch)
}

/// Fits equally spaced positions through the strongest consensus of measured edges.
fn regular_lines(observed: &[f64; 7], weights: &[f32; 7], pitch: f64) -> Option<(f64, f64)> {
    let mut best = Vec::new();
    let mut score = 0.0;

    for first in 0..7 {
        for second in first + 1..7 {
            let step = (observed[second] - observed[first]) / (second - first) as f64;
            if (step - pitch).abs() > (pitch * 0.065).max(1.5) {
                continue;
            }

            let origin = observed[first] - step * (first + 1) as f64;
            let inliers = std::array::from_fn::<_, 7, _>(|index| {
                weights[index] > 2.0
                    && (observed[index] - origin - step * (index + 1) as f64).abs()
                        <= LINE_TOLERANCE + 1e-9
            });

            let support: f32 = inliers
                .iter()
                .zip(weights)
                .filter(|(keep, _)| **keep)
                .map(|(_, weight)| 1.0 + weight.min(50.0) / 500.0)
                .sum();

            if support > score {
                score = support;
                best.clear();
                best.extend(
                    inliers
                        .iter()
                        .enumerate()
                        .filter(|(_, keep)| **keep)
                        .map(|(index, _)| ((index + 1) as f64, observed[index])),
                );
            }
        }
    }

    if best.len() < 5 {
        return None;
    }

    let centre_x = best.iter().map(|&(x, _)| x).sum::<f64>() / best.len() as f64;
    let centre_y = best.iter().map(|&(_, y)| y).sum::<f64>() / best.len() as f64;
    let step = best
        .iter()
        .map(|&(x, y)| (x - centre_x) * (y - centre_y))
        .sum::<f64>()
        / best
            .iter()
            .map(|&(x, _)| (x - centre_x).powi(2))
            .sum::<f64>();

    let supported_steps = best
        .windows(2)
        .filter(|pair| {
            let measured = (pair[1].1 - pair[0].1) / (pair[1].0 - pair[0].0);
            (measured - step).abs() <= LINE_TOLERANCE + 1e-9
        })
        .count();
    (supported_steps >= 3).then_some((centre_y - step * centre_x, step))
}

/// Measures the weakest arm of a crossing at several distances from the intersection.
fn contrast(gray: &Raster, x: f64, y: f64, dx: f64, dy: f64, polarity: f32) -> f32 {
    // this part might need to be rewritten
    // or at least reasoned about

    // These heuristic distances come from the Python reference detector. Sampling at
    // 14%, 21% and 28% of a square keeps probes scale-independent and avoids relying
    // on one potentially noisy pixel near the intersection.
    [0.14, 0.21, 0.28]
        .into_iter()
        .map(|offset| {
            let left = (x - dx * offset).round_ties_even() as isize;
            let right = (x + dx * offset).round_ties_even() as isize;
            let top = (y - dy * offset).round_ties_even() as isize;
            let bottom = (y + dy * offset).round_ties_even() as isize;
            let tl = gray.sample(left, top);
            let tr = gray.sample(right, top);
            let bl = gray.sample(left, bottom);
            let br = gray.sample(right, bottom);
            ((tl - tr) * polarity)
                .min((br - bl) * polarity)
                .min((tl - bl) * polarity)
                .min((br - tr) * polarity)
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Confirms that both arms alternate throughout the grid and stop at its outer edges.
fn crossing_evidence(
    gray: &Raster,
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    polarity: f32,
) -> Option<(usize, f32)> {
    let mut crossings = 0;
    let mut rows = [false; 7];
    let mut cols = [false; 7];
    let mut interior = 0.0;
    let mut sides = [0.0; 4];

    for row in 0_usize..=8 {
        for col in 0_usize..=8 {
            let inside_x = (1..8).contains(&col);
            let inside_y = (1..8).contains(&row);
            if !inside_x && !inside_y {
                continue;
            }

            let sign = if (row + col).is_multiple_of(2) {
                polarity
            } else {
                -polarity
            };

            let value = contrast(gray, x + col as f64 * dx, y + row as f64 * dy, dx, dy, sign);

            if inside_x && inside_y {
                if value >= 6.0 {
                    rows[row - 1] = true;
                    cols[col - 1] = true;
                    crossings += 1;
                }
                interior += value.clamp(0.0, 40.0);
            } else {
                let side = if !inside_y {
                    usize::from(row == 8)
                } else {
                    2 + usize::from(col == 8)
                };
                sides[side] += value.clamp(0.0, 40.0);
            }
        }
    }

    if crossings < MIN_SUPPORTED_CROSSINGS
        || rows.into_iter().filter(|value| *value).count() < 6
        || cols.into_iter().filter(|value| *value).count() < 6
    {
        return None;
    }

    let interior = interior / 49.0;
    let continuation = sides.into_iter().fold(0.0, f32::max) / 7.0;
    (continuation <= 0.85 * interior).then_some((crossings, interior - 0.65 * continuation))
}

/// Allows a narrow crop of an outer edge while keeping all internal grid lines visible.
fn fits_axis_in_image(origin: f64, pitch: f64, extent: usize) -> bool {
    let tolerance = LINE_TOLERANCE.max(pitch * MAX_EDGE_CROP_FRACTION);
    origin >= -tolerance && origin + 8.0 * pitch <= extent as f64 + tolerance
}

/// Refines a candidate and keeps grids within the usable crop tolerance.
pub(super) fn refine(
    gray: &Raster,
    gx: &Raster,
    gy: &Raster,
    candidate: &Candidate,
) -> Option<Fit> {
    let polarity = candidate.response.signum();
    let (x, dx) = fit_axis(gx, candidate.x, candidate.y, candidate.pitch, polarity)?;
    let (y, dy) = fit_axis(gy, candidate.y, candidate.x, candidate.pitch, polarity)?;
    if (8.0 * dx.min(dy)).round_ties_even() < MIN_BOARD_SIDE
        || !fits_axis_in_image(x, dx, gray.width)
        || !fits_axis_in_image(y, dy, gray.height)
    {
        return None;
    }

    let (crossings, strength) = crossing_evidence(gray, x, y, dx, dy, polarity)?;
    Some(Fit {
        x,
        y,
        dx,
        dy,
        crossings,
        strength,
    })
}

/// Identifies duplicate detections by the overlap as a fraction of the smaller board.
pub(super) fn overlaps(a: &Fit, b: &Fit) -> bool {
    let width = ((a.x + 8.0 * a.dx).min(b.x + 8.0 * b.dx) - a.x.max(b.x)).max(0.0);
    let height = ((a.y + 8.0 * a.dy).min(b.y + 8.0 * b.dy) - a.y.max(b.y)).max(0.0);
    width * height > 0.55 * 64.0 * (a.dx * a.dy).min(b.dx * b.dy)
}
