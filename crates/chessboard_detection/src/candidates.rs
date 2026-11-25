//! Multiscale search for regular grids of alternating crossings.

use crate::{
    MIN_BOARD_SIDE,
    raster::{Raster, scaled_with_border},
};

const TEMPLATE_PITCH: usize = 10;
const TEMPLATE_SIZE: usize = 8 * TEMPLATE_PITCH;

/// Describes a candidate origin, pitch and signed checkerboard response.
pub(super) struct Candidate {
    pub x: f64,
    pub y: f64,
    pub pitch: f64,
    pub response: f32,
}

/// Adds the seven alternating three-pixel block differences of the template on one axis.
fn correlate_axis(image: &Raster) -> Raster {
    let width = image.width - TEMPLATE_SIZE + 1;
    let mut pixels = vec![0.0; width * image.height];
    let mut differences = vec![0.0; image.width - 7];
    for (y, output) in pixels.chunks_exact_mut(width).enumerate() {
        let row = image.row(y);
        for (delta, window) in differences.iter_mut().zip(row.windows(8)) {
            *delta = (window[0] + window[1] + window[2]) - (window[5] + window[6] + window[7]);
        }
        // The alternating junction template is an outer product of these one-axis filters.
        for (x, value) in output.iter_mut().enumerate() {
            let deltas = &differences[x + 6..x + 67];
            *value = deltas[0] - deltas[10] + deltas[20] - deltas[30] + deltas[40] - deltas[50]
                + deltas[60];
        }
    }
    Raster {
        width,
        height: image.height,
        pixels,
    }
}

/// Finds local peaks in the 15-pixel neighbourhood, keeping the strongest per scale.
fn scale_peaks(response: &Raster) -> Vec<(f32, usize, usize)> {
    const NORMALIZATION: f32 = 2.0 * 9.0 * 49.0;
    const RADIUS: usize = 7;
    let mut peaks = Vec::new();
    for y in 0..response.height {
        for (x, &value) in response.row(y).iter().enumerate() {
            let magnitude = value.abs();
            if magnitude < 7.0 * NORMALIZATION {
                continue;
            }
            let xs = x.saturating_sub(RADIUS)..(x + RADIUS + 1).min(response.width);
            let ys = y.saturating_sub(RADIUS)..(y + RADIUS + 1).min(response.height);
            if ys.into_iter().any(|row| {
                response.row(row)[xs.clone()]
                    .iter()
                    .any(|v| v.abs() > magnitude)
            }) {
                continue;
            }
            peaks.push((value / NORMALIZATION, x, y));
        }
    }
    peaks.sort_unstable_by(|a, b| {
        b.0.abs()
            .total_cmp(&a.0.abs())
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    peaks.truncate(240);
    peaks
}

/// Searches pitches supported by an upright board at least 80 pixels across.
pub(super) fn propose(image: &Raster) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut pitch = (MIN_BOARD_SIDE - crate::LINE_TOLERANCE) / 8.0;
    let max_pitch = image.width.min(image.height) as f64 / 8.0 * 1.04;
    while pitch <= max_pitch {
        let scale = TEMPLATE_PITCH as f64 / pitch;
        let scaled = scaled_with_border(image, scale);
        let horizontal = correlate_axis(&scaled);
        // Keep the response transposed to avoid copying it again just to read off the peaks.
        let response = correlate_axis(&horizontal.transpose());
        candidates.extend(
            scale_peaks(&response)
                .into_iter()
                .map(|(response, y, x)| Candidate {
                    x: (x as f64 - 3.0) / scale,
                    y: (y as f64 - 3.0) / scale,
                    pitch,
                    response,
                }),
        );
        pitch *= 1.045;
    }
    candidates.sort_by(|a, b| b.response.abs().total_cmp(&a.response.abs()));
    candidates
}
