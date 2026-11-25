//! Detects upright 8×8 chessboard grids without a UI or native image libraries.

mod candidates;
mod fit;
mod raster;

use fit::{Fit, overlaps, refine};
use raster::{Raster, gradients};

const MIN_BOARD_SIDE: f64 = 80.0;
const LINE_TOLERANCE: f64 = 2.0;

/// A grid in source pixel coordinates, ordered from the top-left of the playing area.
#[derive(Debug, PartialEq)]
pub struct Grid {
    /// Nine rows of nine `[x, y]` intersections in image order.
    pub intersections: [[[f64; 2]; 9]; 9],
    /// The number of internal intersections that support the grid.
    pub matched_crossings: usize,
}

/// Detects distinct grids in row-major 8-bit grayscale pixels, in image order.
/// Returns an empty list for mismatched image dimensions or insufficient evidence.
pub fn detect_grids(pixels: &[u8], width: usize, height: usize) -> Vec<Grid> {
    if width.min(height) < MIN_BOARD_SIDE as usize
        || width.checked_mul(height) != Some(pixels.len())
    {
        return Vec::new();
    }
    let image = Raster {
        width,
        height,
        pixels: pixels.iter().map(|&value| f32::from(value)).collect(),
    };
    let (smooth, gx, gy) = gradients(image);
    let mut fits: Vec<_> = candidates::propose(&smooth)
        .iter()
        .filter_map(|candidate| refine(&smooth, &gx, &gy, candidate))
        .collect();
    fits.sort_by(|a, b| {
        b.strength
            .total_cmp(&a.strength)
            .then(b.crossings.cmp(&a.crossings))
    });
    let mut distinct = Vec::new();
    for fit in fits {
        if !distinct.iter().any(|previous| overlaps(&fit, previous)) {
            distinct.push(fit);
        }
    }
    distinct.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
    distinct
        .into_iter()
        .map(|fit| grid(fit, width, height))
        .collect()
}

/// Converts the fitted lines into intersections bounded by the source image.
fn grid(fit: Fit, width: usize, height: usize) -> Grid {
    Grid {
        intersections: std::array::from_fn(|row| {
            std::array::from_fn(|col| {
                [
                    (fit.x + col as f64 * fit.dx).clamp(0.0, width as f64),
                    (fit.y + row as f64 * fit.dy).clamp(0.0, height as f64),
                ]
            })
        }),
        matched_crossings: fit.crossings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Detects complete boards in source coordinates while skipping a thumbnail.
    fn finds_multiple_boards_at_supported_sizes() {
        let (width, height) = (330, 130);
        let mut pixels = vec![55; width * height];
        for (left, top, pitch) in [(9, 12, 8), (92, 16, 10), (199, 26, 12)] {
            for y in 0..8 * pitch {
                for x in 0..8 * pitch {
                    pixels[(top + y) * width + left + x] =
                        if (x / pitch + y / pitch).is_multiple_of(2) {
                            100
                        } else {
                            200
                        };
                }
            }
        }
        let grids = detect_grids(&pixels, width, height);
        assert_eq!(grids.len(), 2);
        for (grid, expected) in grids.iter().zip([[92.0, 16.0], [199.0, 26.0]]) {
            for (coordinate, position) in grid.intersections[0][0].into_iter().zip(expected) {
                assert!((coordinate - position).abs() <= 1.0);
            }
        }
        assert!(detect_grids(&pixels[..pixels.len() - 1], width, height).is_empty());
    }
}
