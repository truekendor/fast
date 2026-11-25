//! Grayscale preprocessing and sampling for the grid detector.

/// Holds an image or a directional gradient in row-major order.
pub(super) struct Raster {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl Raster {
    /// Returns one image row.
    pub fn row(&self, y: usize) -> &[f32] {
        &self.pixels[y * self.width..(y + 1) * self.width]
    }

    /// Samples an image pixel, replicating the nearest pixel beyond the edge.
    pub fn sample(&self, x: isize, y: isize) -> f32 {
        self.row(y.clamp(0, self.height as isize - 1) as usize)
            [x.clamp(0, self.width as isize - 1) as usize]
    }

    /// Swaps the axes so both grid directions can use the same line fitter.
    pub fn transpose(&self) -> Self {
        let mut pixels = vec![0.0; self.pixels.len()];
        for y in 0..self.height {
            for (x, value) in self.row(y).iter().enumerate() {
                pixels[x * self.height + y] = *value;
            }
        }
        Self {
            width: self.height,
            height: self.width,
            pixels,
        }
    }
}

/// Reflects the one-pixel neighbourhood at an image edge without duplicating the edge.
fn neighbours(index: usize, limit: usize) -> [usize; 3] {
    [
        if index == 0 { 1 } else { index - 1 },
        index,
        if index + 1 == limit {
            limit - 2
        } else {
            index + 1
        },
    ]
}

/// Smooths JPEG noise and computes horizontal and transposed vertical Sobel gradients.
pub(super) fn gradients(image: Raster) -> (Raster, Raster, Raster) {
    const SIDE: f32 = 0.189_913_26;
    const CENTRE: f32 = 1.0 - 2.0 * SIDE;
    let (width, height) = (image.width, image.height);
    let mut smooth = vec![0.0; image.pixels.len()];
    let mut row_blur = vec![0.0; image.pixels.len()];
    for (y, output) in row_blur.chunks_exact_mut(width).enumerate() {
        let row = image.row(y);
        for (x, value) in output.iter_mut().enumerate() {
            let [left, _, right] = neighbours(x, width);
            *value = (row[left] + row[right]) * SIDE + row[x] * CENTRE;
        }
    }
    for (y, output) in smooth.chunks_exact_mut(width).enumerate() {
        let [top, _, bottom] = neighbours(y, height);
        for (x, value) in output.iter_mut().enumerate() {
            *value = (row_blur[top * width + x] + row_blur[bottom * width + x]) * SIDE
                + row_blur[y * width + x] * CENTRE;
        }
    }
    let smooth = Raster {
        width,
        height,
        pixels: smooth,
    };
    let mut gx = vec![0.0; smooth.pixels.len()];
    let mut gy = vec![0.0; smooth.pixels.len()];
    for y in 0..height {
        let [top, _, bottom] = neighbours(y, height).map(|row| smooth.row(row));
        for x in 0..width {
            let [left, _, right] = neighbours(x, width);
            gx[y * width + x] = (top[right] - top[left]
                + 2.0 * (smooth.row(y)[right] - smooth.row(y)[left])
                + bottom[right]
                - bottom[left])
                * 0.25;
            gy[x * height + y] =
                (bottom[left] - top[left] + 2.0 * (bottom[x] - top[x]) + bottom[right]
                    - top[right])
                    * 0.25;
        }
    }
    (
        smooth,
        Raster {
            width,
            height,
            pixels: gx,
        },
        Raster {
            width: height,
            height: width,
            pixels: gy,
        },
    )
}

/// Holds the contributing source pixels for one resampled coordinate.
struct Weights {
    start: usize,
    coefficients: Vec<f32>,
}

/// Builds area weights for downscaling and linear weights for upscaling.
fn resampling_weights(length: usize, size: usize, scale: f64) -> Vec<Weights> {
    (0..size)
        .map(|index| {
            if scale >= 1.0 {
                let position = ((index as f64 + 0.5) / scale - 0.5).clamp(0.0, (length - 1) as f64);
                let start = position.floor() as usize;
                let part = position.fract() as f32;
                return Weights {
                    start,
                    coefficients: if start + 1 == length {
                        vec![1.0]
                    } else {
                        vec![1.0 - part, part]
                    },
                };
            }
            let left = index as f64 / scale;
            let right = ((index + 1) as f64 / scale).min(length as f64);
            let start = left.floor() as usize;
            let coefficients = (start..right.ceil() as usize)
                .map(|pixel| {
                    ((right.min((pixel + 1) as f64) - left.max(pixel as f64)) / (right - left))
                        as f32
                })
                .collect();
            Weights {
                start,
                coefficients,
            }
        })
        .collect()
}

/// Resamples the image at the template scale and replicates three border pixels.
pub(super) fn scaled_with_border(image: &Raster, scale: f64) -> Raster {
    const BORDER: usize = 3;
    let width = (image.width as f64 * scale).round_ties_even() as usize;
    let height = (image.height as f64 * scale).round_ties_even() as usize;
    let x_weights = resampling_weights(image.width, width, scale);
    let y_weights = resampling_weights(image.height, height, scale);
    let mut horizontal = vec![0.0; width * image.height];
    for (y, output) in horizontal.chunks_exact_mut(width).enumerate() {
        let row = image.row(y);
        for (value, weights) in output.iter_mut().zip(&x_weights) {
            *value = row[weights.start..]
                .iter()
                .zip(&weights.coefficients)
                .map(|(p, w)| p * w)
                .sum();
        }
    }
    let padded_width = width + BORDER * 2;
    let padded_height = height + BORDER * 2;
    let mut pixels = vec![0.0; padded_width * padded_height];
    for (y, weights) in y_weights.iter().enumerate() {
        let row = &mut pixels[(y + BORDER) * padded_width..(y + BORDER + 1) * padded_width];
        let (left, inside) = row.split_at_mut(BORDER);
        let (inner, right) = inside.split_at_mut(width);
        for (index, weight) in weights.coefficients.iter().enumerate() {
            let y = weights.start + index;
            for (value, sample) in inner
                .iter_mut()
                .zip(&horizontal[y * width..(y + 1) * width])
            {
                *value += sample * weight;
            }
        }
        left.fill(inner[0]);
        right.fill(inner[width - 1]);
    }
    for index in 0..BORDER {
        pixels.copy_within(
            BORDER * padded_width..(BORDER + 1) * padded_width,
            index * padded_width,
        );
        let last = (height + BORDER - 1) * padded_width;
        pixels.copy_within(
            last..last + padded_width,
            (height + BORDER + index) * padded_width,
        );
    }
    Raster {
        width: padded_width,
        height: padded_height,
        pixels,
    }
}
