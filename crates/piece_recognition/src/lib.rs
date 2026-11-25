//! CPU inference for the chessboard image classifier.

use anyhow::bail;
use tract_onnx::prelude::*;

const INPUT_SIDE: usize = 224;
const MODEL: &[u8] = include_bytes!("../assets/pieces.onnx");
const CLASSES: [Option<char>; 13] = [
    Some('p'),
    Some('r'),
    Some('n'),
    Some('b'),
    Some('q'),
    Some('k'),
    Some('P'),
    Some('R'),
    Some('N'),
    Some('B'),
    Some('Q'),
    Some('K'),
    None,
];

/// An initialized MobileNet and piece classifier that can be reused across the board.
pub struct Classifier {
    model: TypedRunnableModel<TypedModel>,
}

/// The MobileNet features and the 13 class probabilities in the original model order.
pub struct Prediction {
    /// MobileNet's feature vector before the Dense layers.
    pub features: [f32; 1024],
    /// Class probabilities ordered as black pieces, white pieces, and an empty square.
    pub probabilities: [f32; 13],
}

impl Prediction {
    /// Returns the most likely FEN piece, or None for an empty square.
    pub fn piece(&self) -> Option<char> {
        let mut best = 0;
        for (index, probability) in self.probabilities.iter().enumerate().skip(1) {
            if probability > &self.probabilities[best] {
                best = index;
            }
        }
        CLASSES[best]
    }
}

impl Classifier {
    /// Prepares the model embedded in Fast without reading a separate model file.
    pub fn embedded() -> TractResult<Self> {
        Self::from_onnx(MODEL)
    }

    /// Loads the exported model from bytes and prepares a reusable CPU execution plan.
    pub fn from_onnx(bytes: &[u8]) -> TractResult<Self> {
        let mut input = bytes;
        let model = tract_onnx::onnx()
            .model_for_read(&mut input)?
            .with_input_fact(0, f32::fact([1, INPUT_SIDE, INPUT_SIDE, 3]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    /// Classifies one tightly packed RGB8 square with the web app's preprocessing.
    pub fn predict(&self, rgb: &[u8], width: usize, height: usize) -> TractResult<Prediction> {
        let pixels = resize_for_model(rgb, width, height)?;
        let input = Tensor::from_shape(&[1, INPUT_SIDE, INPUT_SIDE, 3], &pixels)?;
        let outputs = self.model.run(tvec!(input.into()))?;
        let [features, probabilities] = outputs.as_slice() else {
            bail!("the piece classifier must return features and probabilities");
        };
        Ok(Prediction {
            features: features.as_slice::<f32>()?.try_into()?,
            probabilities: probabilities.as_slice::<f32>()?.try_into()?,
        })
    }
}

/// Computes source coordinates for TensorFlow.js bilinear interpolation with aligned corners.
fn samples(size: usize) -> [(usize, usize, f32); INPUT_SIDE] {
    std::array::from_fn(|index| {
        let source = index as f64 * (size - 1) as f64 / (INPUT_SIDE - 1) as f64;
        let start = source as usize;
        (start, (start + 1).min(size - 1), source.fract() as f32)
    })
}

/// Resizes RGB in channel order and normalizes the channels to the range 0–1.
fn resize_for_model(rgb: &[u8], width: usize, height: usize) -> TractResult<Vec<f32>> {
    if width == 0
        || height == 0
        || width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(3))
            != Some(rgb.len())
    {
        bail!("invalid RGB cell dimensions");
    }
    let columns = samples(width);
    let mut pixels = Vec::with_capacity(INPUT_SIDE * INPUT_SIDE * 3);
    for (top, bottom, dy) in samples(height) {
        for (left, right, dx) in columns {
            for channel in 0..3 {
                let pixel = |x, y| f32::from(rgb[(y * width + x) * 3 + channel]);
                let tl = pixel(left, top);
                let bl = pixel(left, bottom);
                let upper = tl + (pixel(right, top) - tl) * dx;
                let lower = bl + (pixel(right, bottom) - bl) * dx;
                pixels.push((upper + (lower - upper) * dy) / 255.0);
            }
        }
    }
    Ok(pixels)
}
