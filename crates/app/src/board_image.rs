//! Shared image decoding, grid detection and piece recognition for Fast.

use std::path::Path;

use anyhow::Context as _;
use chess_core::shakmaty::{Board, Piece, Square};
use chessboard_detection::Grid;
use image::RgbImage;
use piece_recognition::Classifier;

/// A detected grid and its predicted pieces, with the source image's top treated as rank eight.
pub(crate) struct RecognizedBoard {
    pub grid: Grid,
    pub board: Board,
}

/// Opens a PNG or JPEG and converts it to the RGB format used by the classifier.
pub(crate) fn read(path: &Path) -> anyhow::Result<RgbImage> {
    Ok(image::ImageReader::open(path)
        .with_context(|| format!("could not open {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("could not decode {}", path.display()))?
        .into_rgb8())
}

/// Finds the board's intersections in source pixels.
pub(crate) fn detect(image: &RgbImage, scalar: bool) -> Vec<Grid> {
    // Use the same BT.601 grayscale conversion as the OpenCV benchmark.
    let pixels: Vec<_> = image
        .as_raw()
        .chunks_exact(3)
        .map(|pixel| {
            let (red, green, blue) = (
                u32::from(pixel[0]),
                u32::from(pixel[1]),
                u32::from(pixel[2]),
            );
            ((red * 9798 + green * 19235 + blue * 3735 + 16384) >> 15) as u8
        })
        .collect();
    let detect = if scalar {
        chessboard_detection::detect_grids_scalar
    } else {
        chessboard_detection::detect_grids
    };
    detect(&pixels, image.width() as usize, image.height() as usize)
}

/// Finds every board and reuses one classifier for all of its cells.
pub(crate) fn recognize(image: &RgbImage, scalar: bool) -> anyhow::Result<Vec<RecognizedBoard>> {
    let grids = detect(image, scalar);
    if grids.is_empty() {
        return Ok(Vec::new());
    }
    let classifier = Classifier::embedded().context("could not load the piece classifier")?;
    grids
        .into_iter()
        .map(|grid| recognize_pieces(&classifier, image, grid))
        .collect()
}

/// Crops a region using the same pixel boundaries for previews and inference.
pub(crate) fn crop(image: &RgbImage, start: [f64; 2], end: [f64; 2]) -> RgbImage {
    let [left, top] = start.map(|coordinate| coordinate.round() as u32);
    let [right, bottom] = end.map(|coordinate| coordinate.round() as u32);
    image::imageops::crop_imm(image, left, top, right - left, bottom - top).to_image()
}

/// Classifies the 64 cells and records each piece at its corresponding square.
fn recognize_pieces(
    classifier: &Classifier,
    image: &RgbImage,
    grid: Grid,
) -> anyhow::Result<RecognizedBoard> {
    let mut board = Board::empty();
    for index in 0..64 {
        let (row, col) = (index / 8, index % 8);
        let cell = crop(
            image,
            grid.intersections[row][col],
            grid.intersections[row + 1][col + 1],
        );
        if let Some(piece) = classifier
            .predict(cell.as_raw(), cell.width() as usize, cell.height() as usize)?
            .piece()
            .and_then(Piece::from_char)
        {
            board.set_piece_at(Square::new(((7 - row) * 8 + col) as u32), piece);
        }
    }
    Ok(RecognizedBoard { grid, board })
}
