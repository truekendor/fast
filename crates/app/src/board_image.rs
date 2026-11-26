//! Shared image decoding, grid detection and piece recognition for Fast.

use std::{io::Cursor, path::Path, sync::OnceLock};

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

/// Opens an image file and converts it to the RGB format used by the classifier.
pub(crate) fn read(path: &Path) -> anyhow::Result<RgbImage> {
    Ok(image::ImageReader::open(path)
        .with_context(|| format!("could not open {}", path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("could not decode {}", path.display()))?
        .into_rgb8())
}

/// Decodes an in-memory screenshot using the same image reader as files.
pub(crate) fn read_bytes(bytes: &[u8]) -> anyhow::Result<RgbImage> {
    Ok(image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()
        .context("could not decode the clipboard image")?
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

/// Lazily prepares the embedded model and shares its inference plan between requests.
#[derive(Default)]
pub(crate) struct BoardRecognizer {
    classifier: OnceLock<Result<Classifier, String>>,
}

impl BoardRecognizer {
    /// Loads the embedded classifier only when a board first needs inference.
    fn classifier(&self) -> anyhow::Result<&Classifier> {
        self.classifier
            .get_or_init(|| Classifier::embedded().map_err(|err| format!("{err:#}")))
            .as_ref()
            .map_err(|err| anyhow::anyhow!("could not load the piece classifier: {err}"))
    }

    /// Finds every board and classifies the pieces in each detected grid.
    pub(crate) fn recognize(
        &self,
        image: &RgbImage,
        scalar: bool,
    ) -> anyhow::Result<Vec<RecognizedBoard>> {
        let grids = detect(image, scalar);
        if grids.is_empty() {
            return Ok(Vec::new());
        }
        let classifier = self.classifier()?;
        grids
            .into_iter()
            .map(|grid| recognize_pieces(classifier, image, grid))
            .collect()
    }

    /// Recognizes an axis-aligned square chosen by the user without running grid detection.
    pub(crate) fn recognize_selection(
        &self,
        image: &RgbImage,
        left: u32,
        top: u32,
        side: u32,
    ) -> anyhow::Result<Board> {
        anyhow::ensure!(
            side >= 8
                && side <= image.width().saturating_sub(left)
                && side <= image.height().saturating_sub(top),
            "the board selection must be inside the image and at least eight pixels wide"
        );
        let intersections = std::array::from_fn(|row| {
            std::array::from_fn(|column| {
                [
                    f64::from(left) + f64::from(side) * column as f64 / 8.,
                    f64::from(top) + f64::from(side) * row as f64 / 8.,
                ]
            })
        });
        let grid = Grid {
            intersections,
            matched_crossings: 0,
        };
        Ok(recognize_pieces(self.classifier()?, image, grid)?.board)
    }
}

/// Crops a region using the same pixel boundaries for previews and inference.
fn crop(image: &RgbImage, start: [f64; 2], end: [f64; 2]) -> RgbImage {
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
