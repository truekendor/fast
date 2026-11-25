//! Runs standalone grid and piece recognition through the Fast executable.

use std::{ffi::OsStr, path::Path, process::ExitCode};

use anyhow::{Context, anyhow};
use chess_core::shakmaty::{Board, Piece, Square};
use chessboard_detection::Grid;
use piece_recognition::Classifier;
use serde::Serialize;

/// Represents one detected grid in the JSON output.
#[derive(Serialize)]
struct GridReport {
    intersections: [[[f64; 2]; 9]; 9],
    matched_crossings: usize,
}

/// Represents all grids found in the screenshot.
#[derive(Serialize)]
struct Report {
    grids: Vec<GridReport>,
}

/// Represents the board geometry and its two possible FEN placements.
#[derive(Serialize)]
struct BoardReport {
    grid: GridReport,
    placement: String,
    flipped_placement: String,
}

/// Represents all recognized boards in the screenshot.
#[derive(Serialize)]
struct RecognitionReport {
    boards: Vec<BoardReport>,
}

impl From<Grid> for GridReport {
    /// Copies the grid data into the stable JSON format.
    fn from(grid: Grid) -> Self {
        Self {
            intersections: grid.intersections,
            matched_crossings: grid.matched_crossings,
        }
    }
}

/// Handles an image command before GPUI and profile loading start.
pub(crate) fn run_if_requested() -> Option<ExitCode> {
    let mut args = std::env::args_os().skip(1);
    let command = args.next()?;
    let recognize = match command.to_str()? {
        "--detect-grid" => false,
        "--recognize-board" => true,
        _ => return None,
    };
    let image = args.next();
    let output = args.next();
    let mode = args.next();
    let scalar = mode.as_deref() == Some(OsStr::new("--scalar"));
    let result = match (image, output, args.next()) {
        (Some(image), Some(output), None) if mode.is_none() || scalar => {
            detect(Path::new(&image), Path::new(&output), scalar, recognize)
        }
        _ => Err(anyhow!(
            "usage: fast {} IMAGE OUTPUT.json [--scalar]",
            command.to_string_lossy(),
        )),
    };
    Some(match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::FAILURE
        }
    })
}

/// Decodes the screenshot, detects its grids, and recognizes pieces when requested.
fn detect(input: &Path, output: &Path, scalar: bool, recognize: bool) -> anyhow::Result<()> {
    let image = image::ImageReader::open(input)
        .with_context(|| format!("could not open {}", input.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("could not decode {}", input.display()))?
        .into_rgb8();
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
    let grids = detect(&pixels, image.width() as usize, image.height() as usize);
    if recognize {
        let boards = if grids.is_empty() {
            Vec::new()
        } else {
            let classifier =
                Classifier::embedded().context("could not load the piece classifier")?;
            grids
                .into_iter()
                .map(|grid| recognize_pieces(&classifier, &image, grid))
                .collect::<anyhow::Result<_>>()?
        };
        write_report(output, &RecognitionReport { boards })
    } else {
        let grids = grids.into_iter().map(GridReport::from).collect();
        write_report(output, &Report { grids })
    }
}

/// Classifies the 64 RGB cells in image order and formats their two board orientations.
fn recognize_pieces(
    classifier: &Classifier,
    image: &image::RgbImage,
    grid: Grid,
) -> anyhow::Result<BoardReport> {
    let mut board = Board::empty();
    for index in 0..64 {
        let (row, col) = (index / 8, index % 8);
        let [left, top] = grid.intersections[row][col].map(|coordinate| coordinate.round() as u32);
        let [right, bottom] =
            grid.intersections[row + 1][col + 1].map(|coordinate| coordinate.round() as u32);
        let (width, height) = (right - left, bottom - top);
        let cell = image::imageops::crop_imm(image, left, top, width, height).to_image();
        if let Some(piece) = classifier
            .predict(cell.as_raw(), width as usize, height as usize)?
            .piece()
            .and_then(Piece::from_char)
        {
            board.set_piece_at(Square::new(((7 - row) * 8 + col) as u32), piece);
        }
    }
    let placement = board.board_fen().to_string();
    board.rotate_180();
    Ok(BoardReport {
        grid: grid.into(),
        placement,
        flipped_placement: board.board_fen().to_string(),
    })
}

/// Writes a complete JSON file without replacing an existing path.
fn write_report(output: &Path, report: &impl Serialize) -> anyhow::Result<()> {
    let directory = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    serde_json::to_writer(file.as_file_mut(), report)?;
    file.persist_noclobber(output)
        .with_context(|| format!("could not create {}", output.display()))?;
    Ok(())
}
