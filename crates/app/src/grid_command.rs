//! Runs standalone grid and piece recognition through the Fast executable.

use std::{ffi::OsStr, path::Path, process::ExitCode};

use anyhow::{Context, anyhow};
use chessboard_detection::Grid;
use serde::Serialize;

use crate::board_image;

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
    let image = board_image::read(input)?;
    if recognize {
        let boards = board_image::BoardRecognizer::default()
            .recognize(&image, scalar)?
            .into_iter()
            .map(|board_image::RecognizedBoard { grid, mut board }| {
                let placement = board.board_fen().to_string();
                board.rotate_180();
                BoardReport {
                    grid: grid.into(),
                    placement,
                    flipped_placement: board.board_fen().to_string(),
                }
            })
            .collect();
        write_report(output, &RecognitionReport { boards })
    } else {
        let grids = board_image::detect(&image, scalar)
            .into_iter()
            .map(GridReport::from)
            .collect();
        write_report(output, &Report { grids })
    }
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
