//! Runs standalone grid detection through the Fast executable.

use std::{ffi::OsStr, path::Path, process::ExitCode};

use anyhow::{Context, anyhow};
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

/// Handles a detection request before GPUI and profile loading start.
pub(crate) fn run_if_requested() -> Option<ExitCode> {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(OsStr::new("--detect-grid")) {
        return None;
    }
    let image = args.next();
    let output = args.next();
    let mode = args.next();
    let scalar = mode.as_deref() == Some(OsStr::new("--scalar"));
    let result = match (image, output, args.next()) {
        (Some(image), Some(output), None) if mode.is_none() || scalar => {
            detect(Path::new(&image), Path::new(&output), scalar)
        }
        _ => Err(anyhow!(
            "usage: fast --detect-grid IMAGE OUTPUT.json [--scalar]"
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

/// Decodes the screenshot and writes one JSON result to a new file.
fn detect(input: &Path, output: &Path, scalar: bool) -> anyhow::Result<()> {
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
    let grids: Vec<_> = detect(&pixels, image.width() as usize, image.height() as usize)
        .into_iter()
        .map(|grid| GridReport {
            intersections: grid.intersections,
            matched_crossings: grid.matched_crossings,
        })
        .collect();
    let directory = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(directory)?;
    serde_json::to_writer(file.as_file_mut(), &Report { grids })?;
    file.persist_noclobber(output)
        .with_context(|| format!("could not create {}", output.display()))?;
    Ok(())
}
