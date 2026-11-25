//! Routes Analyse imports by source and reads chess files on a worker thread.

use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use chess_core::{GameTree, parse_epd, parse_pgn};
use gpui::{ClipboardEntry, ClipboardItem, Image};

/// Screenshot data from an external file or the system clipboard.
pub(crate) enum ImageInput {
    File(PathBuf),
    Clipboard(Image),
}

impl ImageInput {
    /// Decodes the image without requiring a temporary file for clipboard content.
    pub(crate) fn read(&self) -> anyhow::Result<image::RgbImage> {
        match self {
            Self::File(path) => crate::board_image::read(path),
            Self::Clipboard(image) => crate::board_image::read_bytes(&image.bytes),
        }
    }
}

/// The supported forms of clipboard input for Analyse.
pub(crate) enum ClipboardImport {
    Files(Vec<PathBuf>),
    Screenshot(ImageInput),
    Text(String),
}

/// Prefers actual file paths and image data over accompanying clipboard text.
pub(crate) fn clipboard_input(item: ClipboardItem) -> Option<ClipboardImport> {
    let text = item.text();
    let mut screenshot = None;
    let mut files = Vec::new();
    for entry in item.entries {
        match entry {
            ClipboardEntry::ExternalPaths(paths) => files.extend_from_slice(paths.paths()),
            ClipboardEntry::Image(image) if screenshot.is_none() => screenshot = Some(image),
            _ => {}
        }
    }
    if !files.is_empty() {
        return Some(ClipboardImport::Files(files));
    }
    screenshot
        .map(|image| ClipboardImport::Screenshot(ImageInput::Clipboard(image)))
        .or_else(|| text.map(ClipboardImport::Text))
}

/// Separates screenshot paths from files to be read as chess notation.
pub(crate) fn split_files(paths: Vec<PathBuf>) -> (Vec<ImageInput>, Vec<PathBuf>) {
    let (images, games): (Vec<_>, Vec<_>) = paths.into_iter().partition(|path| is_screenshot(path));
    (images.into_iter().map(ImageInput::File).collect(), games)
}

/// Checks the formats advertised for drag-and-drop without opening the files.
pub(crate) fn supports_drop(paths: &[PathBuf]) -> bool {
    !paths.is_empty()
        && paths.iter().all(|path| {
            is_screenshot(path) || has_extension(path, &["pgn", "fen", "epd"]) || is_pgn_gzip(path)
        })
}

/// Recognizes the image formats accepted by the screenshot decoder.
fn is_screenshot(path: &Path) -> bool {
    has_extension(path, &["png", "jpeg", "jpg", "bmp", "webp"])
}

/// Matches filename extensions without regard to case.
fn has_extension(path: &Path, formats: &[&str]) -> bool {
    path.extension().is_some_and(|ext| {
        formats
            .iter()
            .any(|format| ext.eq_ignore_ascii_case(format))
    })
}

/// Recognizes the two-part extension used by gzip-compressed PGN files.
fn is_pgn_gzip(path: &Path) -> bool {
    has_extension(path, &["gz"])
        && path
            .file_stem()
            .is_some_and(|stem| has_extension(Path::new(stem), &["pgn"]))
}

/// Reads all games and positions, preserving successful files when another import fails.
pub(crate) fn read_game_files(paths: &[PathBuf]) -> (Vec<GameTree>, Vec<String>) {
    let mut games = Vec::new();
    let mut errors = Vec::new();
    for path in paths {
        let text = match read_game_text(path) {
            Ok(text) => text,
            Err(err) => {
                errors.push(format!("{}: {err}", path.display()));
                continue;
            }
        };
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("epd"))
        {
            let records = parse_epd(&text);
            if records.is_empty() {
                errors.push(format!("{}: no positions", path.display()));
            }
            for record in records {
                match record.parsed {
                    Ok(position) => games.push(GameTree::from_position(position.pos)),
                    Err(err) => errors.push(format!("{}:{}: {err}", path.display(), record.line)),
                }
            }
            continue;
        }
        let text = text.trim_start_matches('\u{feff}').trim();
        if let Ok(tree) = GameTree::from_fen(text) {
            games.push(tree);
            continue;
        }
        match parse_pgn(text) {
            Ok(mut loaded) if !loaded.is_empty() => games.append(&mut loaded),
            Ok(_) => errors.push(format!("{}: no games", path.display())),
            Err(err) => errors.push(format!("{}: {err}", path.display())),
        }
    }
    (games, errors)
}

/// Reads plain notation or decompresses a PGN.gz using Fast's existing gzip support.
fn read_game_text(path: &Path) -> Result<String, String> {
    if is_pgn_gzip(path) {
        let input = File::open(path).map_err(|err| err.to_string())?;
        let mut text = String::new();
        flate2::read::MultiGzDecoder::new(input)
            .read_to_string(&mut text)
            .map_err(|err| err.to_string())?;
        Ok(text)
    } else {
        std::fs::read_to_string(path).map_err(|err| err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::ImageFormat;
    use std::io::{Cursor, Write as _};

    /// Windows can provide a screenshot as BMP alongside clipboard text.
    #[test]
    fn clipboard_bitmap_takes_precedence_over_text_and_decodes() {
        let pixel = image::RgbImage::from_pixel(2, 2, image::Rgb([64, 128, 200]));
        let mut bmp = Cursor::new(Vec::new());
        pixel.write_to(&mut bmp, image::ImageFormat::Bmp).unwrap();
        let mut clipboard = ClipboardItem::new_string("Accompanying text".into());
        clipboard
            .entries
            .push(ClipboardEntry::Image(Image::from_bytes(
                ImageFormat::Bmp,
                bmp.into_inner(),
            )));
        let Some(ClipboardImport::Screenshot(screenshot)) = clipboard_input(clipboard) else {
            panic!("expected clipboard image data");
        };
        assert_eq!(screenshot.read().unwrap(), pixel);
        assert!(matches!(
            clipboard_input(ClipboardItem::new_string("1. e4 *".into())),
            Some(ClipboardImport::Text(text)) if text == "1. e4 *"
        ));
    }

    /// A mixed drop keeps screenshots and opens valid compressed games and EPD rows.
    #[test]
    fn routes_screenshots_and_reads_pgn_gz_and_epd() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("board.PNG");
        let pgn = dir.path().join("test.PGN.GZ");
        let epd = dir.path().join("positions.epd");
        let mut gzip =
            flate2::write::GzEncoder::new(File::create(&pgn).unwrap(), flate2::Compression::fast());
        gzip.write_all(b"[Event \"Imported\"]\n\n1. e4 *").unwrap();
        gzip.finish().unwrap();
        std::fs::write(
            &epd,
            "broken\n8/4k1p1/8/pr2Nb2/3R2P1/8/5PP1/2K5 w - - id \"Position\";\n",
        )
        .unwrap();

        let (images, paths) = split_files(vec![pgn, png.clone(), epd]);
        assert!(matches!(&images[..], [ImageInput::File(path)] if path == &png));
        let (games, errors) = read_game_files(&paths);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].header("Event"), Some("Imported"));
        assert_eq!(
            games[1].fen(GameTree::ROOT),
            "8/4k1p1/8/pr2Nb2/3R2P1/8/5PP1/2K5 w - - 0 1"
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("positions.epd:1"));
    }
    /// Shows the import prompt only when every dragged path has a supported extension.
    #[test]
    fn drop_accepts_supported_formats_and_rejects_mixed_batches() {
        let mut paths = [
            "board.PNG",
            "board.jpeg",
            "board.jpg",
            "board.bmp",
            "board.webp",
            "game.pgn",
            "game.PGN.GZ",
            "position.fen",
            "positions.EPD",
        ]
        .map(PathBuf::from)
        .to_vec();
        assert!(supports_drop(&paths));
        paths.push(PathBuf::from("archive.zip"));
        assert!(!supports_drop(&paths));
        assert!(!supports_drop(&[PathBuf::from("board.txt.gz")]));
        assert!(!supports_drop(&[PathBuf::from("board")]));
        assert!(!supports_drop(&[]));
    }
}
