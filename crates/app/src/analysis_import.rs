//! Routes Analyse imports by source and reads chess files on a worker thread.

use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor, Read},
    ops::ControlFlow,
    path::{Path, PathBuf},
};

use chess_core::{GameTree, parse_epd, pgn::read_pgn_games};
use gpui::{ClipboardEntry, ClipboardItem, Image};

/// Image formats accepted by the screenshot decoder.
pub(crate) const SCREENSHOT_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "webp"];

/// Extensions for uncompressed chess notation files.
pub(crate) const CHESS_FILE_EXTENSIONS: &[&str] = &["pgn", "fen", "epd"];

/// The compound extension for gzip-compressed PGN.
pub(crate) const COMPRESSED_PGN_EXTENSION: &str = "pgn.gz";

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
            is_screenshot(path) || has_extension(path, CHESS_FILE_EXTENSIONS) || is_pgn_gzip(path)
        })
}

/// Recognizes the image formats accepted by the screenshot decoder.
fn is_screenshot(path: &Path) -> bool {
    has_extension(path, SCREENSHOT_EXTENSIONS)
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
pub(crate) fn is_pgn_gzip(path: &Path) -> bool {
    path.file_stem()
        .zip(path.extension())
        .is_some_and(|(stem, ext)| {
            let (inner, outer) = COMPRESSED_PGN_EXTENSION.split_once('.').unwrap();
            ext.eq_ignore_ascii_case(outer) && has_extension(Path::new(stem), &[inner])
        })
}

/// Separates multi-game PGNs from files that should open in Analyse.
pub(crate) fn split_game_collections(paths: Vec<PathBuf>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    paths.into_iter().partition(|path| is_pgn_collection(path))
}

/// Looks for a second game to decide whether the PGN is a collection.
fn is_pgn_collection(path: &Path) -> bool {
    if !has_extension(path, &["pgn"]) && !is_pgn_gzip(path) {
        return false;
    }
    game_input(path).is_ok_and(|input| {
        read_pgn_games(input)
            .take_while(|result| !matches!(result, Err(chess_core::PgnError::Io(_))))
            .nth(1)
            .is_some()
    })
}

/// Reports each game, position or error as it is read, until the receiver closes.
pub(crate) fn read_game_files(
    paths: &[PathBuf],
    mut report: impl FnMut(Result<GameTree, String>) -> ControlFlow<()>,
) {
    for path in paths {
        let mut report = |result: Result<GameTree, String>| {
            report(result.map_err(|err| format!("{}: {err}", path.display())))
        };
        let result = if has_extension(path, &["epd"]) {
            read_epd_file(path, &mut report)
        } else {
            read_game_file(path, &mut report)
        };
        if result.is_break() {
            return;
        }
    }
}

/// Opens plain or gzip-compressed notation without reading the whole file.
pub(crate) fn game_input(path: &Path) -> std::io::Result<BufReader<Box<dyn Read>>> {
    let file = File::open(path)?;
    let input: Box<dyn Read> = if is_pgn_gzip(path) {
        Box::new(flate2::read::MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(BufReader::new(input))
}

/// Allows a standalone FEN in any supported game file, as before.
fn standalone_fen(text: &str) -> Option<GameTree> {
    GameTree::from_fen(text.trim_start_matches('\u{feff}').trim()).ok()
}

/// Keeps successful games from a file even when another game cannot be read.
fn read_game_file(
    path: &Path,
    report: &mut impl FnMut(Result<GameTree, String>) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let mut input = match game_input(path) {
        Ok(input) => input,
        Err(err) => return report(Err(err.to_string())),
    };
    let prefix = match input.fill_buf() {
        Ok(prefix) => prefix,
        Err(err) => return report(Err(err.to_string())),
    };
    // Read ahead only when the prefix could be a standalone FEN.
    let mut text = String::new();
    if std::str::from_utf8(prefix)
        .ok()
        .and_then(standalone_fen)
        .is_some()
    {
        if let Err(err) = input.read_to_string(&mut text) {
            return report(Err(err.to_string()));
        }
        if let Some(tree) = standalone_fen(&text) {
            return report(Ok(tree));
        }
    }
    let mut games = read_pgn_games(Cursor::new(text).chain(input)).peekable();
    if games.peek().is_none() {
        return report(Err("no games".to_owned()));
    }
    for game in games {
        report(game.map_err(|err| err.to_string()))?;
    }
    ControlFlow::Continue(())
}

/// Reports each EPD position or its source line on error.
fn read_epd_file(
    path: &Path,
    report: &mut impl FnMut(Result<GameTree, String>) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => return report(Err(err.to_string())),
    };
    let records = parse_epd(&text);
    if records.is_empty() {
        return report(Err("no positions".to_owned()));
    }
    for record in records {
        report(
            record
                .parsed
                .map(|position| GameTree::from_position(position.pos))
                .map_err(|err| format!("line {}: {err}", record.line)),
        )?;
    }
    ControlFlow::Continue(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::ImageFormat;
    use std::io::{Cursor, Write as _};

    /// Only files with a second PGN record go to Database, regardless of compression.
    #[test]
    fn routes_one_game_to_analyse_and_multiple_games_to_database() {
        let dir = tempfile::tempdir().unwrap();
        let single = dir.path().join("one.pgn");
        let multiple = dir.path().join("multiple.PGN");
        let compressed = dir.path().join("multiple.PGN.GZ");
        let first = b"[Event \"One\"]\n\n1. e4 *\n\n";
        let second = b"[Event \"Broken\"]\n\n1. e5 *\n";
        let games = [first.as_slice(), second.as_slice()].concat();
        std::fs::write(&single, first).unwrap();
        std::fs::write(&multiple, &games).unwrap();
        let mut gzip = flate2::write::GzEncoder::new(
            File::create(&compressed).unwrap(),
            flate2::Compression::fast(),
        );
        gzip.write_all(&games).unwrap();
        gzip.finish().unwrap();

        let (collections, notation) =
            split_game_collections(vec![single.clone(), multiple.clone(), compressed.clone()]);
        assert_eq!(collections, [multiple, compressed]);
        assert_eq!(notation, [single]);
    }

    /// Collects streamed results for assertions.
    fn collect_games(paths: &[PathBuf]) -> (Vec<GameTree>, Vec<String>) {
        let mut games = Vec::new();
        let mut errors = Vec::new();
        read_game_files(paths, |result| {
            match result {
                Ok(game) => games.push(game),
                Err(error) => errors.push(error),
            }
            ControlFlow::Continue(())
        });
        (games, errors)
    }

    /// Keeps valid games on either side of a bad game and proceeds to the next file.
    #[test]
    fn keeps_games_in_and_after_a_file_with_an_illegal_move() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("collection.pgn");
        let second = dir.path().join("another.pgn");
        let fen = dir.path().join("position.fen");
        std::fs::write(
            &first,
            "[Event \"One\"]\n\n1. e4 *\n\n[Event \"Broken\"]\n\n1. e5 *\n\n[Event \"Two\"]\n\n1. d4 *",
        )
        .unwrap();
        std::fs::write(&second, "[Event \"Three\"]\n\n1. c4 *").unwrap();
        std::fs::write(&fen, GameTree::new().fen(GameTree::ROOT)).unwrap();
        let (games, errors) = collect_games(&[first, dir.path().join("missing.pgn"), second, fen]);
        assert_eq!(games.len(), 4);
        for (game, (event, san)) in
            games
                .iter()
                .zip([("One", "e4"), ("Two", "d4"), ("Three", "c4")])
        {
            assert_eq!(game.header("Event"), Some(event));
            assert_eq!(game.node(game.mainline()[0]).san, san);
        }
        assert!(games[3].mainline().is_empty());
        assert_eq!(errors.len(), 2);
        assert!(errors[0].contains("collection.pgn") && errors[0].contains("game 2"));
        assert!(errors[1].contains("missing.pgn"));
    }

    /// Preserves the start of a compressed collection if a later gzip member is damaged.
    #[test]
    fn keeps_games_before_a_gzip_read_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("damaged.pgn.gz");
        let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        gzip.write_all(b"[Event \"One\"]\n\n1. e4 *\n\n").unwrap();
        let mut bytes = gzip.finish().unwrap();
        bytes.extend_from_slice(b"invalid gzip member");
        std::fs::write(&path, bytes).unwrap();

        let (games, errors) = collect_games(&[path]);
        assert_eq!(games.len(), 1);
        assert_eq!(games[0].header("Event"), Some("One"));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("damaged.pgn.gz"));
    }

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
        let (games, errors) = collect_games(&paths);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].header("Event"), Some("Imported"));
        assert_eq!(
            games[1].fen(GameTree::ROOT),
            "8/4k1p1/8/pr2Nb2/3R2P1/8/5PP1/2K5 w - - 0 1"
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("positions.epd: line 1"));
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
