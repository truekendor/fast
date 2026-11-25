//! Exercises the embedded piece classifier through the standalone Fast command.

use std::{path::Path, process::Command};

#[test]
/// Recognizes the same board placement as the web model from a clean working directory.
fn recognizes_screenshot_with_embedded_weights() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../services/chessboard_detection/fixtures/screenshots/images/chessboard_detect_0002.png");
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_fast"))
        .arg("--recognize-board")
        .arg(fixture)
        .arg("position.json")
        .current_dir(temp.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = std::fs::read(temp.path().join("position.json")).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&json).unwrap();
    let boards = report["boards"].as_array().unwrap();
    assert_eq!(boards.len(), 1);
    assert_eq!(boards[0]["placement"], "8/4k1p1/8/pr2Nb2/3R4/6P1/5PP1/2K5");
    assert_eq!(
        boards[0]["flipped_placement"],
        "5K2/1PP5/1P6/4R3/2bN2rp/8/1p1k4/8"
    );
}
