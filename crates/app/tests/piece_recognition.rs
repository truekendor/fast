//! Exercises the embedded piece classifier through the standalone Fast command.

use std::{path::Path, process::Command};

#[test]
/// Recognizes the same board placement as the web model from a clean working directory.
fn recognizes_screenshot_with_embedded_weights() {
    let report = image_command("--recognize-board", "images/chessboard_detect_0002.png");
    let boards = report["boards"].as_array().unwrap();
    assert_eq!(boards.len(), 1);
    assert_eq!(boards[0]["placement"], "8/4k1p1/8/pr2Nb2/3R4/6P1/5PP1/2K5");
    assert_eq!(
        boards[0]["flipped_placement"],
        "5K2/1PP5/1P6/4R3/2bN2rp/8/1p1k4/8"
    );
}

/// Finds the same tightly cropped board at three screenshot resolutions.
#[test]
fn detects_a_board_cropped_at_the_image_edge() {
    for (fixture, scale) in [
        ("cropped/regression-rect-crop-x1.webp", 1.0),
        ("cropped/regression-rect-crop-x2.webp", 2.0),
        ("cropped/regression-rect-crop-x3.webp", 3.0),
    ] {
        let report = image_command("--detect-grid", fixture);
        let grids = report["grids"].as_array().unwrap();
        assert_eq!(grids.len(), 1, "{fixture}");
        let intersections = &grids[0]["intersections"];
        let top_left: [f64; 2] = serde_json::from_value(intersections[0][0].clone()).unwrap();
        let bottom_right: [f64; 2] = serde_json::from_value(intersections[8][8].clone()).unwrap();
        assert!((0.0..=1.0).contains(&(top_left[0] / scale)), "{fixture}");
        assert_eq!(top_left[1], 0.0, "{fixture}");
        assert!(
            (292.0..=296.0).contains(&(bottom_right[0] / scale)),
            "{fixture}"
        );
        assert!(
            (292.0..=294.0).contains(&(bottom_right[1] / scale)),
            "{fixture}"
        );
    }
}

/// Runs Fast on a repository screenshot and loads its JSON output.
fn image_command(command: &str, fixture: &str) -> serde_json::Value {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../services/chessboard_detection/fixtures/screenshots")
        .join(fixture);
    let temp = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_fast"))
        .arg(command)
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
    serde_json::from_slice(&json).unwrap()
}
