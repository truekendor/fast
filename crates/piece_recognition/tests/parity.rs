//! Confirms that the embedded weights reproduce the TensorFlow.js model on RGB cells.

use piece_recognition::Classifier;

#[test]
/// Matches both stages of the original model on reference cells from the screenshot set.
fn embedded_model_matches_web_outputs() {
    let classifier = Classifier::embedded().unwrap();
    let mut fixture = include_bytes!("fixtures/web-output.bin").as_slice();
    for expected_piece in [
        None,
        Some('k'),
        Some('r'),
        Some('N'),
        Some('K'),
        Some('p'),
        Some('K'),
        Some('k'),
    ] {
        let width = u16::from_le_bytes(fixture[..2].try_into().unwrap()) as usize;
        let height = u16::from_le_bytes(fixture[2..4].try_into().unwrap()) as usize;
        let (rgb, tail) = fixture[4..].split_at(width * height * 3);
        let (reference, tail) = tail.split_at((1024 + 13) * 4);
        let prediction = classifier.predict(rgb, width, height).unwrap();
        let expected = reference
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()));
        for (index, (actual, expected)) in prediction
            .features
            .into_iter()
            .chain(prediction.probabilities)
            .zip(expected)
            .enumerate()
        {
            let tolerance = if index < 1024 { 0.0001 } else { 0.00002 };
            assert!(
                (actual - expected).abs() <= tolerance,
                "class {expected_piece:?}, output {index}: {actual} vs {expected}"
            );
        }
        assert_eq!(prediction.piece(), expected_piece);
        fixture = tail;
    }
    assert!(fixture.is_empty());
}
