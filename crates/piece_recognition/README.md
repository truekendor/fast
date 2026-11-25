# Embedded piece recognition

`assets/pieces.onnx` combines the MobileNet feature extractor and the trained Dense head
from [chessboard-image-to-fen at `1955c3ed`](https://github.com/truekendor/chessboard-image-to-fen/tree/1955c3ed1c6db063e9ce819e8b2af81e0334c93c/public/nn).
The exporter translates the graph and tensor layouts and preserves the existing weights.
MobileNet V3 Small was published by Google through [TensorFlow Hub](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/imagenet/mobilenet_v3_small_100_224/feature_vector/5.md).

The ONNX file is 7,220,497 bytes. Its SHA-256 is
`2a95fb0d79c40a7e56060a79e81e4676692bfb92246d1fe7eb3c17fd29290f64`.
Fast embeds it with `include_bytes!` and uses the `tract` CPU backend. Python and Node run
only when converting the model or regenerating the TensorFlow.js reference data.

## Reproduce the export and reference

Download `mobilenet` and `classification_model` from the pinned source and keep their
`model.json` and weight shards under a common directory. With Python `onnx` and `numpy`
installed, run the exporter from the root of the project:

```sh
python -B services/chessboard_detection/export_piece_model.py MODELS crates/piece_recognition/assets/pieces.onnx
```

The separate `piece_model_reference.cjs` script uses `@tensorflow/tfjs` 4.8.0 on CPU.
It reads `cells.json` from a sample directory. Each entry names the cell, its width and
height, and a file containing packed RGB bytes. The script outputs the web model's 1024
features and 13 probabilities as little-endian `f32` values for each sample:

```sh
node services/chessboard_detection/piece_model_reference.cjs MODELS SAMPLES
cargo test --locked -p piece_recognition
```

The Rust regression test includes seven cells from screenshots `0002`, `0094` and `0100`,
and one sample made by blending two of the cells from `0002`.
The fixture stores each cell as little-endian `u16` width and height, packed RGB, then the
1037 little-endian floats from TensorFlow.js. The test compares both outputs and the winning
class at fixed tolerances. A separate local run compared 88 source cells. All classes
agreed, with maximum absolute differences of 0.00006521 for the features and below 0.00000001
for the probabilities. Another 34 blended cells exercised less confident predictions. Their
classes also agreed, with a maximum probability difference of 0.00000919.
