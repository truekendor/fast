# Embedded piece recognition

`assets/pieces.onnx` combines the MobileNet feature extractor and the trained Dense head
from [chessboard-image-to-fen at `1955c3ed`](https://github.com/truekendor/chessboard-image-to-fen/tree/1955c3ed1c6db063e9ce819e8b2af81e0334c93c/public/nn).
The conversion preserved the original trained weights.
MobileNet V3 Small was published by Google through [TensorFlow Hub](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/imagenet/mobilenet_v3_small_100_224/feature_vector/5.md).

The ONNX file is 7,220,497 bytes. Its SHA-256 is
`2a95fb0d79c40a7e56060a79e81e4676692bfb92246d1fe7eb3c17fd29290f64`.
Fast embeds it with `include_bytes!` and runs inference with the `tract` CPU backend.
The [Fast integration test](../app/tests/piece_recognition.rs) checks the FEN placement
produced from a screenshot by the embedded model.
