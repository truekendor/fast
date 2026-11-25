# Chessboard detection demo

Find the best supported 8×8 grid in an upright 2D screenshot. The OpenCV detector looks for
recurring square regions, checks that the two colors alternate across the board, and fits the
grid to their boundaries. It tolerates pieces and some highlighted squares when enough of the
board colors remain visible.

Run these commands from the repository root with Python 3.10 or later, preferably in a virtual
environment. Use a PNG screenshot for the most reliable color matching.

```sh
python -m pip install -r services/chessboard_detection/requirements.txt
python -B -m services.chessboard_detection docs/screenshots/analyse.png --overlay detected-board.png
python -B -m unittest services.chessboard_detection.test_detector services.chessboard_detection.test_line_detector
```

The CLI writes one JSON object to stdout. `grid.intersections` contains nine rows of nine
`[x, y]` points in source pixels, starting at the top-left corner of the playing area, in
image order. The far edge of a board that fills the image can be at `width` or `height`.
`grid.matched_cells` reports how many cells supported the fit. An image with insufficient
evidence yields `{"grid": null}` and does not write an overlay. Unreadable files produce an
error on stderr and a nonzero exit code.

The Python entry point is `detect_grid(image)`, which accepts an 8-bit BGR OpenCV image. The
detector can also be replaced behind the CLI contract when integrating with Fast. The
point layout accommodates a future perspective-aware detector. This demo is intended for
full, axis-aligned boards with solid cell colors. Textured or heavily compressed boards,
clipped boards and perspective views may fail. Piece recognition and board orientation are
outside this geometry service.

## Line detector

The experimental line detector locates alternating horizontal and vertical edges using
OpenCV at several scales. It fits seven internal lines per axis, tests for alternating
contrast at their crossings, and checks for an extra row or column of crossings at the
boundary. The fit allows 2 pixels of error in line position and checks the measured gaps
against the fitted pitch with the same tolerance. Horizontal and vertical pitches must
agree within 2 pixels, with a tighter limit for small boards. A board must measure at least
80 pixels on its shorter side in the source image.

```sh
python -B -m services.chessboard_detection services/chessboard_detection/fixtures/screenshots/images/chessboard_detect_0100.png --detector lines --overlay lines.png
```

The line detector returns `grids`, a list in image order.
Each element contains `intersections` and `matched_crossings`, the number of supporting
internal intersections out of 49. If there is no match, `grids` is empty. Python callers can
use `detect_line_grids(image)`.

## Labelled screenshots

`fixtures/screenshots` contains 15 screenshots selected from the board detection dataset and
their matching YOLO labels. The samples include YouTube frames, textured boards, and screenshots
with up to twelve boards. `labels` uses the same file stems as `images`. Each label row contains
class `0`, followed by normalized center x, center y, width and height. The fixtures are a fixed
regression sample from the training split.

Label the playing area of every complete 8×8 board at least 80 pixels on its shorter side.
Leave clipped fragments outside the annotation set. Five smaller previews in `0002`, `0057`
and `0082` were removed from the images and labels. In screenshots with readable interface
content, pixels outside the labelled boards are replaced with a neutral background. The
video timestamp over the board in `0001` is masked separately. The edited PNGs retain the
original image dimensions and annotation coordinates. The other screenshots retain their
pixels and have no EXIF, application metadata or comments beyond the standard JPEG header.

```sh
python -B -m services.chessboard_detection.benchmark
```

The benchmark runs both detectors by default. Use `--detector colors` or `--detector lines`
to run one. It compares the returned grid bounds with the YOLO boxes. An intersection over
union (IoU) of at least 0.90 counts as a match and each label can be matched once. It reports
frames with a match, matched boards out of all labels, and how many returned grids match a
label. IoU measures the board boundary. The detector tests also compare grid intersections
directly.

On the masked sample the color detector matches 6 of 41 labels. The line detector matches
40 of 41 labels, including all 33 in multi-board frames. It misses the board in `0033`.
