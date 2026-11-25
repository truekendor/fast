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
python -B -m unittest services.chessboard_detection.test_detector
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
