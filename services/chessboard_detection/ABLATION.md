# Line detector validation experiment

The reference version is `5fc308b` on `chessboard-detection-demo`. The experiment uses the
same 15 anonymized frames and 41 labelled boards, including 33 boards in multi-board frames.
The two board-free controls are `position-patterns.png` and `settings.png` from
`docs/screenshots`. The `tournament.png` screenshot contains visible boards and was excluded
from the board-free controls.

Run from the repository root with the OpenCV dependencies installed:

```sh
python -B -m services.chessboard_detection.ablation --csv ablation.csv
```

Each prediction can match one label. The comparison uses outer board bounds at IoU ≥ 0.90
and repeats the match at 0.95 for a stricter localization check. The label files describe
bounding boxes. They provide no measurements for individual internal grid lines.

The line variants share the same candidate search and the 80-pixel minimum board size.
The shared search uses a square template with a regular pitch. Turning off spacing replaces
the robust 2-pixel consensus with a least-squares fit to all sufficiently strong line peaks.
Turning off the square check lets the fitted horizontal and vertical pitches differ. Turning
off the crossing check drops the internal support threshold while retaining contrast scoring.
Turning off the boundary check removes both the rejection and the score penalty for checker
patterns that continue across the fitted edge.

The template-only variant returns coarse candidates after overlap suppression. The separate
contour baseline uses Canny edges, a 5×5 morphological close and rectangularity of at least
0.75. It searches for a bounding rectangle without using the pitch, aspect ratio or pattern
of a chessboard.

An extra synthetic control varied the width of each cell while keeping the height fixed.
A regular 240×240-pixel checkerboard was accepted with either square-check setting.
A 240×224-pixel checkerboard with uniform cells was rejected by the full detector and
accepted when the square check was disabled. Both controls used the same dark background
and a 105-level difference between alternating cells.

## Results

The multi-board column and unmatched predictions use IoU ≥ 0.90. Unmatched predictions
include extra, shifted or duplicate grids. The last column counts all returned grids on the
two board-free screenshots.

| Variant | IoU ≥ 0.90 | IoU ≥ 0.95 | Multi-board matches | Unmatched on positives | Detections on negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full detector | 40/41 | 39/41 | 33/33 | 0 | 0 |
| Without spacing check | 40/41 | 39/41 | 33/33 | 1 | 0 |
| Without square check | 40/41 | 39/41 | 33/33 | 0 | 0 |
| Without crossing check | 40/41 | 39/41 | 33/33 | 83 | 7 |
| Without boundary check | 39/41 | 38/41 | 32/33 | 6 | 0 |
| Without spacing and square checks | 39/41 | 37/41 | 32/33 | 7 | 0 |
| Crossing template only | 22/41 | 18/41 | 17/33 | 331 | 85 |
| Rectangular contours | 31/41 | 31/41 | 26/33 | 7 | 0 |

Removing the spacing check alone preserved all matches and added one grid in `0094` (12
labelled boards). The square check had no effect on these real frames and filtered the
rectangular synthetic control. Removing both checks lost a board in `0050` and produced
seven unmatched grids. Removing the boundary check lost a match in `0012` and returned
four extra grids in `0108`.

The internal crossing check provided the largest reduction in unmatched predictions. The
contour baseline matched nine fewer boards overall and seven fewer in multi-board frames.
These results support keeping the full detector as the default. The square check is the
first candidate for simplification if stretched boards should count as valid inputs.
