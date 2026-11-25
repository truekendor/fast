//! The image import dialog, its worker tasks, and the recognized position preview.

use std::sync::Arc;

use chess_core::{Chess, Color, shakmaty::Board};
use gpui::{App, RenderImage, Size};
use ui::{i18n::t, widgets::dialog};

use super::*;
use crate::{analysis_import::ImageInput, board_image};

mod selection;

use selection::{ImageRect, SquareSelection};

/// The progress or result of importing screenshots into this Analyse tab.
pub(super) enum ImageImport {
    Loading { _task: Task<()> },
    Failed,
    Ready(Box<BoardChoices>),
}

/// The current source image, board selection and position settings.
pub(super) struct BoardChoices {
    sources: Vec<Screenshot>,
    source: usize,
    selected: usize,
    manual: Option<ManualSelection>,
    rotated: bool,
    turn: Color,
    preview: Entity<BoardView>,
}

/// The original RGB pixels, their scaled display copy and all detected boards.
struct Screenshot {
    pixels: Arc<image::RgbImage>,
    display: Arc<RenderImage>,
    candidates: Vec<Candidate>,
}

/// The outline and recognized pieces of an automatically detected board.
struct Candidate {
    outline: ImageRect,
    board: Board,
}

/// A user-selected square and the result for its current bounds.
struct ManualSelection {
    square: SquareSelection,
    result: Option<Board>,
    task: Option<Task<()>>,
    failed: bool,
    focus: FocusHandle,
}

/// Reads a screenshot and preserves it even when automatic detection finds no board.
fn read_screenshot(input: &ImageInput) -> anyhow::Result<Screenshot> {
    let pixels = input.read()?;
    let candidates = match board_image::recognize(&pixels, false) {
        Ok(boards) => boards
            .into_iter()
            .map(|board_image::RecognizedBoard { grid, board }| Candidate {
                outline: ImageRect {
                    start: grid.intersections[0][0],
                    end: grid.intersections[8][8],
                },
                board,
            })
            .collect(),
        Err(err) => {
            log::warn!("Automatic board recognition failed: {err:#}");
            Vec::new()
        }
    };

    // Keep the original pixels for inference and cap the size of the GPU upload.
    let (width, height) = pixels.dimensions();
    let ratio = (1440. / f64::from(width.max(height))).min(1.);
    let scaled = image::imageops::resize(
        &pixels,
        (f64::from(width) * ratio).round().max(1.) as u32,
        (f64::from(height) * ratio).round().max(1.) as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut display = image::DynamicImage::ImageRgb8(scaled).into_rgba8();
    // GPUI consumes BGRA frames.
    for pixel in display.pixels_mut() {
        pixel.0.swap(0, 2);
    }
    Ok(Screenshot {
        pixels: Arc::new(pixels),
        display: Arc::new(RenderImage::new([image::Frame::new(display)])),
        candidates,
    })
}

impl BoardChoices {
    /// Returns the original screenshot currently shown in the dialog.
    fn screenshot(&self) -> &Screenshot {
        &self.sources[self.source]
    }

    /// Returns the current guess with the requested orientation applied.
    fn oriented_board(&self) -> Option<Board> {
        let mut board = match &self.manual {
            Some(manual) => manual.result.clone()?,
            None => self
                .screenshot()
                .candidates
                .get(self.selected)?
                .board
                .clone(),
        };
        if self.rotated {
            board.rotate_180();
        }
        Some(board)
    }

    /// Refreshes the board preview after a selection or orientation change.
    fn update_preview(&self, cx: &mut App) {
        let board = self.oriented_board();
        self.preview
            .update(cx, |view, cx| view.set_editing(board, cx));
    }
}

impl AnalyseScreen {
    /// Opens the file picker for a screenshot, leaving the current game in place.
    pub(crate) fn import_image(&mut self, cx: &mut Context<Self>) {
        if self.image_import.is_some() {
            return;
        }
        let this = cx.entity().downgrade();
        crate::model::pick_path(
            self.model.clone(),
            t("Choose a board image"),
            None,
            cx,
            move |path, cx| {
                let _ = this.update(cx, |screen, cx| {
                    screen.start_image_import(vec![ImageInput::File(path)], cx);
                });
            },
        );
    }

    /// Uses the same preview for images from the picker, clipboard and file drops.
    pub(crate) fn start_image_import(&mut self, inputs: Vec<ImageInput>, cx: &mut Context<Self>) {
        if inputs.is_empty() || self.image_import.is_some() {
            return;
        }

        let task = cx.spawn(async move |this, cx| {
            let (sources, failed) = cx
                .background_spawn(async move {
                    let mut sources = Vec::new();
                    let mut failed = false;
                    for input in inputs {
                        match read_screenshot(&input) {
                            Ok(screenshot) => sources.push(screenshot),
                            Err(err) => {
                                failed = true;
                                log::warn!("Image import failed: {err:#}");
                            }
                        }
                    }
                    (sources, failed)
                })
                .await;
            let _ = this.update(cx, |screen, cx| {
                screen.image_import = Some(if sources.is_empty() {
                    ImageImport::Failed
                } else {
                    if failed {
                        screen.model.toast("Some images could not be read.", cx);
                    }
                    let prefs = screen.board.read(cx).prefs();
                    let choices = BoardChoices {
                        sources,
                        source: 0,
                        selected: 0,
                        manual: None,
                        rotated: false,
                        turn: Color::White,
                        preview: cx.new(|cx| {
                            let mut board = BoardView::new("image-position-preview").read_only();
                            board.set_prefs(prefs, cx);
                            board
                        }),
                    };
                    choices.update_preview(cx);
                    ImageImport::Ready(Box::new(choices))
                });
                cx.notify();
            });
        });
        self.image_import = Some(ImageImport::Loading { _task: task });
        cx.notify();
    }

    /// Applies a change to whichever board is being previewed.
    fn update_image_choice(
        &mut self,
        change: impl FnOnce(&mut BoardChoices),
        cx: &mut Context<Self>,
    ) {
        if let Some(ImageImport::Ready(choices)) = &mut self.image_import {
            change(choices);
            choices.update_preview(cx);
            cx.notify();
        }
    }

    /// Classifies the current square on a worker and discards the result if the square changes.
    fn recognize_image_selection(&mut self, cx: &mut Context<Self>) {
        let Some(ImageImport::Ready(choices)) = &mut self.image_import else {
            return;
        };
        let pixels = choices.screenshot().pixels.clone();
        let Some(manual) = &mut choices.manual else {
            return;
        };
        if manual.task.is_some() {
            return;
        }
        let square = manual.square;
        manual.failed = false;
        manual.task = Some(cx.spawn(async move |this, cx| {
            let [left, top, side] = square.pixel_bounds(pixels.dimensions());
            let result = cx
                .background_spawn(async move {
                    board_image::recognize_selection(&pixels, left, top, side)
                })
                .await;
            let _ = this.update(cx, |screen, cx| {
                screen.update_image_choice(
                    |choices| {
                        if let Some(manual) = &mut choices.manual
                            && manual.square == square
                        {
                            manual.task = None;
                            match result {
                                Ok(board) => manual.result = Some(board),
                                Err(err) => {
                                    log::warn!("Manual board recognition failed: {err:#}");
                                    manual.failed = true;
                                }
                            }
                        }
                    },
                    cx,
                );
            });
        }));
        cx.notify();
    }

    /// Opens a valid position in a new tab or sends an invalid guess to the editor.
    fn open_image_position(
        &mut self,
        position: Result<Chess, PositionEditor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.image_import = None;
        match position {
            Ok(position) => {
                let workspace = self.workspace.clone();
                window.defer(cx, move |window, cx| {
                    let _ = workspace.update(cx, |workspace, cx| {
                        workspace.open_analysis(GameTree::from_position(position), window, cx);
                    });
                });
            }
            Err(editor) => self.show_editor(editor, cx),
        }
        cx.notify();
    }

    /// Draws the screenshot on the left and the inferred position on the right.
    pub(super) fn render_image_import(
        &self,
        viewport: Size<Pixels>,
        cx: &mut Context<Self>,
    ) -> Option<ui::widgets::Modal> {
        let width = (f32::from(viewport.width) - 48.).min(1000.);
        let body = match self.image_import.as_ref()? {
            ImageImport::Loading { .. } => {
                import_message("Looking for chessboards…").into_any_element()
            }
            ImageImport::Failed => {
                import_message("Could not read the image. Use PNG, JPEG, BMP or WebP.")
                    .into_any_element()
            }
            ImageImport::Ready(choices) => h_flex()
                .items_start()
                .gap(px(16.))
                .p(px(16.))
                .child(self.render_screenshot(choices, width - 310., viewport.height, cx))
                .child(self.render_position_preview(choices, cx))
                .into_any_element(),
        };
        let this = cx.entity().downgrade();
        Some(dialog(
            "image-import-dialog",
            "Import position from image",
            width,
            body,
            move |cx| {
                let _ = this.update(cx, |screen, cx| {
                    screen.image_import = None;
                    cx.notify();
                });
            },
        ))
    }

    /// Shows the selected position and the controls for opening it.
    fn render_position_preview(&self, choices: &BoardChoices, cx: &mut Context<Self>) -> gpui::Div {
        let this = cx.entity();
        let mut column = v_flex()
            .w(px(260.))
            .flex_none()
            .gap(px(8.))
            .child(section_label("Recognized position"));
        let Some(board) = choices.oriented_board() else {
            return column.child(import_message("The selected board will appear here."));
        };
        let editor = PositionEditor::from_board(board, choices.turn);
        let position = editor.position().map_err(|_| editor);
        let can_open = position.is_ok();
        let action = if can_open {
            "Open position"
        } else {
            "Edit position"
        };
        column = column
            .child(
                div()
                    .size(px(248.))
                    .flex_none()
                    .child(choices.preview.clone()),
            )
            .child(
                btn("rotate-image-position", "Rotate position")
                    .compact()
                    .on_click(this.listen(|screen, _, cx| {
                        screen.update_image_choice(|choices| choices.rotated = !choices.rotated, cx)
                    })),
            )
            .child(
                h_flex().gap(px(6.)).flex_wrap().children(
                    [
                        (Color::White, "White to move"),
                        (Color::Black, "Black to move"),
                    ]
                    .map(|(turn, label)| {
                        btn(("image-position-turn", usize::from(turn.is_black())), label)
                            .compact()
                            .selected(choices.turn == turn)
                            .on_click(this.listen(move |screen, _, cx| {
                                screen.update_image_choice(|choices| choices.turn = turn, cx)
                            }))
                    }),
                ),
            );
        column
            .when(!can_open, |column| {
                column.child(div().text_as(13., T.fg_2).child(t(
                    "This position needs corrections before it can be opened.",
                )))
            })
            .child(
                btn("confirm-image-position", action)
                    .primary()
                    .on_click(this.listen(move |screen, window, cx| {
                        screen.open_image_position(position.clone(), window, cx)
                    })),
            )
    }
}

/// Displays feedback while loading or when no candidate can be shown.
fn import_message(message: &str) -> gpui::Div {
    div()
        .p(px(16.))
        .text_as(14., T.fg_2)
        .child(t(message).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allows manual selection even in a screenshot with no automatic detection.
    #[test]
    fn keeps_the_original_when_no_board_is_found() {
        let image = image::RgbImage::from_pixel(160, 120, image::Rgb([120; 3]));
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        let screenshot = read_screenshot(&ImageInput::Clipboard(gpui::Image::from_bytes(
            gpui::ImageFormat::Png,
            bytes.into_inner(),
        )))
        .unwrap();
        assert_eq!(screenshot.pixels.as_ref(), &image);
        assert!(screenshot.candidates.is_empty());
    }

    /// Keeps the original image and reports outlines for all of its boards.
    #[test]
    fn imports_original_screenshot_and_all_detected_positions() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../services/chessboard_detection/fixtures/screenshots/images/chessboard_detect_0004.png");
        let screenshot = read_screenshot(&ImageInput::File(path)).unwrap();
        assert_eq!(screenshot.candidates.len(), 3);
        let first = &screenshot.candidates[0];
        assert!(first.outline.start[0] < first.outline.end[0]);
        assert!(first.outline.start[1] < first.outline.end[1]);
        let expected = "8/4k1p1/8/pr2Nb2/3R2P1/8/5PP1/2K5";
        assert_eq!(first.board.board_fen().to_string(), expected);
        let [left, top, side] =
            SquareSelection::centered(screenshot.pixels.dimensions(), Some(first.outline))
                .pixel_bounds(screenshot.pixels.dimensions());
        let manual = board_image::recognize_selection(&screenshot.pixels, left, top, side).unwrap();
        assert_eq!(manual.board_fen().to_string(), expected);
    }
}
