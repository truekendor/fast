//! The image import dialog, its worker tasks, and the recognized position preview.

use std::sync::Arc;

use chess_core::{Chess, Color, shakmaty::Board};
use gpui::{App, RenderImage};
use ui::{i18n::t, widgets::dialog};

use super::*;
use crate::{analysis_import::ImageInput, board_image};

mod selection;

use selection::{ImageRect, SquareSelection};

/// A workspace dialog that previews images before opening an Analyse position.
pub(crate) struct ImageImportDialog {
    model: Entity<AppModel>,
    workspace: WeakEntity<Workspace>,
    recognizer: Arc<board_image::BoardRecognizer>,
    batch_task: Option<Task<()>>,
    image_import: Option<ImageImport>,
    edit_tab: Option<u64>,
}

/// The progress or result of recognizing the screenshots.
enum ImageImport {
    Loading,
    Failed,
    Ready(Box<BoardChoices>),
}

/// The current source image, board selection and position settings.
pub(super) struct BoardChoices {
    sources: Vec<Screenshot>,
    source: usize,
    selected: usize,
    manual: Option<ManualSelection>,
    preview: Entity<BoardView>,
    thumbnails: Vec<Entity<BoardView>>,
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
    options: PositionOptions,
}

/// The orientation and side to move chosen for one recognized position.
#[derive(Clone, Copy)]
struct PositionOptions {
    rotated: bool,
    turn: Color,
}

impl Default for PositionOptions {
    /// Starts a new position in the screenshot's orientation with White to move.
    fn default() -> Self {
        Self {
            rotated: false,
            turn: Color::White,
        }
    }
}

impl PositionOptions {
    /// Returns the board with this position's chosen orientation.
    fn orient(self, board: &Board) -> Board {
        let mut board = board.clone();
        if self.rotated {
            board.rotate_180();
        }
        board
    }
}

/// A user-selected square and the result for its current bounds.
struct ManualSelection {
    square: SquareSelection,
    result: Option<Board>,
    options: PositionOptions,
    task: Option<Task<()>>,
    failed: bool,
    focus: FocusHandle,
}

/// Reads a screenshot and preserves it even when automatic detection finds no board.
fn read_screenshot(
    input: &ImageInput,
    recognizer: &board_image::BoardRecognizer,
) -> anyhow::Result<Screenshot> {
    let pixels = input.read()?;
    let candidates = match recognizer.recognize(&pixels, false) {
        Ok(boards) => boards
            .into_iter()
            .map(|board_image::RecognizedBoard { grid, board }| Candidate {
                outline: ImageRect {
                    start: grid.intersections[0][0],
                    end: grid.intersections[8][8],
                },
                board,
                options: PositionOptions::default(),
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

    /// Selects a screenshot and prepares cards for the positions it contains.
    fn select_screenshot(&mut self, source: usize, cx: &mut App) {
        self.source = source;
        self.selected = 0;
        self.manual = None;
        let mut prefs = self.preview.read(cx).prefs();
        prefs.coordinates = ui::CoordinateStyle::Off;
        self.thumbnails = self.sources[source]
            .candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                cx.new(|cx| {
                    let mut view = BoardView::new(("detected-board-thumbnail", index)).read_only();
                    view.set_prefs(prefs, cx);
                    view.set_editing(Some(candidate.options.orient(&candidate.board)), cx);
                    view
                })
            })
            .collect();
    }

    /// Returns the current guess in its own orientation and with its side to move.
    fn oriented_position(&self) -> Option<(Board, Color)> {
        let (board, options) = match &self.manual {
            Some(manual) => (manual.result.as_ref()?, manual.options),
            None => {
                let candidate = self.screenshot().candidates.get(self.selected)?;
                (&candidate.board, candidate.options)
            }
        };
        Some((options.orient(board), options.turn))
    }

    /// Changes only the options of the detected or manual position being previewed.
    fn edit_position_options(&mut self, change: impl FnOnce(&mut PositionOptions)) {
        if let Some(manual) = &mut self.manual {
            change(&mut manual.options);
        } else if let Some(candidate) = self.sources[self.source].candidates.get_mut(self.selected)
        {
            change(&mut candidate.options);
        }
    }

    /// Refreshes the board preview after a selection or orientation change.
    fn update_preview(&self, cx: &mut App) {
        let board = self.oriented_position().map(|(board, _)| board);
        if self.manual.is_none()
            && let Some(thumbnail) = self.thumbnails.get(self.selected)
        {
            thumbnail.update(cx, |view, cx| view.set_editing(board.clone(), cx));
        }
        self.preview
            .update(cx, |view, cx| view.set_editing(board, cx));
    }
}

impl ImageImportDialog {
    /// Creates a closed import dialog for the workspace.
    pub(crate) fn new(model: Entity<AppModel>, workspace: WeakEntity<Workspace>) -> Self {
        Self {
            model,
            workspace,
            recognizer: Arc::default(),
            batch_task: None,
            image_import: None,
            edit_tab: None,
        }
    }

    /// Opens the file picker for a screenshot, leaving the current game in place.
    pub(crate) fn import_image(&mut self, edit_tab: Option<u64>, cx: &mut Context<Self>) {
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
                    screen.start_image_import(vec![ImageInput::File(path)], edit_tab, cx);
                });
            },
        );
    }

    /// Uses the same preview for images from the picker, clipboard and file drops.
    pub(crate) fn start_image_import(
        &mut self,
        inputs: Vec<ImageInput>,
        edit_tab: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        if inputs.is_empty() || self.image_import.is_some() {
            return;
        }
        self.edit_tab = edit_tab;

        let recognizer = self.recognizer.clone();
        let (send, receive) = smol::channel::bounded(1);
        let worker = cx.background_spawn(async move {
            for input in inputs {
                if send.is_closed() {
                    break;
                }
                let screenshot = read_screenshot(&input, &recognizer);
                if send.send(screenshot).await.is_err() {
                    break;
                }
            }
        });
        self.image_import = Some(ImageImport::Loading);
        self.batch_task = Some(cx.spawn(async move |this, cx| {
            let mut failed = false;
            while let Ok(result) = receive.recv().await {
                match result {
                    Ok(screenshot) => {
                        if this
                            .update(cx, |screen, cx| screen.add_screenshot(screenshot, cx))
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(err) => {
                        failed = true;
                        log::warn!("Image import failed: {err:#}");
                    }
                }
            }
            worker.await;
            let _ = this.update(cx, |screen, cx| {
                screen.batch_task = None;
                if matches!(screen.image_import, Some(ImageImport::Loading)) {
                    screen.image_import = Some(ImageImport::Failed);
                } else if failed {
                    screen.model.toast("Some images could not be read.", cx);
                }
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Shows the first readable screenshot immediately and appends later ones without changing the selection.
    fn add_screenshot(&mut self, screenshot: Screenshot, cx: &mut Context<Self>) {
        match &mut self.image_import {
            Some(ImageImport::Loading) => {
                let prefs = self.model.read(cx).config.board.prefs();
                let mut choices = BoardChoices {
                    sources: vec![screenshot],
                    source: 0,
                    selected: 0,
                    manual: None,
                    preview: cx.new(|cx| {
                        let mut board = BoardView::new("image-position-preview").read_only();
                        board.set_prefs(prefs, cx);
                        board
                    }),
                    thumbnails: Vec::new(),
                };
                choices.select_screenshot(0, cx);
                choices.update_preview(cx);
                self.image_import = Some(ImageImport::Ready(Box::new(choices)));
            }
            Some(ImageImport::Ready(choices)) => choices.sources.push(screenshot),
            _ => return,
        }
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
        let recognizer = self.recognizer.clone();
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
                    recognizer.recognize_selection(&pixels, left, top, side)
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

    /// Opens a valid position or sends an invalid guess to the editor.
    fn open_image_position(
        &mut self,
        position: Result<Chess, PositionEditor>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.batch_task = None;
        self.image_import = None;
        let edit_tab = self.edit_tab.take();
        let workspace = self.workspace.clone();
        window.defer(cx, move |window, cx| {
            let _ = workspace.update(cx, |workspace, cx| match position {
                Ok(position) => {
                    workspace.open_analysis(GameTree::from_position(position), window, cx);
                }
                Err(editor) => {
                    workspace
                        .image_editor_tab(edit_tab, window, cx)
                        .update(cx, |screen, cx| screen.show_editor(editor, cx));
                }
            });
        });
        cx.notify();
    }

    /// Draws the screenshot on the left and the inferred position on the right.
    fn render_image_import(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<ui::widgets::Modal> {
        let viewport = window.viewport_size();
        let width = (f32::from(viewport.width) - 48.).min(1000.);
        let body = match self.image_import.as_ref()? {
            ImageImport::Loading => import_message("Looking for chessboards…").into_any_element(),
            ImageImport::Failed => {
                import_message("Could not read the image. Use PNG, JPEG, BMP or WebP.")
                    .into_any_element()
            }
            ImageImport::Ready(choices) => h_flex()
                .items_start()
                .gap(px(16.))
                .p(px(16.))
                .child(self.render_screenshot(choices, width - 310., window, cx))
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
                    screen.batch_task = None;
                    screen.image_import = None;
                    screen.edit_tab = None;
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
        let Some((board, turn)) = choices.oriented_position() else {
            return column.child(import_message("The selected board will appear here."));
        };
        let editor = PositionEditor::from_board(board, turn);
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
                        screen.update_image_choice(
                            |choices| {
                                choices.edit_position_options(|options| {
                                    options.rotated = !options.rotated
                                })
                            },
                            cx,
                        )
                    })),
            )
            .child(
                h_flex().gap(px(6.)).flex_wrap().children(
                    [
                        (Color::White, "White to move"),
                        (Color::Black, "Black to move"),
                    ]
                    .map(|(side, label)| {
                        btn(("image-position-turn", usize::from(side.is_black())), label)
                            .compact()
                            .selected(turn == side)
                            .on_click(this.listen(move |screen, _, cx| {
                                screen.update_image_choice(
                                    |choices| {
                                        choices.edit_position_options(|options| options.turn = side)
                                    },
                                    cx,
                                )
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

impl Render for ImageImportDialog {
    /// Displays the import dialog above whichever tab the user is viewing.
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .absolute()
            .inset_0()
            .children(self.render_image_import(window, cx))
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
        let screenshot = read_screenshot(
            &ImageInput::Clipboard(gpui::Image::from_bytes(
                gpui::ImageFormat::Png,
                bytes.into_inner(),
            )),
            &board_image::BoardRecognizer::default(),
        )
        .unwrap();
        assert_eq!(screenshot.pixels.as_ref(), &image);
        assert!(screenshot.candidates.is_empty());
    }
}
