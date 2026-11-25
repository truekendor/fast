//! Imports a screenshot through a board selection and confirmation dialog.

use std::sync::Arc;

use chess_core::{Chess, Color, shakmaty::Board};
use gpui::{App, RenderImage, img};
use ui::{i18n::t, widgets::dialog};

use super::*;
use crate::{analysis_import::ImageInput, board_image};

/// The progress or result of importing a screenshot into this Analyse tab.
pub(super) enum ImageImport {
    Loading { _task: Task<()> },
    Failed,
    NoBoards,
    Ready(BoardChoices),
}

/// The boards found in one image and the settings for the selected position.
pub(super) struct BoardChoices {
    candidates: Vec<Candidate>,
    selected: usize,
    rotated: bool,
    turn: Color,
    preview: Entity<BoardView>,
}

/// A board's source crop and recognized pieces.
struct Candidate {
    thumbnail: Arc<RenderImage>,
    board: Board,
}

/// Recognizes boards and makes small previews of the original image on the worker thread.
fn read_candidates(input: &ImageInput) -> anyhow::Result<Vec<Candidate>> {
    let image = input.read()?;
    Ok(board_image::recognize(&image, false)?
        .into_iter()
        .map(|board_image::RecognizedBoard { grid, board }| {
            let crop =
                board_image::crop(&image, grid.intersections[0][0], grid.intersections[8][8]);
            let mut thumbnail = image::DynamicImage::ImageRgb8(crop)
                .thumbnail(224, 224)
                .into_rgba8();
            // GPUI consumes BGRA frames.
            for pixel in thumbnail.pixels_mut() {
                pixel.0.swap(0, 2);
            }
            Candidate {
                thumbnail: Arc::new(RenderImage::new([image::Frame::new(thumbnail)])),
                board,
            }
        })
        .collect())
}

impl Candidate {
    /// Puts the recognized pieces in the orientation chosen by the user.
    fn oriented_board(&self, rotated: bool) -> Board {
        let mut board = self.board.clone();
        if rotated {
            board.rotate_180();
        }
        board
    }
}

impl BoardChoices {
    /// Returns the pieces for the currently selected board and orientation.
    fn oriented_board(&self) -> Board {
        self.candidates[self.selected].oriented_board(self.rotated)
    }

    /// Updates the preview to reflect the selected crop and its orientation.
    fn update_preview(&self, cx: &mut App) {
        let board = self.oriented_board();
        self.preview
            .update(cx, |view, cx| view.set_editing(Some(board), cx));
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

    /// Reuses the board-selection dialog for image files and clipboard screenshots.
    pub(crate) fn start_image_import(&mut self, inputs: Vec<ImageInput>, cx: &mut Context<Self>) {
        if inputs.is_empty() || self.image_import.is_some() {
            return;
        }

        let task = cx.spawn(async move |this, cx| {
            let (candidates, decoded, failed) = cx
                .background_spawn(async move {
                    let mut candidates = Vec::new();
                    let mut decoded = false;
                    let mut failed = false;
                    for input in inputs {
                        match read_candidates(&input) {
                            Ok(found) => {
                                decoded = true;
                                candidates.extend(found);
                            }
                            Err(err) => {
                                failed = true;
                                log::warn!("Image import failed: {err:#}");
                            }
                        }
                    }
                    (candidates, decoded, failed)
                })
                .await;
            let _ = this.update(cx, |screen, cx| {
                screen.image_import = Some(if !decoded {
                    ImageImport::Failed
                } else if candidates.is_empty() {
                    ImageImport::NoBoards
                } else {
                    let prefs = screen.board.read(cx).prefs();
                    let choices = BoardChoices {
                        candidates,
                        selected: 0,
                        rotated: false,
                        turn: Color::White,
                        preview: cx.new(|cx| {
                            let mut board = BoardView::new("image-position-preview").read_only();
                            board.set_prefs(prefs, cx);
                            board
                        }),
                    };
                    choices.update_preview(cx);
                    ImageImport::Ready(choices)
                });
                if decoded && failed {
                    screen.model.toast("Some images could not be read.", cx);
                }
                cx.notify();
            });
        });
        self.image_import = Some(ImageImport::Loading { _task: task });
        cx.notify();
    }

    /// Applies a change to the selection or the settings for the prospective position.
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

    /// Draws the import dialog for the current stage of recognition.
    pub(super) fn render_image_import(&self, cx: &mut Context<Self>) -> Option<ui::widgets::Modal> {
        let body = match self.image_import.as_ref()? {
            ImageImport::Loading { .. } => {
                import_message("Looking for chessboards…").into_any_element()
            }
            ImageImport::Failed => {
                import_message("Could not read the image. Use PNG, JPEG, BMP or WebP.")
                    .into_any_element()
            }
            ImageImport::NoBoards => {
                import_message("No chessboards found in this image.").into_any_element()
            }
            ImageImport::Ready(choices) => {
                self.render_image_choices(choices, cx).into_any_element()
            }
        };
        let this = cx.entity().downgrade();
        Some(dialog(
            "image-import-dialog",
            "Import position from image",
            680.,
            body,
            move |cx| {
                let _ = this.update(cx, |screen, cx| {
                    screen.image_import = None;
                    cx.notify();
                });
            },
        ))
    }

    /// Shows the reconstructed board, controls and source crops.
    fn render_image_choices(&self, choices: &BoardChoices, cx: &mut Context<Self>) -> gpui::Div {
        let this = cx.entity();
        let editor = PositionEditor::from_board(choices.oriented_board(), choices.turn);
        let position = editor.position().map_err(|_| editor);
        let can_open = position.is_ok();
        let action = if can_open {
            "Open position"
        } else {
            "Edit position"
        };
        v_flex()
            .p(px(16.))
            .gap(px(16.))
            .child(
                h_flex()
                    .items_start()
                    .gap(px(16.))
                    .child(
                        div()
                            .size(px(256.))
                            .flex_none()
                            .child(choices.preview.clone()),
                    )
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .gap(px(8.))
                            .child(section_label("Recognized position"))
                            .child(
                                btn("rotate-image-position", "Rotate position")
                                    .compact()
                                    .on_click(this.listen(|screen, _, cx| {
                                        screen.update_image_choice(
                                            |choices| choices.rotated = !choices.rotated,
                                            cx,
                                        )
                                    })),
                            )
                            .child(
                                h_flex().gap(px(6.)).children(
                                    [
                                        (Color::White, "White to move"),
                                        (Color::Black, "Black to move"),
                                    ]
                                    .map(|(turn, label)| {
                                        btn(
                                            ("image-position-turn", usize::from(turn.is_black())),
                                            label,
                                        )
                                        .compact()
                                        .selected(choices.turn == turn)
                                        .on_click(
                                            this.listen(move |screen, _, cx| {
                                                screen.update_image_choice(
                                                    |choices| choices.turn = turn,
                                                    cx,
                                                )
                                            }),
                                        )
                                    }),
                                ),
                            )
                            .when(!can_open, |column| {
                                column.child(div().text_as(13., T.fg_2).child(t(
                                    "This position needs corrections before it can be opened.",
                                )))
                            })
                            .child(btn("confirm-image-position", action).primary().on_click(
                                this.listen(move |screen, window, cx| {
                                    screen.open_image_position(position.clone(), window, cx)
                                }),
                            )),
                    ),
            )
            .child(section_label("Boards in the image"))
            .child(self.render_image_candidates(choices, cx))
    }

    /// Lets the user choose a board by its source crop or its keyboard-accessible button.
    fn render_image_candidates(
        &self,
        choices: &BoardChoices,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let this = cx.entity();
        div()
            .id("image-board-list")
            .max_h(px(188.))
            .overflow_y_scroll()
            .child(
                h_flex().flex_wrap().gap(px(8.)).children(
                    choices
                        .candidates
                        .iter()
                        .enumerate()
                        .map(|(index, candidate)| {
                            let selected = index == choices.selected;
                            let select = || {
                                this.listen(move |screen, _, cx| {
                                    screen
                                        .update_image_choice(|choices| choices.selected = index, cx)
                                })
                            };
                            v_flex()
                                .w(px(116.))
                                .flex_none()
                                .gap(px(4.))
                                .child(
                                    div()
                                        .id(("image-board-thumbnail", index))
                                        .size(px(116.))
                                        .cursor_pointer()
                                        .border_as(if selected { T.accent } else { T.border })
                                        .on_click({
                                            let select = select();
                                            move |_, window, cx| select(window, cx)
                                        })
                                        .child(img(candidate.thumbnail.clone()).size_full()),
                                )
                                .child(
                                    btn(
                                        ("image-board-choice", index),
                                        t("Board {}").replace("{}", &(index + 1).to_string()),
                                    )
                                    .compact()
                                    .selected(selected)
                                    .on_click(select()),
                                )
                        }),
                ),
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

    /// Keeps the screenshot's valid and invalid guesses available for selection.
    #[test]
    fn imports_all_boards_with_their_own_preview_and_position() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../services/chessboard_detection/fixtures/screenshots/images/chessboard_detect_0004.png");
        let candidates = read_candidates(&ImageInput::File(path)).unwrap();
        assert_eq!(candidates.len(), 3);
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.thumbnail.frame_count() == 1)
        );

        let first = &candidates[0];
        let position = PositionEditor::from_board(first.oriented_board(false), Color::White)
            .position()
            .unwrap();
        assert_eq!(
            GameTree::fen_of(&position),
            "8/4k1p1/8/pr2Nb2/3R2P1/8/5PP1/2K5 w - - 0 1"
        );
        assert_eq!(
            first.oriented_board(true).board_fen().to_string(),
            "5K2/1PP5/8/1P2R3/2bN2rp/8/1p1k4/8"
        );
        let incomplete =
            PositionEditor::from_board(candidates[1].oriented_board(false), Color::Black);
        assert!(incomplete.position().is_err());
        assert_eq!(incomplete.turn, Color::Black);
        assert_eq!(incomplete.board, candidates[1].board);
    }
}
