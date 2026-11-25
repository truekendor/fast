//! The screenshot, board outlines, and the movable manual selection.

use std::cell::Cell;

use gpui::{CursorStyle, DragMoveEvent, Empty, KeyDownEvent, img, point};

use super::*;

/// An outline in the original screenshot's pixel coordinates.
#[derive(Clone, Copy)]
pub(super) struct ImageRect {
    pub start: [f64; 2],
    pub end: [f64; 2],
}

/// A square in source pixels, independent of the displayed image's scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SquareSelection {
    left: f64,
    top: f64,
    side: f64,
}

/// A move of the entire square or a resize with the opposite corner anchored.
#[derive(Clone, Copy)]
enum Handle {
    Center,
    Corner { x: f64, y: f64 },
}

/// What the navigation buttons select within the import dialog.
#[derive(Clone, Copy)]
enum Choice {
    Screenshot,
    Board,
}

/// The initial square and pointer position for a drag gesture.
struct SelectionDrag {
    square: SquareSelection,
    pointer: Cell<Point<Pixels>>,
    handle: Handle,
}

impl SquareSelection {
    /// Starts from the selected detection or a centered square in the screenshot.
    pub(super) fn centered(dimensions: (u32, u32), outline: Option<ImageRect>) -> Self {
        let (width, height) = (f64::from(dimensions.0), f64::from(dimensions.1));
        let (x, y, side) = match outline {
            Some(rect) => (
                (rect.start[0] + rect.end[0]) / 2.,
                (rect.start[1] + rect.end[1]) / 2.,
                (rect.end[0] - rect.start[0] + rect.end[1] - rect.start[1]) / 2.,
            ),
            None => (width / 2., height / 2., width.min(height) * 0.6),
        };
        let side = side.clamp(8., width.min(height));
        Self {
            left: (x - side / 2.).clamp(0., width - side),
            top: (y - side / 2.).clamp(0., height - side),
            side,
        }
    }

    /// Moves or resizes the square without crossing the image boundary.
    fn adjust(self, handle: Handle, dx: f64, dy: f64, dimensions: (u32, u32)) -> Self {
        let (width, height) = (f64::from(dimensions.0), f64::from(dimensions.1));
        match handle {
            Handle::Center => Self {
                left: (self.left + dx).clamp(0., width - self.side),
                top: (self.top + dy).clamp(0., height - self.side),
                ..self
            },
            Handle::Corner { x, y } => {
                let anchor_x = if x > 0. {
                    self.left
                } else {
                    self.left + self.side
                };
                let anchor_y = if y > 0. {
                    self.top
                } else {
                    self.top + self.side
                };
                let room_x = if x > 0. { width - anchor_x } else { anchor_x };
                let room_y = if y > 0. { height - anchor_y } else { anchor_y };
                let side =
                    (self.side + (dx * x + dy * y) / 2.).clamp(8., room_x.min(room_y).max(8.));
                Self {
                    left: if x > 0. { anchor_x } else { anchor_x - side },
                    top: if y > 0. { anchor_y } else { anchor_y - side },
                    side,
                }
            }
        }
    }

    /// Rounds to pixels and keeps the same integer side in both dimensions.
    pub(super) fn pixel_bounds(self, (width, height): (u32, u32)) -> [u32; 3] {
        let side = self.side.round() as u32;
        [
            (self.left.round() as u32).min(width - side),
            (self.top.round() as u32).min(height - side),
            side,
        ]
    }
}

/// Gives the frame and its handles a common drag type.
fn drag_selection(
    area: gpui::Stateful<gpui::Div>,
    square: SquareSelection,
    handle: Handle,
) -> gpui::Stateful<gpui::Div> {
    area.on_drag(
        SelectionDrag {
            square,
            pointer: Cell::new(point(px(0.), px(0.))),
            handle,
        },
        |drag, _, window, cx| {
            drag.pointer.set(window.mouse_position());
            cx.new(|_| Empty)
        },
    )
}

impl AnalyseScreen {
    /// Shows the original screenshot and selection controls on the left of the preview.
    pub(super) fn render_screenshot(
        &self,
        choices: &BoardChoices,
        width: f32,
        viewport_height: Pixels,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let this = cx.entity();
        let source = choices.screenshot();
        let (w, h) = source.pixels.dimensions();
        let height = (f32::from(viewport_height) - 205.).clamp(220., 490.);
        let scale = f64::from(width / w as f32).min(f64::from(height / h as f32));
        let draw = |value: f64| px((value * scale) as f32);
        let can_select_manually = w >= 8 && h >= 8;
        let manual = choices.manual.as_ref();
        let screenshot = div()
            .id("original-board-image")
            .relative()
            .flex_none()
            .size_full()
            .w(draw(f64::from(w)))
            .h(draw(f64::from(h)))
            .child(
                img(source.display.clone())
                    .size_full()
                    .object_fit(gpui::ObjectFit::Fill),
            )
            .on_drag_move(cx.listener(
                move |screen, event: &DragMoveEvent<SelectionDrag>, _, cx| {
                    let drag = event.drag(cx);
                    let delta = event.event.position - drag.pointer.get();
                    let square = drag.square.adjust(
                        drag.handle,
                        f64::from(f32::from(delta.x)) / scale,
                        f64::from(f32::from(delta.y)) / scale,
                        (w, h),
                    );
                    screen.move_image_selection(square, cx);
                },
            ))
            .when_some(manual, |image, manual| {
                image.children(self.render_manual_frame(manual, (w, h), scale, cx))
            })
            .when(manual.is_none(), |image| {
                image.children(source.candidates.iter().enumerate().map(|(index, board)| {
                    let rect = board.outline;
                    let active = choices.selected == index;
                    div()
                        .id(("detected-board-outline", index))
                        .absolute()
                        .left(draw(rect.start[0]))
                        .top(draw(rect.start[1]))
                        .w(draw(rect.end[0] - rect.start[0]))
                        .h(draw(rect.end[1] - rect.start[1]))
                        .border(if active { px(3.) } else { px(2.) })
                        .border_color(rgb(if active { T.accent } else { T.fg }))
                        .hover(|style| style.border_color(rgb(T.accent_hover)))
                        .cursor_pointer()
                        .on_click(cx.listener(move |screen, _, _, cx| {
                            screen.update_image_choice(|choices| choices.selected = index, cx)
                        }))
                }))
            });

        v_flex()
            .w(px(width))
            .flex_none()
            .min_w_0()
            .gap(px(8.))
            .child(
                h_flex()
                    .justify_between()
                    .gap(px(8.))
                    .child(section_label("Original screenshot"))
                    .when(choices.sources.len() > 1, |row| {
                        let count = choices.sources.len();
                        row.child(self.image_choice_nav(
                            Choice::Screenshot,
                            choices.source,
                            count,
                            cx,
                        ))
                    }),
            )
            .child(
                div()
                    .w_full()
                    .flex()
                    .justify_center()
                    .bg(rgb(T.bg))
                    .child(screenshot),
            )
            .child(
                h_flex()
                    .gap(px(8.))
                    .flex_wrap()
                    .child(
                        btn(
                            "manual-board",
                            if manual.is_some() {
                                "Use automatic detection"
                            } else {
                                "Select area manually"
                            },
                        )
                        .compact()
                        .disabled(!can_select_manually)
                        .on_click(
                            this.listen(|screen, window, cx| {
                                screen.toggle_image_selection(window, cx)
                            }),
                        ),
                    )
                    .when_some(manual, |row, manual| {
                        row.child(
                            btn(
                                "recognize-selected-area",
                                if manual.task.is_some() {
                                    "Recognizing…"
                                } else {
                                    "Recognize selected area"
                                },
                            )
                            .compact()
                            .disabled(manual.task.is_some())
                            .on_click(
                                this.listen(|screen, _, cx| screen.recognize_image_selection(cx)),
                            ),
                        )
                    })
                    .when(manual.is_none() && source.candidates.len() > 1, |row| {
                        row.child(self.image_choice_nav(
                            Choice::Board,
                            choices.selected,
                            source.candidates.len(),
                            cx,
                        ))
                    }),
            )
            .when(manual.is_none() && source.candidates.is_empty(), |column| {
                column.child(
                    div()
                        .text_as(13., T.fg_2)
                        .child(t("No chessboards found in this image.")),
                )
            })
            .when_some(manual, |column, manual| {
                let label = if manual.failed {
                    "Could not recognize the selected area."
                } else {
                    "Drag the frame or its corners. Use arrow keys to move and + or − to resize."
                };
                column.child(div().text_as(13., T.fg_2).child(t(label)))
            })
    }

    /// Lets the user browse images or detected boards using keyboard-accessible buttons.
    fn image_choice_nav(
        &self,
        choice: Choice,
        index: usize,
        count: usize,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let (kind, label, previous, following) = match choice {
            Choice::Screenshot => (
                "source-image",
                "Image {} of {}",
                "Previous image",
                "Next image",
            ),
            Choice::Board => (
                "detected-board",
                "Board {} of {}",
                "Previous board",
                "Next board",
            ),
        };
        let this = cx.entity();
        let next = |forward, icon, tip, disabled| {
            icon_btn((kind, usize::from(forward)), icon)
                .compact()
                .w(28.)
                .pad(0.)
                .tooltip(tip)
                .disabled(disabled)
                .on_click(this.listen(move |screen, _, cx| {
                    let next = if forward {
                        index + 1
                    } else {
                        index.saturating_sub(1)
                    };
                    screen.update_image_choice(
                        |choices| match choice {
                            Choice::Screenshot => {
                                choices.source = next;
                                choices.selected = 0;
                                choices.manual = None;
                                choices.rotated = false;
                            }
                            Choice::Board => choices.selected = next,
                        },
                        cx,
                    );
                }))
        };
        h_flex()
            .gap(px(6.))
            .child(next(false, "icons/chevron-left.svg", previous, index == 0))
            .child(
                div().text_as(13., T.fg_2).child(
                    t(label)
                        .replacen("{}", &(index + 1).to_string(), 1)
                        .replace("{}", &count.to_string()),
                ),
            )
            .child(next(
                true,
                "icons/chevron-right.svg",
                following,
                index + 1 == count,
            ))
    }

    /// Starts or dismisses a manual square while retaining the automatic results.
    fn toggle_image_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(ImageImport::Ready(choices)) = &mut self.image_import else {
            return;
        };
        if choices.manual.take().is_none() {
            let source = choices.screenshot();
            let (w, h) = source.pixels.dimensions();
            if w < 8 || h < 8 {
                return;
            }
            let square = SquareSelection::centered(
                (w, h),
                source
                    .candidates
                    .get(choices.selected)
                    .map(|board| board.outline),
            );
            let focus = cx.focus_handle().tab_stop(true);
            focus.focus(window, cx);
            choices.manual = Some(ManualSelection {
                square,
                focus,
                result: None,
                task: None,
                failed: false,
            });
        }
        choices.update_preview(cx);
        cx.notify();
    }

    /// Updates the square and clears predictions made from its previous pixels.
    fn move_image_selection(&mut self, square: SquareSelection, cx: &mut Context<Self>) {
        let Some(ImageImport::Ready(choices)) = &mut self.image_import else {
            return;
        };
        let Some(manual) = &mut choices.manual else {
            return;
        };
        if manual.square == square {
            return;
        }
        manual.square = square;
        manual.task = None;
        manual.failed = false;
        if manual.result.take().is_some() {
            choices.update_preview(cx);
        }
        cx.notify();
    }

    /// Draws the drag target, corner handles, and visible keyboard focus for the square.
    fn render_manual_frame(
        &self,
        manual: &ManualSelection,
        dimensions: (u32, u32),
        scale: f64,
        cx: &mut Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        let square = manual.square;
        let draw = |value: f64| px((value * scale) as f32);
        let frame = div()
            .id("manual-image-frame")
            .absolute()
            .left(draw(square.left))
            .top(draw(square.top))
            .size(draw(square.side))
            .border_2()
            .border_color(rgb(T.accent))
            .cursor(CursorStyle::OpenHand)
            .track_focus(&manual.focus)
            .focus_visible(|style| style.border(px(3.)).border_color(rgb(T.accent_hover)))
            .on_key_down(cx.listener(move |screen, event: &KeyDownEvent, _, cx| {
                let step = if event.keystroke.modifiers.shift {
                    8.
                } else {
                    1.
                };
                let (handle, dx, dy) = match event.keystroke.key.as_str() {
                    "left" => (Handle::Center, -step, 0.),
                    "right" => (Handle::Center, step, 0.),
                    "up" => (Handle::Center, 0., -step),
                    "down" => (Handle::Center, 0., step),
                    "+" | "=" => (Handle::Corner { x: 1., y: 1. }, step, step),
                    "-" => (Handle::Corner { x: 1., y: 1. }, -step, -step),
                    _ => return,
                };
                screen.move_image_selection(square.adjust(handle, dx, dy, dimensions), cx);
                cx.stop_propagation();
            }));
        let mut elements = vec![drag_selection(frame, square, Handle::Center).into_any_element()];
        elements.extend(
            [(-1., -1.), (1., -1.), (-1., 1.), (1., 1.)]
                .into_iter()
                .enumerate()
                .map(|(index, (x, y))| {
                    let handle = div()
                        .id(("image-frame-handle", index))
                        .absolute()
                        .left(draw(square.left + if x > 0. { square.side } else { 0. }) - px(7.))
                        .top(draw(square.top + if y > 0. { square.side } else { 0. }) - px(7.))
                        .size(px(14.))
                        .rounded(px(3.))
                        .bg(rgb(T.accent))
                        .border_2()
                        .border_color(rgb(T.accent_fg))
                        .occlude()
                        .cursor(if x == y {
                            CursorStyle::ResizeUpLeftDownRight
                        } else {
                            CursorStyle::ResizeUpRightDownLeft
                        });
                    drag_selection(handle, square, Handle::Corner { x, y }).into_any_element()
                }),
        );
        elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keeps the square inside the source image and fixes the opposite corner when resizing.
    #[test]
    fn moves_and_resizes_the_square_within_source_pixels() {
        let size = (100, 80);
        let initial = SquareSelection {
            left: 20.,
            top: 20.,
            side: 40.,
        };
        assert_eq!(
            initial
                .adjust(Handle::Center, 120., -20., size)
                .pixel_bounds(size),
            [60, 0, 40]
        );
        let upper_left = Handle::Corner { x: -1., y: -1. };
        assert_eq!(
            initial
                .adjust(upper_left, -100., -100., size)
                .pixel_bounds(size),
            [0, 0, 60]
        );
        assert_eq!(
            initial
                .adjust(upper_left, 100., 100., size)
                .pixel_bounds(size),
            [52, 52, 8]
        );
    }
}
