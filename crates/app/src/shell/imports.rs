//! Receives files and clipboard data for Analyse and the new-tab picker.

use std::path::PathBuf;

use gpui::ExternalPaths;

use super::*;
use crate::analysis_import::{
    ClipboardImport, ImageInput, clipboard_input, read_game_files, split_files, supports_drop,
};

/// The destination selected by the area that receives the drop.
pub(super) enum FileDropTarget {
    Analyse,
    AnalyseTab(u64),
    NewTab,
}

/// Shows whether the files currently held over the workspace can be imported.
pub(super) fn file_drop_overlay() -> impl IntoElement {
    div()
        .absolute()
        .inset_0()
        .child(file_drop_prompt(true))
        .child(file_drop_prompt(false))
}

/// Uses the same acceptance check for the overlay and tab strip.
pub(super) fn file_drop_border(files: &ExternalPaths) -> u32 {
    if supports_drop(files.paths()) {
        T.accent
    } else {
        T.warn
    }
}

/// Draws the appropriate prompt only while the cursor is dragging over the area.
fn file_drop_prompt(accepted: bool) -> impl IntoElement {
    let (colour, icon, title, description) = if accepted {
        (
            T.accent,
            "icons/download.svg",
            "Drop files to import",
            "Screenshots and chess files open in Analyse.",
        )
    } else {
        (
            T.warn,
            "icons/x.svg",
            "Unsupported file format",
            "Supported formats: PNG, JPEG, BMP, WebP, PGN, PGN.gz, FEN and EPD.",
        )
    };
    div()
        .absolute()
        .inset_0()
        .flex()
        .items_center()
        .justify_center()
        .p(px(16.))
        .opacity(0.)
        .drag_over::<ExternalPaths>(move |style, files, _, _| {
            style.opacity(if supports_drop(files.paths()) == accepted {
                1.
            } else {
                0.
            })
        })
        .bg(rgb(T.bg).alpha(0.76))
        .child(
            v_flex()
                .max_w_full()
                .w(px(376.))
                .items_center()
                .gap(px(8.))
                .p(px(24.))
                .rounded(px(6.))
                .border_2()
                .border_dashed()
                .border_color(rgb(colour))
                .bg(rgb(T.surface))
                .text_center()
                .child(ui::widgets::icon(icon, px(28.), colour))
                .child(
                    div()
                        .text_as(18., T.fg)
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(ui::i18n::t(title)),
                )
                .child(div().text_as(13., T.fg_2).child(ui::i18n::t(description))),
        )
}

impl Workspace {
    /// Rejects drops that include unsupported formats before changing tabs or starting imports.
    pub(super) fn drop_analysis_files(
        &mut self,
        files: &ExternalPaths,
        target: FileDropTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !supports_drop(files.paths()) {
            self.model.toast("Unsupported file format", cx);
            return;
        }
        if let FileDropTarget::AnalyseTab(id) = target {
            self.activate_tab(id, cx);
        }
        self.import_analysis_files(
            files.paths().to_vec(),
            matches!(target, FileDropTarget::NewTab),
            window,
            cx,
        );
    }

    /// Opens a fresh Analyse tab for clipboard data received by the new-tab picker.
    pub(super) fn paste_analysis(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let from_picker = self.new_tab_menu_open;
        match cx.read_from_clipboard().and_then(clipboard_input) {
            Some(ClipboardImport::Files(paths)) => {
                if !from_picker {
                    self.set_mode(Mode::Analyse, cx);
                }
                self.import_analysis_files(paths, from_picker, window, cx);
            }
            Some(ClipboardImport::Screenshot(image)) => {
                self.preview_imported_images(vec![image], from_picker, window, cx);
            }
            Some(ClipboardImport::Text(text)) => {
                if from_picker {
                    self.new_tab(Mode::Analyse, window, cx);
                } else {
                    self.set_mode(Mode::Analyse, cx);
                }
                self.analyse
                    .update(cx, |screen, cx| screen.load_text(&text, window, cx));
            }
            None => self.model.toast("The clipboard is empty.", cx),
        }
    }

    /// Opens dropped notation in its own tabs and presents screenshots in the import dialog.
    fn import_analysis_files(
        &mut self,
        paths: Vec<PathBuf>,
        new_tab: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (images, game_files) = split_files(paths);
        if game_files.is_empty() {
            self.preview_imported_images(images, new_tab, window, cx);
            return;
        }

        let origin = self.analyse.downgrade();
        cx.spawn_in(window, async move |this, cx| {
            let (games, errors) = cx
                .background_spawn(async move { read_game_files(&game_files) })
                .await;
            let _ = this.update_in(cx, |workspace, window, cx| {
                if !new_tab && games.is_empty() {
                    let _ = origin.update(cx, |screen, cx| {
                        screen.start_image_import(images, false, cx);
                    });
                } else if new_tab || origin.upgrade().is_some() {
                    for tree in games {
                        workspace.open_analysis(tree, window, cx);
                    }
                    workspace.preview_imported_images(images, new_tab, window, cx);
                }
                if !errors.is_empty() {
                    workspace.model.toast(errors.join("\n"), cx);
                }
            });
        })
        .detach();
    }

    /// Starts the preview in the active Analyse tab or in one reserved for the import.
    fn preview_imported_images(
        &mut self,
        images: Vec<ImageInput>,
        new_tab: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if images.is_empty() {
            return;
        }
        if new_tab {
            self.new_tab(Mode::Analyse, window, cx);
        } else {
            self.set_mode(Mode::Analyse, cx);
        }
        self.analyse.update(cx, |screen, cx| {
            screen.start_image_import(images, new_tab, cx)
        });
    }
}
