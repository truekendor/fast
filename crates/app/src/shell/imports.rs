//! Routes opened, dropped and pasted files to Analyse or Database.

use std::{
    ops::ControlFlow,
    path::PathBuf,
    time::{Duration, Instant},
};

use chess_core::GameTree;
use gpui::ExternalPaths;

use super::*;
use crate::analysis_import::{
    CHESS_FILE_EXTENSIONS, COMPRESSED_PGN_EXTENSION, ClipboardImport, ImageInput,
    SCREENSHOT_EXTENSIONS, clipboard_input, read_game_files, split_files, split_game_collections,
    supports_drop,
};
use crate::recovery::Recovery;
use crate::screens::test_positions::SUITE_RUN_IN_PROGRESS;

/// An imported game with a reserved tab and recovery file.
struct ImportedGame {
    id: u64,
    path: PathBuf,
    tree: GameTree,
}

impl ImportedGame {
    /// Writes the recovery snapshot before the tab is added to the session.
    fn save_recovery(&self) -> anyhow::Result<String> {
        let pgn = chess_core::write_pgn(&self.tree);
        Recovery::save(&self.path, pgn.clone())?;
        Ok(pgn)
    }
}

/// The destination selected by the area that receives the drop.
pub(super) enum FileDropTarget {
    Workspace,
    AnalyseTab(u64),
    NewTab,
}

/// What the workspace can do with the files under the cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileDropKind {
    Analyse,
    TestPositions,
    TestPositionsRunning,
    Unsupported,
}

/// Routes a single EPD to Test positions when that screen is visible.
/// The run state is absent over the rest of the workspace.
fn file_drop_kind(paths: &[PathBuf], test_running: Option<bool>) -> FileDropKind {
    if let Some(running) = test_running
        && let [path] = paths
        && path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("epd"))
    {
        return if running {
            FileDropKind::TestPositionsRunning
        } else {
            FileDropKind::TestPositions
        };
    }
    if supports_drop(paths) {
        FileDropKind::Analyse
    } else {
        FileDropKind::Unsupported
    }
}

/// Shows whether the files currently held over the workspace can be imported.
pub(super) fn file_drop_overlay(
    test_positions: Option<&Entity<TestPositionsScreen>>,
) -> impl IntoElement + use<> {
    div()
        .absolute()
        .inset_0()
        .child(file_drop_prompt(FileDropKind::Analyse, test_positions))
        .when(test_positions.is_some(), |overlay| {
            overlay
                .child(file_drop_prompt(
                    FileDropKind::TestPositions,
                    test_positions,
                ))
                .child(file_drop_prompt(
                    FileDropKind::TestPositionsRunning,
                    test_positions,
                ))
        })
        .child(file_drop_prompt(FileDropKind::Unsupported, test_positions))
}

/// Uses the same acceptance check for the overlay and tab strip.
pub(super) fn file_drop_border(files: &ExternalPaths) -> u32 {
    if file_drop_kind(files.paths(), None) == FileDropKind::Analyse {
        T.accent
    } else {
        T.warn
    }
}

/// Draws the appropriate prompt only while the cursor is dragging over the area.
fn file_drop_prompt(
    prompt: FileDropKind,
    test_positions: Option<&Entity<TestPositionsScreen>>,
) -> impl IntoElement + use<> {
    let (colour, icon, title) = match prompt {
        FileDropKind::Analyse => (T.accent, "icons/download.svg", "Drop files to import"),
        FileDropKind::TestPositions => (T.accent, "icons/download.svg", "Drop an .epd file here"),
        FileDropKind::TestPositionsRunning => (T.warn, "icons/x.svg", "Test run in progress"),
        FileDropKind::Unsupported => (T.warn, "icons/x.svg", "Unsupported file format"),
    };
    let test_positions = test_positions.cloned();
    div()
        .absolute()
        .inset_0()
        .flex()
        .items_center()
        .justify_center()
        .p(px(16.))
        .opacity(0.)
        .drag_over::<ExternalPaths>(move |style, files, _, cx| {
            let running = test_positions
                .as_ref()
                .map(|screen| screen.read(cx).is_running());
            style.opacity(if file_drop_kind(files.paths(), running) == prompt {
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
                .when(prompt == FileDropKind::Analyse, |prompt| {
                    prompt.child(
                        div()
                            .text_as(13., T.fg_2)
                            .child(ui::i18n::t("PGNs with multiple games open in Database.")),
                    )
                })
                .when(prompt == FileDropKind::TestPositionsRunning, |prompt| {
                    prompt.child(
                        div()
                            .text_as(13., T.fg_2)
                            .child(ui::i18n::t(SUITE_RUN_IN_PROGRESS)),
                    )
                })
                .when(prompt == FileDropKind::Unsupported, |prompt| {
                    prompt.child(
                        v_flex()
                            .w_full()
                            .gap(px(10.))
                            .pt(px(4.))
                            .child(file_extension_group(
                                "Images",
                                SCREENSHOT_EXTENSIONS.iter().copied(),
                            ))
                            .child(file_extension_group(
                                "Chess files",
                                CHESS_FILE_EXTENSIONS
                                    .iter()
                                    .copied()
                                    .chain([COMPRESSED_PGN_EXTENSION]),
                            )),
                    )
                }),
        )
}

/// Groups supported extensions under a label and displays them as monospaced chips.
fn file_extension_group(
    label: &'static str,
    extensions: impl IntoIterator<Item = &'static str>,
) -> impl IntoElement {
    v_flex()
        .items_start()
        .gap(px(6.))
        .p(px(10.))
        .rounded(px(4.))
        .bg(rgb(T.surface_2))
        .child(ui::widgets::section_label(label))
        .child(
            h_flex()
                .flex_wrap()
                .gap(px(4.))
                .children(extensions.into_iter().map(|ext| {
                    ui::widgets::mono(format!(".{ext}"))
                        .whitespace_nowrap()
                        .text_as(12., T.fg)
                        .px(px(6.))
                        .py(px(2.))
                        .rounded(px(4.))
                        .border_as(T.border_strong)
                        .bg(rgb(T.surface))
                })),
        )
}

impl Workspace {
    /// Exposes the suite screen only while the workspace content shows it.
    pub(super) fn visible_test_positions(&self) -> Option<&Entity<TestPositionsScreen>> {
        self.test_positions
            .as_ref()
            .filter(|_| self.mode == Mode::TestPositions && !self.new_tab_menu_open)
    }

    /// Picks notation files to open in Analyse or import into Database.
    pub(super) fn open_game_files(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let workspace = cx.entity().downgrade();
        let target = window.window_handle();
        crate::model::pick_paths(self.model.clone(), "Open games", cx, move |paths, cx| {
            let _ = target.update(cx, |_, window, cx| {
                let _ = workspace.update(cx, |workspace, cx| {
                    workspace.route_game_files(paths, Vec::new(), None, window, cx);
                });
            });
        });
    }

    /// Routes a drop to its screen or opens the files in Analyse.
    pub(super) fn drop_analysis_files(
        &mut self,
        files: &ExternalPaths,
        target: FileDropTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let test_positions = match target {
            FileDropTarget::Workspace => self.visible_test_positions(),
            _ => None,
        };
        let running = test_positions.map(|screen| screen.read(cx).is_running());
        match file_drop_kind(files.paths(), running) {
            FileDropKind::TestPositions => {
                if let (Some(screen), [path]) = (test_positions, files.paths()) {
                    screen.update(cx, |screen, cx| screen.load(path.clone(), cx));
                }
                return;
            }
            FileDropKind::TestPositionsRunning => {
                self.model.toast(SUITE_RUN_IN_PROGRESS, cx);
                return;
            }
            FileDropKind::Unsupported => {
                self.model.toast("Unsupported file format", cx);
                return;
            }
            FileDropKind::Analyse => {}
        }
        let edit_tab = match target {
            FileDropTarget::Workspace => self.image_edit_target(),
            FileDropTarget::AnalyseTab(id) => Some(id),
            FileDropTarget::NewTab => None,
        };
        self.import_analysis_files(files.paths().to_vec(), edit_tab, window, cx);
    }

    /// Opens pasted notation or previews a screenshot before opening it.
    pub(super) fn paste_analysis(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let from_picker = self.new_tab_menu_open;
        let edit_tab = self.image_edit_target();
        match cx.read_from_clipboard().and_then(clipboard_input) {
            Some(ClipboardImport::Files(paths)) => {
                self.import_analysis_files(paths, edit_tab, window, cx);
            }
            Some(ClipboardImport::Screenshot(image)) => {
                self.preview_imported_images(vec![image], edit_tab, cx);
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
        edit_tab: Option<u64>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (images, game_files) = split_files(paths);
        if game_files.is_empty() {
            self.preview_imported_images(images, edit_tab, cx);
            return;
        }
        if images.is_empty()
            && let Some(id) = edit_tab
        {
            self.activate_tab(id, cx);
        }

        self.route_game_files(game_files, images, edit_tab, window, cx);
    }

    /// Directs collections to Database before any of their games can create tabs.
    fn route_game_files(
        &mut self,
        paths: Vec<PathBuf>,
        images: Vec<ImageInput>,
        edit_tab: Option<u64>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.spawn_in(window, async move |this, cx| {
            let (collections, notation) = cx
                .background_spawn(async move { split_game_collections(paths) })
                .await;
            let _ = this.update_in(cx, |workspace, window, cx| {
                let open_in_analyse = collections.is_empty();
                if !open_in_analyse {
                    workspace.set_mode(Mode::Database, cx);
                    if workspace.database.read(cx).collection_busy() {
                        workspace.new_tab(Mode::Database, window, cx);
                    }
                    workspace.database.update(cx, |database, cx| {
                        database.import_collections(collections, window, cx);
                    });
                }
                if notation.is_empty() {
                    workspace.preview_imported_images(images, edit_tab, cx);
                } else {
                    workspace.open_game_stream(
                        notation,
                        images,
                        edit_tab,
                        open_in_analyse,
                        window,
                        cx,
                    );
                }
            });
        })
        .detach();
    }

    /// Reads files ahead of the UI and adds tabs in short slices so input and redraws can run.
    fn open_game_stream(
        &mut self,
        game_files: Vec<PathBuf>,
        images: Vec<ImageInput>,
        edit_tab: Option<u64>,
        focus_first: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        const GAME_BUFFER: usize = 16;
        const UI_SLICE: Duration = Duration::from_millis(8);
        const FRAME: Duration = Duration::from_millis(16);

        let source_tab = focus_first.then_some((self.active_tab, self.new_tab_menu_open));
        let (tx, rx) = smol::channel::bounded(GAME_BUFFER);
        cx.spawn_in(window, async move |this, cx| {
            let reader = smol::unblock(move || {
                read_game_files(&game_files, |result| match tx.send_blocking(result) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(_) => ControlFlow::Break(()),
                });
            });
            let mut opened = 0;
            let mut failed = 0;
            let mut first_error = None;
            while let Ok(first) = rx.recv().await {
                let batch = this.update_in(cx, |workspace, _, cx| {
                    let results = std::iter::once(first)
                        .chain(std::iter::from_fn(|| rx.try_recv().ok()))
                        .take(GAME_BUFFER);
                    let mut games = Vec::new();
                    for result in results {
                        match result {
                            Ok(tree) => {
                                let (id, path) = workspace.reserve_tab(Mode::Analyse, cx);
                                games.push(ImportedGame {
                                    id,
                                    path: path.expect("Analyse tabs have a recovery file"),
                                    tree,
                                });
                            }
                            Err(error) => {
                                failed += 1;
                                first_error.get_or_insert(error);
                            }
                        }
                    }
                    games
                });
                let Ok(batch) = batch else {
                    return;
                };
                if !batch.is_empty() {
                    let mut prepared = cx
                        .background_spawn(async move {
                            batch
                                .into_iter()
                                .map(|game| {
                                    let saved = game.save_recovery();
                                    (game, saved)
                                })
                                .collect::<Vec<_>>()
                        })
                        .await
                        .into_iter()
                        .peekable();
                    while prepared.peek().is_some() {
                        let alive = this.update_in(cx, |workspace, window, cx| {
                            let start = Instant::now();
                            for (game, saved) in prepared.by_ref() {
                                let recovery_pgn = saved.unwrap_or_else(|err| {
                                    failed += 1;
                                    first_error.get_or_insert_with(|| {
                                        format!(
                                            "{} {err}",
                                            ui::i18n::t("Could not autosave this game:")
                                        )
                                    });
                                    String::new()
                                });
                                let current_tab =
                                    (workspace.active_tab, workspace.new_tab_menu_open);
                                let activate = workspace.tabs.is_empty()
                                    || (opened == 0 && Some(current_tab) == source_tab);
                                workspace.add_imported_game(
                                    game,
                                    recovery_pgn,
                                    activate,
                                    window,
                                    cx,
                                );
                                opened += 1;
                                if start.elapsed() >= UI_SLICE {
                                    break;
                                }
                            }
                            workspace.save_session(cx);
                            cx.notify();
                        });
                        if alive.is_err() {
                            return;
                        }
                        if prepared.peek().is_some() {
                            cx.background_executor().timer(FRAME).await;
                        }
                    }
                }
                if !rx.is_closed() || !rx.is_empty() {
                    cx.background_executor().timer(FRAME).await;
                }
            }
            reader.await;
            let _ = this.update_in(cx, |workspace, _, cx| {
                workspace.preview_imported_images(images, edit_tab, cx);
                if let Some(error) = first_error {
                    let summary =
                        ui::i18n::tf("Opened {} games or positions.", &[&opened.to_string()]);
                    let failures = if failed == 1 {
                        ui::i18n::t("One import error.").to_owned()
                    } else {
                        ui::i18n::tf("{} import errors.", &[&failed.to_string()])
                    };
                    workspace
                        .model
                        .toast(format!("{summary} {failures}\n{error}"), cx);
                }
            });
        })
        .detach();
    }

    /// Adds the game to Analyse while keeping the user's current tab after the first result.
    fn add_imported_game(
        &mut self,
        game: ImportedGame,
        recovery_pgn: String,
        activate: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let ImportedGame { id, path, tree } = game;
        let model = self.model.clone();
        let workspace = cx.entity().downgrade();
        let screen = cx.new(|cx| {
            AnalyseScreen::new_imported(model, workspace, window, cx, path, tree, recovery_pgn)
        });
        self.tabs.push(WorkspaceTab {
            id,
            content: TabContent::Analyse(screen),
            name: None,
        });
        if activate {
            self.activate_tab(id, cx);
        }
    }

    /// Starts a workspace preview without opening or switching tabs.
    fn preview_imported_images(
        &mut self,
        images: Vec<ImageInput>,
        edit_tab: Option<u64>,
        cx: &mut Context<Self>,
    ) {
        self.image_import.update(cx, |dialog, cx| {
            dialog.start_image_import(images, edit_tab, cx)
        });
    }

    /// Remembers the Analyse tab that can edit the guess if the user asks.
    pub(super) fn image_edit_target(&self) -> Option<u64> {
        (self.mode == Mode::Analyse && !self.new_tab_menu_open).then_some(self.active_tab)
    }

    /// Opens the requested Analyse tab for correction, creating one if needed.
    pub(crate) fn image_editor_tab(
        &mut self,
        edit_tab: Option<u64>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Entity<AnalyseScreen> {
        if let Some(id) = edit_tab
            && self
                .tabs
                .iter()
                .any(|tab| tab.id == id && matches!(tab.content, TabContent::Analyse(_)))
        {
            self.activate_tab(id, cx);
        } else {
            self.new_tab(Mode::Analyse, window, cx);
        }
        self.analyse.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A running test blocks replacing its suite, while unrelated files keep their route.
    #[test]
    fn running_test_only_blocks_a_single_epd_dropped_on_its_screen() {
        let suite = [PathBuf::from("suite.EPD")];
        assert_eq!(
            file_drop_kind(&suite, Some(false)),
            FileDropKind::TestPositions
        );
        assert_eq!(
            file_drop_kind(&suite, Some(true)),
            FileDropKind::TestPositionsRunning
        );
        assert_eq!(file_drop_kind(&suite, None), FileDropKind::Analyse);
        assert_eq!(
            file_drop_kind(&[PathBuf::from("games.pgn")], Some(true)),
            FileDropKind::Analyse
        );
        assert_eq!(
            file_drop_kind(&[suite[0].clone(), PathBuf::from("second.epd")], Some(true)),
            FileDropKind::Analyse
        );
        assert_eq!(
            file_drop_kind(&[PathBuf::from("invalid.zip")], Some(true)),
            FileDropKind::Unsupported
        );
    }
}
