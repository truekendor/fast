//! Shared timing for hover tooltips.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use gpui::{
    App, Div, Global, IntoElement, RenderOnce, SharedString, Stateful, StatefulInteractiveElement,
    Window, WindowId,
};
use gpui_component::tooltip::Tooltip;

const QUICK_SHOW_WINDOW: Duration = Duration::from_millis(1000);

/// Keeps track of when each window returns to the regular tooltip delay.
#[derive(Default)]
struct TooltipDeadlines(HashMap<WindowId, Instant>);

impl Global for TooltipDeadlines {}

/// Adds a tooltip to a control or other interactive element.
#[derive(IntoElement)]
pub struct TooltipTrigger {
    trigger: Stateful<Div>,
    text: Option<SharedString>,
}

impl TooltipTrigger {
    /// Keeps the element's normal interaction and adds a tooltip when text is supplied.
    pub fn new(trigger: Stateful<Div>, text: Option<SharedString>) -> Self {
        Self { trigger, text }
    }
}

/// Lets interactive elements share the same tooltip timing as buttons.
pub trait TooltipExt {
    /// Shows translated text on hover using the shared tooltip delay.
    fn with_tooltip(self, text: impl Into<SharedString>) -> TooltipTrigger;
}

impl TooltipExt for Stateful<Div> {
    fn with_tooltip(self, text: impl Into<SharedString>) -> TooltipTrigger {
        TooltipTrigger::new(self, Some(text.into()))
    }
}

impl RenderOnce for TooltipTrigger {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let Some(text) = self.text else {
            return self.trigger;
        };
        let window_id = window.window_handle().window_id();
        let quick_show = cx
            .try_global::<TooltipDeadlines>()
            .and_then(|deadlines| deadlines.0.get(&window_id))
            .is_some_and(|deadline| Instant::now() < *deadline);
        let trigger = self.trigger.tooltip(move |window, cx| {
            make_next_tooltip_immediate(window, cx);
            Tooltip::new(text.clone()).build(window, cx)
        });
        if quick_show {
            trigger.tooltip_show_delay(Duration::ZERO)
        } else {
            trigger
        }
    }
}

/// Starts or renews the window in which nearby tooltips appear without a delay.
fn make_next_tooltip_immediate(window: &mut Window, cx: &mut App) {
    let handle = window.window_handle();
    let window_id = handle.window_id();
    let deadline = Instant::now() + QUICK_SHOW_WINDOW;
    cx.default_global::<TooltipDeadlines>()
        .0
        .insert(window_id, deadline);

    // GPUI refreshes the window when it shows a tooltip. Refresh again when the
    // grace period ends so cached elements go back to GPUI's normal delay.
    cx.spawn(async move |cx| {
        cx.background_executor().timer(QUICK_SHOW_WINDOW).await;
        cx.update(|cx| {
            let deadlines = &mut cx.default_global::<TooltipDeadlines>().0;
            if deadlines.get(&window_id) == Some(&deadline) {
                deadlines.remove(&window_id);
                let _ = handle.update(cx, |_, window, _| window.refresh());
            }
        });
    })
    .detach();
}
