//! Right-to-left support.
//!
//! The vendored gpui (see `vendor/PATCHES.md`) gives every element an
//! inherited `LayoutDirection`. `apply` sets the process default from the
//! settings, so every window — including dialogs, popovers and notifications
//! — lays flex rows out from the right and right-aligns text. Views therefore
//! write ordinary `h_flex()` rows in *reading order* (first child = leading
//! edge) and never reverse them by hand.
//!
//! What still needs care in views:
//! - Directional icons: use [`back_icon`] / [`forward_icon`] / [`chevron_end`]
//!   instead of naming `ChevronLeft`/`ArrowLeft` directly.
//! - Inputs and textareas follow the inherited alignment (vendored
//!   gpui-component/gpui-base patch), so they are right-aligned in RTL with no
//!   extra code; [`input`] spells the same thing out explicitly.
//! - Physical paddings/margins (`pl_*`, `mr_*`, absolute `left`/`right`) do not
//!   mirror; prefer `gap_*` and symmetric `px_*`.
//! - Content that is inherently LTR (code blocks, file paths, waveforms,
//!   progress bars, numbers columns) can opt out with `.ltr()` on its container.

use gpui_kit::component::{
    IconName,
    input::{Input, InputState},
};
use gpui_kit::{App, Entity, LayoutDirection, Styled as _, TextAlign};

use crate::store::AppSettings;

/// Whether the app is laid out right to left.
pub fn is_rtl(cx: &App) -> bool {
    AppSettings::get(cx).rtl
}

/// Make the process-wide default direction match the settings. Call at
/// startup and whenever the `rtl` setting changes (then refresh windows).
pub fn apply(cx: &App) {
    LayoutDirection::set_default(if is_rtl(cx) {
        LayoutDirection::Rtl
    } else {
        LayoutDirection::Ltr
    });
}

/// Text alignment for reading-order start.
pub fn start_align(cx: &App) -> TextAlign {
    if is_rtl(cx) {
        TextAlign::Right
    } else {
        TextAlign::Left
    }
}

/// A single-line `Input` aligned to the reading-order start.
pub fn input(state: &Entity<InputState>, cx: &App) -> Input {
    Input::new(state).text_align(start_align(cx))
}

/// "Back" arrow (points toward the leading edge).
pub fn back_icon(cx: &App) -> IconName {
    if is_rtl(cx) {
        IconName::ArrowRight
    } else {
        IconName::ArrowLeft
    }
}

/// "Forward" arrow (points toward the trailing edge).
pub fn forward_icon(cx: &App) -> IconName {
    if is_rtl(cx) {
        IconName::ArrowLeft
    } else {
        IconName::ArrowRight
    }
}
