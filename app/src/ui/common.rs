//! Small presentational pieces shared by the pages.

use gpui_kit::component::{ActiveTheme as _, Icon, Sizable as _, button::Button, h_flex, v_flex};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{App, IntoElement, ParentElement as _, SharedString, Styled as _, Window, div, px};

use gpui_kit::assets::IconName;

use crate::sidecar::{Sidecar, SidecarGlobal, SidecarStatus};

/// Page header: title, optional muted description, trailing actions.
pub fn page_header(
    title: impl Into<SharedString>,
    description: Option<SharedString>,
    actions: impl IntoElement,
    cx: &App,
) -> impl IntoElement {
    h_flex()
        .w_full()
        .px_6()
        .py_4()
        .gap_4()
        .border_b_1()
        .border_color(cx.theme().border)
        .child(
            v_flex()
                .flex_1()
                .min_w_0()
                .gap_1()
                .child(
                    div()
                        .text_xl()
                        .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                        .child(title.into()),
                )
                .when_some(description, |this, text| {
                    this.child(
                        div()
                            .text_sm()
                            .text_color(cx.theme().muted_foreground)
                            .child(text),
                    )
                }),
        )
        .child(h_flex().gap_2().flex_shrink_0().child(actions))
}

/// Centered empty state with an icon, a title, a hint and an optional action.
pub fn empty_state(
    icon: IconName,
    title: impl Into<SharedString>,
    hint: impl Into<SharedString>,
    action: Option<Button>,
    cx: &App,
) -> impl IntoElement {
    v_flex()
        .size_full()
        .items_center()
        .justify_center()
        .gap_3()
        .p_8()
        .child(
            div()
                .p_3()
                .rounded(cx.theme().radius_lg)
                .bg(cx.theme().muted)
                .child(
                    Icon::new(icon)
                        .large()
                        .text_color(cx.theme().muted_foreground),
                ),
        )
        .child(
            div()
                .text_lg()
                .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                .child(title.into()),
        )
        .child(
            div()
                .max_w(px(420.))
                .text_center()
                .text_sm()
                .text_color(cx.theme().muted_foreground)
                .child(hint.into()),
        )
        .when_some(action, |this, action| this.child(action))
}

/// Shown in place of sidecar-backed content while the sidecar is not online:
/// explains the state and offers a retry.
pub fn sidecar_offline(cx: &App) -> impl IntoElement {
    let sidecar = SidecarGlobal::get(cx);
    let status = sidecar.read(cx).status().clone();
    let (title, hint, busy) = match &status {
        SidecarStatus::Checking => ("در حال بررسی سایدکار…", String::new(), true),
        SidecarStatus::Starting => (
            "سایدکار در حال راه‌اندازی است",
            "بارگذاری بار اول ممکن است چند دقیقه طول بکشد.".to_string(),
            true,
        ),
        SidecarStatus::Offline(err) => ("سایدکار آفلاین است", err.clone(), false),
        SidecarStatus::Online(_) => ("", String::new(), false),
    };
    empty_state(
        IconName::ServerOff,
        title,
        hint,
        Some(
            Button::new("sidecar-retry")
                .label("تلاش دوباره")
                .icon(IconName::RefreshCw)
                .loading(busy)
                .on_click(move |_, _: &mut Window, cx| {
                    sidecar.update(cx, |sidecar: &mut Sidecar, cx| sidecar.retry(cx));
                }),
        ),
        cx,
    )
}

/// Format seconds as `m:ss`.
pub fn format_duration(seconds: f64) -> String {
    let total = seconds.max(0.).round() as u64;
    format!("{}:{:02}", total / 60, total % 60)
}

/// Western digits → Persian digits, for display.
pub fn fa_digits(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '0'..='9' => char::from_u32('۰' as u32 + (c as u32 - '0' as u32)).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// A wrapping row of short metadata items with small separators between
/// them. Each item is its own text element, so bidi reordering never mixes
/// Persian, Latin and digits across items and a wrap never strands a
/// separator at a line start.
pub fn meta_row(items: impl IntoIterator<Item = SharedString>, cx: &App) -> gpui_kit::Div {
    let muted = cx.theme().muted_foreground;
    let mut row = h_flex().flex_wrap().gap_x_1p5().gap_y_0p5().min_w_0();
    for (ix, item) in items.into_iter().enumerate() {
        // The separator travels with the item after it, so a wrap never
        // leaves one dangling at a line end.
        row = row.child(
            h_flex()
                .gap_1p5()
                .when(ix > 0, |this| {
                    this.child(div().size(px(3.)).rounded_full().bg(muted).flex_shrink_0())
                })
                .child(div().whitespace_nowrap().child(item)),
        );
    }
    row.text_xs().text_color(muted)
}

/// Persian label for the sidecar's `busy` field.
pub fn busy_label(busy: &str) -> String {
    match busy {
        "training" => "در حال آموزش".into(),
        "tts" | "synthesizing" => "در حال ساخت صدا".into(),
        "stt" | "transcribing" => "در حال تشخیص گفتار".into(),
        "loading" => "در حال بارگذاری مدل".into(),
        "importing" => "در حال وارد کردن".into(),
        other => format!("مشغول: {other}"),
    }
}
