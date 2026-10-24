//! "Try it": synthesize a text with two run choices (A/B) and play it.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Sizable as _, WindowExt as _,
    button::Button,
    h_flex,
    input::Textarea,
    notification::Notification,
    select::{Select, SelectItem},
    v_flex,
};
use gpui_kit::{
    Context, FontWeight, IntoElement, ParentElement as _, SharedString, Styled as _, Window, div,
};

use super::{VoicesPage, notify_error, player, section};
use crate::sidecar::{Profile, RunStatus, TtsRequest, TtsRun};
use crate::store::AppSettings;

/// Prefilled text for "Try it".
pub(super) const SAMPLE_TEXT: &str =
    "سلام! امروز هوا خیلی خوب است. می‌خواهی با هم کمی قدم بزنیم و درباره‌ی کتاب تازه‌ات حرف بزنیم؟";

const KEY_ZERO: &str = "zero";
const KEY_ACTIVE: &str = "active";

/// One choice in the A/B run pickers. `key` is `zero`, `active` or a run id.
#[derive(Clone)]
pub(super) struct RunChoice {
    key: SharedString,
    title: SharedString,
}

impl SelectItem for RunChoice {
    type Value = SharedString;

    fn title(&self) -> SharedString {
        self.title.clone()
    }

    fn value(&self) -> &Self::Value {
        &self.key
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Slot {
    A,
    B,
}

fn tts_run(key: &str) -> TtsRun {
    match key {
        KEY_ZERO => TtsRun::ZeroShot,
        KEY_ACTIVE => TtsRun::Active,
        id => TtsRun::Run(id.to_string()),
    }
}

impl VoicesPage {
    fn run_choices(&self) -> Vec<RunChoice> {
        let mut choices = vec![RunChoice {
            key: KEY_ZERO.into(),
            title: "بدون آموزش (zero-shot)".into(),
        }];
        if let Some(active) = self.selected_profile().and_then(|p| p.active_run.clone()) {
            choices.push(RunChoice {
                key: KEY_ACTIVE.into(),
                title: format!("مدل فعال ({})", self.run_label(&active)).into(),
            });
        }
        choices.extend(
            self.runs
                .iter()
                .filter(|run| run.status == RunStatus::Done)
                .map(|run| RunChoice {
                    key: run.id.clone().into(),
                    title: self.run_label(&run.id).into(),
                }),
        );
        choices
    }

    /// Refill both pickers, keeping their choice when it still exists.
    pub(super) fn rebuild_run_choices(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let choices = self.run_choices();
        let has_active = choices.iter().any(|c| c.key == KEY_ACTIVE);
        for (state, default) in [
            (self.run_a.clone(), KEY_ZERO),
            (
                self.run_b.clone(),
                if has_active { KEY_ACTIVE } else { KEY_ZERO },
            ),
        ] {
            state.update(cx, |select, cx| {
                select.set_items(choices.clone(), window, cx);
                if select.selected_value().is_none() {
                    select.set_selected_value(&SharedString::from(default), window, cx);
                }
            });
        }
    }

    fn speak(&mut self, slot: Slot, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let text = self.try_text.read(cx).value().trim().to_string();
        if text.is_empty() {
            window.push_notification(Notification::error("اول متنی بنویس."), cx);
            return;
        }
        let Some(player) = player(cx) else {
            window.push_notification(Notification::error("خروجی صدایی پیدا نشد."), cx);
            return;
        };
        let state = match slot {
            Slot::A => &self.run_a,
            Slot::B => &self.run_b,
        };
        let key = state
            .read(cx)
            .selected_value()
            .cloned()
            .unwrap_or_else(|| KEY_ACTIVE.into());
        let request = TtsRequest {
            profile_id: pid,
            text,
            engine: None,
            run: tts_run(&key),
            speed: Some(AppSettings::get(cx).tts_speed),
        };
        self.speaking = Some(slot);
        cx.notify();
        self._speak = Some(self.spawn_job(
            window,
            cx,
            move |client| client.tts(&request),
            move |this, result, window, cx| {
                this.speaking = None;
                match result {
                    Ok(wav) => {
                        // Playing replaces whatever sample was playing.
                        this.playing = None;
                        this._playback_watch = None;
                        player.play(wav);
                    }
                    Err(err) => notify_error(window, cx, "صدا ساخته نشد", &err),
                }
            },
        ));
    }

    fn render_variant(&self, slot: Slot, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let (label, state, id) = match slot {
            Slot::A => ("الف", &self.run_a, "speak-a"),
            Slot::B => ("ب", &self.run_b, "speak-b"),
        };
        let busy = self.speaking.is_some();
        h_flex()
            .flex_1()
            .min_w_0()
            .gap_2()
            .child(
                div()
                    .w_6()
                    .flex_shrink_0()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(cx.theme().muted_foreground)
                    .child(label),
            )
            .child(
                div().flex_1().min_w_0().child(
                    Select::new(state)
                        .small()
                        .accessibility_label(format!("مدل {label}")),
                ),
            )
            .child(
                Button::new(id)
                    .small()
                    .icon(IconName::Volume2)
                    .label(format!("بخوان {label}"))
                    .loading(self.speaking == Some(slot))
                    .disabled(busy && self.speaking != Some(slot))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        if this.speaking.is_none() {
                            this.speak(slot, window, cx)
                        }
                    })),
            )
    }

    pub(super) fn render_try_section(
        &self,
        _profile: &Profile,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let content = v_flex()
            .w_full()
            .gap_3()
            .child(Textarea::new(&self.try_text))
            .child(
                h_flex()
                    .w_full()
                    .gap_4()
                    .flex_wrap()
                    .child(self.render_variant(Slot::A, cx))
                    .child(self.render_variant(Slot::B, cx)),
            );
        section(
            false,
            "امتحان کن",
            Some("یک متن را با دو مدل بشنو و مقایسه کن.".into()),
            None,
            content,
            cx,
        )
    }
}
