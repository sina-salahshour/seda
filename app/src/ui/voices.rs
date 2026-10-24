//! Voices page: a master–detail view of voice profiles.
//!
//! The leading column lists the profiles; the detail pane (the surplus width)
//! shows the selected profile in sections: samples, recording/import,
//! training runs, and "Try it". All sidecar calls run on a worker via
//! [`VoicesPage::spawn_job`]; results that arrive after the selection moved on
//! are dropped.
//!
//! The sections live in submodules (`samples`, `training`, `try_it`) as extra
//! `impl VoicesPage` blocks, so the page keeps one state owner.

mod samples;
mod training;
mod try_it;

use std::cell::Cell;

use anyhow::Result;
use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Icon, IndexPath, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariant, ButtonVariants as _},
    dialog::{DialogAction, DialogClose, DialogFooter},
    form::{field, v_form},
    h_flex,
    input::{InputState, TextareaState},
    list::ListItem,
    menu::{DropdownMenu as _, PopupMenuItem},
    notification::Notification,
    scroll::ScrollableElement as _,
    select::{Select, SelectEvent, SelectItem, SelectState},
    speech::{SpeechEvent, SpeechState},
    tag::Tag,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, App, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _,
    IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _,
    Styled as _, Subscription, Task, Window, div, px,
};

use crate::rtl;
use crate::sidecar::{
    EngineInfo, Profile, ProfilePatch, Run, Sample, Sidecar, SidecarClient, SidecarGlobal,
    engine_defaults, engine_list,
};
use crate::ui::common::{empty_state, fa_digits, format_duration, sidecar_offline};

use samples::ImportJob;
use try_it::{RunChoice, Slot};

/// One engine in the engine pickers.
#[derive(Clone)]
struct EngineItem {
    id: String,
    label: SharedString,
}

impl SelectItem for EngineItem {
    type Value = String;

    fn title(&self) -> SharedString {
        self.label.clone()
    }

    fn value(&self) -> &Self::Value {
        &self.id
    }
}

/// Picker entries: the engine name plus non-commercial / not-ready markers.
fn engine_items(engines: &[EngineInfo]) -> Vec<EngineItem> {
    engines
        .iter()
        .map(|engine| {
            let mut label = engine.name.clone();
            if !engine.is_commercial() {
                label.push_str(" · غیرتجاری");
            }
            if !engine.is_ready() {
                label.push_str(" · مدل دانلود نشده");
            }
            EngineItem {
                id: engine.id.clone(),
                label: label.into(),
            }
        })
        .collect()
}

/// The app-wide audio player, if an output device was found.
fn player(cx: &App) -> Option<crate::audio::Player> {
    cx.try_global::<crate::audio::Player>().cloned()
}

/// RFC 3339 → local `YYYY-MM-DD HH:MM`; the input itself when it doesn't parse.
fn format_time(timestamp: &str) -> String {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|t| {
            t.with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_else(|_| timestamp.to_string())
}

/// Seconds as a short Persian length, e.g. `۴٫۲ ثانیه` or `۲:۳۴`.
///
/// Starts with an RLM so it can lead a line: the vendored bidi drops a
/// leading number run otherwise.
fn format_length(seconds: f64) -> String {
    if seconds < 60. {
        format!("\u{200F}{} ثانیه", fa_decimal(seconds, 1))
    } else {
        format!("\u{200F}{}", fa_digits(&format_duration(seconds)))
    }
}

/// A decimal in Persian digits, e.g. `۰٫۵۱۷`.
fn fa_decimal(value: f64, decimals: usize) -> String {
    fa_digits(&format!("{value:.decimals$}").replace('.', "٫"))
}

/// One metadata item: a single text element that does not wrap inside.
fn meta_text(text: impl Into<SharedString>) -> AnyElement {
    div()
        .whitespace_nowrap()
        .child(text.into())
        .into_any_element()
}

/// A metadata line: separate items in a wrapping row with small muted dots
/// between them, so mixed-direction items never share one bidi run.
fn meta_row(items: impl IntoIterator<Item = AnyElement>, cx: &App) -> gpui_kit::Div {
    let mut row = h_flex()
        .flex_wrap()
        .gap_x_1p5()
        .gap_y_0p5()
        .text_xs()
        .text_color(cx.theme().muted_foreground);
    for (ix, item) in items.into_iter().enumerate() {
        if ix > 0 {
            row = row.child(div().child("·"));
        }
        row = row.child(item);
    }
    row
}

fn notify_error(window: &mut Window, cx: &mut App, what: &str, err: &anyhow::Error) {
    window.push_notification(Notification::error(format!("{what}. {err:#}")), cx);
}

pub struct VoicesPage {
    sidecar: Entity<Sidecar>,
    /// Whether the sidecar was online at the last check.
    online: bool,

    profiles: Vec<Profile>,
    profiles_loading: bool,
    profiles_loaded: bool,
    selected: Option<String>,

    // Detail of the selected profile.
    samples: Vec<Sample>,
    samples_loading: bool,
    runs: Vec<Run>,
    /// Engines from `/health` (or the built-in list while offline).
    engines: Vec<EngineInfo>,
    engine_select: Entity<SelectState<Vec<EngineItem>>>,

    // Samples and recording.
    prompts: Vec<String>,
    prompt_ix: usize,
    prompts_loading: bool,
    speech: Entity<SpeechState>,
    /// The profile the current recording belongs to.
    recording_for: Option<String>,
    /// Profiles with a recorded take being uploaded (one entry per take).
    uploading: Vec<String>,
    imports: Vec<ImportJob>,
    next_import_id: usize,
    /// Sample being fetched for playback.
    fetching_audio: Option<String>,
    /// Sample being played.
    playing: Option<String>,
    _audio_fetch: Option<Task<()>>,
    _playback_watch: Option<Task<()>>,

    // Training.
    steps: Entity<InputState>,
    train_pending: bool,
    polling: bool,
    _poll: Option<Task<()>>,

    // Try it.
    try_text: Entity<TextareaState>,
    run_a: Entity<SelectState<Vec<RunChoice>>>,
    run_b: Entity<SelectState<Vec<RunChoice>>>,
    speaking: Option<Slot>,
    _speak: Option<Task<()>>,

    _subscriptions: Vec<Subscription>,
}

impl VoicesPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let sidecar = SidecarGlobal::get(cx);
        let online = sidecar.read(cx).is_online();

        let engines = engine_list(sidecar.read(cx).health());
        let engine_select = cx.new(|cx| {
            SelectState::new(
                engine_items(&engines),
                Some(IndexPath::default()),
                window,
                cx,
            )
        });
        let speech = cx.new(crate::audio::capture_speech_state);
        let steps = cx.new(|cx| InputState::new(window, cx).placeholder("پیش‌فرض"));
        let try_text = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(3, 8)
                .placeholder("متنی فارسی بنویس تا با این صدا خوانده شود")
                .default_value(try_it::SAMPLE_TEXT)
        });
        let run_a = cx.new(|cx| SelectState::new(Vec::<RunChoice>::new(), None, window, cx));
        let run_b = cx.new(|cx| SelectState::new(Vec::<RunChoice>::new(), None, window, cx));

        let subscriptions = vec![
            cx.observe_in(&sidecar, window, |this: &mut Self, sidecar, window, cx| {
                let online = sidecar.read(cx).is_online();
                if online && !this.online {
                    this.reload_profiles(window, cx);
                }
                this.online = online;
                let engines = engine_list(sidecar.read(cx).health());
                if engines != this.engines {
                    this.engines = engines;
                    let items = engine_items(&this.engines);
                    this.engine_select
                        .update(cx, |select, cx| select.set_items(items, window, cx));
                    this.sync_profile_controls(window, cx);
                }
                cx.notify();
            }),
            cx.subscribe_in(
                &engine_select,
                window,
                |this: &mut Self, _, event: &SelectEvent<Vec<EngineItem>>, window, cx| {
                    let SelectEvent::Confirm(Some(engine)) = event else {
                        return;
                    };
                    let unchanged = this
                        .selected_profile()
                        .is_none_or(|profile| &profile.engine == engine);
                    if !unchanged {
                        this.patch_profile(
                            ProfilePatch {
                                engine: Some(engine.clone()),
                                ..Default::default()
                            },
                            window,
                            cx,
                        );
                    }
                },
            ),
            cx.subscribe_in(
                &speech,
                window,
                |this: &mut Self, _, event: &SpeechEvent, window, cx| {
                    this.on_speech_event(event, window, cx)
                },
            ),
        ];

        let mut this = Self {
            sidecar,
            online,
            profiles: Vec::new(),
            profiles_loading: false,
            profiles_loaded: false,
            selected: None,
            samples: Vec::new(),
            samples_loading: false,
            runs: Vec::new(),
            engines,
            engine_select,
            prompts: Vec::new(),
            prompt_ix: 0,
            prompts_loading: false,
            speech,
            recording_for: None,
            uploading: Vec::new(),
            imports: Vec::new(),
            next_import_id: 0,
            fetching_audio: None,
            playing: None,
            _audio_fetch: None,
            _playback_watch: None,
            steps,
            train_pending: false,
            polling: false,
            _poll: None,
            try_text,
            run_a,
            run_b,
            speaking: None,
            _speak: None,
            _subscriptions: subscriptions,
        };
        if online {
            this.reload_profiles(window, cx);
        }
        this
    }

    /// Run `job` with a sidecar client on a worker, then `done` on the UI
    /// thread. Dropping the returned task cancels the UI half.
    fn spawn_job<T: Send + 'static>(
        &self,
        window: &Window,
        cx: &mut Context<Self>,
        job: impl FnOnce(&SidecarClient) -> Result<T> + Send + 'static,
        done: impl FnOnce(&mut Self, Result<T>, &mut Window, &mut Context<Self>) + 'static,
    ) -> Task<()> {
        let client = SidecarClient::from_settings(cx);
        cx.spawn_in(window, async move |this, cx| {
            let result = cx
                .background_executor()
                .spawn(async move { job(&client) })
                .await;
            this.update_in(cx, |this, window, cx| {
                done(this, result, window, cx);
                cx.notify();
            })
            .ok();
        })
    }

    fn engine(&self, id: &str) -> Option<&EngineInfo> {
        self.engines.iter().find(|engine| engine.id == id)
    }

    /// Short engine name for metadata lines.
    fn engine_short(&self, id: &str) -> String {
        match self.engine(id) {
            Some(engine) => engine.short_name().to_string(),
            None => engine_defaults(id)
                .map(|d| d.short.to_string())
                .unwrap_or_else(|| id.to_string()),
        }
    }

    /// Whether the engine fine-tunes (LoRA); otherwise "training" picks the
    /// best reference clip. Unknown engines count as trainable.
    fn engine_trainable(&self, id: &str) -> bool {
        match self.engine(id) {
            Some(engine) => engine.is_trainable(),
            None => engine_defaults(id).is_none_or(|d| d.trainable),
        }
    }

    fn engine_ready(&self, id: &str) -> bool {
        self.engine(id).is_none_or(|engine| engine.is_ready())
    }

    fn selected_profile(&self) -> Option<&Profile> {
        let id = self.selected.as_deref()?;
        self.profiles.iter().find(|profile| profile.id == id)
    }

    fn is_selected(&self, pid: &str) -> bool {
        self.selected.as_deref() == Some(pid)
    }

    /// What the sidecar is busy with (e.g. training), if anything.
    fn sidecar_busy(&self, cx: &App) -> Option<String> {
        self.sidecar
            .read(cx)
            .health()
            .and_then(|health| health.busy.clone())
    }

    fn reload_profiles(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.profiles_loading = true;
        cx.notify();
        self.spawn_job(
            window,
            cx,
            |client| client.profiles(),
            |this, result, window, cx| {
                this.profiles_loading = false;
                match result {
                    Ok(profiles) => {
                        this.profiles = profiles;
                        this.profiles_loaded = true;
                        let keep = this
                            .selected
                            .clone()
                            .filter(|id| this.profiles.iter().any(|p| &p.id == id));
                        match keep {
                            Some(_) => {
                                this.sync_profile_controls(window, cx);
                                this.load_detail(window, cx);
                            }
                            None => {
                                this.selected = None;
                                if let Some(first) = this.profiles.first().map(|p| p.id.clone()) {
                                    this.select_profile(first, window, cx);
                                } else {
                                    this.clear_detail();
                                }
                            }
                        }
                    }
                    Err(err) => notify_error(window, cx, "فهرست صداها بارگذاری نشد", &err),
                }
            },
        )
        .detach();
    }

    fn clear_detail(&mut self) {
        self.samples.clear();
        self.runs.clear();
        self.polling = false;
        self._poll = None;
    }

    fn select_profile(&mut self, pid: String, window: &mut Window, cx: &mut Context<Self>) {
        if self.is_selected(&pid) {
            return;
        }
        self.selected = Some(pid);
        self.clear_detail();
        self.sync_profile_controls(window, cx);
        self.load_detail(window, cx);
        cx.notify();
    }

    /// Load samples and runs of the selected profile (and prompts once).
    fn load_detail(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        self.samples_loading = true;
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| Ok((client.samples(&job_pid)?, client.runs(&job_pid)?)),
            move |this, result, window, cx| {
                if !this.is_selected(&pid) {
                    return;
                }
                this.samples_loading = false;
                match result {
                    Ok((samples, runs)) => {
                        this.samples = samples;
                        this.runs = runs;
                        this.rebuild_run_choices(window, cx);
                        this.ensure_polling(window, cx);
                    }
                    Err(err) => notify_error(window, cx, "جزئیات صدا بارگذاری نشد", &err),
                }
            },
        )
        .detach();
        if self.prompts.is_empty() {
            self.load_prompts(window, cx);
        }
    }

    /// Fetch one profile (counts, active run) and its samples again.
    fn refresh_profile(&mut self, pid: String, window: &mut Window, cx: &mut Context<Self>) {
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| Ok((client.profile(&job_pid)?, client.samples(&job_pid)?)),
            move |this, result, window, cx| match result {
                Ok((profile, samples)) => {
                    if this.is_selected(&pid) {
                        this.samples = samples;
                    }
                    this.apply_profile(profile, window, cx);
                }
                Err(err) => notify_error(window, cx, "به‌روزرسانی صدا ناموفق بود", &err),
            },
        )
        .detach();
    }

    /// Put a fresh copy of a profile into the list.
    fn apply_profile(&mut self, profile: Profile, window: &mut Window, cx: &mut Context<Self>) {
        let selected = self.is_selected(&profile.id);
        match self.profiles.iter_mut().find(|p| p.id == profile.id) {
            Some(slot) => *slot = profile,
            None => self.profiles.push(profile),
        }
        if selected {
            self.sync_profile_controls(window, cx);
        }
        cx.notify();
    }

    /// Make the engine picker and run pickers match the selected profile.
    fn sync_profile_controls(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(engine) = self
            .selected_profile()
            .map(|p| p.engine.clone())
            .filter(|engine| self.engine(engine).is_some())
        {
            self.engine_select.update(cx, |select, cx| {
                if select.selected_value() != Some(&engine) {
                    select.set_selected_value(&engine, window, cx);
                }
            });
        }
        self.rebuild_run_choices(window, cx);
    }

    fn patch_profile(&mut self, patch: ProfilePatch, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        self.spawn_job(
            window,
            cx,
            move |client| client.update_profile(&pid, &patch),
            |this, result, window, cx| match result {
                Ok(profile) => this.apply_profile(profile, window, cx),
                Err(err) => {
                    notify_error(window, cx, "تغییر ذخیره نشد", &err);
                    // Put the controls back to what the profile says.
                    this.sync_profile_controls(window, cx);
                }
            },
        )
        .detach();
    }

    fn create_profile(
        &mut self,
        name: String,
        engine: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.spawn_job(
            window,
            cx,
            move |client| client.create_profile(&name, Some(&engine)),
            |this, result, window, cx| match result {
                Ok(profile) => {
                    let id = profile.id.clone();
                    this.profiles.push(profile);
                    this.select_profile(id, window, cx);
                }
                Err(err) => notify_error(window, cx, "صدای تازه ساخته نشد", &err),
            },
        )
        .detach();
    }

    fn delete_profile(&mut self, pid: String, window: &mut Window, cx: &mut Context<Self>) {
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.delete_profile(&job_pid),
            move |this, result, window, cx| match result {
                Ok(()) => {
                    let ix = this.profiles.iter().position(|p| p.id == pid);
                    this.profiles.retain(|p| p.id != pid);
                    if this.is_selected(&pid) {
                        this.selected = None;
                        this.clear_detail();
                        let next = ix
                            .and_then(|ix| {
                                this.profiles
                                    .get(ix.min(this.profiles.len().saturating_sub(1)))
                            })
                            .map(|p| p.id.clone());
                        if let Some(next) = next {
                            this.select_profile(next, window, cx);
                        }
                    }
                }
                Err(err) => notify_error(window, cx, "صدا حذف نشد", &err),
            },
        )
        .detach();
    }

    fn open_new_profile_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let name = cx.new(|cx| InputState::new(window, cx).placeholder("مثلاً صدای من"));
        let items = engine_items(&self.engines);
        let engine = cx.new(|cx| SelectState::new(items, Some(IndexPath::default()), window, cx));
        let view = cx.entity().downgrade();
        let initial_focus = Cell::new(true);
        window.open_dialog(cx, move |dialog, window, cx| {
            if initial_focus.replace(false) {
                name.update(cx, |state, cx| state.focus(window, cx));
            }
            let (name_ok, engine_ok, view) = (name.clone(), engine.clone(), view.clone());
            dialog
                .title("صدای تازه")
                .w(px(440.))
                .child(
                    v_form()
                        .child(field().label("نام").child(rtl::input(&name, cx)))
                        .child(
                            field()
                                .label("موتور")
                                .description("بعداً هم می‌توانی موتور را عوض کنی.")
                                .child(Select::new(&engine)),
                        ),
                )
                .on_ok(move |_, window, cx| {
                    let name = name_ok.read(cx).value().trim().to_string();
                    if name.is_empty() {
                        return false;
                    }
                    let engine = engine_ok
                        .read(cx)
                        .selected_value()
                        .cloned()
                        .unwrap_or_else(|| "omnivoice".to_string());
                    view.update(cx, |this, cx| this.create_profile(name, engine, window, cx))
                        .ok();
                    true
                })
                .footer(
                    DialogFooter::new()
                        .child(DialogClose::new().trigger(|button| button.label("انصراف")))
                        .child(
                            DialogAction::new()
                                .child(Button::new("create-voice").primary().label("ساختن")),
                        ),
                )
        });
    }

    fn open_rename_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(profile) = self.selected_profile() else {
            return;
        };
        let current = profile.name.clone();
        let name = cx.new(|cx| InputState::new(window, cx).default_value(current.clone()));
        let view = cx.entity().downgrade();
        let initial_focus = Cell::new(true);
        window.open_dialog(cx, move |dialog, window, cx| {
            if initial_focus.replace(false) {
                name.update(cx, |state, cx| state.focus(window, cx));
            }
            let (name_ok, view, current) = (name.clone(), view.clone(), current.clone());
            dialog
                .title("تغییر نام صدا")
                .w(px(420.))
                .child(v_form().child(field().label("نام").child(rtl::input(&name, cx))))
                .on_ok(move |_, window, cx| {
                    let name = name_ok.read(cx).value().trim().to_string();
                    if name.is_empty() {
                        return false;
                    }
                    if name != current {
                        view.update(cx, |this, cx| {
                            this.patch_profile(
                                ProfilePatch {
                                    name: Some(name),
                                    ..Default::default()
                                },
                                window,
                                cx,
                            )
                        })
                        .ok();
                    }
                    true
                })
                .footer(
                    DialogFooter::new()
                        .child(DialogClose::new().trigger(|button| button.label("انصراف")))
                        .child(
                            DialogAction::new()
                                .child(Button::new("rename-voice").primary().label("ذخیره")),
                        ),
                )
        });
    }

    fn confirm_delete_profile(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(profile) = self.selected_profile() else {
            return;
        };
        let (pid, name) = (profile.id.clone(), profile.name.clone());
        let view = cx.entity().downgrade();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let (pid, view) = (pid.clone(), view.clone());
            alert
                .title(SharedString::from(format!("«{name}» حذف شود؟")))
                .description("همهٔ نمونه‌ها و اجراهای آموزش این صدا برای همیشه پاک می‌شوند.")
                .confirm()
                .ok_text("حذف")
                .ok_variant(ButtonVariant::Danger)
                .cancel_text("انصراف")
                .on_ok(move |_, window, cx| {
                    let pid = pid.clone();
                    view.update(cx, |this, cx| this.delete_profile(pid, window, cx))
                        .ok();
                    true
                })
        });
    }

    // MARK: Render

    fn render_profile_list(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let list: AnyElement = if self.profiles.is_empty() {
            div()
                .px_4()
                .py_6()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(if self.profiles_loading {
                    "در حال بارگذاری…"
                } else {
                    "هنوز صدایی نساخته‌ای."
                })
                .into_any_element()
        } else {
            v_flex()
                .id("voice-profiles")
                .flex_1()
                .min_h_0()
                .px_2()
                .py_2()
                .gap_1()
                .overflow_y_scrollbar()
                .children(self.profiles.iter().map(|profile| {
                    let id = profile.id.clone();
                    let selected = self.is_selected(&profile.id);
                    let meta = meta_row(
                        [
                            meta_text(format!(
                                "\u{200F}{} نمونه",
                                fa_digits(&profile.sample_count.to_string())
                            )),
                            meta_text(format_length(profile.total_seconds)),
                            meta_text(self.engine_short(&profile.engine)),
                        ],
                        cx,
                    );
                    ListItem::new(SharedString::from(format!("voice-{}", profile.id)))
                        .selected(selected)
                        .rounded(theme.radius)
                        .py_2()
                        .accessibility_label(profile.name.clone())
                        .child(
                            h_flex()
                                .w_full()
                                .gap_2()
                                .child(
                                    v_flex()
                                        .flex_1()
                                        .min_w_0()
                                        .gap_0p5()
                                        .child(
                                            div()
                                                .text_sm()
                                                .font_weight(FontWeight::MEDIUM)
                                                .truncate()
                                                .child(profile.name.clone()),
                                        )
                                        .child(meta),
                                )
                                .child(
                                    // Fixed lane, so names line up with or without the marker.
                                    div().size_4().flex_shrink_0().when(
                                        profile.active_run.is_some(),
                                        |this| {
                                            this.child(
                                                Icon::new(IconName::GraduationCap)
                                                    .small()
                                                    .text_color(theme.muted_foreground),
                                            )
                                        },
                                    ),
                                ),
                        )
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.select_profile(id.clone(), window, cx)
                        }))
                }))
                .into_any_element()
        };

        let rtl = rtl::is_rtl(cx);
        let theme = cx.theme();
        v_flex()
            .w_64()
            .flex_shrink_0()
            .h_full()
            .map(|this| {
                if rtl {
                    this.border_l_1()
                } else {
                    this.border_r_1()
                }
            })
            .border_color(theme.border)
            .child(
                h_flex()
                    .px_4()
                    .py_3()
                    .gap_2()
                    .border_b_1()
                    .border_color(theme.border)
                    .child(
                        div()
                            .flex_1()
                            .text_lg()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("صداها"),
                    )
                    .child(
                        Button::new("new-voice")
                            .small()
                            .icon(IconName::Plus)
                            .label("صدای تازه…")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.open_new_profile_dialog(window, cx)
                            })),
                    ),
            )
            .child(list)
    }

    /// «آموزش N»: runs are numbered per profile, oldest = 1.
    fn run_label(&self, rid: &str) -> String {
        match self.runs.iter().position(|run| run.id == rid) {
            Some(ix) => format!("آموزش {}", fa_digits(&(self.runs.len() - ix).to_string())),
            None => "آموزش".to_string(),
        }
    }

    fn render_header(&self, profile: &Profile, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let active = profile
            .active_run
            .as_deref()
            .and_then(|id| self.runs.iter().find(|run| run.id == id));
        let model: SharedString = match (&profile.active_run, active) {
            (None, _) => "بدون آموزش".into(),
            (Some(_), Some(run)) => format!(
                "مدل فعال: {} · {} گام",
                self.run_label(&run.id),
                fa_digits(&run.total_steps.max(run.step).to_string())
            )
            .into(),
            (Some(_), None) => "مدل فعال: آموزش".into(),
        };
        let model_tip: SharedString = match (&profile.active_run, active) {
            (None, _) => "صدا فقط از نمونهٔ مرجع ساخته می‌شود (zero-shot).".into(),
            (Some(id), run) => {
                let mut tip = format!("شناسه: {id}");
                if let Some(loss) = run.and_then(|r| r.loss) {
                    tip.push_str(&format!(" · خطا {}", fa_decimal(loss, 3)));
                }
                tip.into()
            }
        };
        let meta = meta_row(
            [
                meta_text(format!(
                    "\u{200F}{} نمونه",
                    fa_digits(&profile.sample_count.to_string())
                )),
                meta_text(format_length(profile.total_seconds)),
                div()
                    .id("active-model")
                    .tooltip(move |window, cx| Tooltip::new(model_tip.clone()).build(window, cx))
                    .child(Tag::secondary().xsmall().child(model))
                    .into_any_element(),
            ],
            cx,
        )
        .text_sm();

        let theme = cx.theme();
        v_flex()
            .w_full()
            .px_6()
            .py_4()
            .gap_3()
            .border_b_1()
            .border_color(theme.border)
            .child(
                h_flex()
                    .w_full()
                    .gap_1()
                    .child(
                        div()
                            .min_w_0()
                            .text_xl()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(profile.name.clone()),
                    )
                    .child(
                        Button::new("rename-voice")
                            .ghost()
                            .xsmall()
                            .icon(IconName::Pencil)
                            .tooltip("تغییر نام…")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.open_rename_dialog(window, cx)
                            })),
                    )
                    .child(div().flex_1())
                    .child({
                        let view = cx.entity().downgrade();
                        Button::new("voice-menu")
                            .ghost()
                            .small()
                            .icon(IconName::Ellipsis)
                            .accessibility_label("کارهای بیشتر")
                            .dropdown_menu(move |menu, _, _| {
                                let (rename, delete) = (view.clone(), view.clone());
                                menu.item(
                                    PopupMenuItem::new("تغییر نام…")
                                        .icon(IconName::Pencil)
                                        .on_click(move |_, window, cx| {
                                            rename
                                                .update(cx, |this, cx| {
                                                    this.open_rename_dialog(window, cx)
                                                })
                                                .ok();
                                        }),
                                )
                                .separator()
                                .item(
                                    PopupMenuItem::new("حذف صدا…")
                                        .icon(IconName::Trash)
                                        .on_click(move |_, window, cx| {
                                            delete
                                                .update(cx, |this, cx| {
                                                    this.confirm_delete_profile(window, cx)
                                                })
                                                .ok();
                                        }),
                                )
                            })
                    }),
            )
            .child(meta)
            .child(self.render_engine_row(profile))
    }

    /// Engine picker, with a non-commercial note and, when the engine's model
    /// is not downloaded, a link to the Models page.
    fn render_engine_row(&self, profile: &Profile) -> impl IntoElement + use<> {
        let engine = self.engine(&profile.engine).cloned();
        let not_ready = !self.engine_ready(&profile.engine);
        let non_commercial = engine.as_ref().is_some_and(|e| !e.is_commercial());
        let license: SharedString = format!(
            "مجوز {}: فقط استفادهٔ غیرتجاری.",
            engine
                .as_ref()
                .and_then(|e| e.license().map(str::to_string))
                .unwrap_or_default()
        )
        .into();
        h_flex()
            .flex_wrap()
            .gap_3()
            .text_sm()
            .child("موتور صدا")
            .child(
                div().w(px(280.)).flex_shrink_0().child(
                    Select::new(&self.engine_select)
                        .small()
                        .accessibility_label("موتور صدا"),
                ),
            )
            .when(non_commercial, |this| {
                this.child(
                    div()
                        .id("engine-non-commercial")
                        .tooltip(move |window, cx| Tooltip::new(license.clone()).build(window, cx))
                        .child(Tag::warning().xsmall().child("غیرتجاری")),
                )
            })
            .when(not_ready, |this| {
                this.child(Tag::danger().outline().xsmall().child("مدل دانلود نشده"))
                    .child(
                        Button::new("open-models")
                            .xsmall()
                            .outline()
                            .icon(IconName::Boxes)
                            .label("رفتن به مدل‌ها")
                            .on_click(|_, window, cx| {
                                crate::app::show_page(crate::app::Page::Models, window, cx)
                            }),
                    )
            })
    }

    fn render_detail(&self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let Some(profile) = self.selected_profile().cloned() else {
            let (title, hint) = if self.profiles_loaded && self.profiles.is_empty() {
                (
                    "اولین صدایت را بساز",
                    "یک صدا بساز، چند جمله بخوان یا فایل صوتی وارد کن، و بعد آموزش بده.",
                )
            } else if self.profiles_loading {
                ("در حال بارگذاری…", "")
            } else {
                ("صدایی انتخاب نشده", "از فهرست یک صدا انتخاب کن.")
            };
            return empty_state(
                IconName::AudioWaveform,
                title,
                hint,
                (self.profiles_loaded && self.profiles.is_empty()).then(|| {
                    Button::new("empty-new-voice")
                        .icon(IconName::Plus)
                        .label("صدای تازه…")
                        .on_click(cx.listener(|this, _, window, cx| {
                            this.open_new_profile_dialog(window, cx)
                        }))
                }),
                cx,
            )
            .into_any_element();
        };

        v_flex()
            .size_full()
            .child(self.render_header(&profile, cx))
            .child(
                div().flex_1().min_h_0().child(
                    v_flex()
                        .id("voice-detail")
                        .size_full()
                        .overflow_y_scrollbar()
                        .child(
                            v_flex()
                                .px_6()
                                .pb_8()
                                .child(self.render_samples_section(&profile, cx))
                                .child(self.render_record_section(&profile, window, cx))
                                .child(self.render_training_section(&profile, cx))
                                .child(self.render_try_section(&profile, cx)),
                        ),
                ),
            )
            .into_any_element()
    }
}

/// A detail section: title row with optional trailing actions, then content.
/// Sections are separated by a hairline on top (except the first) rather than
/// by cards.
fn section(
    first: bool,
    title: &'static str,
    description: Option<SharedString>,
    actions: Option<AnyElement>,
    content: impl IntoElement,
    cx: &App,
) -> gpui_kit::Div {
    v_flex()
        .w_full()
        .gap_4()
        .py_6()
        .when(!first, |this| {
            this.border_t_1().border_color(cx.theme().border)
        })
        .child(
            h_flex()
                .flex_wrap()
                .gap_3()
                .child(
                    v_flex()
                        .flex_1()
                        .min_w(px(240.))
                        .gap_1()
                        .child(
                            div()
                                .text_base()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(title),
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
                .when_some(actions, |this, actions| {
                    this.child(h_flex().gap_2().flex_shrink_0().child(actions))
                }),
        )
        .child(content)
}

/// Muted one-line hint used for empty lists inside sections.
fn muted_line(text: impl Into<SharedString>, cx: &App) -> gpui_kit::Div {
    div()
        .text_sm()
        .text_color(cx.theme().muted_foreground)
        .child(text.into())
}

impl Render for VoicesPage {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let online = self.sidecar.read(cx).is_online();
        let body: AnyElement = if online {
            h_flex()
                .size_full()
                .items_stretch()
                .child(self.render_profile_list(cx))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .h_full()
                        .child(self.render_detail(window, cx)),
                )
                .into_any_element()
        } else {
            sidecar_offline(cx).into_any_element()
        };
        div()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(body)
    }
}
