//! The "new chat" dialog body: backend (Ollama, Claude Code, an
//! OpenAI-compatible endpoint or opencode), its model and reasoning level,
//! the working directory for agents, voice profile and auto-speak.

use std::path::PathBuf;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, IndexPath, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    dialog::DialogFooter,
    form::{Field, Form},
    h_flex,
    radio::RadioGroup,
    select::{Select, SelectState},
    spinner::Spinner,
    switch::Switch,
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, App, AppContext as _, Context, Entity, IntoElement, ParentElement as _,
    PathPromptOptions, Render, Styled as _, Task, WeakEntity, Window, div,
};

use super::ChatsPage;
use super::util::{
    Choice, Choices, NO_VOICE, ReasoningScale, claude_model_choices, model_choices, non_default,
    optional_model_choices, reasoning_choices, reasoning_setting, reasoning_value, voice_choices,
};
use crate::llm::ollama::OllamaClient;
use crate::llm::openai::OpenAiClient;
use crate::sidecar::{SidecarClient, SidecarGlobal};
use crate::store::{AppSettings, Backend, Session};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ollama,
    Claude,
    OpenAi,
    OpenCode,
}

impl Kind {
    const ALL: [Kind; 4] = [Kind::Ollama, Kind::Claude, Kind::OpenAi, Kind::OpenCode];

    fn label(self) -> &'static str {
        match self {
            Kind::Ollama => "Ollama",
            Kind::Claude => "Claude Code",
            Kind::OpenAi => "سازگار با OpenAI",
            Kind::OpenCode => "opencode",
        }
    }

    fn scale(self) -> ReasoningScale {
        match self {
            Kind::Ollama => ReasoningScale::Ollama,
            Kind::Claude => ReasoningScale::Claude,
            Kind::OpenAi => ReasoningScale::OpenAi,
            Kind::OpenCode => ReasoningScale::OpenCode,
        }
    }

    fn needs_cwd(self) -> bool {
        matches!(self, Kind::Claude | Kind::OpenCode)
    }
}

enum Loadable {
    Loading,
    Ready,
    Failed(String),
}

pub struct NewSessionForm {
    page: WeakEntity<ChatsPage>,
    kind: Kind,
    models: Loadable,
    /// Bumped on every model load, so a slow earlier load cannot overwrite a newer one.
    models_generation: usize,
    model_select: Entity<SelectState<Choices>>,
    endpoint_select: Entity<SelectState<Choices>>,
    reasoning_select: Entity<SelectState<Choices>>,
    cwd: Option<PathBuf>,
    voices: Loadable,
    voice_select: Entity<SelectState<Choices>>,
    auto_speak: bool,
    _tasks: Vec<Task<()>>,
    _subscriptions: Vec<gpui_kit::Subscription>,
}

impl NewSessionForm {
    pub fn new(page: WeakEntity<ChatsPage>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let settings = AppSettings::get(cx).clone();
        let empty = || Choices::new(Vec::<Choice>::new());
        let model_select = cx.new(|cx| SelectState::new(empty(), None, window, cx));
        let reasoning_select = cx.new(|cx| SelectState::new(empty(), None, window, cx));
        let endpoints: Vec<Choice> = settings
            .endpoints
            .iter()
            .map(|e| Choice::new(e.id.clone(), e.name.clone()))
            .collect();
        let has_endpoints = !endpoints.is_empty();
        let endpoint_select = cx.new(|cx| {
            SelectState::new(
                Choices::new(endpoints),
                has_endpoints.then(IndexPath::default),
                window,
                cx,
            )
        });
        let voice_select = cx.new(|cx| {
            SelectState::new(
                Choices::new(voice_choices(&[], None)),
                Some(IndexPath::default()),
                window,
                cx,
            )
        });
        let subscriptions = vec![cx.subscribe_in(
            &endpoint_select,
            window,
            |this, _, event: &gpui_kit::component::select::SelectEvent<Choices>, window, cx| {
                if matches!(event, gpui_kit::component::select::SelectEvent::Confirm(_)) {
                    this.load_models(window, cx);
                }
            },
        )];

        let mut form = Self {
            page,
            kind: Kind::Ollama,
            models: Loadable::Loading,
            models_generation: 0,
            model_select,
            endpoint_select,
            reasoning_select,
            cwd: None,
            voices: Loadable::Loading,
            voice_select,
            auto_speak: settings.default_auto_speak,
            _tasks: Vec::new(),
            _subscriptions: subscriptions,
        };
        form.set_kind(Kind::Ollama, window, cx);
        form.load_voices(window, cx);
        form
    }

    fn set_kind(&mut self, kind: Kind, window: &mut Window, cx: &mut Context<Self>) {
        self.kind = kind;
        let settings = AppSettings::get(cx);
        let current = match kind {
            Kind::Claude => settings.default_claude_effort.clone(),
            _ => None,
        };
        let scale = kind.scale();
        let value = reasoning_value(scale, current.as_deref());
        self.reasoning_select.update(cx, |select, cx| {
            select.set_items(Choices::new(reasoning_choices(scale, current.as_deref())), window, cx);
            select.set_selected_value(&value, window, cx);
        });
        self.load_models(window, cx);
    }

    fn selected_endpoint(&self, cx: &App) -> Option<crate::store::Endpoint> {
        let id = self.endpoint_select.read(cx).selected_value()?.clone();
        AppSettings::get(cx).endpoint(&id).cloned()
    }

    /// Fill the model select for the current kind (fetching the list when needed).
    fn load_models(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.models_generation += 1;
        let generation = self.models_generation;
        let settings = AppSettings::get(cx).clone();
        type Fetch = Box<dyn FnOnce() -> anyhow::Result<Vec<String>> + Send>;
        let (fetch, preferred, optional): (Fetch, Option<String>, bool) = match self.kind {
            Kind::Claude => {
                let current = settings.default_claude_model.clone();
                let value = current.clone().unwrap_or_default();
                self.model_select.update(cx, |select, cx| {
                    select.set_items(Choices::new(claude_model_choices(current.as_deref())), window, cx);
                    select.set_selected_value(&value, window, cx);
                });
                self.models = Loadable::Ready;
                cx.notify();
                return;
            }
            Kind::Ollama => {
                let client = OllamaClient::from_settings(cx);
                (
                    Box::new(move || client.models()),
                    settings.default_ollama_model.clone(),
                    false,
                )
            }
            Kind::OpenAi => {
                let Some(endpoint) = self.selected_endpoint(cx) else {
                    self.models = Loadable::Failed(
                        "هنوز سرویسی تعریف نشده؛ آن را در تنظیمات اضافه کنید.".into(),
                    );
                    cx.notify();
                    return;
                };
                let client = OpenAiClient::from_endpoint(&endpoint);
                (Box::new(move || client.models()), endpoint.default_model.clone(), false)
            }
            Kind::OpenCode => {
                let bin = settings.opencode_bin.clone();
                (
                    Box::new(move || crate::llm::opencode::models(&bin)),
                    None,
                    true,
                )
            }
        };
        self.models = Loadable::Loading;
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { fetch() }).await;
            this.update_in(cx, |this, window, cx| {
                if this.models_generation != generation {
                    return;
                }
                match result {
                    Ok(models) => {
                        let (choices, selected) = if optional {
                            (optional_model_choices(&models, None), Some(String::new()))
                        } else {
                            // A default model the server does not list is still offered.
                            let selected = preferred
                                .clone()
                                .filter(|m| !m.is_empty())
                                .or_else(|| models.first().cloned());
                            (model_choices(&models, preferred.as_deref()), selected)
                        };
                        this.model_select.update(cx, |select, cx| {
                            select.set_items(Choices::new(choices), window, cx);
                            if let Some(model) = &selected {
                                select.set_selected_value(model, window, cx);
                            }
                        });
                        this.models = Loadable::Ready;
                    }
                    Err(err) => this.models = Loadable::Failed(format!("{err:#}")),
                }
                cx.notify();
            })
            .ok();
        });
        self._tasks.push(task);
        cx.notify();
    }

    fn load_voices(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !SidecarGlobal::get(cx).read(cx).is_online() {
            self.voices = Loadable::Failed("سایدکار آفلاین است".into());
            return;
        }
        self.voices = Loadable::Loading;
        let client = SidecarClient::from_settings(cx);
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { client.profiles() }).await;
            this.update_in(cx, |this, window, cx| {
                match result {
                    Ok(profiles) => {
                        let preferred = AppSettings::get(cx).default_voice_profile.clone();
                        let selected = preferred
                            .filter(|id| profiles.iter().any(|p| &p.id == id))
                            .or_else(|| profiles.first().map(|p| p.id.clone()))
                            .unwrap_or_default();
                        this.voice_select.update(cx, |select, cx| {
                            select.set_items(
                                Choices::new(voice_choices(&profiles, None)),
                                window,
                                cx,
                            );
                            select.set_selected_value(&selected, window, cx);
                        });
                        this.voices = Loadable::Ready;
                    }
                    Err(err) => this.voices = Loadable::Failed(format!("{err:#}")),
                }
                cx.notify();
            })
            .ok();
        });
        self._tasks.push(task);
    }

    fn pick_directory(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("انتخاب".into()),
        });
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = paths.await;
            this.update_in(cx, |this, window, cx| match result {
                Ok(Ok(Some(paths))) => {
                    if let Some(path) = paths.into_iter().next() {
                        this.cwd = Some(path);
                        cx.notify();
                    }
                }
                Ok(Ok(None)) | Err(_) => {}
                Ok(Err(err)) => window.push_notification(
                    gpui_kit::component::notification::Notification::error(format!(
                        "باز کردن انتخابگر پوشه ممکن نشد: {err:#}"
                    )),
                    cx,
                ),
            })
            .ok();
        });
        self._tasks.push(task);
    }

    fn selected_model(&self, cx: &App) -> Option<String> {
        self.model_select.read(cx).selected_value().cloned()
    }

    fn reasoning(&self, cx: &App) -> Option<String> {
        let value = self.reasoning_select.read(cx).selected_value().cloned()?;
        reasoning_setting(self.kind.scale(), &value)
    }

    fn backend(&self, cx: &App) -> Option<Backend> {
        let reasoning = self.reasoning(cx);
        match self.kind {
            Kind::Ollama => self
                .selected_model(cx)
                .filter(|m| !m.is_empty())
                .map(|model| Backend::Ollama {
                    model,
                    think: reasoning,
                }),
            Kind::Claude => self.cwd.clone().map(|cwd| Backend::Claude {
                cwd,
                claude_session_id: None,
                model: self.selected_model(cx).as_deref().and_then(non_default),
                effort: reasoning,
                commands: Vec::new(),
            }),
            Kind::OpenAi => {
                let endpoint = self.selected_endpoint(cx)?;
                let model = self.selected_model(cx).filter(|m| !m.is_empty())?;
                Some(Backend::OpenAi {
                    endpoint: endpoint.id,
                    model,
                    reasoning,
                })
            }
            Kind::OpenCode => self.cwd.clone().map(|cwd| Backend::OpenCode {
                cwd,
                model: self.selected_model(cx).as_deref().and_then(non_default),
                variant: reasoning,
                session_id: None,
            }),
        }
    }

    fn create(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(backend) = self.backend(cx) else {
            return;
        };
        let voice = self
            .voice_select
            .read(cx)
            .selected_value()
            .filter(|id| id.as_str() != NO_VOICE)
            .cloned();
        let session = Session::new(backend, voice, self.auto_speak);
        if let Some(page) = self.page.upgrade() {
            page.update(cx, |page, cx| page.add_session(session, window, cx));
        }
        window.close_dialog(cx);
    }

    fn render_models(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        let danger = theme.danger;
        match &self.models {
            Loadable::Loading => h_flex()
                .gap_2()
                .text_sm()
                .text_color(muted)
                .child(Spinner::new().small())
                .child("در حال گرفتن فهرست مدل‌ها…")
                .into_any_element(),
            Loadable::Ready => Select::new(&self.model_select)
                .placeholder("یک مدل انتخاب کنید")
                .into_any_element(),
            Loadable::Failed(err) => v_flex()
                .gap_2()
                .child(div().text_sm().text_color(danger).child(match self.kind {
                    Kind::Ollama => "Ollama در دسترس نیست.",
                    Kind::OpenAi => "فهرست مدل‌های سرویس گرفته نشد.",
                    Kind::OpenCode => "فهرست مدل‌های opencode گرفته نشد.",
                    Kind::Claude => "",
                }))
                .child(div().text_xs().text_color(muted).ltr().child(err.clone()))
                .child(
                    Button::new("retry-models")
                        .small()
                        .outline()
                        .icon(IconName::RefreshCw)
                        .label("تلاش دوباره")
                        .on_click(cx.listener(|this, _, window, cx| this.load_models(window, cx))),
                )
                .into_any_element(),
        }
    }

    fn render_cwd(&self, cx: &mut Context<Self>) -> Field {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        Field::new()
            .label("پوشه‌ی کاری")
            .description(match self.kind {
                Kind::OpenCode => "opencode در این پوشه اجرا می‌شود.",
                _ => "Claude Code در این پوشه اجرا می‌شود.",
            })
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .px_3()
                            .py_1p5()
                            .rounded(theme.radius)
                            .border_1()
                            .border_color(theme.border)
                            .text_sm()
                            .map(|this| match &self.cwd {
                                Some(path) => this.ltr().truncate().child(path.display().to_string()),
                                None => this.text_color(muted).child("هنوز انتخاب نشده"),
                            }),
                    )
                    .child(
                        Button::new("pick-cwd")
                            .outline()
                            .icon(IconName::FolderOpen)
                            .label("انتخاب پوشه…")
                            .on_click(cx.listener(|this, _, window, cx| this.pick_directory(window, cx))),
                    ),
            )
    }
}

impl Render for NewSessionForm {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let can_create = self.backend(cx).is_some();
        let has_endpoints = !AppSettings::get(cx).endpoints.is_empty();

        let backend_field = Field::new().label("پاسخ‌دهنده").child(
            RadioGroup::horizontal("new-session-backend")
                .children(Kind::ALL.map(Kind::label))
                .selected_index(Kind::ALL.iter().position(|k| *k == self.kind))
                .on_change(cx.listener(|this, ix: &usize, window, cx| {
                    let kind = Kind::ALL.get(*ix).copied().unwrap_or(Kind::Ollama);
                    this.set_kind(kind, window, cx);
                })),
        );

        let endpoint_field = (self.kind == Kind::OpenAi).then(|| {
            Field::new().label("سرویس").child(if has_endpoints {
                Select::new(&self.endpoint_select).into_any_element()
            } else {
                div()
                    .text_sm()
                    .text_color(muted)
                    .child("هنوز سرویسی تعریف نشده؛ در تنظیمات، بخش «سرویس‌های سازگار با OpenAI»، یکی اضافه کنید.")
                    .into_any_element()
            })
        });
        let cwd_field = self.kind.needs_cwd().then(|| self.render_cwd(cx));
        let show_models = !(self.kind == Kind::OpenAi && !has_endpoints);
        let model_field = show_models.then(|| Field::new().label("مدل").child(self.render_models(cx)));
        let reasoning_field = Field::new()
            .label("فکر کردن")
            .description("میزان استدلال پیش از پاسخ؛ بیشتر یعنی کندتر و دقیق‌تر.")
            .child(Select::new(&self.reasoning_select));

        let voice_field = Field::new().label("صدا").child(match &self.voices {
            Loadable::Loading => h_flex()
                .gap_2()
                .text_sm()
                .text_color(muted)
                .child(Spinner::new().small())
                .child("در حال گرفتن صداها…")
                .into_any_element(),
            Loadable::Ready => Select::new(&self.voice_select).into_any_element(),
            Loadable::Failed(_) => div()
                .text_sm()
                .text_color(muted)
                .child("سایدکار در دسترس نیست؛ گفتگو بدون صدا ساخته می‌شود. بعداً می‌توانید صدا را از بالای گفتگو انتخاب کنید.")
                .into_any_element(),
        });

        let auto_speak_field = Field::new().label_indent(false).child(
            Switch::new("new-session-auto-speak")
                .label("پاسخ‌ها خودکار با صدا خوانده شوند")
                .checked(self.auto_speak)
                .on_change(cx.listener(|this, checked: &bool, _, cx| {
                    this.auto_speak = *checked;
                    cx.notify();
                })),
        );

        v_flex()
            .gap_5()
            .child(
                Form::new()
                    .child(backend_field)
                    .when_some(endpoint_field, |this, f| this.child(f))
                    .when_some(cwd_field, |this, f| this.child(f))
                    .when_some(model_field, |this, f| this.child(f))
                    .child(reasoning_field)
                    .child(voice_field)
                    .child(auto_speak_field),
            )
            .child(
                DialogFooter::new()
                    .child(
                        Button::new("new-session-cancel")
                            .outline()
                            .label("انصراف")
                            .on_click(|_, window, cx| window.close_dialog(cx)),
                    )
                    .child(
                        Button::new("new-session-create")
                            .primary()
                            .label("شروع گفتگو")
                            .disabled(!can_create)
                            .on_click(cx.listener(|this, _, window, cx| this.create(window, cx))),
                    ),
            )
    }
}
