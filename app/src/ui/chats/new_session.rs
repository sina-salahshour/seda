//! The "new chat" dialog body: backend (Ollama model or Claude Code working
//! directory), voice profile and auto-speak.

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
    AppContext as _, Context, Entity, IntoElement, ParentElement as _, PathPromptOptions, Render,
    Styled as _, Task, WeakEntity, Window, div,
};

use super::ChatsPage;
use super::util::{Choice, Choices, NO_VOICE, model_choices, voice_choices};
use crate::llm::ollama::OllamaClient;
use crate::sidecar::{SidecarClient, SidecarGlobal};
use crate::store::{AppSettings, Backend, Session};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ollama,
    Claude,
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
    model_select: Entity<SelectState<Choices>>,
    cwd: Option<PathBuf>,
    voices: Loadable,
    voice_select: Entity<SelectState<Choices>>,
    auto_speak: bool,
    _tasks: Vec<Task<()>>,
}

impl NewSessionForm {
    pub fn new(page: WeakEntity<ChatsPage>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let settings = AppSettings::get(cx).clone();
        let model_select =
            cx.new(|cx| SelectState::new(Choices::new(Vec::<Choice>::new()), None, window, cx));
        let voice_select = cx.new(|cx| {
            SelectState::new(
                Choices::new(voice_choices(&[], None)),
                Some(IndexPath::default()),
                window,
                cx,
            )
        });

        let mut form = Self {
            page,
            kind: Kind::Ollama,
            models: Loadable::Loading,
            model_select,
            cwd: None,
            voices: Loadable::Loading,
            voice_select,
            auto_speak: settings.default_auto_speak,
            _tasks: Vec::new(),
        };
        form.load_models(window, cx);
        form.load_voices(window, cx);
        form
    }

    fn load_models(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.models = Loadable::Loading;
        let client = OllamaClient::from_settings(cx);
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { client.models() }).await;
            this.update_in(cx, |this, window, cx| {
                match result {
                    Ok(models) => {
                        let preferred = AppSettings::get(cx).default_ollama_model.clone();
                        let selected = preferred
                            .filter(|model| models.contains(model))
                            .or_else(|| models.first().cloned());
                        this.model_select.update(cx, |select, cx| {
                            select.set_items(
                                Choices::new(model_choices(&models, None)),
                                window,
                                cx,
                            );
                            if let Some(model) = &selected {
                                select.set_selected_value(model, window, cx);
                            }
                        });
                        this.models = Loadable::Ready;
                    }
                    Err(err) => {
                        this.models = Loadable::Failed(format!("{err:#}"));
                    }
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

    fn selected_model(&self, cx: &gpui_kit::App) -> Option<String> {
        self.model_select.read(cx).selected_value().cloned()
    }

    fn backend(&self, cx: &gpui_kit::App) -> Option<Backend> {
        match self.kind {
            Kind::Ollama => self
                .selected_model(cx)
                .map(|model| Backend::Ollama { model }),
            Kind::Claude => self.cwd.clone().map(|cwd| Backend::Claude {
                cwd,
                claude_session_id: None,
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
}

impl Render for NewSessionForm {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        let danger = theme.danger;
        let can_create = self.backend(cx).is_some();

        let backend_field = Field::new().label("پاسخ‌دهنده").child(
            RadioGroup::horizontal("new-session-backend")
                .children(["Ollama", "Claude Code"])
                .selected_index(Some(match self.kind {
                    Kind::Ollama => 0,
                    Kind::Claude => 1,
                }))
                .on_change(cx.listener(|this, ix: &usize, _, cx| {
                    this.kind = if *ix == 0 { Kind::Ollama } else { Kind::Claude };
                    cx.notify();
                })),
        );

        let detail_field = match self.kind {
            Kind::Ollama => Field::new().label("مدل").child(match &self.models {
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
                    .child(
                        div()
                            .text_sm()
                            .text_color(danger)
                            .child("Ollama در دسترس نیست."),
                    )
                    .child(div().text_xs().text_color(muted).ltr().child(err.clone()))
                    .child(
                        Button::new("retry-models")
                            .small()
                            .outline()
                            .icon(IconName::RefreshCw)
                            .label("تلاش دوباره")
                            .on_click(
                                cx.listener(|this, _, window, cx| this.load_models(window, cx)),
                            ),
                    )
                    .into_any_element(),
            }),
            Kind::Claude => Field::new()
                .label("پوشه‌ی کاری")
                .description("Claude Code در این پوشه اجرا می‌شود.")
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
                                    Some(path) => {
                                        this.ltr().truncate().child(path.display().to_string())
                                    }
                                    None => this.text_color(muted).child("هنوز انتخاب نشده"),
                                }),
                        )
                        .child(
                            Button::new("pick-cwd")
                                .outline()
                                .icon(IconName::FolderOpen)
                                .label("انتخاب پوشه…")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.pick_directory(window, cx)
                                })),
                        ),
                ),
        };

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
                    .child(detail_field)
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
