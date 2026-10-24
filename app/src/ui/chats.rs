//! Chats page: sessions, streaming replies, push-to-talk, spoken replies.
//!
//! Layout in reading order: the session list (leading side), then the open
//! conversation with its header, the message list and the composer.
//!
//! One reply streams at a time. Its events arrive on an async channel from
//! `llm::*` and are applied in a task the page holds (`Turn`); dropping the
//! turn cancels it. Spoken output goes through one `SpeechQueue` at a time
//! (`Voice`), either fed by the streaming turn or by a "speak" button.

mod new_session;
mod util;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, Sizable as _, WindowExt as _,
    bubble::{Bubble, BubbleVariant},
    button::{Button, ButtonVariant, ButtonVariants as _},
    clipboard::Clipboard,
    dialog::DialogFooter,
    h_flex,
    input::{InputEvent, InputState, Textarea, TextareaState},
    list::ListItem,
    menu::{DropdownMenu as _, PopupMenuItem},
    message::{Message, MessageAlignment, MessageContent, MessageFooter},
    message_scroller::{MessageScroller, MessageScrollerState},
    notification::Notification,
    select::{Select, SelectEvent, SelectState},
    shimmer::ShimmerText,
    speech::{SpeechButton, SpeechEvent, SpeechState, SpeechWaveform},
    spinner::Spinner,
    switch::Switch,
    text::{TextView, TextViewStyle},
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, App, AppContext as _, Context, ElementId, Entity, FocusHandle, Focusable,
    InteractiveElement as _, IntoElement, KeyBinding, ParentElement as _, Render, ScrollHandle,
    SharedString, StatefulInteractiveElement as _, StyleRefinement, Styled as _, Subscription,
    Task, WeakEntity, Window, actions, div, px,
};

use crate::audio::{Player, SpeechQueue};
use crate::llm::claude::{self, ClaudeRequest};
use crate::llm::ollama::{OllamaClient, OllamaMessage};
use crate::llm::openai::{OpenAiClient, OpenAiMessage};
use crate::llm::opencode::{self, OpenCodeRequest};
use crate::llm::skills::{self, SkillInfo};
use crate::llm::{CancelToken, StreamEvent};
use crate::rtl;
use crate::sidecar::{Profile, SidecarClient, SidecarGlobal, TtsRun};
use crate::store::{
    AppSettings, Backend, ChatMessage, PermissionMode, Role, Session, Settings, TaskInfo,
};
use crate::ui::common::{empty_state, meta_row};

use new_session::NewSessionForm;
use util::{
    Choices, NO_VOICE, ReasoningScale, claude_model_choices, model_choices, non_default,
    one_line, optional_model_choices, reasoning_choices, reasoning_setting, reasoning_value,
    relative_time, voice_choices,
};

actions!(chats, [TogglePushToTalk]);

const CONTEXT: &str = "ChatsPage";
/// How often a streaming reply is written to disk.
const STREAM_SAVE_INTERVAL: Duration = Duration::from_secs(3);

/// Disk writes, applied strictly in order by one background task.
enum SaveOp {
    Save(Box<Session>),
    Delete(String),
}

/// The reply that is streaming right now.
struct Turn {
    session_id: String,
    cancel: CancelToken,
    task: Task<()>,
    /// Per agent (`None` = the main one, else the subagent's tool_use id):
    /// the assistant or thinking message the next delta of the same kind
    /// extends. Cleared by a tool call, so the text after it starts anew.
    segments: HashMap<Option<String>, String>,
    /// Per agent: tool messages still waiting for their result, oldest first.
    tools: HashMap<Option<String>, VecDeque<String>>,
    /// No text yet since the turn (or the last tool call) started.
    waiting: bool,
    after_tool: bool,
    last_save: Instant,
}

impl Turn {
    fn is_streaming(&self, message_id: &str) -> bool {
        self.segments.values().any(|id| id == message_id)
    }
}

/// What applying one stream event changed.
#[derive(Default)]
struct Applied {
    /// Main-agent answer text to speak.
    spoken: Option<String>,
    save: bool,
    /// The message that changed (its row is remeasured).
    touched: Option<String>,
    /// Terminal event: nothing more follows.
    finished: bool,
    error: Option<String>,
    commands_changed: bool,
}

/// The open `/` completion popup.
struct Completion {
    items: Vec<SkillInfo>,
    selected: usize,
    scroll: ScrollHandle,
}

/// Spoken output in progress.
struct Voice {
    queue: SpeechQueue,
    /// No more text will be pushed; drop the voice once playback ends.
    finished: bool,
    /// Set when a message's speak button started it.
    message_id: Option<String>,
    _errors: Task<()>,
}

pub struct ChatsPage {
    focus_handle: FocusHandle,
    sessions: Vec<Session>,
    loaded: bool,
    selected: Option<String>,
    scroller: Entity<MessageScrollerState>,
    composer: Entity<TextareaState>,
    speech: Entity<SpeechState>,
    voice_select: Entity<SelectState<Choices>>,
    model_select: Entity<SelectState<Choices>>,
    reasoning_select: Entity<SelectState<Choices>>,
    profiles: Vec<Profile>,
    /// Model lists by source: `ollama`, `opencode`, `openai:<endpoint id>`.
    models: HashMap<String, Vec<String>>,
    models_loading: HashSet<String>,
    /// Thinking blocks and task nodes whose open/closed state the user flipped.
    toggled: HashSet<String>,
    /// Skills and commands found on disk for a working directory.
    skills: Option<(PathBuf, Vec<SkillInfo>)>,
    completion: Option<Completion>,
    /// The composer text the completion popup was dismissed for (Esc).
    completion_dismissed: Option<String>,
    sidecar_online: bool,
    turn: Option<Turn>,
    voice: Option<Voice>,
    voice_generation: usize,
    saver: async_channel::Sender<SaveOp>,
    _tasks: Vec<Task<()>>,
    _subscriptions: Vec<Subscription>,
}

impl ChatsPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.bind_keys([KeyBinding::new("f2", TogglePushToTalk, Some(CONTEXT))]);

        let scroller = cx.new(|cx| MessageScrollerState::new(0, cx));
        let composer = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(1, 8)
                .submit_on_enter(true)
                .placeholder("پیام خود را بنویسید…")
        });
        let speech = cx.new(crate::audio::stt_speech_state);
        let voice_select =
            cx.new(|cx| SelectState::new(Choices::new(voice_choices(&[], None)), None, window, cx));
        let model_select =
            cx.new(|cx| SelectState::new(Choices::new(model_choices(&[], None)), None, window, cx));
        let reasoning_select =
            cx.new(|cx| SelectState::new(Choices::new(model_choices(&[], None)), None, window, cx));

        let (saver, save_rx) = async_channel::unbounded::<SaveOp>();
        let save_task = cx.background_spawn(async move {
            while let Ok(op) = save_rx.recv().await {
                let result = match op {
                    SaveOp::Save(session) => session.save(),
                    SaveOp::Delete(id) => Session::delete(&id),
                };
                if let Err(err) = result {
                    eprintln!("seda: saving a chat session failed: {err:#}");
                }
            }
        });

        let sidecar = SidecarGlobal::get(cx);
        let subscriptions = vec![
            cx.subscribe_in(
                &composer,
                window,
                |this, _, event: &InputEvent, window, cx| match event {
                    InputEvent::PressEnter { shift: false, .. } => this.send(window, cx),
                    InputEvent::Change => {
                        this.update_completion(cx);
                        cx.notify()
                    }
                    _ => {}
                },
            ),
            cx.subscribe_in(
                &speech,
                window,
                |this, _, event: &SpeechEvent, window, cx| match event {
                    SpeechEvent::Final(text) => this.on_transcript(text, window, cx),
                    SpeechEvent::Error(err) => window.push_notification(
                        Notification::error(format!("تشخیص گفتار ناموفق بود: {err}")),
                        cx,
                    ),
                    _ => cx.notify(),
                },
            ),
            cx.observe(&speech, |_, _, cx| cx.notify()),
            cx.subscribe_in(
                &voice_select,
                window,
                |this, _, event: &SelectEvent<Choices>, _, cx| {
                    let SelectEvent::Confirm(Some(value)) = event else {
                        return;
                    };
                    let voice = (value.as_str() != NO_VOICE).then(|| value.clone());
                    this.update_current(cx, |session| session.voice_profile = voice);
                },
            ),
            cx.subscribe_in(
                &model_select,
                window,
                |this, _, event: &SelectEvent<Choices>, _, cx| {
                    let SelectEvent::Confirm(Some(value)) = event else {
                        return;
                    };
                    let value = value.clone();
                    this.update_current(cx, |session| match &mut session.backend {
                        Backend::Ollama { model, .. } | Backend::OpenAi { model, .. } => {
                            *model = value
                        }
                        Backend::Claude { model, .. } | Backend::OpenCode { model, .. } => {
                            *model = non_default(&value)
                        }
                        Backend::Unknown(_) => {}
                    });
                },
            ),
            cx.subscribe_in(
                &reasoning_select,
                window,
                |this, _, event: &SelectEvent<Choices>, _, cx| {
                    let SelectEvent::Confirm(Some(value)) = event else {
                        return;
                    };
                    let value = value.clone();
                    this.update_current(cx, |session| {
                        let Some(scale) = reasoning_scale(&session.backend) else {
                            return;
                        };
                        let setting = reasoning_setting(scale, &value);
                        match &mut session.backend {
                            Backend::Ollama { think: r, .. }
                            | Backend::Claude { effort: r, .. }
                            | Backend::OpenAi { reasoning: r, .. }
                            | Backend::OpenCode { variant: r, .. } => *r = setting,
                            Backend::Unknown(_) => {}
                        }
                    });
                },
            ),
            cx.observe_in(&sidecar, window, |this, sidecar, window, cx| {
                let online = sidecar.read(cx).is_online();
                if online && !this.sidecar_online {
                    this.load_profiles(window, cx);
                }
                this.sidecar_online = online;
                cx.notify();
            }),
        ];

        // The `/` popup takes arrows, Tab, Enter and Esc before the textarea does.
        let weak = cx.entity().downgrade();
        let mut subscriptions = subscriptions;
        subscriptions.push(cx.intercept_keystrokes(move |event, window, cx| {
            let keystroke = &event.keystroke;
            let modifiers = keystroke.modifiers;
            if modifiers.control || modifiers.alt || modifiers.shift || modifiers.platform {
                return;
            }
            let key = keystroke.key.as_str();
            if !matches!(key, "up" | "down" | "tab" | "enter" | "escape") {
                return;
            }
            let handled = weak
                .update(cx, |this, cx| this.completion_key(key, window, cx))
                .unwrap_or(false);
            if handled {
                cx.stop_propagation();
            }
        }));

        let mut page = Self {
            focus_handle: cx.focus_handle(),
            sessions: Vec::new(),
            loaded: false,
            selected: None,
            scroller,
            composer,
            speech,
            voice_select,
            model_select,
            reasoning_select,
            profiles: Vec::new(),
            models: HashMap::new(),
            models_loading: HashSet::new(),
            toggled: HashSet::new(),
            skills: None,
            completion: None,
            completion_dismissed: None,
            sidecar_online: sidecar.read(cx).is_online(),
            turn: None,
            voice: None,
            voice_generation: 0,
            saver,
            _tasks: vec![save_task],
            _subscriptions: subscriptions,
        };
        page.load_sessions(window, cx);
        page.load_models("ollama".into(), window, cx);
        if page.sidecar_online {
            page.load_profiles(window, cx);
        }
        page
    }

    // MARK: Loading

    fn load_sessions(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let task = cx.spawn_in(window, async move |this, cx| {
            let sessions = cx.background_spawn(async { Session::load_all() }).await;
            this.update_in(cx, |this, window, cx| {
                this.sessions = sessions;
                this.loaded = true;
                if let Some(first) = this.sessions.first().map(|s| s.id.clone()) {
                    this.select_session(first, window, cx);
                    // Test hook: SEDA_SEND=<text> sends it in the most recent
                    // session at startup (used for end-to-end checks).
                    // Test hook: SEDA_COMPOSER=<text> prefills the composer and focuses it.
                    if let Ok(text) = std::env::var("SEDA_COMPOSER") {
                        this.composer.update(cx, |composer, cx| {
                            composer.set_value(text, window, cx);
                            composer.focus(window, cx);
                        });
                    }
                    if let Ok(text) = std::env::var("SEDA_SEND") {
                        this.composer
                            .update(cx, |composer, cx| composer.set_value(text, window, cx));
                        this.send(window, cx);
                    }
                }
                cx.notify();
            })
            .ok();
        });
        self._tasks.push(task);
    }

    /// Fetch the model list for `source` (`ollama`, `opencode` or
    /// `openai:<endpoint id>`) unless it is loaded or loading.
    fn load_models(&mut self, source: String, window: &mut Window, cx: &mut Context<Self>) {
        if self.models.contains_key(&source) || !self.models_loading.insert(source.clone()) {
            return;
        }
        let settings = AppSettings::get(cx).clone();
        type Fetch = Box<dyn FnOnce() -> anyhow::Result<Vec<String>> + Send>;
        let fetch: Fetch = if source == "ollama" {
            let client = OllamaClient::from_settings(cx);
            Box::new(move || client.models())
        } else if source == "opencode" {
            let bin = settings.opencode_bin.clone();
            Box::new(move || opencode::models(&bin))
        } else if let Some(endpoint) = source
            .strip_prefix("openai:")
            .and_then(|id| settings.endpoint(id))
        {
            let client = OpenAiClient::from_endpoint(endpoint);
            Box::new(move || client.models())
        } else {
            self.models_loading.remove(&source);
            return;
        };
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { fetch() }).await;
            this.update_in(cx, |this, window, cx| {
                this.models_loading.remove(&source);
                if let Ok(models) = result {
                    this.models.insert(source, models);
                    this.refresh_header_selects(window, cx);
                }
            })
            .ok();
        });
        self._tasks.push(task);
    }

    /// Scan skills and commands for `cwd` (Claude sessions) unless already done.
    fn load_skills(&mut self, cwd: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        if self.skills.as_ref().is_some_and(|(dir, _)| *dir == cwd) {
            return;
        }
        let task = cx.spawn_in(window, async move |this, cx| {
            let dir = cwd.clone();
            let found = cx
                .background_spawn(async move {
                    let home = dirs::home_dir().unwrap_or_default();
                    skills::scan(&home, Some(&dir))
                })
                .await;
            this.update_in(cx, |this, _, cx| {
                this.skills = Some((cwd, found));
                this.update_completion(cx);
                cx.notify();
            })
            .ok();
        });
        self._tasks.push(task);
    }

    fn load_profiles(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let client = SidecarClient::from_settings(cx);
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { client.profiles() }).await;
            this.update_in(cx, |this, window, cx| {
                if let Ok(profiles) = result {
                    this.profiles = profiles;
                    this.refresh_header_selects(window, cx);
                    cx.notify();
                }
            })
            .ok();
        });
        self._tasks.push(task);
    }

    // MARK: Sessions

    fn current(&self) -> Option<&Session> {
        let id = self.selected.as_ref()?;
        self.sessions.iter().find(|s| &s.id == id)
    }

    fn save(&self, session: &Session) {
        _ = self.saver.try_send(SaveOp::Save(Box::new(session.clone())));
    }

    fn save_by_id(&self, id: &str) {
        if let Some(session) = self.sessions.iter().find(|s| s.id == id) {
            self.save(session);
        }
    }

    /// Change the open session and save it.
    fn update_current(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut Session)) {
        let Some(id) = self.selected.clone() else {
            return;
        };
        if let Some(session) = self.sessions.iter_mut().find(|s| s.id == id) {
            f(session);
        }
        self.save_by_id(&id);
        cx.notify();
    }

    fn select_session(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.selected = Some(id);
        let rows = self.row_count();
        self.scroller
            .update(cx, |scroller, cx| scroller.reset(rows, cx));
        self.refresh_header_selects(window, cx);
        let backend = self.current().map(|s| s.backend.clone());
        match backend {
            Some(Backend::OpenAi { endpoint, .. }) => {
                self.load_models(format!("openai:{endpoint}"), window, cx)
            }
            Some(Backend::OpenCode { .. }) => self.load_models("opencode".into(), window, cx),
            Some(Backend::Claude { cwd, .. }) => self.load_skills(cwd, window, cx),
            _ => {}
        }
        self.completion = None;
        self.composer
            .update(cx, |composer, cx| composer.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn add_session(
        &mut self,
        session: Session,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.save(&session);
        let id = session.id.clone();
        self.sessions.insert(0, session);
        self.select_session(id, window, cx);
    }

    fn delete_session(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self.turn.as_ref().is_some_and(|turn| turn.session_id == id) {
            self.stop(cx);
        }
        self.sessions.retain(|s| s.id != id);
        _ = self.saver.try_send(SaveOp::Delete(id.to_string()));
        if self.selected.as_deref() == Some(id) {
            self.selected = None;
            match self.sessions.first().map(|s| s.id.clone()) {
                Some(next) => self.select_session(next, window, cx),
                None => self
                    .scroller
                    .update(cx, |scroller, cx| scroller.reset(0, cx)),
            }
        }
        cx.notify();
    }

    fn refresh_header_selects(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(session) = self.current() else {
            return;
        };
        let voice = session.voice_profile.clone().unwrap_or_default();
        let voices = voice_choices(&self.profiles, Some(&voice));
        let no_models = Vec::new();
        let list = |key: &str| self.models.get(key).unwrap_or(&no_models);
        let models = match &session.backend {
            Backend::Ollama { model, .. } => {
                Some((model_choices(list("ollama"), Some(model)), model.clone()))
            }
            Backend::Claude { model, .. } => Some((
                claude_model_choices(model.as_deref()),
                model.clone().unwrap_or_default(),
            )),
            Backend::OpenAi {
                endpoint, model, ..
            } => Some((
                model_choices(list(&format!("openai:{endpoint}")), Some(model)),
                model.clone(),
            )),
            Backend::OpenCode { model, .. } => Some((
                optional_model_choices(list("opencode"), model.as_deref()),
                model.clone().unwrap_or_default(),
            )),
            Backend::Unknown(_) => None,
        };
        let reasoning = reasoning_scale(&session.backend).map(|scale| {
            let current = reasoning_of(&session.backend);
            (
                reasoning_choices(scale, current.as_deref()),
                reasoning_value(scale, current.as_deref()),
            )
        });
        self.voice_select.update(cx, |select, cx| {
            select.set_items(Choices::new(voices), window, cx);
            select.set_selected_value(&voice, window, cx);
        });
        if let Some((choices, value)) = models {
            self.model_select.update(cx, |select, cx| {
                select.set_items(Choices::new(choices), window, cx);
                select.set_selected_value(&value, window, cx);
            });
        }
        if let Some((choices, value)) = reasoning {
            self.reasoning_select.update(cx, |select, cx| {
                select.set_items(Choices::new(choices), window, cx);
                select.set_selected_value(&value, window, cx);
            });
        }
    }

    fn open_new_session(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let page = cx.entity().downgrade();
        let form = cx.new(|cx| NewSessionForm::new(page, window, cx));
        window.open_dialog(cx, move |dialog, _, _| {
            dialog
                .title("گفتگوی تازه")
                .w(px(560.))
                .close_button(false)
                .child(form.clone())
        });
    }

    fn confirm_delete(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(session) = self.current() else {
            return;
        };
        let id = session.id.clone();
        let title = session.title.clone();
        let page = cx.entity().downgrade();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let page = page.clone();
            let id = id.clone();
            alert
                .title(format!("گفتگوی «{title}» حذف شود؟"))
                .description("همه‌ی پیام‌های این گفتگو برای همیشه پاک می‌شوند.")
                .confirm()
                .ok_text("حذف گفتگو")
                .ok_variant(ButtonVariant::Danger)
                .cancel_text("انصراف")
                .on_ok(move |_, window, cx| {
                    page.update(cx, |page, cx| page.delete_session(&id, window, cx))
                        .ok();
                    true
                })
        });
    }

    fn open_rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(session) = self.current() else {
            return;
        };
        let id = session.id.clone();
        let title = session.title.clone();
        let input = cx.new(|cx| InputState::new(window, cx).default_value(title));
        input.update(cx, |input, cx| input.focus(window, cx));
        let page = cx.entity().downgrade();
        window.open_dialog(cx, move |dialog, _, cx| {
            let page = page.clone();
            let id = id.clone();
            let input_for_save = input.clone();
            dialog
                .title("تغییر نام گفتگو")
                .w(px(420.))
                .close_button(false)
                .child(rtl::input(&input, cx))
                .footer(
                    DialogFooter::new()
                        .child(
                            Button::new("rename-cancel")
                                .outline()
                                .label("انصراف")
                                .on_click(|_, window, cx| window.close_dialog(cx)),
                        )
                        .child(
                            Button::new("rename-save")
                                .primary()
                                .label("ذخیره")
                                .on_click(move |_, window, cx| {
                                    let title = input_for_save.read(cx).value().trim().to_string();
                                    if !title.is_empty() {
                                        page.update(cx, |page, cx| {
                                            if let Some(s) =
                                                page.sessions.iter_mut().find(|s| s.id == id)
                                            {
                                                s.title = title;
                                            }
                                            page.save_by_id(&id);
                                            cx.notify();
                                        })
                                        .ok();
                                    }
                                    window.close_dialog(cx);
                                }),
                        ),
                )
        });
    }

    // MARK: Conversation rows

    /// Whether the open session shows a "thinking" row, and its label.
    fn pending_label(&self) -> Option<&'static str> {
        let turn = self.turn.as_ref()?;
        if Some(&turn.session_id) != self.selected.as_ref() || !turn.waiting {
            return None;
        }
        Some(if turn.after_tool {
            "در حال کار…"
        } else {
            "در حال فکر کردن…"
        })
    }

    /// Rows of the message list: top-level messages only. A subagent's
    /// messages render inside its task node, so a streaming subagent never
    /// inserts rows in the middle of the list.
    fn row_count(&self) -> usize {
        self.current().map_or(0, |s| top_level(s).len()) + usize::from(self.pending_label().is_some())
    }

    /// Keep the message list's row count in step with the open session, and
    /// remeasure the last row plus the row that holds `touched`.
    fn sync_scroller_touching(&mut self, touched: Option<&str>, cx: &mut Context<Self>) {
        let want = self.row_count();
        let touched_row = touched.and_then(|id| self.current().and_then(|s| root_row(s, id)));
        self.scroller.update(cx, |scroller, cx| {
            let have = scroller.item_count();
            if want > have {
                scroller.append(want - have, cx);
            } else if want < have {
                scroller.splice(want..have, 0, cx);
            }
            if want > 0 {
                scroller.remeasure_items(want - 1..want, cx);
            }
            if let Some(row) = touched_row.filter(|row| *row < want) {
                scroller.remeasure_items(row..row + 1, cx);
            }
        });
    }

    fn sync_scroller(&mut self, cx: &mut Context<Self>) {
        self.sync_scroller_touching(None, cx);
    }

    fn can_speak(&self, cx: &App) -> bool {
        self.sidecar_online
            && cx.has_global::<Player>()
            && self.current().is_some_and(|s| s.voice_profile.is_some())
    }

    /// Whether a thinking block or task node is open.
    fn is_expanded(&self, message: &ChatMessage) -> bool {
        let default = match message.role {
            Role::Task => message.task.as_ref().is_some_and(|t| !task_done(&t.status)),
            _ => false,
        };
        default != self.toggled.contains(&message.id)
    }

    fn toggle_expanded(&mut self, message_id: &str, cx: &mut Context<Self>) {
        if !self.toggled.remove(message_id) {
            self.toggled.insert(message_id.to_string());
        }
        self.sync_scroller_touching(Some(message_id), cx);
        cx.notify();
    }

    fn render_row(&self, ix: usize, page: &WeakEntity<Self>, cx: &App) -> AnyElement {
        let Some(session) = self.current() else {
            return div().into_any_element();
        };
        let rows = top_level(session);
        let Some(message) = rows.get(ix).map(|&i| &session.messages[i]) else {
            return match self.pending_label() {
                Some(label) => h_flex()
                    .gap_2()
                    .px_3()
                    .text_sm()
                    .child(ShimmerText::new(label).id("chat-pending"))
                    .into_any_element(),
                None => div().into_any_element(),
            };
        };
        self.render_message(session, message, 0, page, cx)
    }

    /// One message; `depth` > 0 inside a subagent's node.
    fn render_message(
        &self,
        session: &Session,
        message: &ChatMessage,
        depth: usize,
        page: &WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let theme = cx.theme();
        let streaming = self
            .turn
            .as_ref()
            .is_some_and(|turn| turn.is_streaming(&message.id));

        match message.role {
            Role::User => Message::new()
                .id(ElementId::Name(format!("msg-{}", message.id).into()))
                .alignment(MessageAlignment::End)
                .content(
                    MessageContent::new().bubble(
                        Bubble::new()
                            .with_variant(BubbleVariant::Muted)
                            .child(markdown(&message.id, &message.text, false)),
                    ),
                )
                .into_any_element(),
            // A subagent's text: compact, not spoken.
            Role::Assistant if depth > 0 => div()
                .px_3()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(markdown(&message.id, &message.text, streaming))
                .into_any_element(),
            Role::Assistant => {
                let speaking = self
                    .voice
                    .as_ref()
                    .is_some_and(|voice| voice.message_id.as_ref() == Some(&message.id));
                let message_id = message.id.clone();
                let page = page.clone();
                Message::new()
                    .id(ElementId::Name(format!("msg-{}", message.id).into()))
                    .alignment(MessageAlignment::Start)
                    .content(
                        MessageContent::new().bubble(
                            Bubble::new()
                                .with_variant(BubbleVariant::Ghost)
                                .child(markdown(&message.id, &message.text, streaming)),
                        ),
                    )
                    .when(!streaming, |this| {
                        this.footer(
                            MessageFooter::new()
                                .child(
                                    Button::new(ElementId::Name(
                                        format!("speak-{}", message.id).into(),
                                    ))
                                    .ghost()
                                    .xsmall()
                                    .icon(if speaking {
                                        IconName::Square
                                    } else {
                                        IconName::Volume2
                                    })
                                    .tooltip(if speaking {
                                        "توقف خواندن"
                                    } else {
                                        "خواندن با صدا"
                                    })
                                    .disabled(
                                        !speaking && (self.turn.is_some() || !self.can_speak(cx)),
                                    )
                                    .on_click(
                                        move |_, window, cx| {
                                            page.update(cx, |this, cx| {
                                                this.speak_message(&message_id, window, cx)
                                            })
                                            .ok();
                                        },
                                    ),
                                )
                                .child(
                                    Clipboard::new(ElementId::Name(
                                        format!("copy-{}", message.id).into(),
                                    ))
                                    .value(message.text.clone()),
                                ),
                        )
                    })
                    .into_any_element()
            }
            Role::Tool => {
                let mut lines = message.text.lines();
                let call = lines.next().unwrap_or_default().to_string();
                let result: Vec<&str> = lines.collect();
                v_flex()
                    .px_3()
                    .gap_0p5()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(
                        h_flex()
                            .gap_2()
                            .child(Icon::new(IconName::Wrench).xsmall())
                            .child(div().min_w_0().ltr().truncate().child(call)),
                    )
                    .when(!result.is_empty(), |this| {
                        this.child(
                            h_flex().child(
                                div()
                                    .min_w_0()
                                    .ltr()
                                    .truncate()
                                    .opacity(0.8)
                                    .child(result.join(" ")),
                            ),
                        )
                    })
                    .into_any_element()
            }
            Role::Thinking => self.render_thinking(message, streaming, page, cx),
            Role::Task => self.render_task(session, message, depth, page, cx),
            Role::Error => Message::new()
                .id(ElementId::Name(format!("msg-{}", message.id).into()))
                .alignment(MessageAlignment::Start)
                .content(
                    MessageContent::new().bubble(
                        Bubble::new()
                            .with_variant(BubbleVariant::Destructive)
                            .child(
                                h_flex()
                                    .gap_2()
                                    .items_start()
                                    .child(Icon::new(IconName::CircleAlert).small())
                                    .child(div().min_w_0().child(message.text.clone())),
                            ),
                    ),
                )
                .into_any_element(),
        }
    }

    /// A collapsed "thinking" block; click to read it.
    fn render_thinking(
        &self,
        message: &ChatMessage,
        streaming: bool,
        page: &WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let theme = cx.theme();
        let expanded = self.is_expanded(message);
        let id = message.id.clone();
        let page = page.clone();
        let label: SharedString = if streaming { "در حال فکر کردن…" } else { "فکر" }.into();
        v_flex()
            .px_3()
            .gap_1()
            .text_xs()
            .text_color(theme.muted_foreground)
            .child(
                h_flex()
                    .id(ElementId::Name(format!("think-{}", message.id).into()))
                    .gap_1p5()
                    .cursor_pointer()
                    .hover(|this| this.text_color(theme.foreground))
                    .on_click(move |_, _, cx| {
                        page.update(cx, |this, cx| this.toggle_expanded(&id, cx)).ok();
                    })
                    .child(Icon::new(chevron(expanded, cx)).xsmall())
                    .child(Icon::new(IconName::Brain).xsmall())
                    .child(if streaming {
                        ShimmerText::new(label)
                            .id(ElementId::Name(format!("think-shimmer-{}", message.id).into()))
                            .into_any_element()
                    } else {
                        div().child(label).into_any_element()
                    }),
            )
            .when(expanded, |this| {
                this.child(
                    div()
                        .pl_5()
                        .italic()
                        .child(markdown(&message.id, &message.text, streaming)),
                )
            })
            .into_any_element()
    }

    /// A subagent: header (agent type, description, status, summary) and,
    /// when open, its own messages indented under it.
    fn render_task(
        &self,
        session: &Session,
        message: &ChatMessage,
        depth: usize,
        page: &WeakEntity<Self>,
        cx: &App,
    ) -> AnyElement {
        let theme = cx.theme();
        let Some(task) = message.task.as_ref() else {
            return div().into_any_element();
        };
        let expanded = self.is_expanded(message);
        let id = message.id.clone();
        let toggle_page = page.clone();
        let status_icon = match task_state(&task.status) {
            TaskState::Running => Spinner::new().xsmall().into_any_element(),
            TaskState::Done => Icon::new(IconName::CircleCheck)
                .xsmall()
                .text_color(theme.success)
                .into_any_element(),
            TaskState::Failed => Icon::new(IconName::CircleX)
                .xsmall()
                .text_color(theme.danger)
                .into_any_element(),
        };
        let agent_type = if task.agent_type.is_empty() {
            "subagent".to_string()
        } else {
            task.agent_type.clone()
        };
        let children: Vec<AnyElement> = if expanded && depth < 8 {
            session
                .messages
                .iter()
                .filter(|m| m.parent.as_deref() == Some(task.tool_use_id.as_str()))
                .map(|child| self.render_message(session, child, depth + 1, page, cx))
                .collect()
        } else {
            Vec::new()
        };
        let rtl = rtl::is_rtl(cx);
        v_flex()
            .px_3()
            .gap_1()
            .child(
                h_flex()
                    .id(ElementId::Name(format!("task-{}", message.id).into()))
                    .gap_1p5()
                    .min_w_0()
                    .text_xs()
                    .cursor_pointer()
                    .on_click(move |_, _, cx| {
                        toggle_page
                            .update(cx, |this, cx| this.toggle_expanded(&id, cx))
                            .ok();
                    })
                    .child(
                        Icon::new(chevron(expanded, cx))
                            .xsmall()
                            .text_color(theme.muted_foreground),
                    )
                    .child(Icon::new(IconName::Bot).xsmall())
                    .child(
                        div()
                            .flex_shrink_0()
                            .ltr()
                            .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                            .child(agent_type),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .text_color(theme.muted_foreground)
                            .child(task.description.clone()),
                    )
                    .child(status_icon),
            )
            .when_some(
                task.summary.as_ref().filter(|s| !s.trim().is_empty() && !expanded),
                |this, summary| {
                    this.child(
                        div()
                            .pl_5()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .truncate()
                            .child(one_line(summary, 160)),
                    )
                },
            )
            .when(expanded, |this| {
                this.child(
                    v_flex()
                        .ml_1p5()
                        .pl_2()
                        .py_1()
                        .gap_2()
                        .map(|this| if rtl { this.border_r_1() } else { this.border_l_1() })
                        .border_color(theme.border)
                        .when(!message.text.trim().is_empty(), |this| {
                            this.child(
                                div()
                                    .px_3()
                                    .text_xs()
                                    .text_color(theme.muted_foreground)
                                    .opacity(0.8)
                                    .child(one_line(&message.text, 240)),
                            )
                        })
                        .children(children)
                        .when_some(
                            task.summary.as_ref().filter(|s| !s.trim().is_empty()),
                            |this, summary| {
                                this.child(
                                    h_flex()
                                        .px_3()
                                        .gap_1p5()
                                        .items_start()
                                        .text_xs()
                                        .child(Icon::new(IconName::Check).xsmall())
                                        .child(div().min_w_0().child(markdown(
                                            &format!("{}-summary", message.id),
                                            summary,
                                            false,
                                        ))),
                                )
                            },
                        ),
                )
            })
            .into_any_element()
    }

    // MARK: Sending and streaming

    fn send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.turn.is_some() {
            return;
        }
        let text = self.composer.read(cx).value().trim().to_string();
        if text.is_empty() {
            return;
        }
        let Some(id) = self.selected.clone() else {
            return;
        };
        let settings = AppSettings::get(cx).clone();
        let Some(session) = self.sessions.iter_mut().find(|s| s.id == id) else {
            return;
        };
        if let Some(kind) = session.backend.unknown_kind() {
            window.push_notification(
                Notification::warning(format!(
                    "این گفتگو با نسخهٔ تازه‌تری از صدا ساخته شده است ({kind}) و اینجا قابل ادامه نیست."
                )),
                cx,
            );
            return;
        }
        if let Backend::OpenAi { endpoint, .. } = &session.backend
            && settings.endpoint(endpoint).is_none()
        {
            window.push_notification(
                Notification::error("سرویس این گفتگو در تنظیمات پیدا نشد."),
                cx,
            );
            return;
        }
        session
            .messages
            .push(ChatMessage::new(Role::User, text.clone()));
        session.title_from_first_message();
        session.updated_at = crate::store::now();

        let cancel = CancelToken::new();
        let events = start_turn(session, &settings, text, cancel.clone());
        let voice = session.voice_profile.clone().filter(|_| session.auto_speak);
        self.sessions
            .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        self.save_by_id(&id);
        self.completion = None;
        self.composer
            .update(cx, |composer, cx| composer.set_value("", window, cx));

        self.stop_voice();
        if let Some(profile) = voice {
            self.start_voice(profile, None, window, cx);
        }

        let task = cx.spawn_in(window, async move |this, cx| {
            while let Ok(event) = events.recv().await {
                let more = this.update_in(cx, |this, window, cx| this.on_event(event, window, cx));
                if !matches!(more, Ok(true)) {
                    break;
                }
            }
            this.update_in(cx, |this, _, cx| this.finish_turn(cx)).ok();
        });
        self.turn = Some(Turn {
            session_id: id,
            cancel,
            task,
            segments: HashMap::new(),
            tools: HashMap::new(),
            waiting: true,
            after_tool: false,
            last_save: Instant::now(),
        });
        self.sync_scroller(cx);
        self.scroller
            .update(cx, |scroller, cx| scroller.scroll_to_end(cx));
        cx.notify();
    }

    /// Apply one stream event. Returns whether more events are expected.
    fn on_event(
        &mut self,
        event: StreamEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(turn) = self.turn.as_mut() else {
            return false;
        };
        let Some(session) = self.sessions.iter_mut().find(|s| s.id == turn.session_id) else {
            return false;
        };
        let mut applied = apply_event(session, turn, event, None);
        if let Some(delta) = &applied.spoken
            && let Some(voice) = self.voice.as_mut().filter(|v| !v.finished)
        {
            voice.queue.push(delta);
        }
        if let Some(message) = applied.error.take() {
            window.push_notification(Notification::error(message), cx);
        }
        if turn.last_save.elapsed() >= STREAM_SAVE_INTERVAL {
            applied.save = true;
        }
        if applied.save {
            turn.last_save = Instant::now();
            let id = turn.session_id.clone();
            self.save_by_id(&id);
        }
        if applied.commands_changed {
            self.update_completion(cx);
        }
        self.sync_scroller_touching(applied.touched.as_deref(), cx);
        cx.notify();
        !applied.finished
    }

    fn finish_turn(&mut self, cx: &mut Context<Self>) {
        let Some(turn) = self.turn.take() else {
            return;
        };
        // This runs inside the turn's own task: let it end on its own.
        turn.task.detach();
        if let Some(voice) = self.voice.as_mut().filter(|v| !v.finished) {
            voice.queue.finish();
            voice.finished = true;
        }
        if let Some(session) = self.sessions.iter_mut().find(|s| s.id == turn.session_id) {
            session.updated_at = crate::store::now();
            stop_running_tasks(session);
        }
        self.save_by_id(&turn.session_id);
        self.sync_scroller(cx);
        self.scroller.update(cx, |scroller, cx| scroller.remeasure(cx));
        cx.notify();
    }

    /// Stop generating and speaking.
    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(turn) = self.turn.take() {
            turn.cancel.cancel();
            drop(turn.task);
            if let Some(session) = self.sessions.iter_mut().find(|s| s.id == turn.session_id) {
                stop_running_tasks(session);
            }
            self.save_by_id(&turn.session_id);
        }
        self.stop_voice();
        self.sync_scroller(cx);
        self.scroller.update(cx, |scroller, cx| scroller.remeasure(cx));
        cx.notify();
    }

    // MARK: Slash completion

    /// Completions for the open session: skills and commands on disk plus
    /// the ones the last Claude Code init reported.
    fn completion_source(&self) -> Vec<SkillInfo> {
        let Some(session) = self.current() else {
            return Vec::new();
        };
        let Backend::Claude { cwd, commands, .. } = &session.backend else {
            return Vec::new();
        };
        let scanned = self
            .skills
            .as_ref()
            .filter(|(dir, _)| dir == cwd)
            .map(|(_, list)| list.clone())
            .unwrap_or_default();
        skills::merge(scanned, commands)
    }

    /// Open, filter or close the `/` popup for the composer's text.
    fn update_completion(&mut self, cx: &mut Context<Self>) {
        let text = self.composer.read(cx).value().to_string();
        if self.completion_dismissed.as_deref() != Some(text.as_str()) {
            self.completion_dismissed = None;
        }
        let query = text
            .strip_prefix('/')
            .filter(|rest| !rest.contains(char::is_whitespace));
        let Some(query) = query.filter(|_| self.completion_dismissed.is_none()) else {
            self.completion = None;
            return;
        };
        let query = query.to_lowercase();
        let items: Vec<SkillInfo> = self
            .completion_source()
            .into_iter()
            .filter(|s| s.name.to_lowercase().starts_with(&query))
            .collect();
        if items.is_empty() || (items.len() == 1 && items[0].name.to_lowercase() == query) {
            self.completion = None;
            return;
        }
        let previous = self
            .completion
            .as_ref()
            .and_then(|c| c.items.get(c.selected))
            .map(|s| s.name.clone());
        let selected = previous
            .and_then(|name| items.iter().position(|s| s.name == name))
            .unwrap_or(0);
        let scroll = self
            .completion
            .take()
            .map(|c| c.scroll)
            .unwrap_or_default();
        self.completion = Some(Completion {
            items,
            selected,
            scroll,
        });
    }

    /// A key while the popup is open; returns whether it was used.
    fn completion_key(&mut self, key: &str, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let Some(completion) = self.completion.as_mut() else {
            return false;
        };
        if !self.composer.read(cx).focus_handle(cx).is_focused(window) {
            return false;
        }
        let len = completion.items.len();
        match key {
            "up" => completion.selected = (completion.selected + len - 1) % len,
            "down" => completion.selected = (completion.selected + 1) % len,
            "tab" | "enter" => {
                let ix = completion.selected;
                self.accept_completion(ix, window, cx);
                return true;
            }
            "escape" => {
                self.completion = None;
                self.completion_dismissed = Some(self.composer.read(cx).value().to_string());
            }
            _ => return false,
        }
        if let Some(completion) = &self.completion {
            completion.scroll.scroll_to_item(completion.selected);
        }
        cx.notify();
        true
    }

    fn accept_completion(&mut self, ix: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(item) = self.completion.take().and_then(|c| c.items.into_iter().nth(ix)) else {
            return;
        };
        let text = format!("/{} ", item.name);
        let column = text.encode_utf16().count() as u32;
        self.composer.update(cx, |composer, cx| {
            composer.set_value(text, window, cx);
            composer.set_cursor_position(
                gpui_kit::component::input::Position::new(0, column),
                window,
                cx,
            );
        });
        cx.notify();
    }

    fn render_completion(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let completion = self.completion.as_ref()?;
        let theme = cx.theme();
        let rows = completion.items.iter().enumerate().map(|(ix, item)| {
            let selected = ix == completion.selected;
            h_flex()
                .id(ElementId::Name(format!("slash-{}", item.name).into()))
                .ltr()
                .w_full()
                .gap_3()
                .px_2()
                .py_1()
                .rounded(theme.radius)
                .cursor_pointer()
                .when(selected, |this| this.bg(theme.accent))
                .hover(|this| this.bg(theme.accent))
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.accept_completion(ix, window, cx)
                }))
                .child(
                    div()
                        .flex_shrink_0()
                        .text_sm()
                        .font_weight(gpui_kit::FontWeight::MEDIUM)
                        .child(format!("/{}", item.name)),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .text_xs()
                        .text_color(theme.muted_foreground)
                        .child(item.description.clone()),
                )
        });
        Some(
            div()
                .id("slash-completion")
                .max_h(px(240.))
                .overflow_y_scroll()
                .track_scroll(&completion.scroll)
                .p_1()
                .border_b_1()
                .border_color(theme.border)
                .child(v_flex().gap_0p5().children(rows))
                .into_any_element(),
        )
    }

    // MARK: Voice

    fn stop_voice(&mut self) {
        self.voice_generation += 1;
        if let Some(mut voice) = self.voice.take() {
            voice.queue.stop();
        }
    }

    /// Start a speech queue for `profile`. Returns `false` (and explains
    /// why) when speech is not possible right now.
    fn start_voice(
        &mut self,
        profile: String,
        message_id: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(player) = cx.try_global::<Player>().cloned() else {
            if message_id.is_some() {
                window.push_notification(Notification::error("دستگاه پخش صدا پیدا نشد."), cx);
            }
            return false;
        };
        if !SidecarGlobal::get(cx).read(cx).is_online() {
            if message_id.is_some() {
                window.push_notification(
                    Notification::warning("سایدکار آفلاین است؛ خواندن با صدا ممکن نیست."),
                    cx,
                );
            }
            return false;
        }
        self.stop_voice();
        let speed = AppSettings::get(cx).tts_speed;
        let queue = SpeechQueue::new(
            SidecarClient::from_settings(cx),
            player,
            profile,
            TtsRun::Active,
            Some(speed),
        );
        let errors = queue.errors();
        let errors_task = cx.spawn_in(window, async move |this, cx| {
            while let Ok(err) = errors.recv().await {
                let shown = this.update_in(cx, |_, window, cx| {
                    window.push_notification(
                        Notification::error(format!("خواندن با صدا ناموفق بود: {err}")),
                        cx,
                    )
                });
                if shown.is_err() {
                    break;
                }
            }
        });
        self.voice = Some(Voice {
            queue,
            finished: false,
            message_id,
            _errors: errors_task,
        });

        // Drop the voice once it has finished playing, so the stop button
        // and the speak buttons return to rest.
        let generation = self.voice_generation;
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(300))
                    .await;
                let done = this.update(cx, |this, cx| {
                    if this.voice_generation != generation {
                        return true;
                    }
                    let idle = this
                        .voice
                        .as_ref()
                        .is_none_or(|v| v.finished && !v.queue.is_active());
                    if idle {
                        this.voice = None;
                        cx.notify();
                    }
                    idle
                });
                if !matches!(done, Ok(false)) {
                    break;
                }
            }
        })
        .detach();
        cx.notify();
        true
    }

    fn speak_message(&mut self, message_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        if self
            .voice
            .as_ref()
            .is_some_and(|v| v.message_id.as_deref() == Some(message_id))
        {
            self.stop_voice();
            cx.notify();
            return;
        }
        if self.turn.is_some() {
            return;
        }
        let Some(session) = self.current() else {
            return;
        };
        let Some(profile) = session.voice_profile.clone() else {
            window.push_notification(
                Notification::warning("برای این گفتگو صدایی انتخاب نشده است."),
                cx,
            );
            return;
        };
        let Some(text) = session
            .messages
            .iter()
            .find(|m| m.id == message_id)
            .map(|m| m.text.clone())
        else {
            return;
        };
        if self.start_voice(profile, Some(message_id.to_string()), window, cx)
            && let Some(voice) = self.voice.as_mut()
        {
            voice.queue.push(&text);
            voice.queue.finish();
            voice.finished = true;
        }
    }

    // MARK: Push-to-talk

    fn toggle_push_to_talk(
        &mut self,
        _: &TogglePushToTalk,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.selected.is_some() {
            self.speech.update(cx, |speech, cx| speech.toggle(cx));
        }
    }

    fn on_transcript(&mut self, text: &SharedString, window: &mut Window, cx: &mut Context<Self>) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let current = self.composer.read(cx).value();
        let joined = if current.trim().is_empty() {
            text.to_string()
        } else {
            format!("{} {text}", current.trim_end())
        };
        self.composer.update(cx, |composer, cx| {
            composer.set_value(joined, window, cx);
            composer.focus(window, cx);
        });
        if AppSettings::get(cx).ptt_auto_send && self.turn.is_none() {
            self.send(window, cx);
        }
        cx.notify();
    }

    // MARK: Rendering

    fn render_session_list(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let rtl = rtl::is_rtl(cx);
        let busy_session = self.turn.as_ref().map(|t| t.session_id.clone());
        let settings = AppSettings::get(cx).clone();
        let rows = self.sessions.iter().map(|session| {
            let id = session.id.clone();
            let backend = backend_label(&session.backend, &settings);
            let busy = busy_session.as_ref() == Some(&session.id);
            ListItem::new(ElementId::Name(format!("session-{}", session.id).into()))
                .rounded(theme.radius)
                .px_2()
                .py_1p5()
                .selected(self.selected.as_ref() == Some(&session.id))
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.select_session(id.clone(), window, cx)
                }))
                .child(
                    v_flex()
                        .min_w_0()
                        .gap_0p5()
                        .child(
                            h_flex()
                                .gap_2()
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .text_sm()
                                        .truncate()
                                        .child(session.title.clone()),
                                )
                                .when(busy, |this| this.child(Spinner::new().xsmall())),
                        )
                        .child(meta_row(
                            [
                                SharedString::from(backend),
                                relative_time(&session.updated_at).into(),
                            ],
                            cx,
                        )),
                )
        });

        v_flex()
            .w(px(264.))
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
                    .justify_between()
                    .child(
                        div()
                            .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                            .child("گفتگوها"),
                    )
                    .child(
                        Button::new("new-chat")
                            .outline()
                            .small()
                            .icon(IconName::SquarePen)
                            .label("گفتگوی تازه")
                            .on_click(
                                cx.listener(|this, _, window, cx| {
                                    this.open_new_session(window, cx)
                                }),
                            ),
                    ),
            )
            .child(
                div()
                    .id("session-list")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .px_2()
                    .pb_2()
                    .map(|this| {
                        if !self.loaded {
                            this.child(
                                h_flex()
                                    .p_4()
                                    .justify_center()
                                    .child(Spinner::new().small()),
                            )
                        } else if self.sessions.is_empty() {
                            this.child(
                                div()
                                    .p_4()
                                    .text_sm()
                                    .text_color(theme.muted_foreground)
                                    .child("هنوز گفتگویی نیست."),
                            )
                        } else {
                            this.child(v_flex().gap_0p5().children(rows))
                        }
                    }),
            )
    }

    fn render_header(&self, session: &Session, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let page = cx.entity().downgrade();
        let agent = |icon: IconName, name: &'static str, cwd: &std::path::Path| {
            h_flex()
                .gap_1()
                .min_w_0()
                .child(Icon::new(icon).xsmall())
                .child(div().flex_shrink_0().child(name))
                .child(
                    div()
                        .size(px(3.))
                        .mx_0p5()
                        .flex_shrink_0()
                        .rounded_full()
                        .bg(theme.muted_foreground),
                )
                .child(
                    div()
                        .min_w_0()
                        .ltr()
                        .truncate()
                        .child(cwd.display().to_string()),
                )
        };
        let subtitle = match &session.backend {
            Backend::Ollama { .. } => h_flex()
                .gap_1()
                .child(Icon::new(IconName::Bot).xsmall())
                .child("Ollama"),
            Backend::Claude { cwd, .. } => agent(IconName::Terminal, "Claude Code", cwd),
            Backend::OpenCode { cwd, .. } => agent(IconName::Terminal, "opencode", cwd),
            Backend::OpenAi { endpoint, .. } => h_flex()
                .gap_1()
                .child(Icon::new(IconName::Globe).xsmall())
                .child(
                    AppSettings::get(cx)
                        .endpoint(endpoint)
                        .map(|e| e.name.clone())
                        .unwrap_or_else(|| "سرویس حذف‌شده".into()),
                ),
            Backend::Unknown(_) => h_flex()
                .gap_1()
                .child(Icon::new(IconName::CircleAlert).xsmall())
                .child(format!(
                    "پاسخ‌دهندهٔ ناشناخته ({})",
                    session.backend.unknown_kind().unwrap_or_default()
                )),
        };
        let known = session.backend.unknown_kind().is_none();
        // Two rows, so the title keeps its width in a narrow window: title,
        // backend and the menu first; the session controls wrap below.
        v_flex()
            .w_full()
            .px_4()
            .py_2p5()
            .gap_2()
            .border_b_1()
            .border_color(theme.border)
            .child(
                h_flex()
                    .w_full()
                    .gap_3()
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .child(
                                div()
                                    .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                                    .truncate()
                                    .child(session.title.clone()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(theme.muted_foreground)
                                    .child(subtitle),
                            ),
                    )
                    .child(
                        Button::new("session-menu")
                            .ghost()
                            .small()
                            .icon(IconName::Ellipsis)
                            .tooltip("گزینه‌های گفتگو")
                            .dropdown_menu(move |menu, _, _| {
                                let rename = page.clone();
                                let delete = page.clone();
                                menu.item(
                                    PopupMenuItem::new("تغییر نام")
                                        .icon(IconName::Pencil)
                                        .on_click(move |_, window, cx| {
                                            rename
                                                .update(cx, |this, cx| this.open_rename(window, cx))
                                                .ok();
                                        }),
                                )
                                .separator()
                                .item(
                                    PopupMenuItem::new("حذف گفتگو")
                                        .icon(IconName::Trash)
                                        .on_click(move |_, window, cx| {
                                            delete
                                                .update(cx, |this, cx| {
                                                    this.confirm_delete(window, cx)
                                                })
                                                .ok();
                                        }),
                                )
                            }),
                    ),
            )
            .child(
                h_flex()
                    .flex_wrap()
                    .gap_2()
                    .when(known, |this| {
                        this.child(
                            div().w(px(190.)).flex_shrink_0().child(
                                Select::new(&self.model_select)
                                    .small()
                                    .icon(IconName::Bot)
                                    .accessibility_label("مدل")
                                    .disabled(self.turn.is_some()),
                            ),
                        )
                        .child(
                            div().w(px(150.)).flex_shrink_0().child(
                                Select::new(&self.reasoning_select)
                                    .small()
                                    .icon(IconName::Brain)
                                    .accessibility_label("فکر کردن")
                                    .disabled(self.turn.is_some()),
                            ),
                        )
                    })
                    .child(
                        div().w(px(160.)).flex_shrink_0().child(
                            Select::new(&self.voice_select)
                                .small()
                                .icon(IconName::AudioLines)
                                .accessibility_label("صدا"),
                        ),
                    )
                    .child(
                        Switch::new("auto-speak")
                            .label("خواندن خودکار")
                            .tooltip("پاسخ‌ها هم‌زمان با نوشته‌شدن با صدا خوانده شوند")
                            .checked(session.auto_speak)
                            .on_change(cx.listener(|this, checked: &bool, _, cx| {
                                let checked = *checked;
                                this.update_current(cx, |session| session.auto_speak = checked);
                            })),
                    ),
            )
    }

    fn render_composer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let completion = self.render_completion(cx);
        let theme = cx.theme();
        let generating = self.turn.is_some();
        let busy_here = self
            .turn
            .as_ref()
            .is_some_and(|t| Some(&t.session_id) == self.selected.as_ref());
        let show_stop = busy_here || (self.voice.is_some() && !generating);
        let empty = self.composer.read(cx).value().trim().is_empty()
            || self
                .current()
                .is_some_and(|s| s.backend.unknown_kind().is_some());

        let capturing = self.speech.read(cx).status().is_capturing();

        div().w_full().px_4().pb_4().pt_2().child(
            v_flex()
                .w_full()
                .max_w(px(860.))
                .mx_auto()
                .gap_1()
                .p_2()
                .rounded(theme.radius_lg)
                .border_1()
                .border_color(theme.border)
                .bg(theme.background)
                .children(completion)
                .child(Textarea::new(&self.composer).appearance(false))
                .child(
                    h_flex()
                        .gap_2()
                        .child(SpeechButton::new(&self.speech).small())
                        .when(capturing, |this| {
                            this.child(
                                div()
                                    .ltr()
                                    .child(SpeechWaveform::new(&self.speech).small().w(px(120.))),
                            )
                        })
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .overflow_hidden()
                                .child(if capturing {
                                    meta_row(
                                        ["در حال گوش دادن… برای پایان دوباره F2 را بزنید.".into()],
                                        cx,
                                    )
                                } else {
                                    meta_row(
                                        [
                                            "Enter برای ارسال".into(),
                                            "Shift+Enter خط تازه".into(),
                                            "F2 صحبت کردن".into(),
                                        ],
                                        cx,
                                    )
                                }),
                        )
                        .map(|this| {
                            if show_stop {
                                this.child(
                                    Button::new("chat-stop")
                                        .small()
                                        .icon(IconName::Square)
                                        .label("توقف")
                                        .on_click(cx.listener(|this, _, _, cx| this.stop(cx))),
                                )
                            } else {
                                this.child(
                                    Button::new("chat-send")
                                        .primary()
                                        .small()
                                        .icon(IconName::ArrowUp)
                                        .tooltip(if generating {
                                            "پاسخ دیگری در حال نوشته‌شدن است"
                                        } else {
                                            "ارسال"
                                        })
                                        .disabled(empty || generating)
                                        .on_click(
                                            cx.listener(|this, _, window, cx| {
                                                this.send(window, cx)
                                            }),
                                        ),
                                )
                            }
                        }),
                ),
        )
    }

    fn render_conversation(&self, cx: &mut Context<Self>) -> AnyElement {
        let Some(session) = self.current() else {
            return if self.loaded {
                empty_state(
                    IconName::MessagesSquare,
                    "هنوز گفتگویی ندارید",
                    "یک گفتگوی تازه با Ollama، Claude Code، opencode یا یک سرویس سازگار با OpenAI شروع کنید.",
                    Some(
                        Button::new("empty-new-chat")
                            .primary()
                            .icon(IconName::SquarePen)
                            .label("گفتگوی تازه")
                            .on_click(
                                cx.listener(|this, _, window, cx| {
                                    this.open_new_session(window, cx)
                                }),
                            ),
                    ),
                    cx,
                )
                .into_any_element()
            } else {
                div().size_full().into_any_element()
            };
        };

        let body = if session.messages.is_empty() && self.pending_label().is_none() {
            let (icon, title) = match &session.backend {
                Backend::Ollama { model, .. } | Backend::OpenAi { model, .. } => {
                    (IconName::Bot, format!("گفتگو با {model}"))
                }
                Backend::Claude { .. } => (IconName::Terminal, "گفتگو با Claude Code".to_string()),
                Backend::OpenCode { .. } => (IconName::Terminal, "گفتگو با opencode".to_string()),
                Backend::Unknown(_) => (IconName::CircleAlert, "پاسخ‌دهندهٔ ناشناخته".to_string()),
            };
            let hint = if session.voice_profile.is_some() && session.auto_speak {
                "پیامتان را بنویسید یا با F2 بگویید. پاسخ هم‌زمان با صدای انتخاب‌شده خوانده می‌شود."
            } else {
                "پیامتان را بنویسید یا با F2 بگویید."
            };
            empty_state(icon, title, hint, None, cx).into_any_element()
        } else {
            let page = cx.entity().downgrade();
            MessageScroller::new(
                "chat-messages",
                self.scroller.clone(),
                move |ix, _, cx| match page.upgrade() {
                    Some(this) => this.read(cx).render_row(ix, &page, cx),
                    None => div().into_any_element(),
                },
            )
            .with_jump_button_label("رفتن به آخرین پیام")
            .with_list_style(StyleRefinement::default().px_4().py_4())
            .into_any_element()
        };

        v_flex()
            .size_full()
            .min_w_0()
            .child(self.render_header(session, cx))
            .child(div().flex_1().min_h_0().w_full().child(body))
            .child(self.render_composer(cx))
            .into_any_element()
    }
}

/// Start the backend's turn for `prompt` (already appended to `session`).
fn start_turn(
    session: &Session,
    settings: &Settings,
    prompt: String,
    cancel: CancelToken,
) -> async_channel::Receiver<StreamEvent> {
    // Spoken conversation only: no thinking, tools or subagent output.
    let history = session.messages.iter().filter(|m| m.is_history()).map(|m| {
        let role = if m.role == Role::User { "user" } else { "assistant" };
        (role.to_string(), m.text.clone())
    });
    match &session.backend {
        Backend::Ollama { model, think } => {
            let mut messages = vec![OllamaMessage {
                role: "system".into(),
                content: settings.prompt_for(session.backend.uses_tools()),
            }];
            messages.extend(history.map(|(role, content)| OllamaMessage { role, content }));
            OllamaClient::new(&settings.ollama_url).chat_stream(model, messages, think.clone(), cancel)
        }
        Backend::OpenAi {
            endpoint,
            model,
            reasoning,
        } => {
            let mut messages = vec![OpenAiMessage {
                role: "system".into(),
                content: settings.prompt_for(session.backend.uses_tools()),
            }];
            messages.extend(history.map(|(role, content)| OpenAiMessage { role, content }));
            let client = settings
                .endpoint(endpoint)
                .map(OpenAiClient::from_endpoint)
                .unwrap_or_else(|| OpenAiClient::new("", None));
            client.chat_stream(model, messages, reasoning.clone(), cancel)
        }
        Backend::Claude {
            cwd,
            claude_session_id,
            model,
            effort,
            ..
        } => claude::run(
            ClaudeRequest {
                bin: settings.claude_bin.clone(),
                prompt,
                cwd: cwd.clone(),
                resume: claude_session_id.clone(),
                permission_mode: (settings.claude_permission_mode != PermissionMode::Default)
                    .then(|| settings.claude_permission_mode.as_arg().to_string()),
                append_system_prompt: Some(settings.prompt_for(session.backend.uses_tools())),
                model: model.clone(),
                effort: effort.clone(),
            },
            cancel,
        ),
        Backend::OpenCode {
            cwd,
            model,
            variant,
            session_id,
        } => opencode::run(
            OpenCodeRequest {
                bin: settings.opencode_bin.clone(),
                prompt,
                cwd: cwd.clone(),
                session: session_id.clone(),
                model: model.clone(),
                variant: variant.clone(),
                system_prompt: Some(settings.prompt_for(true)),
            },
            cancel,
        ),
        Backend::Unknown(_) => {
            let (tx, rx) = async_channel::bounded(1);
            _ = tx.try_send(StreamEvent::Error("پاسخ‌دهندهٔ ناشناخته".into()));
            rx
        }
    }
}

/// Apply one stream event to the session. `parent` is the subagent the
/// event came from (`None` = the main agent).
fn apply_event(
    session: &mut Session,
    turn: &mut Turn,
    event: StreamEvent,
    parent: Option<String>,
) -> Applied {
    let mut applied = Applied::default();
    let main = parent.is_none();
    match event {
        StreamEvent::SessionId(id) => match &mut session.backend {
            Backend::Claude {
                claude_session_id, ..
            } => {
                *claude_session_id = Some(id);
                applied.save = true;
            }
            Backend::OpenCode { session_id, .. } => {
                *session_id = Some(id);
                applied.save = true;
            }
            _ => {}
        },
        StreamEvent::Commands(list) => {
            if let Backend::Claude { commands, .. } = &mut session.backend
                && *commands != list
            {
                *commands = list;
                applied.save = true;
                applied.commands_changed = true;
            }
        }
        StreamEvent::TextDelta(delta) | StreamEvent::Thinking(delta)
            if delta.is_empty() => {}
        event @ (StreamEvent::TextDelta(_) | StreamEvent::Thinking(_)) => {
            let (role, delta) = match event {
                StreamEvent::TextDelta(d) => (Role::Assistant, d),
                StreamEvent::Thinking(d) => (Role::Thinking, d),
                _ => unreachable!(),
            };
            if main {
                turn.waiting = false;
            }
            let segment = turn
                .segments
                .get(&parent)
                .and_then(|id| session.messages.iter_mut().find(|m| &m.id == id))
                .filter(|m| m.role == role);
            let id = match segment {
                Some(message) => {
                    message.text.push_str(&delta);
                    message.id.clone()
                }
                None => {
                    let message = ChatMessage::new(role, delta.clone()).with_parent(parent.clone());
                    let id = message.id.clone();
                    turn.segments.insert(parent.clone(), id.clone());
                    session.messages.push(message);
                    applied.save = true;
                    id
                }
            };
            if main && role == Role::Assistant {
                applied.spoken = Some(delta);
            }
            applied.touched = Some(id);
        }
        StreamEvent::ToolUse { name, summary } => {
            turn.segments.remove(&parent);
            if main {
                turn.waiting = true;
                turn.after_tool = true;
            }
            let line = if summary.trim().is_empty() {
                name
            } else {
                format!("{name}: {}", one_line(&summary, 200))
            };
            let message = ChatMessage::new(Role::Tool, line).with_parent(parent.clone());
            turn.tools
                .entry(parent)
                .or_default()
                .push_back(message.id.clone());
            applied.touched = Some(message.id.clone());
            session.messages.push(message);
            applied.save = true;
        }
        StreamEvent::ToolResult { summary, is_error } => {
            let pending = turn.tools.get_mut(&parent).and_then(|q| q.pop_front());
            let target = match pending {
                Some(id) => session.messages.iter_mut().find(|m| m.id == id),
                None => session
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == Role::Tool && m.parent == parent),
            };
            if let Some(tool) = target {
                let summary = one_line(&summary, 200);
                if is_error {
                    tool.text.push_str(&format!("\n✗ {summary}"));
                } else if !summary.is_empty() {
                    tool.text.push_str(&format!("\n→ {summary}"));
                }
                applied.touched = Some(tool.id.clone());
            }
        }
        StreamEvent::TaskStarted {
            tool_use_id,
            parent: owner,
            description,
            agent_type,
            prompt,
        } => {
            let owner = owner.or(parent);
            turn.segments.remove(&owner);
            if owner.is_none() {
                turn.waiting = true;
                turn.after_tool = true;
            }
            if find_task(session, &tool_use_id).is_none() {
                let mut message = ChatMessage::new(Role::Task, prompt).with_parent(owner);
                message.task = Some(TaskInfo {
                    tool_use_id,
                    description,
                    agent_type,
                    status: "running".into(),
                    summary: None,
                });
                applied.touched = Some(message.id.clone());
                session.messages.push(message);
                applied.save = true;
            }
        }
        StreamEvent::TaskUpdated {
            tool_use_id,
            status,
            summary,
        } => {
            if let Some(message) = find_task(session, &tool_use_id) {
                if let Some(task) = message.task.as_mut() {
                    // A finished task stays finished; late "running" patches are ignored.
                    if !(task_done(&task.status) && !task_done(&status)) {
                        task.status = status;
                    }
                    if let Some(summary) = summary.filter(|s| !s.trim().is_empty()) {
                        task.summary = Some(summary);
                    }
                }
                applied.touched = Some(message.id.clone());
                applied.save = true;
            }
        }
        StreamEvent::Sub { parent, event } => {
            return apply_event(session, turn, *event, Some(parent));
        }
        StreamEvent::Error(message) => {
            session
                .messages
                .push(ChatMessage::new(Role::Error, message.clone()));
            applied.error = Some(message);
            applied.finished = true;
            applied.save = true;
        }
        StreamEvent::Done => applied.finished = true,
    }
    applied
}

fn find_task<'a>(session: &'a mut Session, tool_use_id: &str) -> Option<&'a mut ChatMessage> {
    session.messages.iter_mut().find(|m| {
        m.role == Role::Task
            && m.task
                .as_ref()
                .is_some_and(|t| t.tool_use_id == tool_use_id)
    })
}

/// Tasks still marked running when a turn ends were stopped with it.
fn stop_running_tasks(session: &mut Session) {
    for task in session.messages.iter_mut().filter_map(|m| m.task.as_mut()) {
        if !task_done(&task.status) {
            task.status = "stopped".into();
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TaskState {
    Running,
    Done,
    Failed,
}

fn task_state(status: &str) -> TaskState {
    match status {
        "completed" | "success" | "done" => TaskState::Done,
        "failed" | "error" | "killed" | "stopped" | "cancelled" => TaskState::Failed,
        _ => TaskState::Running,
    }
}

fn task_done(status: &str) -> bool {
    task_state(status) != TaskState::Running
}

/// Indices of the messages shown as their own rows: the main conversation,
/// plus orphans whose subagent node is missing.
fn top_level(session: &Session) -> Vec<usize> {
    let tasks: HashSet<&str> = session
        .messages
        .iter()
        .filter_map(|m| m.task.as_ref().map(|t| t.tool_use_id.as_str()))
        .collect();
    session
        .messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.parent.as_deref().is_none_or(|p| !tasks.contains(p)))
        .map(|(ix, _)| ix)
        .collect()
}

/// The row that shows message `id` (walking up through its subagent nodes).
fn root_row(session: &Session, id: &str) -> Option<usize> {
    let mut current = session.messages.iter().find(|m| m.id == id)?;
    for _ in 0..16 {
        let Some(parent) = current.parent.as_deref() else {
            break;
        };
        let Some(owner) = session
            .messages
            .iter()
            .find(|m| m.task.as_ref().is_some_and(|t| t.tool_use_id == parent))
        else {
            break;
        };
        current = owner;
    }
    let ix = session.messages.iter().position(|m| m.id == current.id)?;
    top_level(session).iter().position(|&row| row == ix)
}

/// Open/closed chevron; a closed one points along the reading direction.
fn chevron(expanded: bool, cx: &App) -> IconName {
    if expanded {
        IconName::ChevronDown
    } else if rtl::is_rtl(cx) {
        IconName::ChevronLeft
    } else {
        IconName::ChevronRight
    }
}

fn reasoning_scale(backend: &Backend) -> Option<ReasoningScale> {
    match backend {
        Backend::Ollama { .. } => Some(ReasoningScale::Ollama),
        Backend::Claude { .. } => Some(ReasoningScale::Claude),
        Backend::OpenAi { .. } => Some(ReasoningScale::OpenAi),
        Backend::OpenCode { .. } => Some(ReasoningScale::OpenCode),
        Backend::Unknown(_) => None,
    }
}

fn reasoning_of(backend: &Backend) -> Option<String> {
    match backend {
        Backend::Ollama { think: r, .. }
        | Backend::Claude { effort: r, .. }
        | Backend::OpenAi { reasoning: r, .. }
        | Backend::OpenCode { variant: r, .. } => r.clone(),
        Backend::Unknown(_) => None,
    }
}

/// Short label for the session list.
fn backend_label(backend: &Backend, settings: &Settings) -> String {
    match backend {
        Backend::Ollama { model, .. } => model.clone(),
        Backend::Claude { .. } => "Claude Code".into(),
        Backend::OpenAi {
            endpoint, model, ..
        } => match settings.endpoint(endpoint) {
            Some(e) => format!("{} · {model}", e.name),
            None => model.clone(),
        },
        Backend::OpenCode { model, .. } => match model {
            Some(model) => format!("opencode · {model}"),
            None => "opencode".into(),
        },
        Backend::Unknown(_) => backend.unknown_kind().unwrap_or_default(),
    }
}

/// Markdown for one message. Code blocks stay left-to-right in RTL.
fn markdown(id: &str, text: &str, streaming: bool) -> TextView {
    TextView::markdown(
        ElementId::Name(format!("md-{id}").into()),
        SharedString::from(text.to_string()),
    )
    .selectable(true)
    .stream_fade(streaming)
    .style(TextViewStyle {
        code_block: StyleRefinement::default().ltr(),
        ..Default::default()
    })
}

impl Focusable for ChatsPage {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ChatsPage {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        h_flex()
            .id("chats-page")
            .key_context(CONTEXT)
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::toggle_push_to_talk))
            .size_full()
            .items_stretch()
            .bg(cx.theme().background)
            .child(self.render_session_list(cx))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(self.render_conversation(cx)),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn() -> Turn {
        Turn {
            session_id: String::new(),
            cancel: CancelToken::new(),
            task: Task::ready(()),
            segments: HashMap::new(),
            tools: HashMap::new(),
            waiting: true,
            after_tool: false,
            last_save: Instant::now(),
        }
    }

    fn sub(parent: &str, event: StreamEvent) -> StreamEvent {
        StreamEvent::Sub {
            parent: parent.into(),
            event: Box::new(event),
        }
    }

    #[test]
    fn subagent_tree_and_speech() {
        let mut session = Session::new(
            Backend::Claude {
                cwd: "/tmp".into(),
                claude_session_id: None,
                model: None,
                effort: None,
                commands: Vec::new(),
            },
            None,
            true,
        );
        session.messages.push(ChatMessage::new(Role::User, "hi"));
        let mut turn = turn();
        let mut spoken = String::new();
        let events = vec![
            StreamEvent::Thinking("hmm".into()),
            StreamEvent::TextDelta("Sure, ".into()),
            StreamEvent::TaskStarted {
                tool_use_id: "A".into(),
                parent: None,
                description: "look".into(),
                agent_type: "Explore".into(),
                prompt: "find it".into(),
            },
            sub("A", StreamEvent::TextDelta("sub text".into())),
            sub("A", StreamEvent::ToolUse { name: "Bash".into(), summary: "ls".into() }),
            sub("A", StreamEvent::ToolResult { summary: "a.txt".into(), is_error: false }),
            StreamEvent::TaskStarted {
                tool_use_id: "B".into(),
                parent: Some("A".into()),
                description: "nested".into(),
                agent_type: "general-purpose".into(),
                prompt: String::new(),
            },
            sub("B", StreamEvent::Thinking("deep".into())),
            StreamEvent::TaskUpdated { tool_use_id: "B".into(), status: "completed".into(), summary: Some("ok".into()) },
            StreamEvent::TaskUpdated { tool_use_id: "A".into(), status: "completed".into(), summary: None },
            StreamEvent::TaskUpdated { tool_use_id: "A".into(), status: "running".into(), summary: Some("found".into()) },
            StreamEvent::TextDelta("done.".into()),
            StreamEvent::Commands(vec!["clear".into()]),
            StreamEvent::Done,
        ];
        let mut finished = false;
        for event in events {
            let applied = apply_event(&mut session, &mut turn, event, None);
            if let Some(s) = applied.spoken {
                spoken.push_str(&s);
            }
            finished |= applied.finished;
        }
        assert!(finished);
        assert_eq!(spoken, "Sure, done.");
        let roles: Vec<(Role, Option<&str>)> = session
            .messages
            .iter()
            .map(|m| (m.role, m.parent.as_deref()))
            .collect();
        assert_eq!(
            roles,
            vec![
                (Role::User, None),
                (Role::Thinking, None),
                (Role::Assistant, None),
                (Role::Task, None),
                (Role::Assistant, Some("A")),
                (Role::Tool, Some("A")),
                (Role::Task, Some("A")),
                (Role::Thinking, Some("B")),
                (Role::Assistant, None),
            ]
        );
        assert_eq!(session.messages[5].text, "Bash: ls\n→ a.txt");
        let a = session.messages[3].task.as_ref().unwrap();
        assert_eq!((a.status.as_str(), a.summary.as_deref()), ("completed", Some("found")));
        // Only main-conversation user/assistant text goes back as history.
        let history: Vec<&str> = session
            .messages
            .iter()
            .filter(|m| m.is_history())
            .map(|m| m.text.as_str())
            .collect();
        assert_eq!(history, vec!["hi", "Sure, ", "done."]);
        // Rows: subagent messages live inside their task node.
        assert_eq!(top_level(&session), vec![0, 1, 2, 3, 8]);
        let nested = session.messages[7].id.clone();
        assert_eq!(root_row(&session, &nested), Some(3));
        let Backend::Claude { commands, .. } = &session.backend else {
            unreachable!()
        };
        assert_eq!(commands, &vec!["clear".to_string()]);
    }
}
