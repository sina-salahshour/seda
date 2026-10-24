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
    InteractiveElement as _, IntoElement, KeyBinding, ParentElement as _, Render, SharedString,
    StatefulInteractiveElement as _, StyleRefinement, Styled as _, Subscription, Task, WeakEntity,
    Window, actions, div, px,
};

use crate::audio::{Player, SpeechQueue};
use crate::llm::claude::{self, ClaudeRequest};
use crate::llm::ollama::{OllamaClient, OllamaMessage};
use crate::llm::{CancelToken, StreamEvent};
use crate::rtl;
use crate::sidecar::{Profile, SidecarClient, SidecarGlobal, TtsRun};
use crate::store::{AppSettings, Backend, ChatMessage, PermissionMode, Role, Session};
use crate::ui::common::{empty_state, meta_row};

use new_session::NewSessionForm;
use util::{Choices, NO_VOICE, model_choices, one_line, relative_time, voice_choices};

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
    /// The assistant message the next text delta extends; `None` after a
    /// tool call, so the text that follows starts a new message.
    segment: Option<String>,
    /// No text yet since the turn (or the last tool call) started.
    waiting: bool,
    after_tool: bool,
    last_save: Instant,
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
    profiles: Vec<Profile>,
    models: Vec<String>,
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
                    InputEvent::Change => cx.notify(),
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
                    this.update_current(cx, |session| {
                        if let Backend::Ollama { model } = &mut session.backend {
                            *model = value;
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
            profiles: Vec::new(),
            models: Vec::new(),
            sidecar_online: sidecar.read(cx).is_online(),
            turn: None,
            voice: None,
            voice_generation: 0,
            saver,
            _tasks: vec![save_task],
            _subscriptions: subscriptions,
        };
        page.load_sessions(window, cx);
        page.load_models(window, cx);
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

    fn load_models(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let client = OllamaClient::from_settings(cx);
        let task = cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { client.models() }).await;
            this.update_in(cx, |this, window, cx| {
                if let Ok(models) = result {
                    this.models = models;
                    this.refresh_header_selects(window, cx);
                }
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
        let model = match &session.backend {
            Backend::Ollama { model } => Some(model.clone()),
            Backend::Claude { .. } => None,
        };
        let voices = voice_choices(&self.profiles, Some(&voice));
        self.voice_select.update(cx, |select, cx| {
            select.set_items(Choices::new(voices), window, cx);
            select.set_selected_value(&voice, window, cx);
        });
        if let Some(model) = model {
            let models = model_choices(&self.models, Some(&model));
            self.model_select.update(cx, |select, cx| {
                select.set_items(Choices::new(models), window, cx);
                select.set_selected_value(&model, window, cx);
            });
        }
    }

    fn open_new_session(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let page = cx.entity().downgrade();
        let form = cx.new(|cx| NewSessionForm::new(page, window, cx));
        window.open_dialog(cx, move |dialog, _, _| {
            dialog
                .title("گفتگوی تازه")
                .w(px(480.))
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

    fn row_count(&self) -> usize {
        self.current().map_or(0, |s| s.messages.len()) + usize::from(self.pending_label().is_some())
    }

    /// Keep the message list's row count in step with the open session.
    fn sync_scroller(&mut self, cx: &mut Context<Self>) {
        let want = self.row_count();
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
        });
    }

    fn can_speak(&self, cx: &App) -> bool {
        self.sidecar_online
            && cx.has_global::<Player>()
            && self.current().is_some_and(|s| s.voice_profile.is_some())
    }

    fn render_row(&self, ix: usize, page: &WeakEntity<Self>, cx: &App) -> AnyElement {
        let Some(session) = self.current() else {
            return div().into_any_element();
        };
        let theme = cx.theme();
        let Some(message) = session.messages.get(ix) else {
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
        let streaming = self
            .turn
            .as_ref()
            .is_some_and(|turn| turn.segment.as_ref() == Some(&message.id));

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
        session
            .messages
            .push(ChatMessage::new(Role::User, text.clone()));
        session.title_from_first_message();
        session.updated_at = crate::store::now();

        let cancel = CancelToken::new();
        let events = match &session.backend {
            Backend::Ollama { model } => {
                let mut messages = vec![OllamaMessage {
                    role: "system".into(),
                    content: settings.system_prompt.clone(),
                }];
                messages.extend(session.messages.iter().filter_map(|m| {
                    let role = match m.role {
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool | Role::Error => return None,
                    };
                    Some(OllamaMessage {
                        role: role.into(),
                        content: m.text.clone(),
                    })
                }));
                OllamaClient::from_settings(cx).chat_stream(model, messages, cancel.clone())
            }
            Backend::Claude {
                cwd,
                claude_session_id,
            } => claude::run(
                ClaudeRequest {
                    bin: settings.claude_bin.clone(),
                    prompt: text,
                    cwd: cwd.clone(),
                    resume: claude_session_id.clone(),
                    permission_mode: (settings.claude_permission_mode != PermissionMode::Default)
                        .then(|| settings.claude_permission_mode.as_arg().to_string()),
                    append_system_prompt: Some(settings.system_prompt.clone()),
                },
                cancel.clone(),
            ),
        };
        let voice = session.voice_profile.clone().filter(|_| session.auto_speak);
        self.sessions
            .sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        self.save_by_id(&id);
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
            segment: None,
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
        let mut save = false;
        let mut more = true;
        match event {
            StreamEvent::SessionId(id) => {
                if let Backend::Claude {
                    claude_session_id, ..
                } = &mut session.backend
                {
                    *claude_session_id = Some(id);
                    save = true;
                }
            }
            StreamEvent::TextDelta(delta) => {
                turn.waiting = false;
                let segment = turn
                    .segment
                    .as_ref()
                    .and_then(|id| session.messages.iter_mut().find(|m| &m.id == id));
                match segment {
                    Some(message) => message.text.push_str(&delta),
                    None => {
                        let message = ChatMessage::new(Role::Assistant, delta.clone());
                        turn.segment = Some(message.id.clone());
                        session.messages.push(message);
                    }
                }
                if let Some(voice) = self.voice.as_mut().filter(|v| !v.finished) {
                    voice.queue.push(&delta);
                }
                if turn.last_save.elapsed() >= STREAM_SAVE_INTERVAL {
                    save = true;
                }
            }
            StreamEvent::ToolUse { name, summary } => {
                turn.segment = None;
                turn.waiting = true;
                turn.after_tool = true;
                let line = if summary.trim().is_empty() {
                    name
                } else {
                    format!("{name}: {}", one_line(&summary, 200))
                };
                session.messages.push(ChatMessage::new(Role::Tool, line));
                save = true;
            }
            StreamEvent::ToolResult { summary, is_error } => {
                if let Some(last) = session
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == Role::Tool)
                {
                    let summary = one_line(&summary, 200);
                    if is_error {
                        last.text.push_str(&format!("\n✗ {summary}"));
                    } else if !summary.is_empty() {
                        last.text.push_str(&format!("\n→ {summary}"));
                    }
                }
            }
            StreamEvent::Error(message) => {
                session
                    .messages
                    .push(ChatMessage::new(Role::Error, message.clone()));
                window.push_notification(Notification::error(message), cx);
                more = false;
            }
            StreamEvent::Done => more = false,
        }
        if save {
            turn.last_save = Instant::now();
            let id = turn.session_id.clone();
            self.save_by_id(&id);
        }
        self.sync_scroller(cx);
        cx.notify();
        more
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
        }
        self.save_by_id(&turn.session_id);
        self.sync_scroller(cx);
        cx.notify();
    }

    /// Stop generating and speaking.
    fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(turn) = self.turn.take() {
            turn.cancel.cancel();
            drop(turn.task);
            self.save_by_id(&turn.session_id);
        }
        self.stop_voice();
        self.sync_scroller(cx);
        cx.notify();
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
        let rows = self.sessions.iter().map(|session| {
            let id = session.id.clone();
            let backend = match &session.backend {
                Backend::Ollama { model } => model.clone(),
                Backend::Claude { .. } => "Claude Code".to_string(),
            };
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
        let subtitle = match &session.backend {
            Backend::Ollama { .. } => h_flex()
                .gap_1()
                .child(Icon::new(IconName::Bot).xsmall())
                .child("Ollama"),
            Backend::Claude { cwd, .. } => h_flex()
                .gap_1()
                .min_w_0()
                .child(Icon::new(IconName::Terminal).xsmall())
                .child(div().flex_shrink_0().child("Claude Code"))
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
                ),
        };
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
                    .when(matches!(session.backend, Backend::Ollama { .. }), |this| {
                        this.child(
                            div().w(px(170.)).flex_shrink_0().child(
                                Select::new(&self.model_select)
                                    .small()
                                    .icon(IconName::Bot)
                                    .accessibility_label("مدل")
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
        let theme = cx.theme();
        let generating = self.turn.is_some();
        let busy_here = self
            .turn
            .as_ref()
            .is_some_and(|t| Some(&t.session_id) == self.selected.as_ref());
        let show_stop = busy_here || (self.voice.is_some() && !generating);
        let empty = self.composer.read(cx).value().trim().is_empty();
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
                    "یک گفتگوی تازه با یکی از مدل‌های Ollama یا با Claude Code شروع کنید.",
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
                Backend::Ollama { model } => (IconName::Bot, format!("گفتگو با {model}")),
                Backend::Claude { .. } => (IconName::Terminal, "گفتگو با Claude Code".to_string()),
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
