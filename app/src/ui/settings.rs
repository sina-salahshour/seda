//! Settings page: sidecar, Ollama, Claude Code, conversation and appearance.
//! Every control writes straight to `AppSettings`, which saves `settings.json`.

use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Sizable as _, WindowExt as _,
    button::Button,
    form::{Field, Form},
    h_flex,
    input::{Input, InputEvent, InputState, Textarea, TextareaState},
    notification::Notification,
    radio::RadioGroup,
    scroll::ScrollableElement as _,
    select::{Select, SelectEvent, SelectItem, SelectState},
    switch::Switch,
    v_flex,
};
use gpui_kit::{
    AppContext as _, Context, Entity, InteractiveElement as _, IntoElement, ParentElement as _,
    Render, SharedString, Styled as _, Subscription, Window, div, px,
};

use crate::rtl;
use crate::sidecar::{SidecarClient, SidecarGlobal, SidecarStatus};
use crate::store::{AppSettings, PermissionMode, Settings, ThemePreference, data_dir};
use crate::ui::common::page_header;

/// A labelled `Select` choice with a typed value.
#[derive(Clone)]
pub struct Choice<T: Clone + PartialEq + 'static> {
    pub value: T,
    pub label: SharedString,
}

impl<T: Clone + PartialEq + 'static> SelectItem for Choice<T> {
    type Value = T;

    fn title(&self) -> SharedString {
        self.label.clone()
    }

    fn value(&self) -> &T {
        &self.value
    }
}

const SPEEDS: [f32; 6] = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3];

pub struct SettingsPage {
    sidecar_url: Entity<InputState>,
    sidecar_project: Entity<InputState>,
    ollama_url: Entity<InputState>,
    claude_bin: Entity<InputState>,
    system_prompt: Entity<TextareaState>,
    permission: Entity<SelectState<Vec<Choice<PermissionMode>>>>,
    speed: Entity<SelectState<Vec<Choice<u32>>>>,
    input_device: Entity<SelectState<Vec<Choice<Option<String>>>>>,
    unloading: bool,
    _subscriptions: Vec<Subscription>,
}

impl SettingsPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let settings = AppSettings::get(cx).clone();
        let mut subscriptions = Vec::new();

        let mut text_input = |value: String,
                              apply: fn(&mut Settings, String),
                              window: &mut Window,
                              cx: &mut Context<Self>| {
            let state = cx.new(|cx| InputState::new(window, cx).default_value(value));
            subscriptions.push(cx.subscribe_in(
                &state,
                window,
                move |_, state, event: &InputEvent, _, cx| {
                    if matches!(event, InputEvent::Change) {
                        let value = state.read(cx).value().to_string();
                        AppSettings::update(cx, |s| apply(s, value));
                    }
                },
            ));
            state
        };

        let sidecar_url = text_input(
            settings.sidecar_url.clone(),
            |s, v| s.sidecar_url = v.trim().to_string(),
            window,
            cx,
        );
        let sidecar_project = text_input(
            settings.sidecar_project.to_string_lossy().into_owned(),
            |s, v| s.sidecar_project = v.trim().into(),
            window,
            cx,
        );
        let ollama_url = text_input(
            settings.ollama_url.clone(),
            |s, v| s.ollama_url = v.trim().to_string(),
            window,
            cx,
        );
        let claude_bin = text_input(
            settings.claude_bin.clone(),
            |s, v| s.claude_bin = v.trim().to_string(),
            window,
            cx,
        );

        let system_prompt = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(3, 8)
                .default_value(settings.system_prompt.clone())
        });
        subscriptions.push(cx.subscribe_in(
            &system_prompt,
            window,
            |_, state, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    let value = state.read(cx).value().to_string();
                    AppSettings::update(cx, |s| s.system_prompt = value);
                }
            },
        ));

        let permission_choices: Vec<Choice<PermissionMode>> = PermissionMode::ALL
            .iter()
            .map(|mode| Choice {
                value: *mode,
                label: format!("{} — {}", mode.label(), mode.as_arg()).into(),
            })
            .collect();
        let permission_ix = PermissionMode::ALL
            .iter()
            .position(|m| *m == settings.claude_permission_mode)
            .unwrap_or(0);
        let permission = cx.new(|cx| {
            SelectState::new(
                permission_choices,
                Some(gpui_kit::component::IndexPath::new(permission_ix)),
                window,
                cx,
            )
        });
        subscriptions.push(cx.subscribe_in(
            &permission,
            window,
            |_, _, event: &SelectEvent<Vec<Choice<PermissionMode>>>, _, cx| {
                let SelectEvent::Confirm(Some(mode)) = event else {
                    return;
                };
                let mode = *mode;
                AppSettings::update(cx, |s| s.claude_permission_mode = mode);
            },
        ));

        // Speeds as percent, so the value is `Eq`-friendly.
        let speed_choices: Vec<Choice<u32>> = SPEEDS
            .iter()
            .map(|speed| Choice {
                value: (speed * 100.).round() as u32,
                label: format!("×{speed:.1}").into(),
            })
            .collect();
        let speed_ix = SPEEDS
            .iter()
            .position(|s| (s - settings.tts_speed).abs() < 0.01)
            .unwrap_or(2);
        let speed = cx.new(|cx| {
            SelectState::new(
                speed_choices,
                Some(gpui_kit::component::IndexPath::new(speed_ix)),
                window,
                cx,
            )
        });
        subscriptions.push(cx.subscribe_in(
            &speed,
            window,
            |_, _, event: &SelectEvent<Vec<Choice<u32>>>, _, cx| {
                let SelectEvent::Confirm(Some(percent)) = event else {
                    return;
                };
                let speed = *percent as f32 / 100.;
                AppSettings::update(cx, |s| s.tts_speed = speed);
            },
        ));

        // Microphone: the list is probed off the UI thread (ALSA opens devices).
        let current_device = settings.input_device.clone();
        let device_choices = |names: Vec<String>| -> Vec<Choice<Option<String>>> {
            std::iter::once(Choice {
                value: None,
                label: "پیش‌فرض سیستم".into(),
            })
            .chain(names.into_iter().map(|name| Choice {
                label: name.clone().into(),
                value: Some(name),
            }))
            .collect()
        };
        let initial: Vec<String> = current_device.iter().cloned().collect();
        let initial_ix = if current_device.is_some() { 1 } else { 0 };
        let input_device = cx.new(|cx| {
            SelectState::new(
                device_choices(initial),
                Some(gpui_kit::component::IndexPath::new(initial_ix)),
                window,
                cx,
            )
        });
        subscriptions.push(cx.subscribe_in(
            &input_device,
            window,
            |_, _, event: &SelectEvent<Vec<Choice<Option<String>>>>, _, cx| {
                let SelectEvent::Confirm(Some(device)) = event else {
                    return;
                };
                let device = device.clone();
                AppSettings::update(cx, |s| s.input_device = device);
            },
        ));
        {
            let input_device = input_device.clone();
            cx.spawn_in(window, async move |_, cx| {
                let names = cx
                    .background_spawn(async move { crate::audio::input_device_names() })
                    .await;
                _ = cx.update(|window, cx| {
                    let selected = AppSettings::get(cx).input_device.clone();
                    let mut names = names;
                    if let Some(selected) = &selected
                        && !names.contains(selected)
                    {
                        names.push(selected.clone());
                    }
                    input_device.update(cx, |state, cx| {
                        state.set_items(device_choices(names), window, cx);
                        state.set_selected_value(&selected, window, cx);
                    });
                });
            })
            .detach();
        }

        let sidecar = SidecarGlobal::get(cx);
        subscriptions.push(cx.observe(&sidecar, |_, _, cx| cx.notify()));

        Self {
            sidecar_url,
            sidecar_project,
            ollama_url,
            claude_bin,
            system_prompt,
            permission,
            speed,
            input_device,
            unloading: false,
            _subscriptions: subscriptions,
        }
    }

    fn unload_models(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.unloading = true;
        cx.notify();
        let client = SidecarClient::from_settings(cx);
        cx.spawn_in(window, async move |this, cx| {
            let result = cx.background_spawn(async move { client.unload() }).await;
            _ = this.update_in(cx, |this, window, cx| {
                this.unloading = false;
                match result {
                    Ok(()) => window.push_notification(
                        Notification::success("مدل‌ها از حافظهٔ GPU خارج شدند."),
                        cx,
                    ),
                    Err(err) => {
                        window.push_notification(Notification::error(format!("{err:#}")), cx)
                    }
                }
                SidecarGlobal::get(cx).update(cx, |s, cx| s.retry(cx));
                cx.notify();
            });
        })
        .detach();
    }

    fn section(
        title: &'static str,
        description: &'static str,
        cx: &Context<Self>,
    ) -> gpui_kit::Div {
        v_flex()
            .gap_1()
            .pt_6()
            .pb_3()
            .child(
                div()
                    .text_base()
                    .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                    .child(title),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(description),
            )
    }

    /// Status items, each its own element (see `meta_row`).
    fn sidecar_summary(&self, cx: &Context<Self>) -> Vec<SharedString> {
        let sidecar = SidecarGlobal::get(cx);
        let sidecar = sidecar.read(cx);
        let mut items: Vec<SharedString> = Vec::new();
        match sidecar.status() {
            SidecarStatus::Checking => items.push("در حال بررسی…".into()),
            SidecarStatus::Starting => items.push("در حال راه‌اندازی…".into()),
            SidecarStatus::Offline(err) => {
                items.push("آفلاین".into());
                items.push(err.clone().into());
            }
            SidecarStatus::Online(health) => {
                items.push("آنلاین".into());
                items.push(health.device.to_uppercase().into());
                if let Some(gpu) = &health.gpu {
                    items.push(gpu.name.clone().into());
                    items.push(
                        format!("VRAM {} / {} MB", gpu.vram_used_mb, gpu.vram_total_mb).into(),
                    );
                }
                for engine in &health.engines {
                    items.push(if engine.loaded {
                        format!("{} (بارگذاری‌شده)", engine.name).into()
                    } else {
                        engine.name.clone().into()
                    });
                }
                if health.stt_loaded {
                    items.push("Whisper (بارگذاری‌شده)".into());
                }
                if let Some(busy) = &health.busy {
                    items.push(crate::ui::common::busy_label(busy).into());
                }
            }
        }
        if sidecar.owns_process() {
            items.push("اجرا شده توسط صدا".into());
        }
        items
    }
}

impl Render for SettingsPage {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = AppSettings::get(cx).clone();
        let online = SidecarGlobal::get(cx).read(cx).is_online();
        let theme_ix = match settings.theme {
            ThemePreference::System => 0,
            ThemePreference::Light => 1,
            ThemePreference::Dark => 2,
        };

        v_flex()
            .size_full()
            .child(page_header(
                "تنظیمات",
                Some("تغییرها بلافاصله ذخیره می‌شوند.".into()),
                div(),
                cx,
            ))
            .child(
                div()
                    .id("settings-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scrollbar()
                    .child(
                        v_flex()
                            .w_full()
                            .max_w(px(760.))
                            .px_6()
                            .pb_8()
                            // Sidecar
                            .child(Self::section(
                                "سایدکار",
                                "سرور محلی پایتون برای ساخت صدا، آموزش و تشخیص گفتار.",
                                cx,
                            ))
                            .child(
                                Form::new()
                                    .child(
                                        Field::new()
                                            .label("وضعیت")
                                            .child(
                                                h_flex()
                                                    .gap_2()
                                                    .w_full()
                                                    .child(
                                                        crate::ui::common::meta_row(self.sidecar_summary(cx), cx)
                                                            .flex_1()
                                                            .text_sm(),
                                                    )
                                                    .child(
                                                        Button::new("sidecar-check")
                                                            .small()
                                                            .label("بررسی دوباره")
                                                            .on_click(|_, _, cx| {
                                                                SidecarGlobal::get(cx)
                                                                    .update(cx, |s, cx| s.retry(cx));
                                                            }),
                                                    )
                                                    .child(
                                                        Button::new("sidecar-unload")
                                                            .small()
                                                            .label("آزاد کردن GPU")
                                                            .disabled(!online)
                                                            .loading(self.unloading)
                                                            .on_click(cx.listener(
                                                                |this, _, window, cx| {
                                                                    this.unload_models(window, cx)
                                                                },
                                                            )),
                                                    ),
                                            ),
                                    )
                                    .child(
                                        Field::new()
                                            .label("نشانی")
                                            .child(div().ltr().w_full().child(Input::new(&self.sidecar_url))),
                                    )
                                    .child(
                                        Field::new()
                                            .label("پوشهٔ پروژه")
                                            .description("سایدکار با uv از همین پوشه اجرا می‌شود.")
                                            .child(div().ltr().w_full().child(Input::new(&self.sidecar_project))),
                                    )
                                    .child(
                                        Field::new().label_indent(false).child(
                                            Switch::new("sidecar-autostart")
                                                .label("اگر در دسترس نبود، خودکار اجرا شود")
                                                .checked(settings.sidecar_autostart)
                                                .on_change(|checked, _, cx| {
                                                    let checked = *checked;
                                                    AppSettings::update(cx, |s| {
                                                        s.sidecar_autostart = checked
                                                    });
                                                }),
                                        ),
                                    ),
                            )
                            // Models
                            .child(Self::section(
                                "مدل‌های گفتگو",
                                "Ollama برای مدل‌های محلی و Claude Code برای کار روی پروژه‌ها.",
                                cx,
                            ))
                            .child(
                                Form::new()
                                    .child(
                                        Field::new()
                                            .label("نشانی Ollama")
                                            .child(div().ltr().w_full().child(Input::new(&self.ollama_url))),
                                    )
                                    .child(
                                        Field::new()
                                            .label("فایل اجرایی claude")
                                            .child(div().ltr().w_full().child(Input::new(&self.claude_bin))),
                                    )
                                    .child(
                                        Field::new()
                                            .label("مجوزهای Claude Code")
                                            .description("حالت --permission-mode برای هر پیام.")
                                            .child(Select::new(&self.permission)),
                                    ),
                            )
                            // Conversation
                            .child(Self::section(
                                "گفتگو",
                                "پیام سیستمی و رفتار صدا در گفتگوها.",
                                cx,
                            ))
                            .child(
                                Form::new()
                                    .child(
                                        Field::new()
                                            .label("دستور سیستمی")
                                            .description("برای Ollama پیام سیستمی و برای Claude Code با --append-system-prompt.")
                                            .child(Textarea::new(&self.system_prompt)),
                                    )
                                    .child(
                                        Field::new()
                                            .label("میکروفون")
                                            .description("برای ضبط نمونه‌ها و صحبت در گفتگو.")
                                            .child(div().ltr().w_full().child(Select::new(&self.input_device))),
                                    )
                                    .child(
                                        Field::new()
                                            .label("سرعت خواندن")
                                            .child(div().w(px(160.)).child(Select::new(&self.speed))),
                                    )
                                    .child(
                                        Field::new().label_indent(false).child(
                                            Switch::new("default-auto-speak")
                                                .label("پاسخ‌ها در گفتگوهای تازه خوانده شوند")
                                                .checked(settings.default_auto_speak)
                                                .on_change(|checked, _, cx| {
                                                    let checked = *checked;
                                                    AppSettings::update(cx, |s| {
                                                        s.default_auto_speak = checked
                                                    });
                                                }),
                                        ),
                                    )
                                    .child(
                                        Field::new().label_indent(false).child(
                                            Switch::new("ptt-auto-send")
                                                .label("بعد از صحبت، پیام خودکار فرستاده شود")
                                                .checked(settings.ptt_auto_send)
                                                .on_change(|checked, _, cx| {
                                                    let checked = *checked;
                                                    AppSettings::update(cx, |s| {
                                                        s.ptt_auto_send = checked
                                                    });
                                                }),
                                        ),
                                    ),
                            )
                            // Appearance
                            .child(Self::section("ظاهر", "جهت و رنگ‌بندی برنامه.", cx))
                            .child(
                                Form::new()
                                    .child(
                                        Field::new().label_indent(false).child(
                                            Switch::new("rtl")
                                                .label("چیدمان راست‌به‌چپ")
                                                .checked(settings.rtl)
                                                .on_change(|checked, window, cx| {
                                                    let checked = *checked;
                                                    AppSettings::update(cx, |s| s.rtl = checked);
                                                    rtl::apply(cx);
                                                    window.refresh();
                                                }),
                                        ),
                                    )
                                    .child(
                                        Field::new().label("پوسته").child(
                                            RadioGroup::new("theme")
                                                .children(["هماهنگ با سیستم", "روشن", "تیره"])
                                                .selected_index(Some(theme_ix))
                                                .on_change(|ix, window, cx| {
                                                    let theme = match *ix {
                                                        1 => ThemePreference::Light,
                                                        2 => ThemePreference::Dark,
                                                        _ => ThemePreference::System,
                                                    };
                                                    AppSettings::update(cx, |s| s.theme = theme);
                                                    crate::apply_theme(Some(window), cx);
                                                }),
                                        ),
                                    )
                                    .child(
                                        Field::new().label("پوشهٔ داده").child(
                                            div()
                                                .ltr()
                                                .text_sm()
                                                .text_color(cx.theme().muted_foreground)
                                                .child(data_dir().to_string_lossy().into_owned()),
                                        ),
                                    ),
                            ),
                    ),
            )
    }
}
