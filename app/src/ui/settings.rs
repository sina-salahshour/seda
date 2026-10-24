//! Settings page: sidecar, Ollama, Claude Code, conversation and appearance.
//! Every control writes straight to `AppSettings`, which saves `settings.json`.

use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariant, ButtonVariants as _},
    dialog::DialogFooter,
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
use gpui_kit::assets::IconName;
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AppContext as _, Context, Entity, InteractiveElement as _, IntoElement, ParentElement as _,
    Render, SharedString, Styled as _, Subscription, Window, div, px,
};

use crate::rtl;
use crate::sidecar::{SidecarClient, SidecarGlobal, SidecarStatus};
use crate::store::{AppSettings, Endpoint, PermissionMode, Settings, ThemePreference, data_dir};
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
    opencode_bin: Entity<InputState>,
    agent_prompt: Entity<TextareaState>,
    claude_model: Entity<SelectState<Vec<Choice<Option<String>>>>>,
    claude_effort: Entity<SelectState<Vec<Choice<Option<String>>>>>,
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

        let opencode_bin = text_input(
            settings.opencode_bin.clone(),
            |s, v| s.opencode_bin = v.trim().to_string(),
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

        let agent_prompt = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(3, 10)
                .default_value(settings.agent_prompt.clone())
        });
        subscriptions.push(cx.subscribe_in(
            &agent_prompt,
            window,
            |_, state, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    let value = state.read(cx).value().to_string();
                    AppSettings::update(cx, |s| s.agent_prompt = value);
                }
            },
        ));

        let mut optional_select = |choices: Vec<(&'static str, &'static str)>,
                                   default_label: &'static str,
                                   current: Option<String>,
                                   apply: fn(&mut Settings, Option<String>),
                                   window: &mut Window,
                                   cx: &mut Context<Self>| {
            let mut items = vec![Choice {
                value: None,
                label: default_label.into(),
            }];
            items.extend(choices.into_iter().map(|(value, label)| Choice {
                value: Some(value.to_string()),
                label: label.into(),
            }));
            if let Some(current) = &current
                && !items.iter().any(|c| c.value.as_ref() == Some(current))
            {
                items.push(Choice {
                    value: Some(current.clone()),
                    label: current.clone().into(),
                });
            }
            let ix = items.iter().position(|c| c.value == current).unwrap_or(0);
            let state = cx.new(|cx| {
                SelectState::new(
                    items,
                    Some(gpui_kit::component::IndexPath::new(ix)),
                    window,
                    cx,
                )
            });
            subscriptions.push(cx.subscribe_in(
                &state,
                window,
                move |_, _, event: &SelectEvent<Vec<Choice<Option<String>>>>, _, cx| {
                    let SelectEvent::Confirm(Some(value)) = event else {
                        return;
                    };
                    let value = value.clone();
                    AppSettings::update(cx, |s| apply(s, value));
                },
            ));
            state
        };
        let claude_model = optional_select(
            vec![
                ("fable", "fable"),
                ("opus", "opus"),
                ("sonnet", "sonnet"),
                ("haiku", "haiku"),
            ],
            "پیش‌فرض CLI",
            settings.default_claude_model.clone(),
            |s, v| s.default_claude_model = v,
            window,
            cx,
        );
        let claude_effort = optional_select(
            vec![
                ("low", "کم — low"),
                ("medium", "متوسط — medium"),
                ("high", "زیاد — high"),
                ("xhigh", "خیلی زیاد — xhigh"),
                ("max", "بیشینه — max"),
            ],
            "پیش‌فرض",
            settings.default_claude_effort.clone(),
            |s, v| s.default_claude_effort = v,
            window,
            cx,
        );

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
            opencode_bin,
            agent_prompt,
            claude_model,
            claude_effort,
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

    fn confirm_remove_endpoint(endpoint: &Endpoint, window: &mut Window, cx: &mut Context<Self>) {
        let id = endpoint.id.clone();
        let name = endpoint.name.clone();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let id = id.clone();
            alert
                .title(format!("سرویس «{name}» حذف شود؟"))
                .description("گفتگوهای قبلی که از این سرویس استفاده می‌کردند دیگر پاسخ نمی‌دهند.")
                .confirm()
                .ok_text("حذف سرویس")
                .ok_variant(ButtonVariant::Danger)
                .cancel_text("انصراف")
                .on_ok(move |_, _, cx| {
                    let id = id.clone();
                    AppSettings::update(cx, |s| s.endpoints.retain(|e| e.id != id));
                    true
                })
        });
    }

    /// Add (`None`) or edit an OpenAI-compatible endpoint.
    fn open_endpoint_dialog(
        existing: Option<Endpoint>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let make = |value: String, placeholder: &'static str, masked: bool, window: &mut Window, cx: &mut Context<Self>| {
            cx.new(|cx| {
                InputState::new(window, cx)
                    .placeholder(placeholder)
                    .masked(masked)
                    .default_value(value)
            })
        };
        let name = make(
            existing.as_ref().map(|e| e.name.clone()).unwrap_or_default(),
            "مثلاً OpenRouter",
            false,
            window,
            cx,
        );
        let base_url = make(
            existing.as_ref().map(|e| e.base_url.clone()).unwrap_or_default(),
            "https://api.openai.com/v1",
            false,
            window,
            cx,
        );
        let api_key = make(
            existing.as_ref().and_then(|e| e.api_key.clone()).unwrap_or_default(),
            "اختیاری",
            true,
            window,
            cx,
        );
        let model = make(
            existing.as_ref().and_then(|e| e.default_model.clone()).unwrap_or_default(),
            "اختیاری",
            false,
            window,
            cx,
        );
        name.update(cx, |input, cx| input.focus(window, cx));
        let editing = existing.is_some();
        let existing_id = existing.map(|e| e.id);
        window.open_dialog(cx, move |dialog, _, cx| {
            let (name, base_url, api_key, model) =
                (name.clone(), base_url.clone(), api_key.clone(), model.clone());
            let existing_id = existing_id.clone();
            let field = |label: &'static str, input: Input, ltr: bool| {
                let input = if ltr {
                    div().ltr().w_full().child(input)
                } else {
                    div().w_full().child(input)
                };
                v_flex()
                    .gap_1()
                    .child(div().text_sm().child(label))
                    .child(input)
            };
            dialog
                .title(if editing { "ویرایش سرویس" } else { "افزودن سرویس" })
                .w(px(460.))
                .close_button(false)
                .child(
                    v_flex()
                        .gap_3()
                        .child(field("نام", rtl::input(&name, cx), false))
                        .child(field("نشانی پایه", Input::new(&base_url), true))
                        .child(field("کلید API", Input::new(&api_key), true))
                        .child(field("مدل پیش‌فرض", Input::new(&model), true)),
                )
                .footer(
                    DialogFooter::new()
                        .child(
                            Button::new("endpoint-cancel")
                                .outline()
                                .label("انصراف")
                                .on_click(|_, window, cx| window.close_dialog(cx)),
                        )
                        .child(
                            Button::new("endpoint-save")
                                .primary()
                                .label("ذخیره")
                                .on_click(move |_, window, cx| {
                                    let name = name.read(cx).value().trim().to_string();
                                    let url = base_url
                                        .read(cx)
                                        .value()
                                        .trim()
                                        .trim_end_matches('/')
                                        .to_string();
                                    if name.is_empty() || url.is_empty() {
                                        window.push_notification(
                                            Notification::error("نام و نشانی پایه لازم است."),
                                            cx,
                                        );
                                        return;
                                    }
                                    let non_empty = |v: String| {
                                        let v = v.trim().to_string();
                                        (!v.is_empty()).then_some(v)
                                    };
                                    let api_key = non_empty(api_key.read(cx).value().to_string());
                                    let default_model = non_empty(model.read(cx).value().to_string());
                                    let existing_id = existing_id.clone();
                                    AppSettings::update(cx, |s| {
                                        let id = existing_id
                                            .clone()
                                            .unwrap_or_else(crate::store::new_id);
                                        let endpoint = Endpoint {
                                            id: id.clone(),
                                            name,
                                            base_url: url,
                                            api_key,
                                            default_model,
                                        };
                                        match s.endpoints.iter_mut().find(|e| e.id == id) {
                                            Some(slot) => *slot = endpoint,
                                            None => s.endpoints.push(endpoint),
                                        }
                                    });
                                    window.close_dialog(cx);
                                }),
                        ),
                )
        });
    }

    fn endpoint_row(endpoint: &Endpoint, cx: &Context<Self>) -> gpui_kit::Div {
        let theme = cx.theme();
        let muted = theme.muted_foreground;
        let edit_target = endpoint.clone();
        let remove_target = endpoint.clone();
        let mut details = v_flex().flex_1().min_w_0().gap_0p5().child(
            div()
                .font_weight(gpui_kit::FontWeight::MEDIUM)
                .child(endpoint.name.clone()),
        );
        details = details.child(
            div()
                .ltr()
                .text_sm()
                .text_color(muted)
                .child(endpoint.base_url.clone()),
        );
        let mut meta = h_flex().gap_2().text_xs().text_color(muted);
        if let Some(model) = &endpoint.default_model {
            meta = meta.child(div().ltr().child(model.clone()));
        }
        if endpoint.api_key.is_some() {
            meta = meta.child(
                h_flex()
                    .gap_1()
                    .child(gpui_kit::component::Icon::new(IconName::KeyRound).size_3())
                    .child("کلید تنظیم شده · ••••••"),
            );
        }
        details = details.child(meta);
        h_flex()
            .w_full()
            .gap_2()
            .items_center()
            .px_3()
            .py_2()
            .rounded(theme.radius)
            .border_1()
            .border_color(theme.border)
            .child(details)
            .child(
                Button::new(SharedString::from(format!("endpoint-edit-{}", endpoint.id)))
                    .ghost()
                    .small()
                    .icon(IconName::Pencil)
                    .tooltip("ویرایش")
                    .on_click(cx.listener(move |_, _, window, cx| {
                        Self::open_endpoint_dialog(Some(edit_target.clone()), window, cx)
                    })),
            )
            .child(
                Button::new(SharedString::from(format!("endpoint-remove-{}", endpoint.id)))
                    .ghost()
                    .small()
                    .icon(IconName::Trash)
                    .tooltip("حذف")
                    .on_click(cx.listener(move |_, _, window, cx| {
                        Self::confirm_remove_endpoint(&remove_target, window, cx)
                    })),
            )
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
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
                                            .label("فایل اجرایی opencode")
                                            .child(div().ltr().w_full().child(Input::new(&self.opencode_bin))),
                                    )
                                    .child(
                                        Field::new()
                                            .label("مدل پیش‌فرض Claude")
                                            .child(Select::new(&self.claude_model)),
                                    )
                                    .child(
                                        Field::new()
                                            .label("تلاش فکری پیش‌فرض Claude")
                                            .child(Select::new(&self.claude_effort)),
                                    )
                                    .child(
                                        Field::new()
                                            .label("مجوزهای Claude Code")
                                            .description("حالت --permission-mode برای هر پیام.")
                                            .child(Select::new(&self.permission)),
                                    ),
                            )
                            // OpenAI-compatible endpoints
                            .child(Self::section(
                                "سرویس‌های سازگار با OpenAI",
                                "هر API با /chat/completions، مثل OpenAI، OpenRouter، LM Studio یا vLLM.",
                                cx,
                            ))
                            .child(
                                v_flex()
                                    .w_full()
                                    .gap_2()
                                    .when(settings.endpoints.is_empty(), |this| {
                                        this.child(
                                            div()
                                                .text_sm()
                                                .text_color(cx.theme().muted_foreground)
                                                .child("هنوز سرویسی اضافه نشده."),
                                        )
                                    })
                                    .children(
                                        settings
                                            .endpoints
                                            .iter()
                                            .map(|endpoint| Self::endpoint_row(endpoint, cx)),
                                    )
                                    .child(
                                        h_flex().child(
                                            Button::new("endpoint-add")
                                                .small()
                                                .icon(IconName::Plus)
                                                .label("افزودن سرویس")
                                                .on_click(cx.listener(|_, _, window, cx| {
                                                    Self::open_endpoint_dialog(None, window, cx)
                                                })),
                                        ),
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
                                            .description("برای همهٔ بک‌اندها اعمال می‌شود: پیام سیستمی در Ollama و سرویس‌های OpenAI، و --append-system-prompt در Claude Code.")
                                            .child(Textarea::new(&self.system_prompt)),
                                    )
                                    .child(
                                        Field::new()
                                            .label("دستور عامل")
                                            .description("برای Claude Code و opencode، بعد از دستور سیستمی اضافه می‌شود: پاسخ‌ها کوتاه و گفتنی، کار با خود عامل، زیرعامل فقط برای کارهای بزرگ.")
                                            .child(
                                                v_flex()
                                                    .w_full()
                                                    .gap_2()
                                                    .child(Textarea::new(&self.agent_prompt))
                                                    .child(
                                                        h_flex().child(
                                                            Button::new("agent-prompt-reset")
                                                                .small()
                                                                .outline()
                                                                .label("بازگردانی پیش‌فرض")
                                                                .on_click(cx.listener(
                                                                    |this, _, window, cx| {
                                                                        this.agent_prompt.update(cx, |state, cx| {
                                                                            state.set_value(Settings::default().agent_prompt, window, cx)
                                                                        });
                                                                        AppSettings::update(cx, |s| {
                                                                            s.agent_prompt = Settings::default().agent_prompt
                                                                        });
                                                                    },
                                                                )),
                                                        ),
                                                    ),
                                            ),
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
