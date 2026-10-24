//! Training section: train / retrain, the run list with live progress, and
//! per-run actions (set active, cancel, log, delete).

use std::time::Duration;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariant, ButtonVariants as _},
    h_flex,
    menu::{DropdownMenu as _, PopupMenuItem},
    notification::Notification,
    progress::Progress,
    scroll::ScrollableElement as _,
    tag::Tag,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, Context, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _,
    SharedString, StatefulInteractiveElement as _, Styled as _, Window, div, px,
};

use super::{
    VoicesPage, fa_decimal, format_time, meta_row, meta_text, muted_line, notify_error, section,
};
use crate::sidecar::{Profile, ProfilePatch, Run, RunStatus, SidecarClient};
use crate::ui::common::fa_digits;

const POLL_INTERVAL: Duration = Duration::from_secs(1);

fn status_tag(status: RunStatus) -> Tag {
    let tag = match status {
        RunStatus::Queued | RunStatus::Cancelled => Tag::secondary(),
        RunStatus::Running => Tag::info(),
        RunStatus::Done => Tag::success(),
        RunStatus::Failed => Tag::danger(),
    };
    tag.outline().xsmall().child(status.label())
}

impl VoicesPage {
    fn any_run_active(&self) -> bool {
        self.runs.iter().any(|run| run.status.is_active())
    }

    fn train(&mut self, from_run: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let steps_text = self.steps.read(cx).value().trim().to_string();
        let steps = if steps_text.is_empty() {
            None
        } else {
            // Accept Persian digits too.
            let ascii: String = steps_text
                .chars()
                .map(|c| match c {
                    '۰'..='۹' => char::from_u32(c as u32 - '۰' as u32 + '0' as u32).unwrap_or(c),
                    _ => c,
                })
                .collect();
            match ascii.parse::<u32>() {
                Ok(steps) if steps > 0 => Some(steps),
                _ => {
                    window.push_notification(
                        Notification::error("تعداد گام‌ها باید یک عدد مثبت باشد."),
                        cx,
                    );
                    return;
                }
            }
        };
        self.train_pending = true;
        cx.notify();
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.train(&job_pid, steps, from_run.as_deref()),
            move |this, result, window, cx| {
                this.train_pending = false;
                match result {
                    Ok(run) => {
                        if this.is_selected(&pid) {
                            this.runs.retain(|r| r.id != run.id);
                            this.runs.insert(0, run);
                            this.ensure_polling(window, cx);
                        }
                    }
                    Err(err) => notify_error(window, cx, "آموزش شروع نشد", &err),
                }
            },
        )
        .detach();
    }

    /// Reload the run list (newest first) of the selected profile.
    fn reload_runs(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.runs(&job_pid),
            move |this, result, window, cx| {
                if !this.is_selected(&pid) {
                    return;
                }
                match result {
                    Ok(runs) => {
                        this.runs = runs;
                        this.rebuild_run_choices(window, cx);
                        this.ensure_polling(window, cx);
                    }
                    Err(err) => notify_error(window, cx, "فهرست اجراها بارگذاری نشد", &err),
                }
            },
        )
        .detach();
    }

    /// Start the 1 s poll of active runs unless it is running or nothing is active.
    pub(super) fn ensure_polling(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.polling || !self.any_run_active() {
            return;
        }
        let Some(pid) = self.selected.clone() else {
            return;
        };
        self.polling = true;
        self._poll = Some(cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(POLL_INTERVAL).await;
                let next = this.update(cx, |this, cx| {
                    if !this.is_selected(&pid) {
                        return None;
                    }
                    let ids: Vec<String> = this
                        .runs
                        .iter()
                        .filter(|run| run.status.is_active())
                        .map(|run| run.id.clone())
                        .collect();
                    (!ids.is_empty()).then(|| (SidecarClient::from_settings(cx), ids))
                });
                let Ok(Some((client, ids))) = next else {
                    break;
                };
                let job_pid = pid.clone();
                let results = cx
                    .background_executor()
                    .spawn(async move {
                        ids.iter()
                            .map(|id| client.run(&job_pid, id))
                            .collect::<Vec<_>>()
                    })
                    .await;
                let applied = this.update_in(cx, |this, window, cx| {
                    this.apply_polled_runs(&pid, results, window, cx)
                });
                if applied.is_err() {
                    break;
                }
            }
            this.update(cx, |this, _| this.polling = false).ok();
        }));
    }

    fn apply_polled_runs(
        &mut self,
        pid: &str,
        results: Vec<anyhow::Result<Run>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.is_selected(pid) {
            return;
        }
        let mut finished = false;
        // Errors are usually transient (sidecar busy); the next tick retries.
        for run in results.into_iter().flatten() {
            let label = self.run_label(&run.id);
            if let Some(slot) = self.runs.iter_mut().find(|r| r.id == run.id) {
                if slot.status.is_active() && !run.status.is_active() {
                    finished = true;
                    match run.status {
                        RunStatus::Done => window.push_notification(
                            Notification::success(format!("{label} تمام شد.")),
                            cx,
                        ),
                        RunStatus::Failed => window.push_notification(
                            Notification::error(if run.message.is_empty() {
                                format!("{label} ناموفق بود.")
                            } else {
                                format!("{label} ناموفق بود. {}", run.message)
                            }),
                            cx,
                        ),
                        _ => {}
                    }
                }
                *slot = run;
            }
        }
        if finished {
            self.refresh_profile(pid.to_string(), window, cx);
            self.reload_runs(window, cx);
        }
        cx.notify();
    }

    fn cancel_run(&mut self, rid: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.cancel_run(&job_pid, &rid),
            move |this, result, window, cx| match result {
                Ok(run) => {
                    if this.is_selected(&pid)
                        && let Some(slot) = this.runs.iter_mut().find(|r| r.id == run.id)
                    {
                        *slot = run;
                    }
                }
                Err(err) => notify_error(window, cx, "اجرا متوقف نشد", &err),
            },
        )
        .detach();
    }

    fn open_run_log(&mut self, rid: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let label = self.run_label(&rid);
        let job_rid = rid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.run_log(&pid, &job_rid),
            move |_, result, window, cx| match result {
                Ok(log) => {
                    let log: SharedString = if log.trim().is_empty() {
                        "(گزارش خالی است)".into()
                    } else {
                        log.into()
                    };
                    window.open_dialog(cx, move |dialog, _, cx| {
                        dialog
                            .title(SharedString::from(format!("گزارش {label}")))
                            .w(px(780.))
                            .child(
                                div()
                                    .id("run-log")
                                    .ltr()
                                    .w_full()
                                    .max_h(px(480.))
                                    .p_3()
                                    .rounded(cx.theme().radius)
                                    .bg(cx.theme().muted)
                                    .font_family(cx.theme().mono_font_family.clone())
                                    .text_xs()
                                    .overflow_y_scrollbar()
                                    .child(log.clone()),
                            )
                    });
                }
                Err(err) => notify_error(window, cx, "گزارش بارگذاری نشد", &err),
            },
        )
        .detach();
    }

    fn confirm_delete_run(&mut self, rid: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let label = self.run_label(&rid);
        let is_active_model = self
            .selected_profile()
            .is_some_and(|p| p.active_run.as_deref() == Some(rid.as_str()));
        let view = cx.entity().downgrade();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let (pid, rid2, view) = (pid.clone(), rid.clone(), view.clone());
            alert
                .title(SharedString::from(format!("«{label}» حذف شود؟")))
                .description(if is_active_model {
                    "این مدل فعال است. بعد از حذف، صدا بدون آموزش (zero-shot) ساخته می‌شود."
                } else {
                    "مدل آموزش‌دیده و گزارش این اجرا پاک می‌شوند."
                })
                .confirm()
                .ok_text("حذف")
                .ok_variant(ButtonVariant::Danger)
                .cancel_text("انصراف")
                .on_ok(move |_, window, cx| {
                    let (pid, rid) = (pid.clone(), rid2.clone());
                    view.update(cx, |this, cx| this.delete_run(pid, rid, window, cx))
                        .ok();
                    true
                })
        });
    }

    fn delete_run(
        &mut self,
        pid: String,
        rid: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (job_pid, job_rid) = (pid.clone(), rid.clone());
        self.spawn_job(
            window,
            cx,
            move |client| client.delete_run(&job_pid, &job_rid),
            move |this, result, window, cx| match result {
                Ok(()) => {
                    if this.is_selected(&pid) {
                        this.runs.retain(|r| r.id != rid);
                        this.rebuild_run_choices(window, cx);
                    }
                    this.refresh_profile(pid, window, cx);
                }
                Err(err) => notify_error(window, cx, "اجرا حذف نشد", &err),
            },
        )
        .detach();
    }

    fn set_active_run(&mut self, rid: Option<String>, window: &mut Window, cx: &mut Context<Self>) {
        self.patch_profile(
            ProfilePatch {
                active_run: Some(rid),
                ..Default::default()
            },
            window,
            cx,
        );
    }

    // MARK: Render

    fn render_run_row(
        &self,
        run: &Run,
        profile: &Profile,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let is_model = profile.active_run.as_deref() == Some(run.id.as_str());
        let active = run.status.is_active();

        let mut meta = vec![
            div()
                .ltr()
                .whitespace_nowrap()
                .child(fa_digits(&format_time(&run.created_at)))
                .into_any_element(),
        ];
        if run.sample_count > 0 {
            meta.push(meta_text(format!(
                "\u{200F}{} نمونه",
                fa_digits(&run.sample_count.to_string())
            )));
        }
        if !active && run.total_steps > 0 {
            meta.push(meta_text(format!(
                "\u{200F}{} گام",
                fa_digits(&run.step.max(run.total_steps).to_string())
            )));
        }
        if !active && let Some(loss) = run.loss {
            meta.push(meta_text(format!("خطا {}", fa_decimal(loss, 3))));
        }
        let meta = meta_row(meta, cx);
        let label = self.run_label(&run.id);
        let raw_id: SharedString = format!("شناسه: {}", run.id).into();

        let progress_line = active.then(|| {
            let mut numbers = vec![meta_text(if run.total_steps > 0 {
                format!(
                    "گام {} از {}",
                    fa_digits(&run.step.to_string()),
                    fa_digits(&run.total_steps.to_string())
                )
            } else {
                format!("{}٪", fa_digits(&format!("{:.0}", run.progress * 100.)))
            })];
            if let Some(loss) = run.loss {
                numbers.push(meta_text(format!("خطا {}", fa_decimal(loss, 4))));
            }
            h_flex()
                .w_full()
                .gap_3()
                .child(
                    div().flex_1().min_w_0().ltr().child(
                        Progress::new(SharedString::from(format!("run-progress-{}", run.id)))
                            .small()
                            .loading(run.status == RunStatus::Queued)
                            .value((run.progress * 100.) as f32)
                            .accessibility_label("پیشرفت آموزش"),
                    ),
                )
                .child(meta_row(numbers, cx).flex_shrink_0())
        });

        let rid = run.id.clone();
        let view = cx.entity().downgrade();
        let menu_rid = run.id.clone();

        v_flex()
            .w_full()
            .gap_2()
            .py_3()
            .border_b_1()
            .border_color(theme.border)
            .child(
                h_flex()
                    .w_full()
                    .gap_3()
                    .child(status_tag(run.status))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .gap_0p5()
                            .child(
                                h_flex()
                                    .gap_2()
                                    .child(
                                        div()
                                            .id(SharedString::from(format!("run-title-{}", run.id)))
                                            .text_sm()
                                            .font_weight(FontWeight::MEDIUM)
                                            .tooltip(move |window, cx| {
                                                Tooltip::new(raw_id.clone()).build(window, cx)
                                            })
                                            .child(label),
                                    )
                                    .when(is_model, |this| {
                                        this.child(Tag::primary().xsmall().child("مدل فعال"))
                                    }),
                            )
                            .child(meta),
                    )
                    .map(|this| {
                        if active {
                            let rid = rid.clone();
                            this.child(
                                Button::new(SharedString::from(format!("cancel-{}", run.id)))
                                    .small()
                                    .icon(IconName::CircleStop)
                                    .label("توقف")
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.cancel_run(rid.clone(), window, cx)
                                    })),
                            )
                        } else if run.status == RunStatus::Done && !is_model {
                            let rid = rid.clone();
                            this.child(
                                Button::new(SharedString::from(format!("activate-{}", run.id)))
                                    .small()
                                    .outline()
                                    .label("فعال کن")
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.set_active_run(Some(rid.clone()), window, cx)
                                    })),
                            )
                        } else {
                            this
                        }
                    })
                    .child(
                        Button::new(SharedString::from(format!("run-menu-{}", run.id)))
                            .ghost()
                            .small()
                            .icon(IconName::Ellipsis)
                            .accessibility_label("کارهای بیشتر")
                            .dropdown_menu(move |menu, _, _| {
                                let (log, delete) = (view.clone(), view.clone());
                                let (log_rid, del_rid) = (menu_rid.clone(), menu_rid.clone());
                                menu.item(
                                    PopupMenuItem::new("گزارش…")
                                        .icon(IconName::FileText)
                                        .on_click(move |_, window, cx| {
                                            let rid = log_rid.clone();
                                            log.update(cx, |this, cx| {
                                                this.open_run_log(rid, window, cx)
                                            })
                                            .ok();
                                        }),
                                )
                                .separator()
                                .item(
                                    PopupMenuItem::new("حذف اجرا…")
                                        .icon(IconName::Trash)
                                        .disabled(active)
                                        .on_click(move |_, window, cx| {
                                            let rid = del_rid.clone();
                                            delete
                                                .update(cx, |this, cx| {
                                                    this.confirm_delete_run(rid, window, cx)
                                                })
                                                .ok();
                                        }),
                                )
                            }),
                    ),
            )
            .when_some(progress_line, |this, line| this.child(line))
            .when(!run.models.is_empty(), |this| {
                this.child(meta_row(
                    std::iter::once(meta_text("مدل‌ها:")).chain(run.models.iter().map(
                        |(role, model)| {
                            div()
                                .ltr()
                                .whitespace_nowrap()
                                .child(format!("{role}: {model}"))
                                .into_any_element()
                        },
                    )),
                    cx,
                ))
            })
            .when(!run.message.trim().is_empty(), |this| {
                this.child(
                    div()
                        .text_xs()
                        .text_color(if run.status == RunStatus::Failed {
                            theme.danger
                        } else {
                            theme.muted_foreground
                        })
                        .child(run.message.clone()),
                )
            })
    }

    pub(super) fn render_training_section(
        &self,
        profile: &Profile,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let muted_foreground = cx.theme().muted_foreground;
        // Engines without fine-tuning "train" by picking the best reference clip.
        let picks_reference = !self.engine_trainable(&profile.engine);
        let engine_ready = self.engine_ready(&profile.engine);
        let busy = self.sidecar_busy(cx);
        let running = self.any_run_active();
        let blocked = busy.is_some() || running || self.train_pending || !engine_ready;
        let no_samples = profile.sample_count == 0 && self.samples.is_empty();
        let can_retrain = !picks_reference && profile.active_run.is_some();

        let reason: Option<String> = if running {
            Some("یک آموزش در جریان است.".into())
        } else if let Some(busy) = &busy {
            Some(format!("سایدکار مشغول است: {busy}"))
        } else if !engine_ready {
            Some("مدل این موتور هنوز دانلود نشده؛ از صفحهٔ «مدل‌ها» دانلودش کن.".into())
        } else if no_samples {
            Some("برای آموزش اول چند نمونه اضافه کن.".into())
        } else {
            None
        };

        let controls =
            h_flex()
                .w_full()
                .gap_3()
                .flex_wrap()
                .when(!picks_reference, |this| {
                    this.child(
                        h_flex().gap_2().text_sm().child("تعداد گام‌ها").child(
                            div()
                                .w(px(110.))
                                .child(crate::rtl::input(&self.steps, cx).small()),
                        ),
                    )
                })
                .child({
                    let button = Button::new("train-fresh")
                        .small()
                        .icon(IconName::GraduationCap)
                        .label(if picks_reference {
                            "انتخاب بهترین مرجع"
                        } else {
                            "آموزش از صفر"
                        })
                        .loading(self.train_pending)
                        .disabled(blocked || no_samples)
                        .on_click(cx.listener(|this, _, window, cx| this.train(None, window, cx)));
                    if can_retrain {
                        button
                    } else {
                        button.primary()
                    }
                })
                .when(can_retrain, |this| {
                    let from = profile.active_run.clone();
                    this.child(
                        Button::new("train-continue")
                            .small()
                            .primary()
                            .icon(IconName::RotateCcw)
                            .label("ادامهٔ آموزش")
                            .tooltip("از مدل فعال ادامه می‌دهد، با همهٔ نمونه‌های فعلی")
                            .disabled(blocked || no_samples)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.train(from.clone(), window, cx)
                            })),
                    )
                })
                .child(div().flex_1())
                .when(profile.active_run.is_some(), |this| {
                    this.child(
                        Button::new("use-zero-shot")
                            .small()
                            .ghost()
                            .label("استفاده بدون آموزش")
                            .tooltip("صدا فقط از نمونهٔ مرجع ساخته می‌شود (zero-shot)")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.set_active_run(None, window, cx)
                            })),
                    )
                });

        let runs: AnyElement = if self.runs.is_empty() {
            muted_line(
                if self.samples_loading {
                    "در حال بارگذاری…"
                } else {
                    "هنوز آموزشی اجرا نشده. بدون آموزش هم صدا از روی نمونهٔ مرجع ساخته می‌شود."
                },
                cx,
            )
            .into_any_element()
        } else {
            v_flex()
                .w_full()
                .children(
                    self.runs
                        .iter()
                        .map(|run| self.render_run_row(run, profile, cx)),
                )
                .into_any_element()
        };

        let content = v_flex()
            .w_full()
            .gap_3()
            .child(controls)
            .when_some(reason, |this, reason| {
                this.child(div().text_xs().text_color(muted_foreground).child(reason))
            })
            .child(runs);

        section(
            false,
            "آموزش",
            Some(if picks_reference {
                format!(
                    "برای {}، «آموزش» بهترین تکهٔ کوتاه نمونه‌ها را به‌عنوان مرجع انتخاب می‌کند.",
                    self.engine_short(&profile.engine)
                )
                .into()
            } else {
                "یک LoRA روی همهٔ نمونه‌های فعلی آموزش می‌دهد. وقتی تمام شود، مدل تازه فعال می‌شود."
                    .into()
            }),
            None,
            content,
            cx,
        )
    }
}
