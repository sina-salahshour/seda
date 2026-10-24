//! Samples section (list, playback, transcript edits, reference) and the
//! record/import section.

use std::{cell::Cell, path::PathBuf, time::Duration};

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariant, ButtonVariants as _},
    dialog::{DialogAction, DialogClose, DialogFooter},
    h_flex,
    input::{Textarea, TextareaState},
    menu::{DropdownMenu as _, PopupMenuItem},
    speech::{SpeechButton, SpeechEvent, SpeechStatus, SpeechWaveform},
    spinner::Spinner,
    tag::Tag,
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, AppContext as _, Context, FontWeight, IntoElement, ParentElement as _,
    PathPromptOptions, SharedString, Styled as _, Window, div, px,
};

use super::{
    VoicesPage, format_length, meta_row, meta_text, muted_line, notify_error, player, section,
};
use crate::rtl;
use crate::sidecar::{Profile, ProfilePatch, Sample, SidecarClient};
use crate::ui::common::fa_digits;

/// How many prompt sentences to fetch at a time.
const PROMPT_BATCH: usize = 10;

/// An audio file being imported (POST, then Whisper transcription).
pub(super) struct ImportJob {
    id: usize,
    pid: String,
    name: String,
    status: ImportStatus,
}

enum ImportStatus {
    Waiting,
    Running,
    Failed(String),
}

impl VoicesPage {
    // MARK: Prompts

    pub(super) fn load_prompts(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.prompts_loading {
            return;
        }
        self.prompts_loading = true;
        self.spawn_job(
            window,
            cx,
            |client| client.prompts(PROMPT_BATCH),
            |this, result, window, cx| {
                this.prompts_loading = false;
                match result {
                    Ok(prompts) => this.prompts.extend(prompts),
                    Err(err) => notify_error(window, cx, "جمله‌های پیشنهادی بارگذاری نشد", &err),
                }
            },
        )
        .detach();
    }

    fn current_prompt(&self) -> Option<&str> {
        self.prompts.get(self.prompt_ix).map(String::as_str)
    }

    fn step_prompt(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        if forward {
            if self.prompt_ix + 1 < self.prompts.len() {
                self.prompt_ix += 1;
            }
            // Keep a few sentences ahead.
            if self.prompt_ix + 2 >= self.prompts.len() {
                self.load_prompts(window, cx);
            }
        } else {
            self.prompt_ix = self.prompt_ix.saturating_sub(1);
        }
        cx.notify();
    }

    // MARK: Recording

    pub(super) fn on_speech_event(
        &mut self,
        event: &SpeechEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            SpeechEvent::Started => {
                self.recording_for = self.selected.clone();
                // Don't record the sample playing back.
                if self.playing.take().is_some()
                    && let Some(player) = player(cx)
                {
                    player.stop();
                }
            }
            SpeechEvent::Final(path) => {
                let pid = self.recording_for.take().or_else(|| self.selected.clone());
                if let Some(pid) = pid
                    && !path.trim().is_empty()
                {
                    self.upload_recording(pid, PathBuf::from(path.as_ref()), window, cx);
                }
            }
            SpeechEvent::Cancelled => self.recording_for = None,
            SpeechEvent::Error(err) => {
                self.recording_for = None;
                notify_error(window, cx, "ضبط ناموفق بود", &anyhow::anyhow!("{err}"));
            }
            SpeechEvent::Partial(_) => {}
        }
        cx.notify();
    }

    fn upload_recording(
        &mut self,
        pid: String,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let text = self.current_prompt().map(str::to_string);
        self.uploading.push(pid.clone());
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| {
                let result = client.add_sample(&job_pid, &path, text.as_deref(), "recorded");
                // The temp WAV is ours; the sidecar keeps its own converted copy.
                let _ = std::fs::remove_file(&path);
                result
            },
            move |this, result, window, cx| {
                if let Some(ix) = this.uploading.iter().position(|p| *p == pid) {
                    this.uploading.remove(ix);
                }
                match result {
                    Ok(_) => {
                        if this.is_selected(&pid) {
                            this.step_prompt(true, window, cx);
                        }
                        this.refresh_profile(pid, window, cx);
                    }
                    Err(err) => notify_error(window, cx, "نمونهٔ ضبط‌شده افزوده نشد", &err),
                }
            },
        )
        .detach();
    }

    // MARK: Import

    fn import_files(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let picked = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("وارد کردن".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let paths = match picked.await {
                Ok(Ok(Some(paths))) => paths,
                Ok(Ok(None)) | Err(_) => return,
                Ok(Err(err)) => {
                    this.update_in(cx, |_, window, cx| {
                        notify_error(window, cx, "پنجرهٔ انتخاب فایل باز نشد", &err)
                    })
                    .ok();
                    return;
                }
            };
            let Ok(jobs) = this.update(cx, |this, cx| {
                let jobs: Vec<(usize, PathBuf)> = paths
                    .into_iter()
                    .map(|path| {
                        let id = this.next_import_id;
                        this.next_import_id += 1;
                        this.imports.push(ImportJob {
                            id,
                            pid: pid.clone(),
                            name: path
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_else(|| path.display().to_string()),
                            status: ImportStatus::Waiting,
                        });
                        (id, path)
                    })
                    .collect();
                cx.notify();
                jobs
            }) else {
                return;
            };
            // One at a time: each may run Whisper on the GPU.
            for (id, path) in jobs {
                let Ok(client) = this.update(cx, |this, cx| {
                    if let Some(job) = this.imports.iter_mut().find(|j| j.id == id) {
                        job.status = ImportStatus::Running;
                    }
                    cx.notify();
                    SidecarClient::from_settings(cx)
                }) else {
                    return;
                };
                let job_pid = pid.clone();
                let result = cx
                    .background_executor()
                    .spawn(async move { client.add_sample(&job_pid, &path, None, "imported") })
                    .await;
                let applied = this.update_in(cx, |this, window, cx| {
                    match result {
                        Ok(_) => {
                            this.imports.retain(|j| j.id != id);
                            this.refresh_profile(pid.clone(), window, cx);
                        }
                        Err(err) => {
                            if let Some(job) = this.imports.iter_mut().find(|j| j.id == id) {
                                job.status = ImportStatus::Failed(format!("{err:#}"));
                            }
                        }
                    }
                    cx.notify();
                });
                if applied.is_err() {
                    return;
                }
            }
        })
        .detach();
    }

    // MARK: Samples

    fn toggle_playback(&mut self, sid: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(player) = player(cx) else {
            window.push_notification(
                gpui_kit::component::notification::Notification::error("خروجی صدایی پیدا نشد."),
                cx,
            );
            return;
        };
        if self.playing.as_deref() == Some(sid.as_str()) {
            player.stop();
            self.playing = None;
            self._playback_watch = None;
            cx.notify();
            return;
        }
        let Some(pid) = self.selected.clone() else {
            return;
        };
        self.fetching_audio = Some(sid.clone());
        cx.notify();
        let job_sid = sid.clone();
        self._audio_fetch = Some(self.spawn_job(
            window,
            cx,
            move |client| client.sample_audio(&pid, &job_sid),
            move |this, result, window, cx| {
                if this.fetching_audio.as_deref() == Some(sid.as_str()) {
                    this.fetching_audio = None;
                }
                match result {
                    Ok(wav) => {
                        player.play(wav);
                        this.playing = Some(sid);
                        this.watch_playback(window, cx);
                    }
                    Err(err) => notify_error(window, cx, "نمونه پخش نشد", &err),
                }
            },
        ));
    }

    /// Clear `playing` once the player goes quiet.
    pub(super) fn watch_playback(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self._playback_watch = Some(cx.spawn_in(window, async move |this, cx| {
            // The player thread needs a moment to start.
            cx.background_executor()
                .timer(Duration::from_millis(600))
                .await;
            loop {
                let playing = this.update(cx, |_, cx| player(cx).is_some_and(|p| p.is_playing()));
                if !matches!(playing, Ok(true)) {
                    break;
                }
                cx.background_executor()
                    .timer(Duration::from_millis(250))
                    .await;
            }
            this.update(cx, |this, cx| {
                this.playing = None;
                cx.notify();
            })
            .ok();
        }));
    }

    fn open_transcript_dialog(
        &mut self,
        sample: &Sample,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let sid = sample.id.clone();
        let current = sample.text.clone();
        let text = cx.new(|cx| {
            TextareaState::new(window, cx)
                .auto_grow(3, 10)
                .default_value(current.clone())
        });
        let view = cx.entity().downgrade();
        let initial_focus = Cell::new(true);
        window.open_dialog(cx, move |dialog, window, cx| {
            if initial_focus.replace(false) {
                text.update(cx, |state, cx| state.focus(window, cx));
            }
            let (text_ok, view, pid, sid, current) = (
                text.clone(),
                view.clone(),
                pid.clone(),
                sid.clone(),
                current.clone(),
            );
            dialog
                .title("ویرایش متن نمونه")
                .w(px(560.))
                .child(
                    v_flex()
                        .gap_2()
                        .child(Textarea::new(&text))
                        .child(muted_line(
                            "متن باید دقیقاً همان چیزی باشد که در نمونه گفته شده.",
                            cx,
                        )),
                )
                .on_ok(move |_, window, cx| {
                    let value = text_ok.read(cx).value().trim().to_string();
                    if value != current.trim() {
                        let (pid, sid) = (pid.clone(), sid.clone());
                        view.update(cx, |this, cx| {
                            this.save_transcript(pid, sid, value, window, cx)
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
                                .child(Button::new("save-transcript").primary().label("ذخیره")),
                        ),
                )
        });
    }

    fn save_transcript(
        &mut self,
        pid: String,
        sid: String,
        text: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let job_pid = pid.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.update_sample(&job_pid, &sid, &text),
            move |this, result, window, cx| match result {
                Ok(sample) => {
                    if this.is_selected(&pid)
                        && let Some(slot) = this.samples.iter_mut().find(|s| s.id == sample.id)
                    {
                        *slot = sample;
                    }
                }
                Err(err) => notify_error(window, cx, "متن ذخیره نشد", &err),
            },
        )
        .detach();
    }

    fn confirm_delete_sample(
        &mut self,
        sample: &Sample,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(pid) = self.selected.clone() else {
            return;
        };
        let sid = sample.id.clone();
        let preview: SharedString = if sample.text.trim().is_empty() {
            "این نمونه متن ندارد.".into()
        } else {
            let text: String = sample.text.chars().take(120).collect();
            format!("«{text}»").into()
        };
        let view = cx.entity().downgrade();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let (pid, sid, view) = (pid.clone(), sid.clone(), view.clone());
            alert
                .title("این نمونه حذف شود؟")
                .description(preview.clone())
                .confirm()
                .ok_text("حذف")
                .ok_variant(ButtonVariant::Danger)
                .cancel_text("انصراف")
                .on_ok(move |_, window, cx| {
                    let (pid, sid) = (pid.clone(), sid.clone());
                    view.update(cx, |this, cx| this.delete_sample(pid, sid, window, cx))
                        .ok();
                    true
                })
        });
    }

    fn delete_sample(
        &mut self,
        pid: String,
        sid: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let (job_pid, job_sid) = (pid.clone(), sid.clone());
        self.spawn_job(
            window,
            cx,
            move |client| client.delete_sample(&job_pid, &job_sid),
            move |this, result, window, cx| match result {
                Ok(()) => {
                    if this.is_selected(&pid) {
                        this.samples.retain(|s| s.id != sid);
                    }
                    this.refresh_profile(pid, window, cx);
                }
                Err(err) => notify_error(window, cx, "نمونه حذف نشد", &err),
            },
        )
        .detach();
    }

    // MARK: Render

    fn render_sample_row(
        &self,
        sample: &Sample,
        is_reference: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let playing = self.playing.as_deref() == Some(sample.id.as_str());
        let fetching = self.fetching_audio.as_deref() == Some(sample.id.as_str());
        let source = match sample.source.as_str() {
            "recorded" => "ضبط‌شده",
            "imported" => "واردشده",
            _ => "",
        };
        let mut meta = vec![meta_text(format_length(sample.duration))];
        if !source.is_empty() {
            meta.push(meta_text(source));
        }
        if is_reference {
            meta.push(
                Tag::secondary()
                    .xsmall()
                    .child(
                        h_flex()
                            .gap_1()
                            .child(Icon::new(IconName::Star).xsmall())
                            .child("مرجع"),
                    )
                    .into_any_element(),
            );
        }
        let meta = meta_row(meta, cx);

        let play_sid = sample.id.clone();
        let edit_sample = sample.clone();
        let view = cx.entity().downgrade();
        let menu_sample = sample.clone();

        h_flex()
            .w_full()
            .gap_3()
            .py_2()
            .items_start()
            .border_b_1()
            .border_color(theme.border)
            .child(
                Button::new(SharedString::from(format!("play-{}", sample.id)))
                    .ghost()
                    .small()
                    .icon(if playing {
                        IconName::Square
                    } else {
                        IconName::Play
                    })
                    .loading(fetching)
                    .accessibility_label(if playing { "توقف" } else { "پخش" })
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.toggle_playback(play_sid.clone(), window, cx)
                    })),
            )
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .gap_1()
                    .child(if sample.text.trim().is_empty() {
                        div()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .child("بدون متن")
                    } else {
                        div().text_sm().child(sample.text.clone())
                    })
                    .child(meta),
            )
            .child(
                Button::new(SharedString::from(format!("edit-{}", sample.id)))
                    .ghost()
                    .small()
                    .icon(IconName::Pencil)
                    .tooltip("ویرایش متن…")
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.open_transcript_dialog(&edit_sample, window, cx)
                    })),
            )
            .child(
                Button::new(SharedString::from(format!("sample-menu-{}", sample.id)))
                    .ghost()
                    .small()
                    .icon(IconName::Ellipsis)
                    .accessibility_label("کارهای بیشتر")
                    .dropdown_menu(move |menu, _, _| {
                        let (reference, delete) = (view.clone(), view.clone());
                        let (ref_sid, del_sample) = (menu_sample.id.clone(), menu_sample.clone());
                        menu.item(
                            PopupMenuItem::new("استفاده به‌عنوان مرجع")
                                .icon(IconName::Star)
                                .disabled(is_reference)
                                .on_click(move |_, window, cx| {
                                    let sid = ref_sid.clone();
                                    reference
                                        .update(cx, |this, cx| {
                                            this.patch_profile(
                                                ProfilePatch {
                                                    reference_sample: Some(Some(sid)),
                                                    ..Default::default()
                                                },
                                                window,
                                                cx,
                                            )
                                        })
                                        .ok();
                                }),
                        )
                        .separator()
                        .item(
                            PopupMenuItem::new("حذف نمونه…")
                                .icon(IconName::Trash)
                                .on_click(move |_, window, cx| {
                                    delete
                                        .update(cx, |this, cx| {
                                            this.confirm_delete_sample(&del_sample, window, cx)
                                        })
                                        .ok();
                                }),
                        )
                    }),
            )
    }

    pub(super) fn render_samples_section(
        &self,
        profile: &Profile,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let content: AnyElement = if self.samples.is_empty() {
            muted_line(
                if self.samples_loading {
                    "در حال بارگذاری…"
                } else {
                    "هنوز نمونه‌ای نیست. چند جمله ضبط کن یا فایل صوتی وارد کن."
                },
                cx,
            )
            .into_any_element()
        } else {
            v_flex()
                .w_full()
                .children({
                    let mut rows = Vec::with_capacity(self.samples.len());
                    for sample in &self.samples {
                        let is_reference = profile.reference_sample.as_deref() == Some(&sample.id);
                        rows.push(self.render_sample_row(sample, is_reference, cx));
                    }
                    rows
                })
                .into_any_element()
        };
        let total = self.samples.iter().map(|s| s.duration).sum::<f64>();
        let description = (!self.samples.is_empty()).then(|| {
            SharedString::from(format!(
                "\u{200F}{} نمونه، روی هم {}. نمونهٔ مرجع برای صداسازی بدون آموزش به کار می‌رود.",
                fa_digits(&self.samples.len().to_string()),
                format_length(total)
            ))
        });
        section(true, "نمونه‌ها", description, None, content, cx)
    }

    pub(super) fn render_record_section(
        &self,
        _profile: &Profile,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
        let _ = window;
        let theme = cx.theme();
        let speech = self.speech.read(cx);
        let status = speech.status();
        let available = speech.is_available(cx);

        // The sentence to read, shown large.
        let prompt: AnyElement = match self.current_prompt() {
            Some(text) => div()
                .text_2xl()
                .font_weight(FontWeight::MEDIUM)
                .line_height(gpui_kit::relative(1.6))
                .child(text.to_string())
                .into_any_element(),
            None => div()
                .text_base()
                .text_color(theme.muted_foreground)
                .child(if self.prompts_loading {
                    "در حال آوردن جمله‌ها…"
                } else {
                    "جمله‌ای در دسترس نیست. هر چه بخوانی، سایدکار خودش متنش را می‌نویسد."
                })
                .into_any_element(),
        };
        let position = (!self.prompts.is_empty()).then(|| {
            format!(
                "\u{200F}{} از {}",
                fa_digits(&(self.prompt_ix + 1).to_string()),
                fa_digits(&self.prompts.len().to_string())
            )
        });

        let (title, detail): (&str, &str) = match status {
            _ if !available && !status.is_active() => {
                ("ضبط در دسترس نیست", "ورودی صدایی برای ضبط پیدا نشد.")
            }
            SpeechStatus::Idle => ("آماده", "دکمهٔ میکروفون را بزن، جمله را بخوان و دوباره بزن."),
            SpeechStatus::Connecting | SpeechStatus::Recording => {
                ("در حال ضبط…", "وقتی جمله تمام شد دوباره دکمه را بزن.")
            }
            SpeechStatus::Stopping => ("در حال ذخیره…", ""),
        };
        let pid = self.selected.as_deref().unwrap_or_default();
        let uploading = self.uploading.iter().filter(|p| *p == pid).count();
        let capturing = status.is_capturing();

        let recorder = h_flex()
            .w_full()
            .gap_3()
            .px_3()
            .py_2()
            .rounded(theme.radius_lg)
            .border_1()
            .border_color(if capturing { theme.ring } else { theme.border })
            .child(SpeechButton::new(&self.speech).large())
            .child(
                v_flex()
                    .map(|this| {
                        if status.is_active() {
                            this.w(px(200.)).flex_shrink_0()
                        } else {
                            this.flex_1()
                        }
                    })
                    .min_w_0()
                    .gap_0p5()
                    .child(div().text_sm().font_weight(FontWeight::MEDIUM).child(title))
                    .when(!detail.is_empty(), |this| {
                        this.child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .child(detail),
                        )
                    }),
            )
            .when(status.is_active(), |this| {
                this.child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .ltr()
                        .child(SpeechWaveform::new(&self.speech).small()),
                )
                .child(
                    Button::new("discard-recording")
                        .ghost()
                        .small()
                        .label("دور ریختن")
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.speech.update(cx, |speech, cx| speech.cancel(cx));
                        })),
                )
            });

        let imports: Vec<&ImportJob> = self.imports.iter().filter(|j| j.pid == pid).collect();
        let import_list = v_flex().gap_1().children(imports.into_iter().map(|job| {
            let id = job.id;
            h_flex()
                .gap_2()
                .text_sm()
                .child(match &job.status {
                    ImportStatus::Waiting => Icon::new(IconName::Clock)
                        .small()
                        .text_color(theme.muted_foreground)
                        .into_any_element(),
                    ImportStatus::Running => Spinner::new().small().into_any_element(),
                    ImportStatus::Failed(_) => Icon::new(IconName::CircleAlert)
                        .small()
                        .text_color(theme.danger)
                        .into_any_element(),
                })
                .child(
                    div()
                        .ltr()
                        .truncate()
                        .max_w(px(260.))
                        .child(job.name.clone()),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .text_xs()
                        .text_color(match job.status {
                            ImportStatus::Failed(_) => theme.danger,
                            _ => theme.muted_foreground,
                        })
                        .child(match &job.status {
                            ImportStatus::Waiting => "در صف".to_string(),
                            ImportStatus::Running => "در حال تبدیل و نوشتن متن…".to_string(),
                            ImportStatus::Failed(err) => err.clone(),
                        }),
                )
                .when(matches!(job.status, ImportStatus::Failed(_)), |this| {
                    this.child(
                        Button::new(SharedString::from(format!("dismiss-import-{id}")))
                            .ghost()
                            .xsmall()
                            .icon(IconName::X)
                            .accessibility_label("بستن")
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.imports.retain(|j| j.id != id);
                                cx.notify();
                            })),
                    )
                })
        }));

        let content = v_flex()
            .w_full()
            .gap_4()
            .child(
                v_flex()
                    .w_full()
                    .gap_3()
                    .px_5()
                    .py_4()
                    .rounded(theme.radius_lg)
                    .bg(theme.muted)
                    .child(
                        h_flex()
                            .gap_2()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child(div().flex_1().child("این جمله را بلند و طبیعی بخوان"))
                            .when_some(position, |this, position| this.child(position))
                            .child(
                                Button::new("prompt-prev")
                                    .ghost()
                                    .xsmall()
                                    .icon(rtl::back_icon(cx))
                                    .tooltip("جملهٔ قبلی")
                                    .disabled(self.prompt_ix == 0 || self.prompts.is_empty())
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.step_prompt(false, window, cx)
                                    })),
                            )
                            .child(
                                Button::new("prompt-next")
                                    .ghost()
                                    .xsmall()
                                    .icon(rtl::forward_icon(cx))
                                    .tooltip("جملهٔ بعدی")
                                    .disabled(self.prompts.is_empty())
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.step_prompt(true, window, cx)
                                    })),
                            ),
                    )
                    .child(prompt),
            )
            .child(recorder)
            .when(uploading > 0, |this| {
                this.child(
                    h_flex()
                        .gap_2()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(Spinner::new().small())
                        .child("در حال افزودن نمونهٔ ضبط‌شده…"),
                )
            })
            .child(import_list);

        let actions = Button::new("import-audio")
            .small()
            .icon(IconName::Upload)
            .label("وارد کردن فایل صوتی…")
            .on_click(cx.listener(|this, _, window, cx| this.import_files(window, cx)))
            .into_any_element();

        section(
            false,
            "ضبط و افزودن",
            Some("هر جمله یک نمونه می‌شود. فایل‌های واردشده خودکار متن‌نویسی می‌شوند و ممکن است کمی طول بکشد.".into()),
            Some(actions),
            content,
            cx,
        )
    }
}
