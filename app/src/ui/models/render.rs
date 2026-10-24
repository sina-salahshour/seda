//! Rendering of the Models page.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    ActiveTheme as _, Disableable as _, Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    input::Input,
    progress::Progress,
    scroll::ScrollableElement as _,
    select::Select,
    tag::Tag,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, App, Context, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _,
    Render, SharedString, StatefulInteractiveElement as _, Styled as _, Window, div, px,
};

use super::ModelsPage;
use crate::sidecar::{ModelInfo, ModelKind, ModelStatus};
use crate::ui::common::{empty_state, fa_digits, page_header, sidecar_offline};

/// A decimal in Persian digits, e.g. `۱٫۲`.
fn fa_decimal(value: f64, decimals: usize) -> String {
    fa_digits(&format!("{value:.decimals$}").replace('.', "٫"))
}

/// Megabytes as a short Persian size, e.g. `۸۵۰ مگابایت` or `۱٫۲ گیگابایت`.
/// Starts with an RLM so the number never leads the bidi run.
pub(super) fn format_size(mb: f64) -> String {
    if mb >= 1024. {
        format!("\u{200F}{} گیگابایت", fa_decimal(mb / 1024., 1))
    } else {
        format!(
            "\u{200F}{} مگابایت",
            fa_digits(&format!("{:.0}", mb.max(0.)))
        )
    }
}

/// `۵۰۰ از ۲٬۰۰۰ مگابایت`-style progress amount, both numbers in the unit of the total.
pub(super) fn format_progress(done_mb: f64, total_mb: Option<i64>) -> String {
    match total_mb.filter(|t| *t > 0) {
        Some(total) if total >= 1024 => format!(
            "\u{200F}{} از {} گیگابایت",
            fa_decimal(done_mb / 1024., 1),
            fa_decimal(total as f64 / 1024., 1)
        ),
        Some(total) => format!(
            "\u{200F}{} از {} مگابایت",
            fa_digits(&format!("{:.0}", done_mb.max(0.))),
            fa_digits(&total.to_string())
        ),
        None => format_size(done_mb),
    }
}

fn status_tag(status: ModelStatus) -> Tag {
    let tag = match status {
        ModelStatus::Missing => Tag::secondary(),
        ModelStatus::Downloading => Tag::info(),
        ModelStatus::Ready => Tag::success(),
        ModelStatus::Error => Tag::danger(),
    };
    tag.outline().xsmall().child(status.label())
}

/// One metadata item that does not wrap inside.
fn meta_text(text: impl Into<SharedString>) -> AnyElement {
    div()
        .whitespace_nowrap()
        .child(text.into())
        .into_any_element()
}

/// Left-to-right metadata (repo ids, paths).
fn meta_ltr(text: impl Into<SharedString>) -> AnyElement {
    div()
        .ltr()
        .min_w_0()
        .truncate()
        .child(text.into())
        .into_any_element()
}

/// A wrapping row of metadata items with dots between them.
fn meta_row(items: impl IntoIterator<Item = AnyElement>, cx: &App) -> gpui_kit::Div {
    let mut row = h_flex()
        .flex_wrap()
        .min_w_0()
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

/// A page section: title and description, optional trailing actions, content.
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

fn muted_line(text: impl Into<SharedString>, cx: &App) -> gpui_kit::Div {
    div()
        .text_sm()
        .text_color(cx.theme().muted_foreground)
        .child(text.into())
}

impl ModelsPage {
    fn engine_short_name(&self, engine: &str, cx: &App) -> String {
        self.engines(cx)
            .iter()
            .find(|e| e.id == engine)
            .map(|e| e.short_name().to_string())
            .unwrap_or_else(|| engine.to_string())
    }

    fn render_model_row(&self, model: &ModelInfo, cx: &mut Context<Self>) -> AnyElement {
        let theme = cx.theme();
        let pending = self.pending.contains(&model.id);
        let key = |prefix: &str| SharedString::from(format!("{prefix}-{}", model.id));

        let mut tags: Vec<AnyElement> = Vec::new();
        if model.in_use {
            tags.push(
                Tag::primary()
                    .xsmall()
                    .child("در حال استفاده")
                    .into_any_element(),
            );
        }
        if model.is_local() {
            tags.push(
                Tag::info()
                    .outline()
                    .xsmall()
                    .child("محلی")
                    .into_any_element(),
            );
        }
        if !model.commercial {
            tags.push(
                div()
                    .id(key("nc"))
                    .tooltip(|window, cx| {
                        Tooltip::new("مجوز این مدل استفادهٔ تجاری را اجازه نمی‌دهد.")
                            .build(window, cx)
                    })
                    .child(Tag::warning().xsmall().child("غیرتجاری"))
                    .into_any_element(),
            );
        }
        if !model.license.is_empty() {
            tags.push(
                Tag::secondary()
                    .outline()
                    .xsmall()
                    .child(div().ltr().child(model.license.clone()))
                    .into_any_element(),
            );
        }
        if model.gated {
            tags.push(
                div()
                    .id(key("gated"))
                    .tooltip(|window, cx| {
                        Tooltip::new(
                            "پیش از دانلود باید در Hugging Face شرایط مدل را بپذیری و توکن را این‌جا بگذاری.",
                        )
                        .build(window, cx)
                    })
                    .child(Tag::secondary().outline().xsmall().child("نیاز به پذیرش شرایط"))
                    .into_any_element(),
            );
        }

        let mut meta: Vec<AnyElement> = Vec::new();
        if let Some(engine) = &model.engine {
            meta.push(meta_text(self.engine_short_name(engine, cx)));
        }
        if let Some(size) = model.size_mb {
            meta.push(meta_text(format_size(size as f64)));
        }
        match (&model.repo, &model.path) {
            (_, Some(path)) if model.is_local() => meta.push(meta_ltr(path.clone())),
            (Some(repo), _) => meta.push(meta_ltr(repo.clone())),
            _ => {}
        }

        let id = model.id.clone();
        let catalog = !model.is_local();
        let action: Option<AnyElement> =
            match model.status {
                ModelStatus::Missing if catalog => Some(
                    Button::new(key("download"))
                        .small()
                        .icon(IconName::Download)
                        .label("دانلود")
                        .loading(pending)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.download(id.clone(), window, cx)
                        }))
                        .into_any_element(),
                ),
                ModelStatus::Downloading => Some(
                    Button::new(key("cancel"))
                        .small()
                        .icon(IconName::X)
                        .label("لغو")
                        .loading(pending)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.cancel(id.clone(), window, cx)
                        }))
                        .into_any_element(),
                ),
                ModelStatus::Ready if catalog => Some(
                    Button::new(key("delete"))
                        .ghost()
                        .small()
                        .icon(IconName::Trash)
                        .tooltip("حذف از دیسک…")
                        .loading(pending)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.confirm_delete(id.clone(), window, cx)
                        }))
                        .into_any_element(),
                ),
                ModelStatus::Error if catalog => Some(
                    Button::new(key("retry"))
                        .small()
                        .icon(IconName::RotateCcw)
                        .label("تلاش دوباره")
                        .loading(pending)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.download(id.clone(), window, cx)
                        }))
                        .into_any_element(),
                ),
                _ => None,
            };

        let progress = (model.status == ModelStatus::Downloading).then(|| {
            let fraction = model.progress();
            let mut numbers = vec![meta_text(format_progress(
                model.downloaded_mb,
                model.size_mb,
            ))];
            if let Some(fraction) = fraction {
                numbers.push(meta_text(format!(
                    "{}٪",
                    fa_digits(&format!("{:.0}", fraction * 100.))
                )));
            }
            h_flex()
                .w_full()
                .gap_3()
                .child(
                    div().flex_1().min_w_0().ltr().child(
                        Progress::new(key("progress"))
                            .small()
                            .loading(fraction.is_none())
                            .value((fraction.unwrap_or(0.) * 100.) as f32)
                            .accessibility_label("پیشرفت دانلود"),
                    ),
                )
                .child(meta_row(numbers, cx).flex_shrink_0())
        });

        let theme_danger = theme.danger;
        let muted = theme.muted_foreground;
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
                    .items_start()
                    .child(div().pt_0p5().child(status_tag(model.status)))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .gap_1()
                            .child(
                                h_flex()
                                    .flex_wrap()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(FontWeight::MEDIUM)
                                            .child(model.display_name().to_string()),
                                    )
                                    .children(tags),
                            )
                            .when(!model.description.trim().is_empty(), |this| {
                                this.child(
                                    div()
                                        .text_xs()
                                        .text_color(muted)
                                        .child(model.description.clone()),
                                )
                            })
                            .when(!meta.is_empty(), |this| this.child(meta_row(meta, cx))),
                    )
                    .when_some(action, |this, action| {
                        this.child(div().flex_shrink_0().child(action))
                    }),
            )
            .when_some(progress, |this, line| this.child(line))
            .when(model.status == ModelStatus::Error, |this| {
                this.child(
                    div().text_xs().text_color(theme_danger).child(
                        model
                            .error
                            .clone()
                            .filter(|e| !e.trim().is_empty())
                            .unwrap_or_else(|| "دانلود ناموفق بود.".into()),
                    ),
                )
            })
            .into_any_element()
    }

    fn render_model_list(
        &self,
        kind: ModelKind,
        empty: &'static str,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let mut models: Vec<&ModelInfo> = self.models.iter().filter(|m| m.kind == kind).collect();
        // Catalog entries first, local ones after; otherwise the sidecar's order.
        models.sort_by_key(|m| m.is_local());
        if models.is_empty() {
            return muted_line(
                if self.loading && !self.loaded {
                    "در حال بارگذاری…"
                } else {
                    empty
                },
                cx,
            )
            .into_any_element();
        }
        let rows: Vec<AnyElement> = models
            .into_iter()
            .map(|m| self.render_model_row(m, cx))
            .collect();
        v_flex().w_full().children(rows).into_any_element()
    }

    fn render_tts_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let engines = self.engines(cx);
        let overrides: Vec<AnyElement> = engines
            .iter()
            .filter(|engine| self.engine_has_choices(&engine.id))
            .filter_map(|engine| {
                let (_, select) = self
                    .engine_selects
                    .iter()
                    .find(|(id, _)| *id == engine.id)?;
                Some(
                    h_flex()
                        .gap_3()
                        .text_sm()
                        .child(
                            div()
                                .w(px(160.))
                                .flex_shrink_0()
                                .child(format!("مدل {}", engine.short_name())),
                        )
                        .child(
                            div().w(px(320.)).flex_shrink_0().child(
                                Select::new(select)
                                    .small()
                                    .disabled(self.config.is_none() || self.config_pending)
                                    .accessibility_label("مدل موتور"),
                            ),
                        )
                        .into_any_element(),
                )
            })
            .collect();
        let has_overrides = !overrides.is_empty();
        let content = v_flex()
            .w_full()
            .gap_3()
            .when(has_overrides, |this| {
                this.child(v_flex().gap_2().children(overrides))
            })
            .child(self.render_model_list(ModelKind::Tts, "مدل گفتاری در فهرست نیست.", cx));
        section(
            true,
            "ساخت گفتار (TTS)",
            Some(if has_overrides {
                "مدل‌های موتورهای صدا. برای هر موتور می‌توانی به‌جای مدل پیش‌فرض، مدل آمادهٔ دیگری انتخاب کنی."
                    .into()
            } else {
                "مدل‌های موتورهای صدا. موتوری که مدلش دانلود نشده، تا دانلود آن آماده نیست.".into()
            }),
            None,
            content,
            cx,
        )
    }

    fn render_stt_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let picker = h_flex().gap_3().text_sm().child("مدل فعال").child(
            div().w(px(320.)).flex_shrink_0().child(
                Select::new(&self.stt_select)
                    .small()
                    .placeholder("مدل آماده‌ای نیست")
                    .disabled(self.config.is_none() || self.config_pending)
                    .accessibility_label("مدل تشخیص گفتار"),
            ),
        );
        let content = v_flex()
            .w_full()
            .gap_3()
            .child(picker)
            .child(self.render_model_list(ModelKind::Stt, "مدل تشخیص گفتاری در فهرست نیست.", cx));
        section(
            false,
            "تشخیص گفتار (STT)",
            Some("برای نوشتن متن نمونه‌ها و صحبت در گفتگو. فقط مدل‌های آماده قابل انتخاب‌اند.".into()),
            None,
            content,
            cx,
        )
    }

    fn render_aux_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let content = self.render_model_list(ModelKind::Aux, "مدل کمکی‌ای در فهرست نیست.", cx);
        section(
            false,
            "مدل‌های کمکی",
            Some("مدل‌هایی که موتورها کنارشان لازم دارند، مثل تبدیل نویسه به واج.".into()),
            None,
            content,
            cx,
        )
    }

    fn render_folders_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let theme = cx.theme();
        let dirs = self
            .config
            .as_ref()
            .map(|c| c.model_dirs.clone())
            .unwrap_or_default();
        let list: AnyElement = if dirs.is_empty() {
            muted_line("پوشه‌ای اضافه نشده.", cx).into_any_element()
        } else {
            v_flex()
                .w_full()
                .children(dirs.into_iter().enumerate().map(|(ix, dir)| {
                    let found = self
                        .models
                        .iter()
                        .filter(|m| {
                            m.is_local() && m.path.as_deref().is_some_and(|p| p.starts_with(&dir))
                        })
                        .count();
                    let remove_dir = dir.clone();
                    h_flex()
                        .w_full()
                        .gap_3()
                        .py_2()
                        .border_b_1()
                        .border_color(theme.border)
                        .child(
                            Icon::new(IconName::Folder)
                                .small()
                                .text_color(theme.muted_foreground),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .ltr()
                                .text_sm()
                                .truncate()
                                .child(dir),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme.muted_foreground)
                                .whitespace_nowrap()
                                .child(if found == 0 {
                                    "مدلی پیدا نشد".to_string()
                                } else {
                                    format!("\u{200F}{} مدل", fa_digits(&found.to_string()))
                                }),
                        )
                        .child(
                            Button::new(SharedString::from(format!("remove-dir-{ix}")))
                                .ghost()
                                .xsmall()
                                .icon(IconName::Trash)
                                .tooltip("برداشتن پوشه")
                                .disabled(self.config_pending)
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.remove_folder(remove_dir.clone(), window, cx)
                                })),
                        )
                }))
                .into_any_element()
        };
        section(
            false,
            "پوشه‌های مدل",
            Some(
                "مدل‌هایی که در این پوشه‌ها پیدا شوند با برچسب «محلی» در فهرست‌های بالا می‌آیند. فایل‌ها پاک نمی‌شوند."
                    .into(),
            ),
            Some(
                Button::new("add-model-dir")
                    .small()
                    .icon(IconName::FolderPlus)
                    .label("افزودن پوشه…")
                    .disabled(self.config.is_none() || self.config_pending)
                    .on_click(cx.listener(|this, _, window, cx| this.add_folder(window, cx)))
                    .into_any_element(),
            ),
            list,
            cx,
        )
    }

    fn render_token_section(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let token_set = self.config.as_ref().is_some_and(|c| c.hf_token_set);
        let empty = self.hf_token.read(cx).value().trim().is_empty();
        let status = if token_set {
            Tag::success().outline().xsmall().child("توکن ذخیره شده")
        } else {
            Tag::secondary()
                .outline()
                .xsmall()
                .child("توکنی ذخیره نشده")
        };
        let content = v_flex()
            .w_full()
            .gap_3()
            .child(h_flex().gap_2().child(status))
            .child(
                h_flex()
                    .w_full()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .max_w(px(420.))
                            .ltr()
                            .child(Input::new(&self.hf_token).small().mask_toggle()),
                    )
                    .child(
                        Button::new("save-hf-token")
                            .small()
                            .primary()
                            .icon(IconName::KeyRound)
                            .label("ذخیره")
                            .loading(self.token_pending)
                            .disabled(empty || self.config.is_none())
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.save_token(false, window, cx)
                            })),
                    )
                    .child(
                        Button::new("clear-hf-token")
                            .small()
                            .ghost()
                            .label("پاک کردن")
                            .disabled(!token_set || self.token_pending)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.save_token(true, window, cx)
                            })),
                    ),
            )
            .child(muted_line(
                "ParsVoice-XTTS گیت‌دار است: اول در Hugging Face وارد شو و شرایط مدل را بپذیر، بعد دانلود کن.",
                cx,
            ));
        section(
            false,
            "توکن Hugging Face",
            Some("برای دانلود مدل‌های گیت‌دار. توکن فقط در سایدکار نگه داشته می‌شود.".into()),
            None,
            content,
            cx,
        )
    }
}

impl Render for ModelsPage {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let online = self.sidecar.read(cx).is_online();
        let refresh = Button::new("models-refresh")
            .small()
            .ghost()
            .icon(IconName::RefreshCw)
            .label("بارگذاری دوباره")
            .loading(self.loading)
            .disabled(!online)
            .on_click(cx.listener(|this, _, window, cx| this.reload(window, cx)));
        let body: AnyElement = if !online {
            sidecar_offline(cx).into_any_element()
        } else if let Some(err) = self.load_error.clone().filter(|_| self.models.is_empty()) {
            let hint = if err.contains("sidecar 404") {
                "این نسخهٔ سایدکار مدیریت مدل ندارد. سایدکار را به‌روز کن.".to_string()
            } else {
                err
            };
            empty_state(
                IconName::TriangleAlert,
                "فهرست مدل‌ها بارگذاری نشد",
                hint,
                Some(
                    Button::new("models-retry")
                        .icon(IconName::RefreshCw)
                        .label("تلاش دوباره")
                        .loading(self.loading)
                        .on_click(cx.listener(|this, _, window, cx| this.reload(window, cx))),
                ),
                cx,
            )
            .into_any_element()
        } else {
            div()
                .id("models-scroll")
                .flex_1()
                .min_h_0()
                .overflow_y_scrollbar()
                .child(
                    v_flex()
                        .w_full()
                        .max_w(px(880.))
                        .px_6()
                        .pb_8()
                        .child(self.render_tts_section(cx))
                        .child(self.render_stt_section(cx))
                        .child(self.render_aux_section(cx))
                        .child(self.render_folders_section(cx))
                        .child(self.render_token_section(cx)),
                )
                .into_any_element()
        };

        let header = page_header(
            "مدل‌ها",
            Some("دانلود و مدیریت مدل‌های ساخت گفتار، تشخیص گفتار و مدل‌های کمکی.".into()),
            refresh,
            cx,
        );

        v_flex()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(header)
            .child(div().flex_1().min_h_0().flex().flex_col().child(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(format_size(850.), "\u{200F}۸۵۰ مگابایت");
        assert_eq!(format_size(1536.), "\u{200F}۱٫۵ گیگابایت");
        assert_eq!(
            format_progress(500., Some(800)),
            "\u{200F}۵۰۰ از ۸۰۰ مگابایت"
        );
        assert_eq!(
            format_progress(1024., Some(2048)),
            "\u{200F}۱٫۰ از ۲٫۰ گیگابایت"
        );
        assert_eq!(format_progress(12., None), "\u{200F}۱۲ مگابایت");
    }
}
