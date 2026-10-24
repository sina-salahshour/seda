//! Persian/RTL rendering check: labels, wrapped paragraphs, styled runs,
//! markdown, Input and Textarea, in an RTL layout.
//! `cargo run --example rtl_spike`

use std::borrow::Cow;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState, Textarea, TextareaState};
use gpui_kit::component::text::TextView;
use gpui_kit::component::{ActiveTheme, IconName, Theme, ThemeMode, h_flex, v_flex};
use gpui_kit::*;

const FA: &str = "سلام دنیا! این یک آزمایش است.";
const MIX: &str = "من با Claude Code و مدل qwen3.8 در Ollama صحبت می‌کنم (نسخه 2.5).";
const LATIN_FIRST: &str = "Ollama برای مدل‌های محلی و Claude Code برای کار روی پروژه‌ها.";
const LONG: &str = "این یک پاراگراف طولانی است که باید در چند خط شکسته شود تا ببینیم ترتیب کلمات در خطوط شکسته شده درست است یا نه، همراه با English words در وسط.";

struct Spike {
    input: Entity<InputState>,
    area: Entity<TextareaState>,
}

impl Render for Spike {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            .p_4()
            .gap_3()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .child(
                h_flex()
                    .gap_2()
                    .child(Button::new("a").primary().label("اول"))
                    .child(Button::new("b").label("دوم"))
                    .child(gpui_kit::component::Icon::new(IconName::ChevronLeft)),
            )
            .child(div().child("۴ ساعت پیش"))
            .child(div().whitespace_nowrap().child("۴ ساعت پیش (nowrap)"))
            .child(div().child("4 ساعت پیش"))
            .child(div().w(px(200.)).child("۱۲ دقیقه پیش"))
            .child(div().child(FA))
            .child(div().child(MIX))
            .child(div().child(LATIN_FIRST))
            .child(div().w(px(380.)).child(LONG))
            .child(
                StyledText::new(MIX).with_highlights([(
                    10..21,
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        ..Default::default()
                    },
                )]),
            )
            .child(Input::new(&self.input))
            .child(Textarea::new(&self.area).h(px(100.)))
            .child(TextView::markdown(
                "md",
                format!("## سرتیتر\n\nاین **متن پررنگ** است. {LONG}\n\n- مورد اول\n- second item\n\n```\nfn main() {{}}\n```\n"),
            ))
    }
}

fn main() {
    gpui_kit::application()
        .with_assets(gpui_kit::assets::Assets)
        .run(move |cx| {
            gpui_kit::init(cx);
            LayoutDirection::set_default(LayoutDirection::Rtl);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(
                        include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf").as_slice(),
                    ),
                    Cow::Borrowed(include_bytes!("../assets/fonts/Vazirmatn-Bold.ttf").as_slice()),
                ])
                .expect("fonts");
            Theme::change(ThemeMode::Dark, None, cx);
            Theme::update(cx, |t| t.font_family = "Vazirmatn".into());
            let opts = WindowOptions {
                window_bounds: Some(WindowBounds::centered(size(px(800.), px(800.)), cx)),
                app_id: Some("seda-spike".into()),
                ..Default::default()
            };
            gpui_kit::open_window(opts, cx, |window, cx| {
                cx.new(|cx| Spike {
                    input: cx.new(|cx| {
                        let mut state = InputState::new(window, cx).default_value(MIX);
                        state.set_text_align(TextAlign::Right, cx);
                        state
                    }),
                    area: cx.new(|cx| {
                        TextareaState::new(window, cx).default_value(format!("{FA}\n{LONG}"))
                    }),
                })
            })
            .expect("window");
        });
}
