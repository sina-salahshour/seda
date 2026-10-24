//! Seda (صدا): clone your Persian voice and talk with local Ollama models and
//! Claude Code by voice. See SPEC.md at the repository root.

mod app;
mod assets;
mod audio;
mod llm;
mod rtl;
mod sidecar;
mod store;
mod ui;

use std::borrow::Cow;

use gpui_kit::component::{Theme, ThemeMode};
use gpui_kit::*;

use crate::store::{AppSettings, Settings, ThemePreference};

const FONTS: [&[u8]; 4] = [
    include_bytes!("../assets/fonts/Vazirmatn-Regular.ttf"),
    include_bytes!("../assets/fonts/Vazirmatn-Medium.ttf"),
    include_bytes!("../assets/fonts/Vazirmatn-SemiBold.ttf"),
    include_bytes!("../assets/fonts/Vazirmatn-Bold.ttf"),
];

actions!(seda, [Quit]);

fn main() {
    gpui_kit::application()
        .with_assets(assets::AppAssets)
        .run(move |cx| {
            gpui_kit::init(cx);

            AppSettings::init(Settings::load(), cx);
            rtl::apply(cx);

            if let Err(err) = cx
                .text_system()
                .add_fonts(FONTS.iter().map(|bytes| Cow::Borrowed(*bytes)).collect())
            {
                eprintln!("seda: loading Vazirmatn failed: {err:#}");
            }
            apply_theme(None, cx);
            // Vazirmatn carries Latin glyphs too, so it is the whole UI font.
            Theme::update(cx, |theme| theme.font_family = "Vazirmatn".into());

            // SEDA_NO_AUDIO=1 runs without an output device (e.g. for screenshots).
            if std::env::var_os("SEDA_NO_AUDIO").is_none() {
                match audio::Player::new() {
                    Ok(player) => cx.set_global(player),
                    Err(err) => eprintln!("seda: no audio output: {err:#}"),
                }
            }
            sidecar::install_signal_handler();
            sidecar::Sidecar::init(cx);

            cx.on_action(|_: &Quit, cx| cx.quit());
            cx.bind_keys([KeyBinding::new("ctrl-q", Quit, None)]);
            cx.on_window_closed(|cx, _| {
                if cx.windows().is_empty() {
                    cx.quit();
                }
            })
            .detach();

            let options = WindowOptions {
                window_bounds: Some(initial_window_bounds(cx)),
                window_min_size: Some(MIN_WINDOW),
                // SEDA_APP_ID lets test instances use their own Wayland app id.
                app_id: Some(std::env::var("SEDA_APP_ID").unwrap_or_else(|_| "seda".into())),
                titlebar: Some(TitlebarOptions {
                    title: Some("صدا — Seda".into()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            gpui_kit::open_window(options, cx, |window, cx| {
                cx.new(|cx| app::SedaApp::new(window, cx))
            })
            .expect("failed to open the main window");
        });
}

const DEFAULT_WINDOW: Size<Pixels> = Size {
    width: px(1280.),
    height: px(800.),
};
const MIN_WINDOW: Size<Pixels> = Size {
    width: px(900.),
    height: px(600.),
};

/// The saved (or default 16:10, 1280×800) size, scaled down with its aspect
/// ratio kept to fit 90% of the display's visible area, centered.
fn initial_window_bounds(cx: &App) -> WindowBounds {
    let settings = AppSettings::get(cx);
    let wanted = settings
        .window_size
        .filter(|(w, h)| *w >= 100. && *h >= 100.)
        .map(|(w, h)| size(px(w).max(MIN_WINDOW.width), px(h).max(MIN_WINDOW.height)))
        .unwrap_or(DEFAULT_WINDOW);
    // Wayland does not say which output a new window lands on, and gpui
    // reports fractional-scale outputs at the integer scale (a 2560×1600
    // panel at 1.6 shows up as 1280×800), so fit the smallest display: 90% of
    // it, but never below the 16:10 default when the default fits it whole.
    let mut fitted = wanted;
    let smallest = cx
        .displays()
        .into_iter()
        .map(|display| display.visible_bounds().size)
        .min_by(|a, b| (a.width * a.height.as_f32()).cmp(&(b.width * b.height.as_f32())));
    if let Some(visible) = smallest {
        let mut limit = size(visible.width * 0.9, visible.height * 0.9);
        if DEFAULT_WINDOW.width <= visible.width && DEFAULT_WINDOW.height <= visible.height {
            limit = size(
                limit.width.max(DEFAULT_WINDOW.width),
                limit.height.max(DEFAULT_WINDOW.height),
            );
        }
        let scale = (limit.width.as_f32() / wanted.width.as_f32())
            .min(limit.height.as_f32() / wanted.height.as_f32())
            .min(1.0);
        fitted = size(wanted.width * scale, wanted.height * scale);
    }
    let bounds = Bounds::centered(None, fitted, cx);
    if settings.window_maximized {
        WindowBounds::Maximized(bounds)
    } else {
        WindowBounds::Windowed(bounds)
    }
}

/// Apply the theme preference from the settings.
pub fn apply_theme(window: Option<&mut Window>, cx: &mut App) {
    match AppSettings::get(cx).theme {
        ThemePreference::System => Theme::sync_system_appearance(window, cx),
        ThemePreference::Light => Theme::change(ThemeMode::Light, window, cx),
        ThemePreference::Dark => Theme::change(ThemeMode::Dark, window, cx),
    }
    Theme::update(cx, |theme| theme.font_family = "Vazirmatn".into());
}
