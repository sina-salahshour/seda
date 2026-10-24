//! The window's root view: sidebar navigation + the active page.

use gpui_kit::component::{
    ActiveTheme as _, Icon, Side, Sizable as _, TitleBar, h_flex,
    sidebar::{Sidebar, SidebarFooter, SidebarGroup, SidebarHeader, SidebarMenu, SidebarMenuItem},
    v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    App, AppContext as _, Context, Entity, Global, InteractiveElement as _, IntoElement,
    ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _,
    Subscription, WeakEntity, Window, div, px,
};

use gpui_kit::assets::IconName;

use crate::rtl;
use crate::sidecar::{SidecarGlobal, SidecarStatus};
use crate::store::AppSettings;
use crate::ui::{chats::ChatsPage, models::ModelsPage, settings::SettingsPage, voices::VoicesPage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Voices,
    Chats,
    Models,
    Settings,
}

/// The main window's root view, so any page can switch pages.
struct AppGlobal(WeakEntity<SedaApp>);

impl Global for AppGlobal {}

/// Switch the main window to `page` (e.g. a "go to Models" link).
pub fn show_page(page: Page, window: &mut Window, cx: &mut App) {
    let Some(app) = cx.try_global::<AppGlobal>().and_then(|g| g.0.upgrade()) else {
        return;
    };
    app.update(cx, |this, cx| this.show(page, window, cx));
}

pub struct SedaApp {
    page: Page,
    voices: Entity<VoicesPage>,
    chats: Entity<ChatsPage>,
    models: Entity<ModelsPage>,
    settings: Entity<SettingsPage>,
    _subscriptions: Vec<Subscription>,
    _save_bounds: Option<gpui_kit::Task<()>>,
}

impl SedaApp {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let voices = cx.new(|cx| VoicesPage::new(window, cx));
        let chats = cx.new(|cx| ChatsPage::new(window, cx));
        let models = cx.new(|cx| ModelsPage::new(window, cx));
        let settings = cx.new(|cx| SettingsPage::new(window, cx));
        let root = cx.entity().downgrade();
        cx.set_global(AppGlobal(root));
        let sidecar = SidecarGlobal::get(cx);
        let subscriptions = vec![
            cx.observe(&sidecar, |_, _, cx| cx.notify()),
            cx.observe_window_bounds(window, |this, window, cx| this.remember_bounds(window, cx)),
        ];
        // SEDA_PAGE=voices|chats|models|settings picks the start page (handy for screenshots).
        let page = match std::env::var("SEDA_PAGE").as_deref() {
            Ok("voices") => Page::Voices,
            Ok("models") => Page::Models,
            Ok("settings") => Page::Settings,
            _ => Page::Chats,
        };
        if page == Page::Models {
            models.update(cx, |models, cx| models.on_show(window, cx));
        }
        Self {
            page,
            voices,
            chats,
            models,
            settings,
            _subscriptions: subscriptions,
            _save_bounds: None,
        }
    }

    /// Save the window size and maximized state, debounced, so the next
    /// launch restores them.
    fn remember_bounds(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let maximized = window.is_maximized();
        let size = window.bounds().size;
        let size = (size.width.as_f32().round(), size.height.as_f32().round());
        self._save_bounds = Some(cx.spawn(async move |_, cx| {
            cx.background_executor()
                .timer(std::time::Duration::from_millis(600))
                .await;
            cx.update(|cx| {
                AppSettings::update(cx, |s| {
                    s.window_maximized = maximized;
                    if !maximized {
                        s.window_size = Some(size);
                    }
                });
            });
        }));
    }

    fn menu_item(
        &self,
        page: Page,
        label: &'static str,
        icon: IconName,
        cx: &mut Context<Self>,
    ) -> SidebarMenuItem {
        SidebarMenuItem::new(label)
            .icon(icon)
            .active(self.page == page)
            .on_click(cx.listener(move |this, _, window, cx| this.show(page, window, cx)))
    }

    pub fn show(&mut self, page: Page, window: &mut Window, cx: &mut Context<Self>) {
        if page == Page::Models && self.page != Page::Models {
            self.models
                .update(cx, |models, cx| models.on_show(window, cx));
        }
        self.page = page;
        cx.notify();
    }

    /// Window title bar: drag area (double-click maximizes/restores) and the
    /// minimize / maximize / close buttons. Rows follow the layout direction,
    /// so in RTL the title sits on the right and the controls on the left,
    /// with close at the outer edge.
    fn render_title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let page = match self.page {
            Page::Chats => "گفتگوها",
            Page::Voices => "صداها",
            Page::Models => "مدل‌ها",
            Page::Settings => "تنظیمات",
        };
        let muted = cx.theme().muted_foreground;
        // The component pads its physical left (macOS traffic lights); our
        // own symmetric padding goes on the title instead.
        TitleBar::new().pl_0().child(
            h_flex()
                .h_full()
                .px_3()
                .gap_2()
                .text_sm()
                .child(Icon::new(IconName::AudioLines).small().text_color(muted))
                .child(div().font_weight(gpui_kit::FontWeight::MEDIUM).child("صدا"))
                .child(div().text_color(muted).child("·"))
                .child(div().text_color(muted).child(page)),
        )
    }

    fn render_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let sidecar = SidecarGlobal::get(cx);
        let status = sidecar.read(cx).status().clone();
        let busy = sidecar.read(cx).health().and_then(|h| h.busy.clone());
        let (color, title, details): (_, String, Vec<SharedString>) = match &status {
            SidecarStatus::Checking => (
                cx.theme().muted_foreground,
                "سایدکار".into(),
                vec!["در حال بررسی…".into()],
            ),
            SidecarStatus::Starting => (
                cx.theme().warning,
                "سایدکار".into(),
                vec!["در حال راه‌اندازی…".into()],
            ),
            SidecarStatus::Offline(_) => {
                (cx.theme().danger, "سایدکار".into(), vec!["آفلاین".into()])
            }
            SidecarStatus::Online(health) => {
                let loaded: Vec<&str> = health
                    .engines
                    .iter()
                    .filter(|e| e.loaded)
                    .map(|e| e.name.as_str())
                    .collect();
                let mut details: Vec<SharedString> = vec![health.device.to_uppercase().into()];
                if let Some(gpu) = &health.gpu {
                    details.push(
                        format!(
                            "{:.1}/{:.1} GB",
                            gpu.vram_used_mb as f64 / 1024.,
                            gpu.vram_total_mb as f64 / 1024.
                        )
                        .into(),
                    );
                }

                let title = if loaded.is_empty() {
                    "سایدکار آماده".to_string()
                } else {
                    loaded.join("، ")
                };
                (cx.theme().success, title, details)
            }
        };
        h_flex()
            .id("sidecar-status")
            .gap_2()
            .w_full()
            .min_w_0()
            .child(div().size(px(8.)).flex_shrink_0().rounded_full().bg(color))
            .child(
                v_flex()
                    .min_w_0()
                    .child(div().text_sm().truncate().child(title))
                    .child(crate::ui::common::meta_row(details, cx))
                    .when_some(busy, |this, busy| {
                        this.child(
                            div()
                                .text_xs()
                                .text_color(cx.theme().warning)
                                .child(crate::ui::common::busy_label(&busy)),
                        )
                    }),
            )
            .when(!matches!(status, SidecarStatus::Online(_)), |this| {
                this.cursor_pointer().on_click(move |_, _, cx| {
                    sidecar.update(cx, |s, cx| s.retry(cx));
                })
            })
    }
}

impl Render for SedaApp {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let rtl = rtl::is_rtl(cx);
        let page = match self.page {
            Page::Voices => self.voices.clone().into_any_element(),
            Page::Chats => self.chats.clone().into_any_element(),
            Page::Models => self.models.clone().into_any_element(),
            Page::Settings => self.settings.clone().into_any_element(),
        };
        let body = h_flex()
            .flex_1()
            .min_h_0()
            .w_full()
            .items_stretch()
            .child(
                Sidebar::new("seda-sidebar")
                    .side(if rtl { Side::Right } else { Side::Left })
                    .collapsible(false)
                    .w(px(232.))
                    .header(
                        SidebarHeader::new().child(
                            h_flex()
                                .gap_2()
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .size_8()
                                        .flex_shrink_0()
                                        .rounded(cx.theme().radius)
                                        .bg(cx.theme().sidebar_primary)
                                        .text_color(cx.theme().sidebar_primary_foreground)
                                        .child(Icon::new(IconName::AudioLines).small()),
                                )
                                .child(
                                    v_flex()
                                        .child(
                                            div()
                                                .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                                                .child("صدا"),
                                        )
                                        .child(
                                            div()
                                                .text_xs()
                                                .text_color(cx.theme().muted_foreground)
                                                .child("گفتگو با صدای خودت"),
                                        ),
                                ),
                        ),
                    )
                    .child(
                        SidebarGroup::new("برنامه").child(
                            SidebarMenu::new()
                                .child(self.menu_item(
                                    Page::Chats,
                                    "گفتگوها",
                                    IconName::MessagesSquare,
                                    cx,
                                ))
                                .child(self.menu_item(
                                    Page::Voices,
                                    "صداها",
                                    IconName::AudioWaveform,
                                    cx,
                                ))
                                .child(self.menu_item(Page::Models, "مدل‌ها", IconName::Boxes, cx))
                                .child(self.menu_item(
                                    Page::Settings,
                                    "تنظیمات",
                                    IconName::Settings,
                                    cx,
                                )),
                        ),
                    )
                    .footer(SidebarFooter::new().child(self.render_status(cx))),
            )
            .child(div().flex_1().min_w_0().h_full().child(page));
        v_flex()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .font_family(cx.theme().font_family.clone())
            .child(self.render_title_bar(cx))
            .child(body)
    }
}
