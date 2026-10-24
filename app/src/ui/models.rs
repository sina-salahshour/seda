//! Models page: the sidecar's model manager.
//!
//! Lists the TTS, STT and auxiliary models from `GET /models` with their
//! license and status, and lets the user download (with live progress),
//! cancel, delete and retry them. It also edits `/models/config`: the STT
//! model, per-engine model overrides, extra model folders (local models found
//! there show up in the lists) and the Hugging Face token.
//!
//! `GET /models` is fetched when the page is shown and polled every second
//! only while something is downloading. The rendering lives in `render`.

mod render;

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use gpui_kit::component::{
    WindowExt as _,
    button::ButtonVariant,
    input::InputState,
    notification::Notification,
    select::{SelectEvent, SelectItem, SelectState},
};
use gpui_kit::{
    AppContext as _, Context, Entity, PathPromptOptions, SharedString, Subscription, Task, Window,
};

use crate::sidecar::{
    EngineInfo, ModelInfo, ModelKind, ModelStatus, ModelsConfig, ModelsConfigPatch, Sidecar,
    SidecarClient, SidecarGlobal, engine_list,
};

const POLL_INTERVAL: Duration = Duration::from_secs(1);

/// A labelled `Select` choice with a typed value.
#[derive(Clone)]
pub(crate) struct Choice<T: Clone + PartialEq + 'static> {
    value: T,
    label: SharedString,
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

type SttSelect = Entity<SelectState<Vec<Choice<String>>>>;
/// `None` = the catalog default.
type EngineSelect = Entity<SelectState<Vec<Choice<Option<String>>>>>;

fn notify_error(window: &mut Window, cx: &mut gpui_kit::App, what: &str, err: &anyhow::Error) {
    window.push_notification(Notification::error(format!("{what}. {err:#}")), cx);
}

pub struct ModelsPage {
    sidecar: Entity<Sidecar>,
    online: bool,
    /// The page was shown at least once (so reconnects reload it).
    shown: bool,

    models: Vec<ModelInfo>,
    config: Option<ModelsConfig>,
    loading: bool,
    loaded: bool,
    /// Why `GET /models` failed (e.g. an older sidecar without it).
    load_error: Option<String>,
    /// Model ids with a download/cancel/delete request in flight.
    pending: HashSet<String>,
    config_pending: bool,
    token_pending: bool,
    polling: bool,
    _poll: Option<Task<()>>,

    stt_select: SttSelect,
    /// One override picker per engine id, created as engines appear.
    engine_selects: Vec<(String, EngineSelect)>,
    hf_token: Entity<InputState>,

    _subscriptions: Vec<Subscription>,
}

impl ModelsPage {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let sidecar = SidecarGlobal::get(cx);
        let online = sidecar.read(cx).is_online();
        let stt_select =
            cx.new(|cx| SelectState::new(Vec::<Choice<String>>::new(), None, window, cx));
        let hf_token = cx.new(|cx| InputState::new(window, cx).masked(true).placeholder("hf_…"));

        let subscriptions = vec![
            cx.observe_in(&sidecar, window, |this: &mut Self, sidecar, window, cx| {
                let online = sidecar.read(cx).is_online();
                if online && !this.online && this.shown {
                    this.reload(window, cx);
                }
                this.online = online;
                this.sync_engine_selects(window, cx);
                cx.notify();
            }),
            cx.subscribe_in(
                &stt_select,
                window,
                |this: &mut Self, _, event: &SelectEvent<Vec<Choice<String>>>, window, cx| {
                    let SelectEvent::Confirm(Some(id)) = event else {
                        return;
                    };
                    if this.config.as_ref().is_some_and(|c| &c.stt_model != id) {
                        this.patch_config(
                            ModelsConfigPatch {
                                stt_model: Some(id.clone()),
                                ..Default::default()
                            },
                            "مدل تشخیص گفتار عوض نشد",
                            window,
                            cx,
                        );
                    }
                },
            ),
        ];

        Self {
            sidecar,
            online,
            shown: false,
            models: Vec::new(),
            config: None,
            loading: false,
            loaded: false,
            load_error: None,
            pending: HashSet::new(),
            config_pending: false,
            token_pending: false,
            polling: false,
            _poll: None,
            stt_select,
            engine_selects: Vec::new(),
            hf_token,
            _subscriptions: subscriptions,
        }
    }

    /// Called by the app whenever this page becomes the visible one.
    pub fn on_show(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.shown = true;
        if self.sidecar.read(cx).is_online() {
            self.reload(window, cx);
        }
    }

    /// Run `job` with a sidecar client on a worker, then `done` on the UI thread.
    fn spawn_job<T: Send + 'static>(
        &self,
        window: &Window,
        cx: &mut Context<Self>,
        job: impl FnOnce(&SidecarClient) -> Result<T> + Send + 'static,
        done: impl FnOnce(&mut Self, Result<T>, &mut Window, &mut Context<Self>) + 'static,
    ) {
        let client = SidecarClient::from_settings(cx);
        cx.spawn_in(window, async move |this, cx| {
            let result = cx
                .background_executor()
                .spawn(async move { job(&client) })
                .await;
            this.update_in(cx, |this, window, cx| {
                done(this, result, window, cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn engines(&self, cx: &gpui_kit::App) -> Vec<EngineInfo> {
        engine_list(self.sidecar.read(cx).health())
    }

    fn model(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }

    fn any_downloading(&self) -> bool {
        self.models
            .iter()
            .any(|m| m.status == ModelStatus::Downloading)
    }

    /// Fetch the model list and the config.
    fn reload(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.loading {
            return;
        }
        self.loading = true;
        cx.notify();
        self.spawn_job(
            window,
            cx,
            |client| Ok((client.models(), client.models_config())),
            |this, result, window, cx| {
                this.loading = false;
                let (models, config) = result.unwrap_or_else(|err| {
                    let config = Err(anyhow::anyhow!("{err:#}"));
                    (Err(err), config)
                });
                match models {
                    Ok(models) => {
                        this.load_error = None;
                        this.loaded = true;
                        this.apply_models(models, window, cx);
                    }
                    Err(err) => this.load_error = Some(format!("{err:#}")),
                }
                if let Ok(config) = config {
                    this.config = Some(config);
                }
                this.sync_selects(window, cx);
            },
        );
    }

    /// Replace the list, announce finished downloads, and keep polling while
    /// anything is still downloading.
    fn apply_models(
        &mut self,
        models: Vec<ModelInfo>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let mut changed_ready = false;
        for model in &models {
            let Some(old) = self.model(&model.id) else {
                continue;
            };
            if old.status == model.status {
                continue;
            }
            changed_ready |= old.status == ModelStatus::Ready || model.status == ModelStatus::Ready;
            if old.status != ModelStatus::Downloading {
                continue;
            }
            match model.status {
                ModelStatus::Ready => window.push_notification(
                    Notification::success(format!("«{}» دانلود شد.", model.display_name())),
                    cx,
                ),
                ModelStatus::Error => window.push_notification(
                    Notification::error(format!(
                        "دانلود «{}» ناموفق بود. {}",
                        model.display_name(),
                        model.error.as_deref().unwrap_or_default()
                    )),
                    cx,
                ),
                _ => {}
            }
        }
        let ready_before: Vec<&str> = ready_ids(&self.models);
        let ready_after: Vec<&str> = ready_ids(&models);
        changed_ready |= ready_before != ready_after;
        self.models = models;
        if changed_ready {
            self.sync_selects(window, cx);
            // Engine `ready` flags in /health follow the downloads.
            self.sidecar.update(cx, |sidecar, cx| sidecar.retry(cx));
        }
        self.ensure_polling(window, cx);
        cx.notify();
    }

    fn replace_model(&mut self, model: ModelInfo, window: &mut Window, cx: &mut Context<Self>) {
        let mut models = self.models.clone();
        match models.iter_mut().find(|m| m.id == model.id) {
            Some(slot) => *slot = model,
            None => models.push(model),
        }
        self.apply_models(models, window, cx);
    }

    fn ensure_polling(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.polling || !self.any_downloading() {
            return;
        }
        self.polling = true;
        self._poll = Some(cx.spawn_in(window, async move |this, cx| {
            loop {
                cx.background_executor().timer(POLL_INTERVAL).await;
                // Decide and clear `polling` in one update, so a download
                // started right after still gets a fresh loop.
                let next = this.update(cx, |this, cx| {
                    let go = this.any_downloading() && this.sidecar.read(cx).is_online();
                    if !go {
                        this.polling = false;
                    }
                    go.then(|| SidecarClient::from_settings(cx))
                });
                let Ok(Some(client)) = next else {
                    return;
                };
                let result = cx
                    .background_executor()
                    .spawn(async move { client.models() })
                    .await;
                let applied = this.update_in(cx, |this, window, cx| {
                    // Errors are usually transient; the next tick retries.
                    if let Ok(models) = result {
                        this.apply_models(models, window, cx);
                    }
                });
                if applied.is_err() {
                    return;
                }
            }
        }));
    }

    // MARK: Model actions

    fn download(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        if !self.pending.insert(id.clone()) {
            return;
        }
        cx.notify();
        let job_id = id.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.download_model(&job_id),
            move |this, result, window, cx| {
                this.pending.remove(&id);
                match result {
                    Ok(model) => this.replace_model(model, window, cx),
                    Err(err) => {
                        // 409: already downloading; just show the current state.
                        if !format!("{err}").contains("sidecar 409") {
                            notify_error(window, cx, "دانلود شروع نشد", &err);
                        }
                        this.reload(window, cx);
                    }
                }
            },
        );
    }

    fn cancel(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        if !self.pending.insert(id.clone()) {
            return;
        }
        cx.notify();
        let job_id = id.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.cancel_model_download(&job_id),
            move |this, result, window, cx| {
                this.pending.remove(&id);
                match result {
                    Ok(model) => this.replace_model(model, window, cx),
                    Err(err) => notify_error(window, cx, "دانلود لغو نشد", &err),
                }
            },
        );
    }

    fn confirm_delete(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(model) = self.model(&id) else {
            return;
        };
        let name = model.display_name().to_string();
        let in_use = model.in_use;
        let view = cx.entity().downgrade();
        window.open_alert_dialog(cx, move |alert, _, _| {
            let (id, view) = (id.clone(), view.clone());
            alert
                .title(SharedString::from(format!("«{name}» از دیسک حذف شود؟")))
                .description(if in_use {
                    "این مدل الان در حال استفاده است. برای استفادهٔ دوباره باید از نو دانلود شود."
                } else {
                    "نسخهٔ دانلودشده از حافظهٔ Hugging Face پاک می‌شود. بعداً می‌توانی دوباره دانلودش کنی."
                })
                .confirm()
                .ok_text("حذف")
                .ok_variant(ButtonVariant::Danger)
                .cancel_text("انصراف")
                .on_ok(move |_, window, cx| {
                    let id = id.clone();
                    view.update(cx, |this, cx| this.delete(id, window, cx)).ok();
                    true
                })
        });
    }

    fn delete(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        if !self.pending.insert(id.clone()) {
            return;
        }
        cx.notify();
        let job_id = id.clone();
        self.spawn_job(
            window,
            cx,
            move |client| client.delete_model(&job_id),
            move |this, result, window, cx| {
                this.pending.remove(&id);
                if let Err(err) = result {
                    notify_error(window, cx, "مدل حذف نشد", &err);
                }
                this.reload(window, cx);
            },
        );
    }

    // MARK: Config

    fn patch_config(
        &mut self,
        patch: ModelsConfigPatch,
        failed: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.config_pending = true;
        cx.notify();
        self.spawn_job(
            window,
            cx,
            move |client| client.update_models_config(&patch),
            move |this, result, window, cx| {
                this.config_pending = false;
                match result {
                    Ok(config) => {
                        this.config = Some(config);
                        // `in_use` and local models follow the config.
                        this.reload(window, cx);
                        this.sidecar.update(cx, |sidecar, cx| sidecar.retry(cx));
                    }
                    Err(err) => notify_error(window, cx, failed, &err),
                }
                // Put the pickers back to what the config says.
                this.sync_selects(window, cx);
            },
        );
    }

    fn set_engine_model(
        &mut self,
        engine: String,
        model: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(config) = &self.config else {
            return;
        };
        if config.engine_models.get(&engine) == model.as_ref() {
            return;
        }
        let mut map = config.engine_models.clone();
        match model {
            Some(model) => map.insert(engine, model),
            None => map.remove(&engine),
        };
        self.patch_config(
            ModelsConfigPatch {
                engine_models: Some(map),
                ..Default::default()
            },
            "مدل موتور عوض نشد",
            window,
            cx,
        );
    }

    fn add_folder(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let paths = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("افزودن".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let result = paths.await;
            this.update_in(cx, |this, window, cx| match result {
                Ok(Ok(Some(paths))) => {
                    let Some(path) = paths.into_iter().next() else {
                        return;
                    };
                    let path = path.to_string_lossy().into_owned();
                    let mut dirs = this
                        .config
                        .as_ref()
                        .map(|c| c.model_dirs.clone())
                        .unwrap_or_default();
                    if dirs.contains(&path) {
                        return;
                    }
                    dirs.push(path);
                    this.patch_config(
                        ModelsConfigPatch {
                            model_dirs: Some(dirs),
                            ..Default::default()
                        },
                        "پوشه اضافه نشد",
                        window,
                        cx,
                    );
                }
                Ok(Ok(None)) | Err(_) => {}
                Ok(Err(err)) => window.push_notification(
                    Notification::error(format!("باز کردن انتخابگر پوشه ممکن نشد: {err:#}")),
                    cx,
                ),
            })
            .ok();
        })
        .detach();
    }

    fn remove_folder(&mut self, dir: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(config) = &self.config else {
            return;
        };
        let dirs: Vec<String> = config
            .model_dirs
            .iter()
            .filter(|d| **d != dir)
            .cloned()
            .collect();
        self.patch_config(
            ModelsConfigPatch {
                model_dirs: Some(dirs),
                ..Default::default()
            },
            "پوشه حذف نشد",
            window,
            cx,
        );
    }

    /// Save the typed token, or clear the stored one with `clear`.
    fn save_token(&mut self, clear: bool, window: &mut Window, cx: &mut Context<Self>) {
        let token = if clear {
            String::new()
        } else {
            let token = self.hf_token.read(cx).value().trim().to_string();
            if token.is_empty() {
                return;
            }
            token
        };
        self.token_pending = true;
        cx.notify();
        self.spawn_job(
            window,
            cx,
            move |client| {
                client.update_models_config(&ModelsConfigPatch {
                    hf_token: Some(token),
                    ..Default::default()
                })
            },
            move |this, result, window, cx| {
                this.token_pending = false;
                match result {
                    Ok(config) => {
                        this.config = Some(config);
                        this.hf_token
                            .update(cx, |state, cx| state.set_value("", window, cx));
                        window.push_notification(
                            Notification::success(if clear {
                                "توکن Hugging Face پاک شد."
                            } else {
                                "توکن Hugging Face ذخیره شد."
                            }),
                            cx,
                        );
                    }
                    Err(err) => notify_error(window, cx, "توکن ذخیره نشد", &err),
                }
            },
        );
    }

    // MARK: Pickers

    fn sync_selects(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.sync_stt_select(window, cx);
        self.sync_engine_selects(window, cx);
    }

    fn sync_stt_select(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let current = self.config.as_ref().map(|c| c.stt_model.clone());
        let mut items: Vec<Choice<String>> = self
            .models
            .iter()
            .filter(|m| m.kind == ModelKind::Stt && m.status == ModelStatus::Ready)
            .map(|m| Choice {
                value: m.id.clone(),
                label: model_choice_label(m),
            })
            .collect();
        if let Some(current) = current.as_ref().filter(|c| !c.is_empty())
            && !items.iter().any(|item| &item.value == current)
        {
            let label = match self.model(current) {
                Some(model) => format!("{} (آماده نیست)", model.display_name()),
                None => current.clone(),
            };
            items.push(Choice {
                value: current.clone(),
                label: label.into(),
            });
        }
        self.stt_select.update(cx, |select, cx| {
            select.set_items(items, window, cx);
            match &current {
                Some(current) => select.set_selected_value(current, window, cx),
                None => select.set_selected_index(None, window, cx),
            }
        });
    }

    /// Make one override picker per engine and fill it with the engine's
    /// ready models.
    fn sync_engine_selects(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        for engine in self.engines(cx) {
            let items = self.engine_choices(&engine.id);
            let current = self
                .config
                .as_ref()
                .and_then(|c| c.engine_models.get(&engine.id).cloned());
            let select = match self.engine_selects.iter().find(|(id, _)| *id == engine.id) {
                Some((_, select)) => select.clone(),
                None => {
                    let select = cx.new(|cx| {
                        SelectState::new(Vec::<Choice<Option<String>>>::new(), None, window, cx)
                    });
                    let engine_id = engine.id.clone();
                    self._subscriptions.push(cx.subscribe_in(
                        &select,
                        window,
                        move |this: &mut Self,
                              _,
                              event: &SelectEvent<Vec<Choice<Option<String>>>>,
                              window,
                              cx| {
                            let SelectEvent::Confirm(Some(model)) = event else {
                                return;
                            };
                            this.set_engine_model(engine_id.clone(), model.clone(), window, cx);
                        },
                    ));
                    self.engine_selects
                        .push((engine.id.clone(), select.clone()));
                    select
                }
            };
            select.update(cx, |select, cx| {
                select.set_items(items, window, cx);
                select.set_selected_value(&current, window, cx);
            });
        }
    }

    /// Ready models of `engine`, after the "catalog default" choice. An
    /// override that is not ready (any more) stays listed so it shows.
    fn engine_choices(&self, engine: &str) -> Vec<Choice<Option<String>>> {
        let current = self
            .config
            .as_ref()
            .and_then(|c| c.engine_models.get(engine));
        let mut items = vec![Choice {
            value: None,
            label: "پیش‌فرض (کاتالوگ)".into(),
        }];
        items.extend(
            self.models
                .iter()
                .filter(|m| {
                    m.engine.as_deref() == Some(engine)
                        && (m.status == ModelStatus::Ready || Some(&m.id) == current)
                })
                .map(|m| Choice {
                    value: Some(m.id.clone()),
                    label: model_choice_label(m),
                }),
        );
        if let Some(current) = current
            && !items.iter().any(|c| c.value.as_ref() == Some(current))
        {
            items.push(Choice {
                value: Some(current.clone()),
                label: current.clone().into(),
            });
        }
        items
    }

    /// Whether an engine has something to choose besides the default.
    fn engine_has_choices(&self, engine: &str) -> bool {
        self.engine_choices(engine).len() > 1
    }
}

fn ready_ids(models: &[ModelInfo]) -> Vec<&str> {
    models
        .iter()
        .filter(|m| m.status == ModelStatus::Ready)
        .map(|m| m.id.as_str())
        .collect()
}

fn model_choice_label(model: &ModelInfo) -> SharedString {
    if model.is_local() {
        format!("{} (محلی)", model.display_name()).into()
    } else {
        model.display_name().to_string().into()
    }
}
