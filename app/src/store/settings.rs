use std::path::PathBuf;

use anyhow::Result;
use gpui_kit::{App, AppContext as _, Global};
use serde::{Deserialize, Serialize};

/// Claude Code `--permission-mode` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    #[default]
    Default,
    AcceptEdits,
    Plan,
    BypassPermissions,
}

impl PermissionMode {
    pub const ALL: [PermissionMode; 4] = [
        Self::Default,
        Self::AcceptEdits,
        Self::Plan,
        Self::BypassPermissions,
    ];

    /// The CLI value.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::Plan => "plan",
            Self::BypassPermissions => "bypassPermissions",
        }
    }

    /// Persian label for pickers.
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "پیش‌فرض (پرسیدن)",
            Self::AcceptEdits => "پذیرش ویرایش‌ها",
            Self::Plan => "فقط برنامه‌ریزی",
            Self::BypassPermissions => "بدون پرسش (خطرناک)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemePreference {
    #[default]
    System,
    Light,
    Dark,
}

pub const DEFAULT_SYSTEM_PROMPT: &str = "تو یک دستیار صوتی فارسی‌زبان هستی. به فارسی طبیعی و محاوره‌ای جواب بده، با جمله‌های کوتاه که خواندنشان با صدا راحت باشد. وقتی پاسخ با صدا خوانده می‌شود از جدول مارک‌داون و فهرست‌های طولانی استفاده نکن.";

/// Appended after the system prompt for agents that use tools (Claude Code,
/// opencode): they talk through voice, so they should work quietly and report briefly.
pub const DEFAULT_AGENT_PROMPT: &str = "تو رابط صوتی و فکری کاربر هستی و پاسخ‌هایت با صدا برای او خوانده می‌شوند؛ پس کوتاه، روشن و محاوره‌ای جواب بده. کارها را خودت انجام بده. فقط وقتی سراغ زیرعامل (subagent) برو که کار بزرگ است، می‌شود آن را موازی جلو برد، خروجی‌اش ذهن و زمینهٔ تو را پر می‌کند، یا خود کاربر خواسته است. خروجی زیرعامل‌ها خوانده نمی‌شود؛ نتیجه‌شان را با چند جملهٔ قابل‌گفتن برای کاربر خلاصه کن. هرگز کد یا لاگ طولانی را در پاسخ نچسبان؛ به‌جایش بگو چه کردی و نتیجه چه شد.";

/// An OpenAI-compatible API (`{base_url}/chat/completions`, `{base_url}/models`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    /// Stable id that sessions refer to.
    pub id: String,
    pub name: String,
    /// e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
}

/// `settings.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub sidecar_url: String,
    pub sidecar_autostart: bool,
    /// The `sidecar/` uv project that `uv run --project <it> seda-sidecar` starts.
    pub sidecar_project: PathBuf,
    pub ollama_url: String,
    /// Default system prompt (Ollama) / appended prompt (Claude).
    pub system_prompt: String,
    pub claude_bin: String,
    pub claude_permission_mode: PermissionMode,
    /// Defaults for new Claude Code sessions (`--model`, `--effort`).
    pub default_claude_model: Option<String>,
    pub default_claude_effort: Option<String>,
    /// Appended after `system_prompt` for tool-using agents (Claude Code, opencode).
    pub agent_prompt: String,
    /// Path to the `opencode` binary.
    pub opencode_bin: String,
    /// OpenAI-compatible endpoints.
    pub endpoints: Vec<Endpoint>,
    /// Right-to-left layout (default on).
    pub rtl: bool,
    pub theme: ThemePreference,
    /// TTS playback speed sent to `/tts`.
    pub tts_speed: f32,
    /// Push-to-talk sends the transcript right away.
    pub ptt_auto_send: bool,
    /// Last window size in logical pixels, restored (and re-clamped) at launch.
    pub window_size: Option<(f32, f32)>,
    pub window_maximized: bool,
    /// cpal input device name; `None` = the system default input.
    pub input_device: Option<String>,
    /// Defaults for new chat sessions.
    pub default_voice_profile: Option<String>,
    pub default_ollama_model: Option<String>,
    pub default_auto_speak: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
        Self {
            sidecar_url: "http://127.0.0.1:8765".into(),
            sidecar_autostart: true,
            // `make install` bakes in the installed copy; dev builds use the repo's.
            sidecar_project: option_env!("SEDA_SIDECAR_DIR")
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from)
                .or_else(|| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .map(|p| p.join("sidecar"))
                })
                .unwrap_or_else(|| home.join("sidecar")),
            ollama_url: "http://127.0.0.1:11434".into(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.into(),
            claude_bin: home
                .join(".local/bin/claude")
                .to_string_lossy()
                .into_owned(),
            claude_permission_mode: PermissionMode::Default,
            default_claude_model: None,
            default_claude_effort: None,
            agent_prompt: DEFAULT_AGENT_PROMPT.into(),
            opencode_bin: home
                .join(".opencode/bin/opencode")
                .to_string_lossy()
                .into_owned(),
            endpoints: Vec::new(),
            rtl: true,
            theme: ThemePreference::System,
            tts_speed: 1.0,
            ptt_auto_send: false,
            input_device: None,
            window_size: None,
            window_maximized: false,
            default_voice_profile: None,
            default_ollama_model: None,
            default_auto_speak: true,
        }
    }
}

impl Settings {
    pub fn path() -> PathBuf {
        super::data_dir().join("settings.json")
    }

    /// Load `settings.json`, falling back to defaults for anything missing or broken.
    pub fn load() -> Self {
        std::fs::read(Self::path())
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        super::write_atomic(&Self::path(), &serde_json::to_vec_pretty(self)?)
    }

    pub fn endpoint(&self, id: &str) -> Option<&Endpoint> {
        self.endpoints.iter().find(|e| e.id == id)
    }

    /// The system prompt for a backend: the voice style, plus the agent
    /// prompt for tool-using agents.
    pub fn prompt_for(&self, uses_tools: bool) -> String {
        let base = self.system_prompt.trim();
        let agent = self.agent_prompt.trim();
        if uses_tools && !agent.is_empty() {
            if base.is_empty() {
                agent.to_string()
            } else {
                format!("{base}\n\n{agent}")
            }
        } else {
            base.to_string()
        }
    }
}

/// The live settings, as a GPUI global.
///
/// Read with `AppSettings::get(cx)`; change with `AppSettings::update`, which
/// saves in the background and refreshes every window.
pub struct AppSettings(Settings);

impl Global for AppSettings {}

impl AppSettings {
    pub fn init(settings: Settings, cx: &mut App) {
        cx.set_global(AppSettings(settings));
    }

    pub fn get(cx: &App) -> &Settings {
        &cx.global::<AppSettings>().0
    }

    pub fn update(cx: &mut App, f: impl FnOnce(&mut Settings)) {
        let settings = &mut cx.global_mut::<AppSettings>().0;
        let before = settings.clone();
        f(settings);
        if *settings == before {
            return;
        }
        let snapshot = settings.clone();
        cx.background_spawn(async move {
            if let Err(err) = snapshot.save() {
                eprintln!("seda: saving settings failed: {err:#}");
            }
        })
        .detach();
        cx.refresh_windows();
    }
}
