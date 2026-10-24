//! JSON shapes of the sidecar HTTP API (SPEC.md, "Sidecar HTTP API").

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Health {
    pub ok: bool,
    pub device: String,
    pub gpu: Option<GpuInfo>,
    #[serde(default)]
    pub engines: Vec<EngineInfo>,
    #[serde(default)]
    pub stt_loaded: bool,
    #[serde(default)]
    pub busy: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: i64,
    pub vram_used_mb: i64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct EngineInfo {
    pub id: String,
    pub name: String,
    pub loaded: bool,
}

/// TTS engines the sidecar knows.
pub const ENGINES: [(&str, &str); 2] = [
    ("omnivoice", "OmniVoice (کیفیت بالا، GPU)"),
    ("pocket_v2", "Pocket TTS فارسی (سریع، CPU)"),
];

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub engine: String,
    pub created_at: String,
    pub active_run: Option<String>,
    pub reference_sample: Option<String>,
    #[serde(default)]
    pub sample_count: u32,
    #[serde(default)]
    pub total_seconds: f64,
}

/// `PATCH /profiles/{pid}`. `None` leaves a field alone; `Some(None)` sends `null`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ProfilePatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_run: Option<Option<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_sample: Option<Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Sample {
    pub id: String,
    #[serde(default)]
    pub text: String,
    pub duration: f64,
    pub created_at: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl RunStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Queued => "در صف",
            Self::Running => "در حال آموزش",
            Self::Done => "تمام شد",
            Self::Failed => "ناموفق",
            Self::Cancelled => "لغو شد",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Run {
    pub id: String,
    pub profile_id: String,
    pub engine: String,
    pub status: RunStatus,
    #[serde(default)]
    pub progress: f64,
    #[serde(default)]
    pub step: u32,
    #[serde(default)]
    pub total_steps: u32,
    pub loss: Option<f64>,
    #[serde(default)]
    pub message: String,
    pub created_at: String,
    pub finished_at: Option<String>,
    #[serde(default)]
    pub sample_count: u32,
    #[serde(default)]
    pub params: serde_json::Value,
}

/// Which adapter `/tts` uses.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TtsRun {
    /// Omit `run_id`: the profile's `active_run`.
    #[default]
    Active,
    /// `run_id: null`: zero-shot, no adapter.
    ZeroShot,
    /// A specific run.
    Run(String),
}

#[derive(Debug, Clone)]
pub struct TtsRequest {
    pub profile_id: String,
    pub text: String,
    pub engine: Option<String>,
    pub run: TtsRun,
    pub speed: Option<f32>,
}
