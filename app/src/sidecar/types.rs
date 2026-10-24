//! JSON shapes of the sidecar HTTP API (SPEC.md, "Sidecar HTTP API").

use std::collections::BTreeMap;

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
    /// Id of the STT model in use (newer sidecars).
    #[serde(default)]
    pub stt_model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: i64,
    pub vram_used_mb: i64,
}

/// One TTS engine in `/health`. The capability fields are newer additions:
/// an older sidecar leaves them out, and the accessors then fall back to
/// [`ENGINE_DEFAULTS`].
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct EngineInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub loaded: bool,
    #[serde(default)]
    pub uses_gpu: Option<bool>,
    #[serde(default)]
    pub trainable: Option<bool>,
    /// Whether its models are downloaded (`None`: unknown, older sidecar).
    #[serde(default)]
    pub ready: Option<bool>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub commercial: Option<bool>,
    /// Id of the main model the engine uses.
    #[serde(default)]
    pub model: Option<String>,
}

/// Built-in knowledge about an engine, used when `/health` does not say.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineDefaults {
    pub id: &'static str,
    pub label: &'static str,
    pub short: &'static str,
    pub uses_gpu: bool,
    /// LoRA fine-tuning (otherwise "training" picks the best reference clip).
    pub trainable: bool,
    pub license: &'static str,
    pub commercial: bool,
}

/// TTS engines the sidecar knows: the fallback list when `/health` has none.
pub const ENGINE_DEFAULTS: [EngineDefaults; 3] = [
    EngineDefaults {
        id: "omnivoice",
        label: "OmniVoice (کیفیت بالا، GPU)",
        short: "OmniVoice",
        uses_gpu: true,
        trainable: true,
        license: "Apache-2.0",
        commercial: true,
    },
    EngineDefaults {
        id: "pocket_v2",
        label: "Pocket TTS فارسی (سریع، CPU)",
        short: "Pocket TTS",
        uses_gpu: false,
        trainable: false,
        license: "CC-BY-NC-4.0",
        commercial: false,
    },
    EngineDefaults {
        id: "xtts",
        label: "ParsVoice-XTTS",
        short: "XTTS",
        uses_gpu: true,
        trainable: false,
        license: "CPML",
        commercial: false,
    },
];

pub fn engine_defaults(id: &str) -> Option<&'static EngineDefaults> {
    ENGINE_DEFAULTS.iter().find(|e| e.id == id)
}

impl EngineInfo {
    fn defaults(&self) -> Option<&'static EngineDefaults> {
        engine_defaults(&self.id)
    }

    pub fn is_trainable(&self) -> bool {
        self.trainable
            .or_else(|| self.defaults().map(|d| d.trainable))
            .unwrap_or(false)
    }

    pub fn is_commercial(&self) -> bool {
        self.commercial
            .or_else(|| self.defaults().map(|d| d.commercial))
            .unwrap_or(true)
    }

    /// Unknown (older sidecar) counts as ready.
    pub fn is_ready(&self) -> bool {
        self.ready.unwrap_or(true)
    }

    pub fn license(&self) -> Option<&str> {
        self.license
            .as_deref()
            .or_else(|| self.defaults().map(|d| d.license))
    }

    /// Short name for metadata lines.
    pub fn short_name(&self) -> &str {
        self.defaults().map(|d| d.short).unwrap_or(&self.name)
    }

    fn from_defaults(d: &EngineDefaults) -> Self {
        Self {
            id: d.id.into(),
            name: d.label.into(),
            loaded: false,
            uses_gpu: Some(d.uses_gpu),
            trainable: Some(d.trainable),
            ready: None,
            license: Some(d.license.into()),
            commercial: Some(d.commercial),
            model: None,
        }
    }
}

/// The engine list: from `/health` when it has one, else the built-in list.
pub fn engine_list(health: Option<&Health>) -> Vec<EngineInfo> {
    match health {
        Some(health) if !health.engines.is_empty() => health.engines.clone(),
        _ => ENGINE_DEFAULTS
            .iter()
            .map(EngineInfo::from_defaults)
            .collect(),
    }
}

// MARK: Models

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Tts,
    Stt,
    #[default]
    #[serde(other)]
    Aux,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    Local,
    /// Also any source this app does not know yet.
    #[default]
    #[serde(other)]
    Catalog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Downloading,
    Ready,
    Error,
    /// Also any status this app does not know yet.
    #[default]
    #[serde(other)]
    Missing,
}

impl ModelStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Missing => "دانلود نشده",
            Self::Downloading => "در حال دانلود",
            Self::Ready => "آماده",
            Self::Error => "خطا",
        }
    }
}

/// One entry of `GET /models`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub kind: ModelKind,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub source: ModelSource,
    #[serde(default)]
    pub license: String,
    #[serde(default = "default_true")]
    pub commercial: bool,
    #[serde(default)]
    pub gated: bool,
    #[serde(default)]
    pub size_mb: Option<i64>,
    #[serde(default)]
    pub status: ModelStatus,
    #[serde(default)]
    pub downloaded_mb: f64,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub in_use: bool,
}

fn default_true() -> bool {
    true
}

impl ModelInfo {
    pub fn is_local(&self) -> bool {
        self.source == ModelSource::Local
    }

    pub fn display_name(&self) -> &str {
        if self.name.is_empty() {
            &self.id
        } else {
            &self.name
        }
    }

    /// Download progress in 0..=1, when the total size is known.
    pub fn progress(&self) -> Option<f64> {
        let total = self.size_mb.filter(|s| *s > 0)? as f64;
        Some((self.downloaded_mb / total).clamp(0., 1.))
    }
}

/// `GET /models/config`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize, Serialize)]
pub struct ModelsConfig {
    #[serde(default)]
    pub stt_model: String,
    #[serde(default)]
    pub model_dirs: Vec<String>,
    /// Engine id → model id overrides (absent = the catalog default).
    #[serde(default)]
    pub engine_models: BTreeMap<String, String>,
    #[serde(default)]
    pub hf_token_set: bool,
}

/// `PATCH /models/config`. `None` leaves a field alone; `hf_token: Some("")` clears it.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelsConfigPatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stt_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_dirs: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_models: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_token: Option<String>,
}

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
    /// Role → model id the run used (newer sidecars).
    #[serde(default)]
    pub models: BTreeMap<String, String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_health_parses_with_fallbacks() {
        let json = r#"{"ok": true, "device": "cuda",
            "gpu": {"name": "RTX", "vram_total_mb": 8000, "vram_used_mb": 100},
            "engines": [{"id": "omnivoice", "name": "OmniVoice", "loaded": false},
                        {"id": "pocket_v2", "name": "Pocket", "loaded": true}],
            "stt_loaded": false, "busy": null}"#;
        let health: Health = serde_json::from_str(json).unwrap();
        assert_eq!(health.stt_model, None);
        let [omni, pocket] = &health.engines[..] else {
            panic!("two engines")
        };
        assert!(omni.is_trainable() && omni.is_commercial() && omni.is_ready());
        assert!(!pocket.is_trainable() && !pocket.is_commercial() && pocket.is_ready());
        assert_eq!(pocket.short_name(), "Pocket TTS");
    }

    #[test]
    fn new_health_parses() {
        let json = r#"{"ok": true, "device": "cpu", "gpu": null, "stt_model": "whisper",
            "engines": [{"id": "xtts", "name": "ParsVoice-XTTS", "loaded": false,
                "uses_gpu": true, "trainable": true, "ready": false,
                "license": "CPML", "commercial": false, "model": "x/y"}]}"#;
        let health: Health = serde_json::from_str(json).unwrap();
        assert_eq!(health.stt_model.as_deref(), Some("whisper"));
        let xtts = &health.engines[0];
        assert!(xtts.is_trainable() && !xtts.is_ready() && !xtts.is_commercial());
        assert_eq!(xtts.model.as_deref(), Some("x/y"));
        assert_eq!(engine_list(Some(&health)).len(), 1);
        assert_eq!(engine_list(None).len(), ENGINE_DEFAULTS.len());
    }

    #[test]
    fn models_parse() {
        let json = r#"[
            {"id": "k2-fsa/OmniVoice", "kind": "tts", "engine": "omnivoice", "name": "OmniVoice",
             "description": "d", "repo": "k2-fsa/OmniVoice", "path": null, "source": "catalog",
             "license": "Apache-2.0", "commercial": true, "gated": false, "size_mb": 2000,
             "status": "downloading", "downloaded_mb": 500.5, "error": null, "in_use": true},
            {"id": "local:/m/w", "kind": "stt", "engine": null, "name": "w", "description": "",
             "repo": null, "path": "/m/w", "source": "local", "license": "", "commercial": false,
             "gated": false, "size_mb": null, "status": "ready", "downloaded_mb": 0,
             "error": null, "in_use": false},
            {"id": "minimal"},
            {"id": "future", "kind": "vocoder", "status": "verifying", "source": "remote"}
        ]"#;
        let models: Vec<ModelInfo> = serde_json::from_str(json).unwrap();
        assert_eq!(models[0].status, ModelStatus::Downloading);
        assert!((models[0].progress().unwrap() - 0.25025).abs() < 1e-9);
        assert!(models[1].is_local() && models[1].kind == ModelKind::Stt);
        assert_eq!(models[1].progress(), None);
        assert_eq!(models[2].display_name(), "minimal");
        assert!(models[2].commercial);
        assert_eq!(models[3].kind, ModelKind::Aux);
        assert_eq!(models[3].status, ModelStatus::Missing);
        assert_eq!(models[3].source, ModelSource::Catalog);
    }

    #[test]
    fn models_config_roundtrip() {
        let config: ModelsConfig = serde_json::from_str(
            r#"{"stt_model": "w", "model_dirs": ["/a"], "engine_models": {"xtts": "m"},
                "hf_token_set": true}"#,
        )
        .unwrap();
        assert_eq!(
            config.engine_models.get("xtts").map(String::as_str),
            Some("m")
        );
        assert!(config.hf_token_set);
        assert_eq!(
            serde_json::from_str::<ModelsConfig>("{}").unwrap(),
            ModelsConfig::default()
        );

        let patch = ModelsConfigPatch {
            hf_token: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(serde_json::to_string(&patch).unwrap(), r#"{"hf_token":""}"#);
    }

    #[test]
    fn run_models_optional() {
        let base = r#""id": "r", "profile_id": "p", "engine": "omnivoice", "status": "done",
            "loss": null, "created_at": "t", "finished_at": null"#;
        let old: Run = serde_json::from_str(&format!("{{{base}}}")).unwrap();
        assert!(old.models.is_empty());
        let new: Run = serde_json::from_str(&format!(
            r#"{{{base}, "models": {{"base": "k2-fsa/OmniVoice"}}}}"#
        ))
        .unwrap();
        assert_eq!(new.models["base"], "k2-fsa/OmniVoice");
    }
}
