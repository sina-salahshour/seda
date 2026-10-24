//! Persistence the Rust app owns: `settings.json` and `sessions/*.json` under
//! `$SEDA_DATA` (default `~/.local/share/seda`). All functions here are
//! blocking file I/O; call them from a background task.

mod sessions;
mod settings;

use std::path::PathBuf;

pub use sessions::{Backend, ChatMessage, Role, Session};
pub use settings::{AppSettings, PermissionMode, Settings, ThemePreference};

/// `$SEDA_DATA`, or `~/.local/share/seda`.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SEDA_DATA").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("seda")
}

/// Scratch directory for recordings and synthesized audio.
pub fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("seda");
    _ = std::fs::create_dir_all(&dir);
    dir
}

/// A short lowercase random id (8 hex chars), as the SPEC prescribes.
pub fn new_id() -> String {
    format!("{:08x}", rand::random::<u32>())
}

/// Now as an RFC 3339 UTC timestamp.
pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Write `bytes` to `path` atomically (temp file + rename).
pub(crate) fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
