use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Who answers in a chat session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Backend {
    Ollama {
        model: String,
    },
    Claude {
        cwd: PathBuf,
        /// From the `system/init` event; passed back as `--resume`.
        #[serde(default)]
        claude_session_id: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    /// A compact tool-use line from Claude Code (`text` holds the summary).
    Tool,
    /// A failed turn, shown inline.
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub role: Role,
    pub text: String,
    pub created_at: String,
}

impl ChatMessage {
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            id: super::new_id(),
            role,
            text: text.into(),
            created_at: super::now(),
        }
    }
}

/// `sessions/<id>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub backend: Backend,
    /// Sidecar profile id used for TTS.
    #[serde(default)]
    pub voice_profile: Option<String>,
    #[serde(default)]
    pub auto_speak: bool,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
}

impl Session {
    pub fn new(backend: Backend, voice_profile: Option<String>, auto_speak: bool) -> Self {
        let now = super::now();
        Self {
            id: super::new_id(),
            title: "گفتگوی تازه".into(),
            created_at: now.clone(),
            updated_at: now,
            backend,
            voice_profile,
            auto_speak,
            messages: Vec::new(),
        }
    }

    pub fn dir() -> PathBuf {
        super::data_dir().join("sessions")
    }

    pub fn path(&self) -> PathBuf {
        Self::dir().join(format!("{}.json", self.id))
    }

    /// Every session on disk, most recently updated first. Unreadable files are skipped.
    pub fn load_all() -> Vec<Session> {
        let Ok(entries) = std::fs::read_dir(Self::dir()) else {
            return Vec::new();
        };
        let mut sessions: Vec<Session> = entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
            .filter_map(|entry| std::fs::read(entry.path()).ok())
            .filter_map(|bytes| serde_json::from_slice(&bytes).ok())
            .collect();
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sessions
    }

    pub fn save(&self) -> Result<()> {
        super::write_atomic(&self.path(), &serde_json::to_vec_pretty(self)?)
    }

    pub fn delete(id: &str) -> Result<()> {
        let path = Self::dir().join(format!("{id}.json"));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Set the title from the first user message, if it is still the default.
    pub fn title_from_first_message(&mut self) {
        if self
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .count()
            != 1
        {
            return;
        }
        if let Some(first) = self.messages.iter().find(|m| m.role == Role::User) {
            let title: String = first.text.chars().take(40).collect();
            let title = title.trim().to_string();
            if !title.is_empty() {
                self.title = title;
            }
        }
    }
}
