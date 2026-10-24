use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Who answers in a chat session.
///
/// Every field added after the first release is `#[serde(default)]`, so older
/// session files keep parsing. A `kind` this build does not know loads as
/// [`Backend::Unknown`], which keeps the raw JSON and writes it back unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Backend {
    Ollama {
        model: String,
        /// `None`/`"off"` = no thinking, `"on"` = thinking, `"low"|"medium"|"high"` = effort.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        think: Option<String>,
    },
    Claude {
        cwd: PathBuf,
        /// From the `system/init` event; passed back as `--resume`.
        #[serde(default)]
        claude_session_id: Option<String>,
        /// `--model` (alias or full name); `None` = the CLI default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// `--effort` (`low|medium|high|xhigh|max`); `None` = the CLI default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effort: Option<String>,
        /// Slash commands + skills from the last `system/init`, for autocomplete.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        commands: Vec<String>,
    },
    /// An OpenAI-compatible `/chat/completions` endpoint from the settings.
    #[serde(rename = "openai")]
    OpenAi {
        /// `Endpoint::id` in the settings.
        endpoint: String,
        model: String,
        /// `reasoning_effort` (`low|medium|high`); `None` = not sent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    /// The opencode CLI (`opencode run --format json`).
    #[serde(rename = "opencode")]
    OpenCode {
        cwd: PathBuf,
        /// `-m provider/model`; `None` = opencode's default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// `--variant` (reasoning effort); `None` = the model default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        variant: Option<String>,
        /// opencode's `sessionID`; passed back as `--session`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
    },
    /// A backend from a newer build: shown read-only, saved back verbatim.
    #[serde(skip)]
    Unknown(serde_json::Value),
}

impl Backend {
    /// The raw `kind` tag of an unknown backend, for display.
    pub fn unknown_kind(&self) -> Option<String> {
        match self {
            Backend::Unknown(raw) => Some(
                raw.get("kind")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("?")
                    .to_string(),
            ),
            _ => None,
        }
    }

    /// Whether the backend runs an agent with tools (gets the agent prompt).
    pub fn uses_tools(&self) -> bool {
        matches!(self, Backend::Claude { .. } | Backend::OpenCode { .. })
    }
}

fn deserialize_backend<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Backend, D::Error> {
    let raw = serde_json::Value::deserialize(d)?;
    match Backend::deserialize(&raw) {
        Ok(backend) => Ok(backend),
        Err(err) => {
            eprintln!("seda: unknown chat backend ({err}); the session is read-only");
            Ok(Backend::Unknown(raw))
        }
    }
}

fn serialize_backend<S: serde::Serializer>(backend: &Backend, s: S) -> Result<S::Ok, S::Error> {
    match backend {
        Backend::Unknown(raw) => raw.serialize(s),
        known => known.serialize(s),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    /// A compact tool-use line (`text` holds the call, then `→ result` / `✗ error`).
    Tool,
    /// A failed turn, shown inline.
    Error,
    /// Reasoning text, shown collapsed; never spoken or sent back.
    Thinking,
    /// A subagent (`task` holds its details); its own messages point at it
    /// through `parent`.
    Task,
}

/// A subagent node of the conversation tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskInfo {
    /// The spawning tool call's id; children carry it as `parent`.
    pub tool_use_id: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub agent_type: String,
    /// `running`, `completed`, `failed`, `killed`, …
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub role: Role,
    pub text: String,
    pub created_at: String,
    /// The subagent (its `TaskInfo::tool_use_id`) this message belongs to;
    /// `None` for the main conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Set for `Role::Task`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskInfo>,
}

impl ChatMessage {
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            id: super::new_id(),
            role,
            text: text.into(),
            created_at: super::now(),
            parent: None,
            task: None,
        }
    }

    pub fn with_parent(mut self, parent: Option<String>) -> Self {
        self.parent = parent;
        self
    }

    /// Part of the spoken conversation (sent back as history).
    pub fn is_history(&self) -> bool {
        self.parent.is_none() && matches!(self.role, Role::User | Role::Assistant)
    }
}

/// `sessions/<id>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(
        deserialize_with = "deserialize_backend",
        serialize_with = "serialize_backend"
    )]
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
            .filter_map(|entry| {
                let path = entry.path();
                let bytes = std::fs::read(&path).ok()?;
                match serde_json::from_slice(&bytes) {
                    Ok(session) => Some(session),
                    Err(err) => {
                        eprintln!("seda: skipping unreadable session {}: {err}", path.display());
                        None
                    }
                }
            })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_files_still_parse() {
        let ollama = r#"{"id":"a1","title":"t","created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z",
            "backend":{"kind":"ollama","model":"qwen3.8:latest"},"voice_profile":null,"auto_speak":true,
            "messages":[{"id":"m1","role":"user","text":"سلام","created_at":"2026-01-01T00:00:00Z"},
                        {"id":"m2","role":"tool","text":"Bash: ls","created_at":"2026-01-01T00:00:00Z"}]}"#;
        let session: Session = serde_json::from_str(ollama).unwrap();
        assert_eq!(
            session.backend,
            Backend::Ollama { model: "qwen3.8:latest".into(), think: None }
        );
        assert_eq!(session.messages[0].parent, None);
        assert_eq!(session.messages[1].task, None);

        let claude = r#"{"id":"a2","title":"t","created_at":"x","updated_at":"x",
            "backend":{"kind":"claude","cwd":"/tmp","claude_session_id":"abc"}}"#;
        let session: Session = serde_json::from_str(claude).unwrap();
        assert_eq!(
            session.backend,
            Backend::Claude {
                cwd: "/tmp".into(),
                claude_session_id: Some("abc".into()),
                model: None,
                effort: None,
                commands: Vec::new(),
            }
        );
        assert!(session.messages.is_empty());
    }

    #[test]
    fn new_backends_round_trip() {
        for backend in [
            Backend::OpenAi { endpoint: "e1".into(), model: "gpt".into(), reasoning: Some("low".into()) },
            Backend::OpenCode {
                cwd: "/w".into(),
                model: Some("opencode/x".into()),
                variant: None,
                session_id: Some("ses_1".into()),
            },
            Backend::Ollama { model: "m".into(), think: Some("high".into()) },
        ] {
            let mut session = Session::new(backend.clone(), None, false);
            let mut task = ChatMessage::new(Role::Task, "");
            task.task = Some(TaskInfo {
                tool_use_id: "toolu_1".into(),
                description: "d".into(),
                agent_type: "Explore".into(),
                status: "completed".into(),
                summary: Some("ok".into()),
            });
            session.messages.push(task);
            session
                .messages
                .push(ChatMessage::new(Role::Thinking, "hmm").with_parent(Some("toolu_1".into())));
            let json = serde_json::to_string(&session).unwrap();
            let back: Session = serde_json::from_str(&json).unwrap();
            assert_eq!(back, session, "{json}");
        }
        let json = serde_json::to_value(Backend::OpenAi {
            endpoint: "e".into(),
            model: "m".into(),
            reasoning: None,
        })
        .unwrap();
        assert_eq!(json["kind"], "openai");
    }

    #[test]
    fn unknown_backend_is_kept() {
        let text = r#"{"id":"a3","title":"t","created_at":"x","updated_at":"x",
            "backend":{"kind":"gemini","model":"g","extra":[1,2]}}"#;
        let session: Session = serde_json::from_str(text).unwrap();
        assert_eq!(session.backend.unknown_kind().as_deref(), Some("gemini"));
        let json = serde_json::to_value(&session).unwrap();
        assert_eq!(json["backend"]["extra"], serde_json::json!([1, 2]));
    }
}
