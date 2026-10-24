//! Streaming chat backends: Ollama (`/api/chat`, NDJSON), OpenAI-compatible
//! endpoints (`/chat/completions`, SSE), Claude Code (`claude -p …
//! --output-format stream-json`) and opencode (`opencode run --format json`).
//! Each turn runs on its own worker thread and reports [`StreamEvent`]s
//! through an async channel, so the UI awaits them on the foreground executor
//! without blocking.

pub mod claude;
pub mod ollama;
pub mod openai;
pub mod opencode;
pub mod skills;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// One event of a streaming reply.
///
/// `Thinking` and everything inside `Sub` are shown only: they are never
/// spoken and never sent back as conversation history.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// The agent's session id (Claude Code `system/init`, opencode
    /// `sessionID`); keep it for `--resume` / `--session`.
    SessionId(String),
    /// Slash commands and skills the agent offers (Claude Code `system/init`
    /// `slash_commands` + `skills`, deduplicated, without the leading `/`).
    Commands(Vec<String>),
    /// More assistant text.
    TextDelta(String),
    /// More reasoning text (thinking blocks, `<think>` spans, `reasoning_content`).
    Thinking(String),
    /// A tool call, compactly: e.g. name `Bash`, summary `ls -la`.
    ToolUse { name: String, summary: String },
    /// A tool finished; `summary` is a short first line of its output.
    ToolResult { summary: String, is_error: bool },
    /// A subagent started. `tool_use_id` is the spawning tool call's id and
    /// identifies the task from now on; `parent` is the task that spawned it
    /// (`None` for the top-level agent).
    TaskStarted {
        tool_use_id: String,
        parent: Option<String>,
        description: String,
        agent_type: String,
        prompt: String,
    },
    /// A subagent's status changed (`running`, `completed`, `failed`,
    /// `killed`, …); `summary` is its short final report when known.
    TaskUpdated {
        tool_use_id: String,
        status: String,
        summary: Option<String>,
    },
    /// `TextDelta`, `Thinking`, `ToolUse` or `ToolResult` from inside the
    /// subagent whose task is `parent` (its spawning tool_use id).
    Sub {
        parent: String,
        event: Box<StreamEvent>,
    },
    /// The turn finished normally.
    Done,
    /// The turn failed; nothing more follows.
    Error(String),
}

/// Cooperative cancellation shared between the UI and a worker thread.
#[derive(Clone, Default, Debug)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Sends events of one turn and guarantees exactly one terminal event
/// (`Done` or `Error`): after it, everything else is dropped. Shared between
/// a worker thread and a cancellation watcher so `Done` reaches the UI
/// promptly even while the worker is blocked on I/O.
#[derive(Clone)]
pub(crate) struct Emitter {
    tx: async_channel::Sender<StreamEvent>,
    finished: Arc<std::sync::Mutex<bool>>,
    cancel: CancelToken,
}

impl Emitter {
    pub(crate) fn new(cancel: CancelToken) -> (Self, async_channel::Receiver<StreamEvent>) {
        let (tx, rx) = async_channel::unbounded();
        let emitter = Self {
            tx,
            finished: Arc::new(std::sync::Mutex::new(false)),
            cancel,
        };
        (emitter, rx)
    }

    /// Send one event. Non-terminal events are dropped once the turn is
    /// cancelled or finished. Returns `false` when nothing more should be sent.
    pub(crate) fn send(&self, event: StreamEvent) -> bool {
        let terminal = matches!(event, StreamEvent::Done | StreamEvent::Error(_));
        let mut finished = match self.finished.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if *finished {
            return false;
        }
        if self.cancel.is_cancelled() {
            // A cancelled turn always ends with `Done`.
            *finished = true;
            let _ = self.tx.send_blocking(StreamEvent::Done);
            return false;
        }
        if terminal {
            *finished = true;
        }
        self.tx.send_blocking(event).is_ok() && !terminal
    }

    pub(crate) fn is_finished(&self) -> bool {
        match self.finished.lock() {
            Ok(guard) => *guard,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Whether the worker should stop (finished, cancelled, or the receiver is gone).
    pub(crate) fn should_stop(&self) -> bool {
        self.cancel.is_cancelled() || self.tx.is_closed() || self.is_finished()
    }

    /// Poll the cancel token every 100 ms on its own thread; once it is set,
    /// emit `Done` right away and run `on_cancel` (e.g. kill a child process).
    /// The watcher exits when the turn finishes by itself.
    pub(crate) fn watch_cancel(&self, on_cancel: impl FnOnce() + Send + 'static) {
        let emitter = self.clone();
        std::thread::spawn(move || {
            loop {
                if emitter.cancel.is_cancelled() {
                    emitter.send(StreamEvent::Done);
                    on_cancel();
                    return;
                }
                if emitter.is_finished() || emitter.tx.is_closed() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });
    }
}

/// Splits streamed text into answer text and `<think>…</think>` reasoning.
/// Stateful across chunks, so a tag cut in two by the network still matches.
#[derive(Debug, Default)]
pub struct ThinkSplitter {
    inside: bool,
    /// Text held back because it may be the start of a tag.
    pending: String,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

impl ThinkSplitter {
    /// Events (`TextDelta` / `Thinking`) for one more chunk.
    pub fn feed(&mut self, chunk: &str) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let mut text = std::mem::take(&mut self.pending);
        text.push_str(chunk);
        loop {
            let tag = if self.inside { THINK_CLOSE } else { THINK_OPEN };
            if let Some(pos) = text.find(tag) {
                self.emit(&text[..pos], &mut events);
                self.inside = !self.inside;
                text.drain(..pos + tag.len());
                continue;
            }
            // Hold back a suffix that could be the start of the tag.
            let split = text.len() - partial_tag_suffix(&text, tag);
            self.pending = text.split_off(split);
            self.emit(&text, &mut events);
            break;
        }
        events
    }

    /// Flush whatever is held back at the end of the stream.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let rest = std::mem::take(&mut self.pending);
        self.emit(&rest, &mut events);
        events
    }

    fn emit(&self, text: &str, events: &mut Vec<StreamEvent>) {
        let event = if self.inside {
            StreamEvent::Thinking(text.to_string())
        } else {
            StreamEvent::TextDelta(text.to_string())
        };
        push_event(events, event);
    }
}

/// Append, merging with the previous event of the same kind.
fn push_event(events: &mut Vec<StreamEvent>, event: StreamEvent) {
    match (events.last_mut(), &event) {
        (Some(StreamEvent::TextDelta(a)), StreamEvent::TextDelta(b))
        | (Some(StreamEvent::Thinking(a)), StreamEvent::Thinking(b)) => a.push_str(b),
        _ => {
            let empty = matches!(&event, StreamEvent::TextDelta(t) | StreamEvent::Thinking(t) if t.is_empty());
            if !empty {
                events.push(event);
            }
        }
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&n| text.len() >= n && text.is_char_boundary(text.len() - n) && text.ends_with(&tag[..n]))
        .unwrap_or(0)
}

/// `~/x` → `$HOME/x`; anything else unchanged.
pub(crate) fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => std::path::PathBuf::from(home)
            .join(rest)
            .to_string_lossy()
            .into_owned(),
        _ => path.to_string(),
    }
}

/// One-line description of a tool call (`Bash` → its command, `Read` → the
/// path, …; otherwise the input as JSON). Tool names match case-insensitively.
pub(crate) fn tool_summary(name: &str, input: &serde_json::Value) -> String {
    use serde_json::Value;
    let fields: &[&str] = match name.to_ascii_lowercase().as_str() {
        "bash" => &["command"],
        "read" | "edit" | "write" | "multiedit" | "notebookedit" => &["file_path", "filePath"],
        "grep" | "glob" => &["pattern"],
        "webfetch" => &["url"],
        "websearch" => &["query"],
        "skill" => &["skill", "name"],
        _ => &[],
    };
    let text = fields
        .iter()
        .find_map(|f| input.get(*f).and_then(Value::as_str))
        .map(str::to_string)
        .unwrap_or_else(|| match input {
            Value::Null => String::new(),
            other => other.to_string(),
        });
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate(&one_line, 80)
}

/// First non-empty line of a tool result (`content` is a string or text blocks).
pub(crate) fn result_summary(content: &serde_json::Value) -> String {
    use serde_json::Value;
    let text = match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default();
    truncate(line, 80)
}

/// Shorten `text` to at most `max` characters, adding `…` when cut.
pub(crate) fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_all(chunks: &[&str]) -> (String, String) {
        let mut splitter = ThinkSplitter::default();
        let mut events: Vec<StreamEvent> = chunks.iter().flat_map(|c| splitter.feed(c)).collect();
        events.extend(splitter.finish());
        let mut text = String::new();
        let mut thinking = String::new();
        for event in events {
            match event {
                StreamEvent::TextDelta(t) => text.push_str(&t),
                StreamEvent::Thinking(t) => thinking.push_str(&t),
                other => panic!("unexpected {other:?}"),
            }
        }
        (text, thinking)
    }

    #[test]
    fn think_tags_across_chunks() {
        assert_eq!(
            split_all(&["<thi", "nk>hmm, ", "let me</th", "ink>Hello", " <b>world</b>"]),
            ("Hello <b>world</b>".into(), "hmm, let me".into())
        );
        // Every split point of the same text gives the same answer.
        let full = "<think>a<b</think>x < y";
        for i in 0..=full.len() {
            let (a, b) = full.split_at(i);
            assert_eq!(split_all(&[a, b]), ("x < y".into(), "a<b".into()), "split at {i}");
        }
        assert_eq!(split_all(&["no tags at all"]), ("no tags at all".into(), String::new()));
        // Unclosed thinking stays thinking.
        assert_eq!(split_all(&["<think>still going"]), (String::new(), "still going".into()));
        // A held-back `<` at the end is flushed.
        assert_eq!(split_all(&["a <"]), ("a <".into(), String::new()));
    }

    #[test]
    fn summaries() {
        assert_eq!(tool_summary("Read", &serde_json::json!({"file_path":"/a/b.rs"})), "/a/b.rs");
        assert_eq!(tool_summary("bash", &serde_json::json!({"command":"ls  -la"})), "ls -la");
        assert_eq!(result_summary(&serde_json::json!("\n first\nsecond")), "first");
    }
}
