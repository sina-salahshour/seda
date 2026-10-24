//! Streaming chat backends: Ollama (`/api/chat`, NDJSON) and Claude Code
//! (`claude -p … --output-format stream-json`). Each turn runs on its own
//! worker thread and reports [`StreamEvent`]s through an async channel, so the
//! UI awaits them on the foreground executor without blocking.

pub mod claude;
pub mod ollama;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// One event of a streaming reply.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Claude Code's session id (from the `system/init` event); keep it for `--resume`.
    SessionId(String),
    /// More assistant text.
    TextDelta(String),
    /// A tool call, compactly: e.g. name `Bash`, summary `ls -la`.
    ToolUse { name: String, summary: String },
    /// A tool finished; `summary` is a short first line of its output.
    ToolResult { summary: String, is_error: bool },
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
