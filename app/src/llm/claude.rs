//! Claude Code CLI in print mode with stream-json output.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::{
    CancelToken, Emitter, StreamEvent, expand_home, result_summary, tool_summary, truncate,
};

#[derive(Debug, Clone)]
pub struct ClaudeRequest {
    /// Path to the `claude` binary.
    pub bin: String,
    pub prompt: String,
    pub cwd: PathBuf,
    /// `--resume <id>` when continuing a session.
    pub resume: Option<String>,
    /// `--permission-mode <mode>`.
    pub permission_mode: Option<String>,
    /// `--append-system-prompt <text>`.
    pub append_system_prompt: Option<String>,
    /// `--model <name>` (alias like `sonnet` or a full model id).
    pub model: Option<String>,
    /// `--effort <level>` (`low`, `medium`, `high`, …).
    pub effort: Option<String>,
}

/// Spawn `claude -p <prompt> --output-format stream-json --verbose
/// --include-partial-messages [...]` in `cwd` on a worker thread and stream
/// its events. Setting `cancel` kills the process (and emits `Done`).
pub fn run(request: ClaudeRequest, cancel: CancelToken) -> async_channel::Receiver<StreamEvent> {
    let (emitter, rx) = Emitter::new(cancel);
    std::thread::spawn(move || run_blocking(request, emitter));
    rx
}

fn build_command(request: &ClaudeRequest) -> Command {
    let mut cmd = Command::new(expand_home(request.bin.trim()));
    cmd.arg("-p").arg(&request.prompt).args([
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
    ]);
    if let Some(id) = request.resume.as_deref().filter(|s| !s.is_empty()) {
        cmd.args(["--resume", id]);
    }
    if let Some(mode) = request.permission_mode.as_deref().filter(|s| !s.is_empty()) {
        cmd.args(["--permission-mode", mode]);
    }
    if let Some(model) = request
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        cmd.args(["--model", model]);
    }
    if let Some(effort) = request
        .effort
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        cmd.args(["--effort", effort]);
    }
    if let Some(text) = request
        .append_system_prompt
        .as_deref()
        .filter(|s| !s.trim().is_empty())
    {
        cmd.args(["--append-system-prompt", text]);
    }
    cmd.current_dir(&request.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

fn run_blocking(request: ClaudeRequest, emitter: Emitter) {
    if emitter.should_stop() {
        emitter.send(StreamEvent::Done);
        return;
    }
    let mut child = match build_command(&request).spawn() {
        Ok(child) => child,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let msg = if request.cwd.is_dir() {
                format!(
                    "Claude Code پیدا نشد؛ مسیر claude را در تنظیمات بررسی کنید. \
                     (Claude Code binary not found: {})",
                    request.bin
                )
            } else {
                format!(
                    "پوشهٔ کاری وجود ندارد. (Working directory not found: {})",
                    request.cwd.display()
                )
            };
            emitter.send(StreamEvent::Error(msg));
            return;
        }
        Err(err) => {
            emitter.send(StreamEvent::Error(format!(
                "اجرای Claude Code ناموفق بود. (Failed to start {}: {err})",
                request.bin
            )));
            return;
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = String::new();
        if let Some(mut stderr) = stderr {
            let _ = stderr.read_to_string(&mut buf);
        }
        buf
    });

    let child = Arc::new(Mutex::new(child));
    {
        let child = child.clone();
        emitter.watch_cancel(move || kill(&child));
    }

    let mut parser = Parser::default();
    let mut got_result = false;
    if let Some(stdout) = stdout {
        for line in BufReader::new(stdout).lines() {
            if emitter.should_stop() {
                break;
            }
            let Ok(line) = line else { break };
            for event in parser.feed(&line) {
                let terminal = matches!(event, StreamEvent::Done | StreamEvent::Error(_));
                got_result |= terminal;
                emitter.send(event);
            }
            if got_result {
                break;
            }
        }
    }

    if emitter.should_stop() || got_result {
        // Cancelled or finished: don't wait on a process that may linger.
        if emitter.should_stop() && !got_result {
            emitter.send(StreamEvent::Done);
        }
        kill(&child);
        let _ = stderr_thread.join();
        return;
    }

    // stdout closed without a `result` event: the process failed.
    let status = child.lock().ok().and_then(|mut c| c.wait().ok());
    let stderr_text = stderr_thread.join().unwrap_or_default();
    let tail = stderr_tail(&stderr_text);
    let status = status.map(|s| s.to_string()).unwrap_or_default();
    let msg = if tail.is_empty() {
        format!("Claude Code بدون پاسخ بسته شد. (claude exited without a result: {status})")
    } else {
        format!("Claude Code خطا داد. (claude failed: {status})\n{tail}")
    };
    emitter.send(StreamEvent::Error(msg));
}

/// Kill and reap the child; harmless if it already exited.
fn kill(child: &Arc<Mutex<Child>>) {
    let mut child = match child.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Ok(None) = child.try_wait() {
        let _ = child.kill();
    }
    let _ = child.wait();
}

/// Last few non-empty lines of stderr.
fn stderr_tail(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(6);
    truncate(&lines[start..].join("\n"), 600)
}

/// Delta bookkeeping for the message an agent is currently streaming.
#[derive(Default)]
struct MessageState {
    /// Id from the last `message_start`, when partial messages are on.
    msg_id: Option<String>,
    /// A `text_delta` was streamed for that message.
    saw_text: bool,
    /// A `thinking_delta` was streamed for that message.
    saw_thinking: bool,
}

/// Turns stream-json lines into [`StreamEvent`]s.
#[derive(Default)]
struct Parser {
    /// Per agent: `None` is the top level, `Some(id)` the subagent spawned
    /// by tool call `id`.
    states: HashMap<Option<String>, MessageState>,
    /// `Agent`/`Task` tool_use ids already announced with `TaskStarted`.
    tasks: HashSet<String>,
    /// `system/task_*` task id → spawning tool_use id.
    task_ids: HashMap<String, String>,
}

fn str_of<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

/// Inner events of a subagent are wrapped in `Sub`.
fn wrap(parent: &Option<String>, event: StreamEvent) -> StreamEvent {
    match parent {
        Some(p) => StreamEvent::Sub {
            parent: p.clone(),
            event: Box::new(event),
        },
        None => event,
    }
}

impl Parser {
    fn feed(&mut self, line: &str) -> Vec<StreamEvent> {
        let line = line.trim();
        if line.is_empty() {
            return Vec::new();
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            return Vec::new();
        };
        // Set on everything a subagent (spawned by that tool call) produces.
        let parent = str_of(&value, "parent_tool_use_id").map(str::to_string);
        let str_at = |ptr: &str| value.pointer(ptr).and_then(Value::as_str);
        match str_at("/type").unwrap_or_default() {
            "system" => self.system(&value, &parent),
            "stream_event" => {
                let state = self.states.entry(parent.clone()).or_default();
                match str_at("/event/type").unwrap_or_default() {
                    "message_start" => {
                        *state = MessageState {
                            msg_id: str_at("/event/message/id").map(str::to_string),
                            ..MessageState::default()
                        };
                        Vec::new()
                    }
                    "content_block_delta" => match str_at("/event/delta/type") {
                        Some("text_delta") => match str_at("/event/delta/text") {
                            Some(text) if !text.is_empty() => {
                                state.saw_text = true;
                                vec![wrap(&parent, StreamEvent::TextDelta(text.to_string()))]
                            }
                            _ => Vec::new(),
                        },
                        Some("thinking_delta") => match str_at("/event/delta/thinking") {
                            Some(text) if !text.is_empty() => {
                                state.saw_thinking = true;
                                vec![wrap(&parent, StreamEvent::Thinking(text.to_string()))]
                            }
                            _ => Vec::new(),
                        },
                        _ => Vec::new(),
                    },
                    _ => Vec::new(),
                }
            }
            "assistant" => self.assistant(&value, &parent),
            "user" => content_blocks(&value)
                .filter(|b| str_of(b, "type") == Some("tool_result"))
                .map(|b| {
                    let is_error = b.get("is_error").and_then(Value::as_bool).unwrap_or(false);
                    match str_of(b, "tool_use_id").filter(|id| self.tasks.contains(*id)) {
                        // A subagent returned; the notification carries its summary.
                        Some(id) => StreamEvent::TaskUpdated {
                            tool_use_id: id.to_string(),
                            status: if is_error { "failed" } else { "completed" }.into(),
                            summary: None,
                        },
                        None => wrap(
                            &parent,
                            StreamEvent::ToolResult {
                                summary: result_summary(b.get("content").unwrap_or(&Value::Null)),
                                is_error,
                            },
                        ),
                    }
                })
                .collect(),
            "result" if parent.is_none() => result_event(&value),
            _ => Vec::new(),
        }
    }

    fn system(&mut self, value: &Value, parent: &Option<String>) -> Vec<StreamEvent> {
        match str_of(value, "subtype").unwrap_or_default() {
            "init" => {
                let mut events = Vec::new();
                if let Some(id) = str_of(value, "session_id") {
                    events.push(StreamEvent::SessionId(id.to_string()));
                }
                let mut seen = HashSet::new();
                let names: Vec<String> = ["slash_commands", "skills"]
                    .iter()
                    .filter_map(|k| value.get(*k).and_then(Value::as_array))
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(|n| n.trim().trim_start_matches('/'))
                    .filter(|n| !n.is_empty() && seen.insert(n.to_string()))
                    .map(str::to_string)
                    .collect();
                if !names.is_empty() {
                    events.push(StreamEvent::Commands(names));
                }
                events
            }
            "task_started" => {
                let Some(tool_use_id) = str_of(value, "tool_use_id") else {
                    return Vec::new();
                };
                if let Some(task_id) = str_of(value, "task_id") {
                    self.task_ids
                        .insert(task_id.to_string(), tool_use_id.to_string());
                }
                if !self.tasks.insert(tool_use_id.to_string()) {
                    return Vec::new();
                }
                vec![StreamEvent::TaskStarted {
                    tool_use_id: tool_use_id.to_string(),
                    parent: parent.clone(),
                    description: str_of(value, "description").unwrap_or_default().to_string(),
                    agent_type: str_of(value, "subagent_type")
                        .filter(|s| !s.is_empty())
                        .unwrap_or("general-purpose")
                        .to_string(),
                    prompt: str_of(value, "prompt").unwrap_or_default().to_string(),
                }]
            }
            "task_updated" => {
                let tool_use_id = str_of(value, "task_id").and_then(|t| self.task_ids.get(t));
                let status = value.pointer("/patch/status").and_then(Value::as_str);
                match (tool_use_id, status) {
                    (Some(id), Some(status)) => vec![StreamEvent::TaskUpdated {
                        tool_use_id: id.clone(),
                        status: status.to_string(),
                        summary: None,
                    }],
                    _ => Vec::new(),
                }
            }
            "task_notification" => {
                let tool_use_id = str_of(value, "tool_use_id")
                    .map(str::to_string)
                    .or_else(|| {
                        str_of(value, "task_id").and_then(|t| self.task_ids.get(t).cloned())
                    });
                let (Some(tool_use_id), Some(status)) = (tool_use_id, str_of(value, "status"))
                else {
                    return Vec::new();
                };
                vec![StreamEvent::TaskUpdated {
                    tool_use_id,
                    status: status.to_string(),
                    summary: str_of(value, "summary")
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string),
                }]
            }
            _ => Vec::new(),
        }
    }

    fn assistant(&mut self, value: &Value, parent: &Option<String>) -> Vec<StreamEvent> {
        let state = self.states.entry(parent.clone()).or_default();
        // Deltas count only for the message they were streamed for.
        let msg_id = value.pointer("/message/id").and_then(Value::as_str);
        let same = msg_id.is_none() || state.msg_id.as_deref() == msg_id;
        let (saw_text, saw_thinking) = (same && state.saw_text, same && state.saw_thinking);
        let mut events = Vec::new();
        for block in content_blocks(value) {
            match str_of(block, "type") {
                Some("text") if !saw_text => {
                    if let Some(text) = str_of(block, "text").filter(|t| !t.is_empty()) {
                        events.push(wrap(parent, StreamEvent::TextDelta(text.to_string())));
                    }
                }
                Some("thinking") if !saw_thinking => {
                    // Redacted thinking has only a signature: nothing to show.
                    if let Some(text) = str_of(block, "thinking").filter(|t| !t.trim().is_empty()) {
                        events.push(wrap(parent, StreamEvent::Thinking(text.to_string())));
                    }
                }
                Some("tool_use") => {
                    let name = str_of(block, "name").unwrap_or("tool");
                    let input = block.get("input").cloned().unwrap_or(Value::Null);
                    match str_of(block, "id") {
                        Some(id) if matches!(name, "Agent" | "Task") => {
                            if self.tasks.insert(id.to_string()) {
                                events.push(StreamEvent::TaskStarted {
                                    tool_use_id: id.to_string(),
                                    parent: parent.clone(),
                                    description: str_of(&input, "description")
                                        .unwrap_or_default()
                                        .to_string(),
                                    agent_type: str_of(&input, "subagent_type")
                                        .filter(|s| !s.is_empty())
                                        .unwrap_or("general-purpose")
                                        .to_string(),
                                    prompt: str_of(&input, "prompt")
                                        .unwrap_or_default()
                                        .to_string(),
                                });
                            }
                        }
                        _ => events.push(wrap(
                            parent,
                            StreamEvent::ToolUse {
                                name: name.to_string(),
                                summary: tool_summary(name, &input),
                            },
                        )),
                    }
                }
                _ => {}
            }
        }
        events
    }
}

/// The top-level `result` event: `Done`, or `Error` with the best message.
fn result_event(value: &Value) -> Vec<StreamEvent> {
    let is_error = value
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let subtype = str_of(value, "subtype").unwrap_or("error");
    if !is_error && subtype == "success" {
        return vec![StreamEvent::Done];
    }
    let text = str_of(value, "result")
        .filter(|t| !t.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            value.get("errors").and_then(Value::as_array).map(|errs| {
                errs.iter()
                    .map(|e| {
                        e.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| e.to_string())
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
        })
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| subtype.to_string());
    vec![StreamEvent::Error(format!("Claude Code: {text}"))]
}

fn content_blocks(value: &Value) -> impl Iterator<Item = &Value> {
    value
        .pointer("/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: &str = include_str!("testdata/claude1.jsonl");
    const TOOL: &str = include_str!("testdata/claude2.jsonl");
    const SUBAGENT: &str = include_str!("testdata/claude3.jsonl");
    const NESTED: &str = include_str!("testdata/claude4.jsonl");

    fn parse_all(text: &str) -> Vec<StreamEvent> {
        let mut parser = Parser::default();
        text.lines().flat_map(|l| parser.feed(l)).collect()
    }

    /// Events without the (long) `Commands` list of a real `system/init`.
    fn parse_no_commands(text: &str) -> Vec<StreamEvent> {
        let events = parse_all(text);
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::Commands(names) if names.iter().any(|n| n == "clear")
        )));
        events
            .into_iter()
            .filter(|e| !matches!(e, StreamEvent::Commands(_)))
            .collect()
    }

    #[test]
    fn plain_reply() {
        assert_eq!(
            parse_no_commands(PLAIN),
            vec![
                StreamEvent::SessionId("ae00ee0a-9769-436b-8644-58d50c9ce3d4".into()),
                StreamEvent::TextDelta("Hi there, fri".into()),
                StreamEvent::TextDelta("end!".into()),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn tool_call() {
        let events = parse_no_commands(TOOL);
        assert_eq!(
            events,
            vec![
                StreamEvent::SessionId("fb6ad4f9-5622-440e-a6bd-bdc32561024e".into()),
                StreamEvent::ToolUse {
                    name: "Bash".into(),
                    summary: "ls".into()
                },
                StreamEvent::ToolResult {
                    summary: "claude1.err".into(),
                    is_error: false
                },
                StreamEvent::TextDelta("The directory has 8 ent".into()),
                StreamEvent::TextDelta("ries.".into()),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn fallback_text_without_deltas() {
        // Older CLIs without partial messages: only the full assistant event.
        let without_deltas: String = PLAIN
            .lines()
            .filter(|l| !l.contains("\"content_block_delta\""))
            .map(|l| format!("{l}\n"))
            .collect();
        let events = parse_all(&without_deltas);
        assert!(events.contains(&StreamEvent::TextDelta("Hi there, friend!".into())));
    }

    #[test]
    fn errors_and_summaries() {
        let mut p = Parser::default();
        let err = r#"{"type":"result","subtype":"error_during_execution","is_error":true,"errors":["boom"]}"#;
        assert_eq!(
            p.feed(err),
            vec![StreamEvent::Error("Claude Code: boom".into())]
        );
        let err =
            r#"{"type":"result","subtype":"success","is_error":true,"result":"Not logged in"}"#;
        assert_eq!(
            p.feed(err),
            vec![StreamEvent::Error("Claude Code: Not logged in".into())]
        );
        let user = r#"{"type":"user","message":{"content":[{"type":"tool_result","content":[{"type":"text","text":"\n  first line\nsecond"}],"is_error":true}]}}"#;
        assert_eq!(
            p.feed(user),
            vec![StreamEvent::ToolResult {
                summary: "first line".into(),
                is_error: true
            }]
        );
        assert_eq!(
            tool_summary("Read", &serde_json::json!({"file_path":"/a/b.rs"})),
            "/a/b.rs"
        );
        assert_eq!(
            tool_summary("Grep", &serde_json::json!({"pattern":"fn main"})),
            "fn main"
        );
        let long = tool_summary(
            "TodoWrite",
            &serde_json::json!({"todos":[{"content":"x".repeat(200)}]}),
        );
        assert!(long.starts_with("{\"todos\"") && long.chars().count() == 80);
        assert!(p.feed("not json").is_empty());
    }

    fn sub(parent: &str, event: StreamEvent) -> StreamEvent {
        StreamEvent::Sub {
            parent: parent.into(),
            event: Box::new(event),
        }
    }

    fn task_updated(id: &str, status: &str, summary: Option<&str>) -> StreamEvent {
        StreamEvent::TaskUpdated {
            tool_use_id: id.into(),
            status: status.into(),
            summary: summary.map(str::to_string),
        }
    }

    /// Real capture (CLI 2.1.310, no partial messages) of one Agent call.
    #[test]
    fn subagent_capture() {
        const AGENT: &str = "toolu_016GadyR8RAQ953kKXR5XPZt";
        assert_eq!(
            parse_all(SUBAGENT),
            vec![
                StreamEvent::SessionId("14d5d802-b58e-44d0-abe3-e1e4cc9862e0".into()),
                StreamEvent::Commands(
                    [
                        "find-skills",
                        "swarm",
                        "tempad-dev:figma-canvas-authoring",
                        "clear",
                        "compact",
                        "model",
                        "dataviz",
                        "anthropic-skills:pdf",
                        "code-review",
                    ]
                    .map(String::from)
                    .to_vec()
                ),
                StreamEvent::TaskStarted {
                    tool_use_id: AGENT.into(),
                    parent: None,
                    description: "ZEBRAMARKER probe".into(),
                    agent_type: "general-purpose".into(),
                    prompt: "Look at your own system prompt. Does it contain the word \
                             ZEBRAMARKER? Reply only YES or NO."
                        .into(),
                },
                task_updated(AGENT, "completed", None),
                task_updated(AGENT, "completed", Some("NO")),
                task_updated(AGENT, "completed", None),
                StreamEvent::TextDelta("The subagent answered: **NO**.".into()),
                StreamEvent::Done,
            ]
        );
    }

    /// Partial messages with thinking, a subagent with tools, and a nested subagent.
    #[test]
    fn nested_subagents() {
        const A: &str = "toolu_01AgentAAAAAAAAAAAAAAAA";
        const B: &str = "toolu_01AgentBBBBBBBBBBBBBBBB";
        assert_eq!(
            parse_all(NESTED),
            vec![
                StreamEvent::SessionId("5b1c2f0e-8d7a-4c39-9e61-2a4f0c7d9b13".into()),
                StreamEvent::Commands(
                    ["review", "clear", "compact", "anthropic-skills:pdf"]
                        .map(String::from)
                        .to_vec()
                ),
                StreamEvent::Thinking("Let me ".into()),
                StreamEvent::Thinking("plan.".into()),
                StreamEvent::TextDelta("I'll dele".into()),
                StreamEvent::TextDelta("gate.".into()),
                StreamEvent::TaskStarted {
                    tool_use_id: A.into(),
                    parent: None,
                    description: "Explore repo".into(),
                    agent_type: "Explore".into(),
                    prompt: "List the files.".into(),
                },
                sub(A, StreamEvent::TextDelta("Checking.".into())),
                sub(
                    A,
                    StreamEvent::ToolUse {
                        name: "Bash".into(),
                        summary: "ls -la".into()
                    }
                ),
                sub(
                    A,
                    StreamEvent::ToolResult {
                        summary: "Cargo.toml".into(),
                        is_error: false
                    }
                ),
                sub(A, StreamEvent::Thinking("Need more detail.".into())),
                StreamEvent::TaskStarted {
                    tool_use_id: B.into(),
                    parent: Some(A.into()),
                    description: "Read src".into(),
                    agent_type: "general-purpose".into(),
                    prompt: "Summarize src/.".into(),
                },
                sub(B, StreamEvent::TextDelta("src has main.rs.".into())),
                task_updated(B, "completed", None),
                task_updated(B, "completed", Some("src has main.rs.")),
                task_updated(B, "completed", None),
                sub(A, StreamEvent::TextDelta("Done exploring.".into())),
                task_updated(A, "completed", None),
                task_updated(A, "completed", None),
                StreamEvent::Thinking("Summarize.".into()),
                StreamEvent::TextDelta("All good.".into()),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn task_started_without_tool_use_and_failed_task() {
        let mut p = Parser::default();
        let started = r#"{"type":"system","subtype":"task_started","task_id":"t1","tool_use_id":"toolu_X","description":"bg job","prompt":"do it"}"#;
        assert_eq!(
            p.feed(started),
            vec![StreamEvent::TaskStarted {
                tool_use_id: "toolu_X".into(),
                parent: None,
                description: "bg job".into(),
                agent_type: "general-purpose".into(),
                prompt: "do it".into(),
            }]
        );
        assert!(p.feed(started).is_empty());
        let updated = r#"{"type":"system","subtype":"task_updated","task_id":"t1","patch":{"status":"killed"}}"#;
        assert_eq!(
            p.feed(updated),
            vec![task_updated("toolu_X", "killed", None)]
        );
        let result = r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_X","content":"boom","is_error":true}]},"parent_tool_use_id":null}"#;
        assert_eq!(
            p.feed(result),
            vec![task_updated("toolu_X", "failed", None)]
        );
    }

    #[test]
    fn model_and_effort_args() {
        let mut request = ClaudeRequest {
            bin: "claude".into(),
            prompt: "hi".into(),
            cwd: std::env::temp_dir(),
            resume: None,
            permission_mode: None,
            append_system_prompt: None,
            model: Some("sonnet".into()),
            effort: Some("high".into()),
        };
        let args = |r: &ClaudeRequest| -> Vec<String> {
            build_command(r)
                .get_args()
                .map(|a| a.to_string_lossy().into_owned())
                .collect()
        };
        let got = args(&request);
        assert!(
            got.windows(2).any(|w| w == ["--model", "sonnet"]),
            "{got:?}"
        );
        assert!(got.windows(2).any(|w| w == ["--effort", "high"]), "{got:?}");
        request.model = Some(" ".into());
        request.effort = None;
        let got = args(&request);
        assert!(
            !got.iter().any(|a| a == "--model" || a == "--effort"),
            "{got:?}"
        );
    }

    #[test]
    fn missing_binary() {
        let rx = run(
            ClaudeRequest {
                bin: "/nonexistent/claude".into(),
                prompt: "hi".into(),
                cwd: std::env::temp_dir(),
                resume: None,
                permission_mode: None,
                append_system_prompt: None,
                model: None,
                effort: None,
            },
            CancelToken::new(),
        );
        match rx.recv_blocking() {
            Ok(StreamEvent::Error(msg)) => assert!(msg.contains("not found"), "{msg}"),
            other => panic!("unexpected {other:?}"),
        }
    }

    fn collect(rx: async_channel::Receiver<StreamEvent>) -> Vec<StreamEvent> {
        std::iter::from_fn(|| rx.recv_blocking().ok()).collect()
    }

    /// Live check: `cargo test claude_live -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn claude_live() {
        let dir = std::env::temp_dir().join(format!("seda-claude-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let req = |prompt: &str, resume: Option<String>| ClaudeRequest {
            bin: "~/.local/bin/claude".into(),
            prompt: prompt.into(),
            cwd: dir.clone(),
            resume,
            permission_mode: None,
            append_system_prompt: None,
            model: None,
            effort: None,
        };
        let t = std::time::Instant::now();
        let events = collect(run(req("Say hi in 3 words.", None), CancelToken::new()));
        eprintln!("turn 1 ({:?}): {events:?}", t.elapsed());
        assert_eq!(events.last(), Some(&StreamEvent::Done));
        let id = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::SessionId(id) => Some(id.clone()),
                _ => None,
            })
            .expect("session id");
        let t = std::time::Instant::now();
        let events = collect(run(
            req("What did you just say?", Some(id)),
            CancelToken::new(),
        ));
        eprintln!("turn 2 ({:?}): {events:?}", t.elapsed());
        assert_eq!(events.last(), Some(&StreamEvent::Done));
    }

    /// Cancel right after start: `cargo test claude_cancel_live -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn claude_cancel_live() {
        let cancel = CancelToken::new();
        let rx = run(
            ClaudeRequest {
                bin: "~/.local/bin/claude".into(),
                prompt: "Write a 500 word story.".into(),
                cwd: std::env::temp_dir(),
                resume: None,
                permission_mode: None,
                append_system_prompt: None,
                model: None,
                effort: None,
            },
            cancel.clone(),
        );
        let t = std::time::Instant::now();
        let mut deltas = 0;
        while let Ok(ev) = rx.recv_blocking() {
            if let StreamEvent::TextDelta(_) = ev {
                deltas += 1;
                if deltas == 3 {
                    cancel.cancel();
                }
            } else if ev == StreamEvent::Done || matches!(ev, StreamEvent::Error(_)) {
                eprintln!("end after {deltas} deltas ({:?}): {ev:?}", t.elapsed());
                assert_eq!(ev, StreamEvent::Done);
                break;
            }
        }
        assert!(rx.recv_blocking().is_err());
    }
}
