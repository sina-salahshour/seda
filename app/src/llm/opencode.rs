//! The opencode CLI: `opencode run --format json --thinking … -- <prompt>`.
//!
//! opencode prints one JSON object per line:
//! `{"type": "step_start"|"text"|"reasoning"|"tool_use"|"step_finish"|"error",
//! "sessionID": "…", "part": {…}}`. Parts arrive whole (text when it ends,
//! tools when they complete or fail), and there is no end-of-turn event: the
//! turn is over when the process exits.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use serde_json::Value;

use super::{CancelToken, Emitter, StreamEvent, expand_home, result_summary, tool_summary, truncate};

#[derive(Debug, Clone)]
pub struct OpenCodeRequest {
    /// Path to the `opencode` binary (`~/` is expanded).
    pub bin: String,
    pub prompt: String,
    pub cwd: PathBuf,
    /// `--session <id>` when continuing.
    pub session: Option<String>,
    /// `-m provider/model`.
    pub model: Option<String>,
    /// `--variant <effort>`.
    pub variant: Option<String>,
    /// Instructions for a new session (opencode has no append-system-prompt
    /// flag): prepended to the first prompt only, when `session` is `None`.
    pub system_prompt: Option<String>,
}

/// Spawn `opencode run` on a worker thread and stream its events. Setting
/// `cancel` kills the process (and emits `Done`).
pub fn run(request: OpenCodeRequest, cancel: CancelToken) -> async_channel::Receiver<StreamEvent> {
    let (emitter, rx) = Emitter::new(cancel);
    std::thread::spawn(move || run_blocking(request, emitter));
    rx
}

/// `opencode models` → `provider/model` names. Blocking; call off the UI thread.
pub fn models(bin: &str) -> Result<Vec<String>> {
    let bin_path = expand_home(bin.trim());
    let output = Command::new(&bin_path)
        .arg("models")
        .stdin(Stdio::null())
        .output()
        .map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                anyhow!("opencode پیدا نشد. (opencode binary not found: {bin})")
            } else {
                anyhow!("اجرای opencode ناموفق بود. (Failed to start {bin}: {err})")
            }
        })?;
    if !output.status.success() {
        let tail = stderr_tail(&String::from_utf8_lossy(&output.stderr));
        return Err(anyhow!(
            "فهرست مدل‌های opencode دریافت نشد. (opencode models failed: {})\n{tail}",
            output.status
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && l.contains('/'))
        .map(str::to_string)
        .collect())
}

/// The prompt actually sent: the system prompt goes in front of the first
/// message of a new session only.
fn full_prompt(request: &OpenCodeRequest) -> String {
    let new_session = request.session.as_deref().is_none_or(|s| s.is_empty());
    match request.system_prompt.as_deref().map(str::trim) {
        Some(system) if new_session && !system.is_empty() => {
            format!("{system}\n\n---\n\n{}", request.prompt)
        }
        _ => request.prompt.clone(),
    }
}

fn build_command(request: &OpenCodeRequest) -> Command {
    let mut cmd = Command::new(expand_home(request.bin.trim()));
    cmd.args(["run", "--format", "json", "--thinking"]);
    if let Some(model) = request.model.as_deref().filter(|s| !s.trim().is_empty()) {
        cmd.args(["-m", model.trim()]);
    }
    if let Some(variant) = request.variant.as_deref().filter(|s| !s.trim().is_empty()) {
        cmd.args(["--variant", variant.trim()]);
    }
    if let Some(id) = request.session.as_deref().filter(|s| !s.is_empty()) {
        cmd.args(["-s", id]);
    }
    cmd.arg("--dir").arg(&request.cwd);
    // `--` so a prompt starting with `-` is not taken for a flag.
    cmd.arg("--").arg(full_prompt(request));
    cmd.current_dir(&request.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

fn run_blocking(request: OpenCodeRequest, emitter: Emitter) {
    if emitter.should_stop() {
        emitter.send(StreamEvent::Done);
        return;
    }
    if !request.cwd.is_dir() {
        emitter.send(StreamEvent::Error(format!(
            "پوشهٔ کاری وجود ندارد. (Working directory not found: {})",
            request.cwd.display()
        )));
        return;
    }
    let mut child = match build_command(&request).spawn() {
        Ok(child) => child,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            emitter.send(StreamEvent::Error(format!(
                "opencode پیدا نشد؛ مسیر opencode را در تنظیمات بررسی کنید. \
                 (opencode binary not found: {})",
                request.bin
            )));
            return;
        }
        Err(err) => {
            emitter.send(StreamEvent::Error(format!(
                "اجرای opencode ناموفق بود. (Failed to start {}: {err})",
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
    let mut got_error = false;
    if let Some(stdout) = stdout {
        for line in BufReader::new(stdout).lines() {
            if emitter.should_stop() {
                break;
            }
            let Ok(line) = line else { break };
            for event in parser.feed(&line) {
                got_error |= matches!(event, StreamEvent::Error(_));
                emitter.send(event);
            }
            if got_error {
                break;
            }
        }
    }

    if emitter.should_stop() || got_error {
        if !emitter.is_finished() {
            emitter.send(StreamEvent::Done);
        }
        kill(&child);
        let _ = stderr_thread.join();
        return;
    }

    // stdout closed: the turn is over once the process exits.
    let status = child.lock().ok().and_then(|mut c| c.wait().ok());
    let stderr_text = stderr_thread.join().unwrap_or_default();
    if status.is_some_and(|s| s.success()) {
        emitter.send(StreamEvent::Done);
        return;
    }
    let tail = stderr_tail(&stderr_text);
    let status = status.map(|s| s.to_string()).unwrap_or_default();
    let msg = if tail.is_empty() {
        format!("opencode ناموفق بسته شد. (opencode failed: {status})")
    } else {
        format!("opencode خطا داد. (opencode failed: {status})\n{tail}")
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

/// Turns `opencode run --format json` lines into [`StreamEvent`]s.
#[derive(Default)]
struct Parser {
    session_sent: bool,
    /// A text part was emitted this turn (the next one gets a paragraph break).
    text_sent: bool,
    /// Text/reasoning part ids already emitted.
    parts_seen: HashSet<String>,
    /// Tool call ids whose `ToolUse` / `TaskStarted` was emitted.
    calls_started: HashSet<String>,
    /// Tool call ids whose result was emitted.
    calls_finished: HashSet<String>,
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
        let mut events = Vec::new();
        if !self.session_sent {
            let id = value
                .get("sessionID")
                .or_else(|| value.pointer("/part/sessionID"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            if let Some(id) = id {
                self.session_sent = true;
                events.push(StreamEvent::SessionId(id.to_string()));
            }
        }
        let part = value.get("part").unwrap_or(&Value::Null);
        match value.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" => {
                if self.first_time(part)
                    && let Some(text) = part.get("text").and_then(Value::as_str)
                    && !text.trim().is_empty()
                {
                    let text = if self.text_sent && !text.starts_with(char::is_whitespace) {
                        format!("\n\n{text}")
                    } else {
                        text.to_string()
                    };
                    self.text_sent = true;
                    events.push(StreamEvent::TextDelta(text));
                }
            }
            "reasoning" => {
                if self.first_time(part)
                    && let Some(text) = part.get("text").and_then(Value::as_str)
                    && !text.trim().is_empty()
                {
                    events.push(StreamEvent::Thinking(text.to_string()));
                }
            }
            "tool_use" => self.tool(part, &mut events),
            "error" => events.push(StreamEvent::Error(error_message(&value))),
            _ => {}
        }
        events
    }

    /// Whether this text/reasoning part (by id) has not been emitted yet.
    fn first_time(&mut self, part: &Value) -> bool {
        match part.get("id").and_then(Value::as_str) {
            Some(id) => self.parts_seen.insert(id.to_string()),
            None => true,
        }
    }

    fn tool(&mut self, part: &Value, events: &mut Vec<StreamEvent>) {
        let str_at = |ptr: &str| part.pointer(ptr).and_then(Value::as_str).unwrap_or_default();
        let tool = part.get("tool").and_then(Value::as_str).unwrap_or("tool");
        let call_id = str_at("/callID").to_string();
        let status = str_at("/state/status");
        let input = part.pointer("/state/input").cloned().unwrap_or(Value::Null);
        let input_empty = match &input {
            Value::Null => true,
            Value::Object(map) => map.is_empty(),
            _ => false,
        };
        let is_task = tool == "task";
        // A pending call may not have its input yet; wait for a later update.
        let can_start = !(status == "pending" && input_empty);
        if can_start && (call_id.is_empty() || self.calls_started.insert(call_id.clone())) {
            if is_task {
                let field = |k: &str| input.get(k).and_then(Value::as_str).unwrap_or_default();
                let agent_type = Some(field("subagent_type"))
                    .filter(|s| !s.is_empty())
                    .unwrap_or("general");
                events.push(StreamEvent::TaskStarted {
                    tool_use_id: call_id.clone(),
                    parent: None,
                    description: field("description").to_string(),
                    agent_type: agent_type.to_string(),
                    prompt: field("prompt").to_string(),
                });
            } else {
                let title = str_at("/state/title").trim();
                let summary = if title.is_empty() {
                    tool_summary(tool, &input)
                } else {
                    truncate(&title.split_whitespace().collect::<Vec<_>>().join(" "), 80)
                };
                events.push(StreamEvent::ToolUse { name: tool.to_string(), summary });
            }
        }
        let finished = matches!(status, "completed" | "error");
        if !finished || !(call_id.is_empty() || self.calls_finished.insert(call_id.clone())) {
            return;
        }
        if status == "completed" {
            let output = part.pointer("/state/output").cloned().unwrap_or(Value::Null);
            if is_task {
                events.push(StreamEvent::TaskUpdated {
                    tool_use_id: call_id,
                    status: "completed".into(),
                    summary: Some(task_summary(output.as_str().unwrap_or_default())),
                });
            } else {
                events.push(StreamEvent::ToolResult {
                    summary: result_summary(&output),
                    is_error: false,
                });
            }
        } else {
            let error = match part.pointer("/state/error") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(),
                Some(other) => error_text(other).unwrap_or_else(|| other.to_string()),
            };
            let line = error
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("error");
            let line = truncate(line, 80);
            if is_task {
                events.push(StreamEvent::TaskUpdated {
                    tool_use_id: call_id,
                    status: "failed".into(),
                    summary: Some(line),
                });
            } else {
                events.push(StreamEvent::ToolResult { summary: line, is_error: true });
            }
        }
    }
}

/// The subagent's report without opencode's `task_id:` header and
/// `<task_result>` tags, as one short paragraph.
fn task_summary(output: &str) -> String {
    let text = output
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with("task_id:")
                && *l != "<task_result>"
                && *l != "</task_result>"
        })
        .collect::<Vec<_>>()
        .join(" ");
    let text = text.replace("<task_result>", "").replace("</task_result>", "");
    truncate(text.trim(), 300)
}

/// Message of an `error` event: `error.data.message`, `error.message`,
/// `error.name`, or `error` as a string.
fn error_message(value: &Value) -> String {
    let detail = match value.get("error") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(err) => error_text(err),
        None => value.get("message").and_then(Value::as_str).map(str::to_string),
    };
    match detail.filter(|s| !s.trim().is_empty()) {
        Some(msg) => format!("opencode خطا داد. (opencode error: {msg})"),
        None => "opencode خطا داد. (opencode error)".into(),
    }
}

fn error_text(err: &Value) -> Option<String> {
    ["/data/message", "/message", "/name"]
        .iter()
        .find_map(|p| err.pointer(p).and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CAPTURE: &str = include_str!("testdata/opencode1.jsonl");

    fn parse_all(text: &str) -> Vec<StreamEvent> {
        let mut parser = Parser::default();
        text.lines().flat_map(|l| parser.feed(l)).collect()
    }

    fn line(value: Value) -> String {
        value.to_string()
    }

    #[test]
    fn capture_tool_then_text() {
        assert_eq!(
            parse_all(CAPTURE),
            vec![
                StreamEvent::SessionId("ses_e90b51a21ffeyDSzVMmSfh6sLN".into()),
                StreamEvent::ToolUse { name: "bash".into(), summary: "ls".into() },
                StreamEvent::ToolResult { summary: "a.txt".into(), is_error: false },
                StreamEvent::TextDelta("done".into()),
            ]
        );
    }

    #[test]
    fn reasoning_and_multiple_texts() {
        let lines = [
            line(json!({"type":"reasoning","sessionID":"s1","part":{"id":"r1","type":"reasoning","text":"hmm"}})),
            line(json!({"type":"text","sessionID":"s1","part":{"id":"t1","type":"text","text":"one"}})),
            // Same part again: ignored.
            line(json!({"type":"text","sessionID":"s1","part":{"id":"t1","type":"text","text":"one"}})),
            line(json!({"type":"text","sessionID":"s1","part":{"id":"t2","type":"text","text":"two"}})),
            line(json!({"type":"text","sessionID":"s1","part":{"id":"t3","type":"text","text":"\nthree"}})),
            line(json!({"type":"step_finish","sessionID":"s1","part":{}})),
            "not json".into(),
            line(json!({"type":"weird","sessionID":"s1"})),
        ];
        assert_eq!(
            parse_all(&lines.join("\n")),
            vec![
                StreamEvent::SessionId("s1".into()),
                StreamEvent::Thinking("hmm".into()),
                StreamEvent::TextDelta("one".into()),
                StreamEvent::TextDelta("\n\ntwo".into()),
                StreamEvent::TextDelta("\nthree".into()),
            ]
        );
    }

    #[test]
    fn task_tool() {
        let input = json!({"description":"Find files","prompt":"look around","subagent_type":"explore"});
        let lines = [
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"task","callID":"c1",
                "state":{"status":"running","input":input,"title":"Find files"}}})),
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"task","callID":"c1",
                "state":{"status":"completed","input":input,
                "output":"task_id: ses_x\n\n<task_result>\nFound 3 files.\nAll good.\n</task_result>"}}})),
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"task","callID":"c2",
                "state":{"status":"error","input":{"description":"d","prompt":"p"},"error":"boom\nstack"}}})),
        ];
        assert_eq!(
            parse_all(&lines.join("\n")),
            vec![
                StreamEvent::SessionId("s".into()),
                StreamEvent::TaskStarted {
                    tool_use_id: "c1".into(),
                    parent: None,
                    description: "Find files".into(),
                    agent_type: "explore".into(),
                    prompt: "look around".into(),
                },
                StreamEvent::TaskUpdated {
                    tool_use_id: "c1".into(),
                    status: "completed".into(),
                    summary: Some("Found 3 files. All good.".into()),
                },
                StreamEvent::TaskStarted {
                    tool_use_id: "c2".into(),
                    parent: None,
                    description: "d".into(),
                    agent_type: "general".into(),
                    prompt: "p".into(),
                },
                StreamEvent::TaskUpdated {
                    tool_use_id: "c2".into(),
                    status: "failed".into(),
                    summary: Some("boom".into()),
                },
            ]
        );
    }

    #[test]
    fn tool_error_and_repeated_call_id() {
        let lines = [
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"read","callID":"c1",
                "state":{"status":"pending","input":{}}}})),
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"read","callID":"c1",
                "state":{"status":"running","input":{"filePath":"/a/b.rs"}}}})),
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"read","callID":"c1",
                "state":{"status":"error","input":{"filePath":"/a/b.rs"},"error":"File not found: /a/b.rs\nmore"}}})),
            line(json!({"type":"tool_use","sessionID":"s","part":{"type":"tool","tool":"read","callID":"c1",
                "state":{"status":"error","input":{"filePath":"/a/b.rs"},"error":"File not found: /a/b.rs"}}})),
        ];
        assert_eq!(
            parse_all(&lines.join("\n")),
            vec![
                StreamEvent::SessionId("s".into()),
                StreamEvent::ToolUse { name: "read".into(), summary: "/a/b.rs".into() },
                StreamEvent::ToolResult { summary: "File not found: /a/b.rs".into(), is_error: true },
            ]
        );
    }

    #[test]
    fn error_events() {
        let parse = |v: Value| parse_all(&line(v));
        let last = |events: Vec<StreamEvent>| match events.last() {
            Some(StreamEvent::Error(msg)) => msg.clone(),
            other => panic!("unexpected {other:?}"),
        };
        let msg = last(parse(json!({"type":"error","sessionID":"s",
            "error":{"name":"APIError","data":{"message":"rate limited"}}})));
        assert!(msg.contains("rate limited"), "{msg}");
        let msg = last(parse(json!({"type":"error","error":{"name":"ProviderAuthError"}})));
        assert!(msg.contains("ProviderAuthError"), "{msg}");
        let msg = last(parse(json!({"type":"error","error":{"message":"bad"}})));
        assert!(msg.contains("bad"), "{msg}");
        let msg = last(parse(json!({"type":"error","error":"plain"})));
        assert!(msg.contains("plain"), "{msg}");
    }

    fn request() -> OpenCodeRequest {
        OpenCodeRequest {
            bin: "~/.opencode/bin/opencode".into(),
            prompt: "-hello".into(),
            cwd: PathBuf::from("/tmp"),
            session: None,
            model: Some("opencode/big-pickle".into()),
            variant: Some("high".into()),
            system_prompt: Some("Be brief.".into()),
        }
    }

    fn args(req: &OpenCodeRequest) -> Vec<String> {
        build_command(req)
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn command_line() {
        let req = request();
        let cmd = build_command(&req);
        assert!(cmd.get_program().to_string_lossy().ends_with("/.opencode/bin/opencode"));
        assert_eq!(
            args(&req),
            [
                "run", "--format", "json", "--thinking", "-m", "opencode/big-pickle", "--variant",
                "high", "--dir", "/tmp", "--", "Be brief.\n\n---\n\n-hello",
            ]
        );
        // Continuing: session flag, no system prompt.
        let req = OpenCodeRequest {
            session: Some("ses_1".into()),
            model: None,
            variant: Some(String::new()),
            ..request()
        };
        assert_eq!(
            args(&req),
            ["run", "--format", "json", "--thinking", "-s", "ses_1", "--dir", "/tmp", "--", "-hello"]
        );
        // New session with a blank system prompt: prompt unchanged.
        let req = OpenCodeRequest { system_prompt: Some("  ".into()), ..request() };
        assert_eq!(args(&req).last().map(String::as_str), Some("-hello"));
    }

    #[test]
    fn missing_binary() {
        let rx = run(
            OpenCodeRequest {
                bin: "/nonexistent/opencode".into(),
                cwd: std::env::temp_dir(),
                ..request()
            },
            CancelToken::new(),
        );
        match rx.recv_blocking() {
            Ok(StreamEvent::Error(msg)) => assert!(msg.contains("not found"), "{msg}"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(models("/nonexistent/opencode").unwrap_err().to_string().contains("not found"));
    }
}
