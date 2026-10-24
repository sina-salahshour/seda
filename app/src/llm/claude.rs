//! Claude Code CLI in print mode with stream-json output.

use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::{CancelToken, Emitter, StreamEvent, truncate};

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
}

/// Spawn `claude -p <prompt> --output-format stream-json --verbose
/// --include-partial-messages [...]` in `cwd` on a worker thread and stream
/// its events. Setting `cancel` kills the process (and emits `Done`).
pub fn run(request: ClaudeRequest, cancel: CancelToken) -> async_channel::Receiver<StreamEvent> {
    let (emitter, rx) = Emitter::new(cancel);
    std::thread::spawn(move || run_blocking(request, emitter));
    rx
}

/// `~/x` → `$HOME/x`; anything else unchanged.
fn expand_home(path: &str) -> String {
    match (path.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home)
            .join(rest)
            .to_string_lossy()
            .into_owned(),
        _ => path.to_string(),
    }
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

/// Turns stream-json lines into [`StreamEvent`]s.
#[derive(Default)]
struct Parser {
    /// A `text_delta` was streamed for the current assistant message.
    saw_delta: bool,
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
        // Skip events from inside subagents (Task tool); only the top-level turn is shown.
        if value
            .get("parent_tool_use_id")
            .is_some_and(|p| !p.is_null())
        {
            return Vec::new();
        }
        let str_at = |ptr: &str| value.pointer(ptr).and_then(Value::as_str);
        match str_at("/type").unwrap_or_default() {
            "system" if str_at("/subtype") == Some("init") => str_at("/session_id")
                .map(|id| vec![StreamEvent::SessionId(id.to_string())])
                .unwrap_or_default(),
            "stream_event" => match str_at("/event/type").unwrap_or_default() {
                "message_start" => {
                    self.saw_delta = false;
                    Vec::new()
                }
                "content_block_delta" if str_at("/event/delta/type") == Some("text_delta") => {
                    match str_at("/event/delta/text") {
                        Some(text) if !text.is_empty() => {
                            self.saw_delta = true;
                            vec![StreamEvent::TextDelta(text.to_string())]
                        }
                        _ => Vec::new(),
                    }
                }
                _ => Vec::new(),
            },
            "assistant" => {
                let mut events = Vec::new();
                for block in content_blocks(&value) {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") if !self.saw_delta => {
                            if let Some(text) = block.get("text").and_then(Value::as_str)
                                && !text.is_empty()
                            {
                                events.push(StreamEvent::TextDelta(text.to_string()));
                            }
                        }
                        Some("tool_use") => {
                            let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
                            let input = block.get("input").cloned().unwrap_or(Value::Null);
                            events.push(StreamEvent::ToolUse {
                                name: name.to_string(),
                                summary: tool_summary(name, &input),
                            });
                        }
                        _ => {}
                    }
                }
                // Each assistant event closes one message; the next starts fresh.
                self.saw_delta = false;
                events
            }
            "user" => content_blocks(&value)
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                .map(|b| StreamEvent::ToolResult {
                    summary: result_summary(b.get("content").unwrap_or(&Value::Null)),
                    is_error: b.get("is_error").and_then(Value::as_bool).unwrap_or(false),
                })
                .collect(),
            "result" => {
                let is_error = value
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let subtype = str_at("/subtype").unwrap_or("error");
                if is_error || subtype != "success" {
                    let text = str_at("/result")
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
                } else {
                    vec![StreamEvent::Done]
                }
            }
            _ => Vec::new(),
        }
    }
}

fn content_blocks(value: &Value) -> impl Iterator<Item = &Value> {
    value
        .pointer("/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

/// One-line description of a tool call.
fn tool_summary(name: &str, input: &Value) -> String {
    let field = match name {
        "Bash" => Some("command"),
        "Read" | "Edit" | "Write" | "MultiEdit" | "NotebookEdit" => Some("file_path"),
        "Grep" | "Glob" => Some("pattern"),
        "WebFetch" => Some("url"),
        "WebSearch" => Some("query"),
        _ => None,
    };
    let text = field
        .and_then(|f| input.get(f))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| match input {
            Value::Null => String::new(),
            other => other.to_string(),
        });
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate(&one_line, 80)
}

/// First non-empty line of a tool result (`content` is a string or text blocks).
fn result_summary(content: &Value) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: &str = include_str!("testdata/claude1.jsonl");
    const TOOL: &str = include_str!("testdata/claude2.jsonl");

    fn parse_all(text: &str) -> Vec<StreamEvent> {
        let mut parser = Parser::default();
        text.lines().flat_map(|l| parser.feed(l)).collect()
    }

    #[test]
    fn plain_reply() {
        assert_eq!(
            parse_all(PLAIN),
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
        let events = parse_all(TOOL);
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
