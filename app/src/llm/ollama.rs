//! Ollama HTTP API: model list and streaming chat.
//!
//! Thinking: the request's `think` field is `false` by default, `true`, or a
//! level (`low|medium|high`); see [`think_value`]. Replies carry reasoning in
//! `message.thinking` (-> [`StreamEvent::Thinking`]); models that inline
//! `<think>` tags in `message.content` are split by a [`ThinkSplitter`].

use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use serde_json::{Value, json};

use super::{CancelToken, Emitter, StreamEvent, ThinkSplitter};

/// One message of the conversation sent to `/api/chat`.
#[derive(Debug, Clone, PartialEq)]
pub struct OllamaMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct OllamaClient {
    base: String,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base: base_url.trim_end_matches('/').to_string(),
        }
    }

    pub fn from_settings(cx: &gpui_kit::App) -> Self {
        Self::new(&crate::store::AppSettings::get(cx).ollama_url)
    }

    /// `GET /api/tags` → model names. Blocking; call off the UI thread.
    pub fn models(&self) -> Result<Vec<String>> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(3))
            .timeout(Duration::from_secs(15))
            .build();
        let url = format!("{}/api/tags", self.base);
        let body: Value = match agent.get(&url).call() {
            Ok(resp) => resp
                .into_json()
                .context("Ollama: invalid /api/tags response")?,
            Err(err) => return Err(anyhow!(http_error_message(err))),
        };
        Ok(parse_tags(&body))
    }

    /// Start `POST /api/chat` with `stream: true` on a worker thread.
    /// Emits `TextDelta`s, then `Done` or `Error`. Stops early (sending
    /// `Done`) once `cancel` is set. `think`: `None`/`"off"`/`""` disables
    /// thinking, `"on"` enables it, `"low"|"medium"|"high"` sets the level;
    /// the reasoning is emitted as `Thinking` events.
    pub fn chat_stream(
        &self,
        model: &str,
        messages: Vec<OllamaMessage>,
        think: Option<String>,
        cancel: CancelToken,
    ) -> async_channel::Receiver<StreamEvent> {
        let (emitter, rx) = Emitter::new(cancel);
        // Report `Done` promptly on cancel even while the model is still loading.
        emitter.watch_cancel(|| {});
        let url = format!("{}/api/chat", self.base);
        let model = model.to_string();
        std::thread::spawn(move || run_chat(&url, &model, &messages, think.as_deref(), &emitter));
        rx
    }
}

/// The `think` request field for a UI level.
fn think_value(think: Option<&str>) -> Value {
    match think.map(str::trim) {
        Some("on") => json!(true),
        Some(level @ ("low" | "medium" | "high")) => json!(level),
        _ => json!(false),
    }
}

fn run_chat(
    url: &str,
    model: &str,
    messages: &[OllamaMessage],
    think: Option<&str>,
    emitter: &Emitter,
) {
    // Connect timeout only: a large model can take minutes to load and answer.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .build();
    let messages: Vec<Value> = messages
        .iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        // Off by default: skips the reasoning phase of thinking models (qwen3...).
        "think": think_value(think),
    });

    let mut result = agent.post(url).send_json(body.clone());
    if let Err(ureq::Error::Status(400, _)) = &result {
        // Models without thinking support reject `think`; retry without it.
        if let Some(obj) = body.as_object_mut() {
            obj.remove("think");
        }
        if emitter.should_stop() {
            emitter.send(StreamEvent::Done);
            return;
        }
        result = agent.post(url).send_json(body);
    }
    let response = match result {
        Ok(resp) => resp,
        Err(err) => {
            emitter.send(StreamEvent::Error(http_error_message(err)));
            return;
        }
    };

    let reader = BufReader::new(response.into_reader());
    let mut parser = ChatParser::default();
    for line in reader.lines() {
        if emitter.should_stop() {
            // Dropping the reader closes the connection, which stops generation.
            emitter.send(StreamEvent::Done);
            return;
        }
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                emitter.send(StreamEvent::Error(format!("Ollama: {err}")));
                return;
            }
        };
        for event in parser.parse_line(&line) {
            if !emitter.send(event) {
                return;
            }
        }
    }
    // The stream ended without `done: true`.
    emitter.send(StreamEvent::Error(
        "Ollama: پاسخ ناتمام ماند (stream ended early)".into(),
    ));
}

/// Names from a `/api/tags` body.
fn parse_tags(body: &Value) -> Vec<String> {
    body.get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m.get("name").or_else(|| m.get("model")))
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Stateful parser of the NDJSON lines of a streaming `/api/chat` response.
#[derive(Default)]
struct ChatParser {
    splitter: ThinkSplitter,
}

impl ChatParser {
    /// Events from one NDJSON line.
    fn parse_line(&mut self, line: &str) -> Vec<StreamEvent> {
        let line = line.trim();
        if line.is_empty() {
            return Vec::new();
        }
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(err) => return vec![StreamEvent::Error(format!("Ollama: invalid JSON: {err}"))],
        };
        if let Some(error) = value.get("error") {
            let msg = error
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| error.to_string());
            return vec![StreamEvent::Error(format!("Ollama: {msg}"))];
        }
        let mut events = Vec::new();
        if let Some(text) = value.pointer("/message/thinking").and_then(Value::as_str)
            && !text.is_empty()
        {
            events.push(StreamEvent::Thinking(text.to_string()));
        }
        if let Some(text) = value.pointer("/message/content").and_then(Value::as_str) {
            events.extend(self.splitter.feed(text));
        }
        if value.get("done").and_then(Value::as_bool) == Some(true) {
            events.extend(self.splitter.finish());
            events.push(StreamEvent::Done);
        }
        events
    }
}

/// A readable message for a failed request, including Ollama's `{"error"}` body.
fn http_error_message(err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, resp) => {
            let text = resp.into_string().unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string))
                .unwrap_or(text);
            format!("Ollama: HTTP {code}: {}", detail.trim())
        }
        ureq::Error::Transport(t) => {
            format!("Ollama در دسترس نیست (Ollama unreachable): {t}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_chat_line(line: &str) -> Vec<StreamEvent> {
        ChatParser::default().parse_line(line)
    }

    #[test]
    fn tags() {
        let body = json!({"models":[{"name":"qwen3.8:latest","model":"qwen3.8:latest"},{"model":"llama3:8b"}]});
        assert_eq!(parse_tags(&body), vec!["qwen3.8:latest", "llama3:8b"]);
        assert!(parse_tags(&json!({})).is_empty());
    }

    #[test]
    fn chat_lines() {
        let delta = r#"{"model":"q","created_at":"x","message":{"role":"assistant","content":"سلام"},"done":false}"#;
        assert_eq!(
            parse_chat_line(delta),
            vec![StreamEvent::TextDelta("سلام".into())]
        );
        let thinking = r#"{"model":"q","message":{"role":"assistant","content":"","thinking":"hmm"},"done":false}"#;
        assert_eq!(
            parse_chat_line(thinking),
            vec![StreamEvent::Thinking("hmm".into())]
        );
        let done = r#"{"model":"q","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}"#;
        assert_eq!(parse_chat_line(done), vec![StreamEvent::Done]);
        let err = r#"{"error":"model 'x' not found"}"#;
        assert_eq!(
            parse_chat_line(err),
            vec![StreamEvent::Error("Ollama: model 'x' not found".into())]
        );
        assert!(parse_chat_line("  ").is_empty());
    }

    #[test]
    fn inline_think_split_across_lines() {
        let mut p = ChatParser::default();
        let l1 = r#"{"message":{"role":"assistant","content":"<think>pl"},"done":false}"#;
        let l2 = r#"{"message":{"role":"assistant","content":"an</think>Hi <"},"done":false}"#;
        let l3 = r#"{"message":{"role":"assistant","content":""},"done":true}"#;
        let mut events = p.parse_line(l1);
        events.extend(p.parse_line(l2));
        events.extend(p.parse_line(l3));
        assert_eq!(
            events,
            vec![
                StreamEvent::Thinking("pl".into()),
                StreamEvent::Thinking("an".into()),
                StreamEvent::TextDelta("Hi ".into()),
                StreamEvent::TextDelta("<".into()),
                StreamEvent::Done
            ]
        );
    }

    #[test]
    fn think_values() {
        assert_eq!(think_value(None), json!(false));
        assert_eq!(think_value(Some("")), json!(false));
        assert_eq!(think_value(Some("off")), json!(false));
        assert_eq!(think_value(Some("on")), json!(true));
        assert_eq!(think_value(Some("low")), json!("low"));
        assert_eq!(think_value(Some("medium")), json!("medium"));
        assert_eq!(think_value(Some("high")), json!("high"));
        assert_eq!(think_value(Some("bogus")), json!(false));
    }

    #[test]
    fn cancelled_emitter_sends_single_done() {
        let cancel = CancelToken::new();
        let (emitter, rx) = Emitter::new(cancel.clone());
        assert!(emitter.send(StreamEvent::TextDelta("a".into())));
        cancel.cancel();
        assert!(!emitter.send(StreamEvent::TextDelta("b".into())));
        assert!(!emitter.send(StreamEvent::Error("late".into())));
        drop(emitter);
        let events: Vec<_> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert_eq!(
            events,
            vec![StreamEvent::TextDelta("a".into()), StreamEvent::Done]
        );
    }

    /// Live check against a local Ollama: `cargo test ollama_live -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn ollama_live() {
        let client = OllamaClient::new("http://127.0.0.1:11434");
        let t = std::time::Instant::now();
        let models = client.models().expect("models");
        eprintln!("models ({:?}): {models:?}", t.elapsed());
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.8:latest".into());
        let t = std::time::Instant::now();
        let rx = client.chat_stream(
            &model,
            vec![OllamaMessage {
                role: "user".into(),
                content: "Say hello in one short sentence.".into(),
            }],
            None,
            CancelToken::new(),
        );
        let mut first = None;
        let mut text = String::new();
        let mut last = None;
        while let Ok(ev) = rx.recv_blocking() {
            match ev {
                StreamEvent::TextDelta(d) => {
                    first.get_or_insert(t.elapsed());
                    text.push_str(&d);
                }
                other => {
                    last = Some(other);
                    break;
                }
            }
        }
        eprintln!(
            "first delta {first:?}, total {:?}, end {last:?}\ntext: {text}",
            t.elapsed()
        );
        assert_eq!(last, Some(StreamEvent::Done));
        assert!(!text.is_empty());
    }

    /// Cancel mid-stream: `cargo test ollama_cancel_live -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn ollama_cancel_live() {
        let client = OllamaClient::new("http://127.0.0.1:11434");
        let cancel = CancelToken::new();
        let rx = client.chat_stream(
            "qwen3.8:latest",
            vec![OllamaMessage {
                role: "user".into(),
                content: "Count from 1 to 200.".into(),
            }],
            None,
            cancel.clone(),
        );
        let mut n = 0;
        let t = std::time::Instant::now();
        while let Ok(ev) = rx.recv_blocking() {
            match ev {
                StreamEvent::TextDelta(_) => {
                    n += 1;
                    if n == 5 {
                        cancel.cancel();
                    }
                }
                other => {
                    eprintln!("after {n} deltas ({:?}): {other:?}", t.elapsed());
                    assert_eq!(other, StreamEvent::Done);
                    break;
                }
            }
        }
        assert!(rx.recv_blocking().is_err(), "nothing after Done");
    }
}
