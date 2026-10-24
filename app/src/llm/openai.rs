//! OpenAI-compatible endpoints: `GET {base}/models` and streaming
//! `POST {base}/chat/completions` (SSE). Works with OpenAI, OpenRouter,
//! vLLM, LM Studio, DeepSeek and friends. Reasoning arrives as
//! `delta.reasoning_content` / `delta.reasoning` or inline `<think>` tags and
//! becomes [`StreamEvent::Thinking`].

use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::{Context as _, Result, anyhow};
use serde_json::{Value, json};

use super::{CancelToken, Emitter, StreamEvent, ThinkSplitter};

/// One message of the conversation sent to `/chat/completions`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiMessage {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct OpenAiClient {
    base: String,
    api_key: Option<String>,
}

impl OpenAiClient {
    /// `base_url` is e.g. `https://api.openai.com/v1` (trailing `/` ignored);
    /// an empty or missing `api_key` sends no `Authorization` header.
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            base: base_url.trim().trim_end_matches('/').to_string(),
            api_key: api_key.filter(|k| !k.trim().is_empty()),
        }
    }

    pub fn from_endpoint(endpoint: &crate::store::Endpoint) -> Self {
        Self::new(&endpoint.base_url, endpoint.api_key.clone())
    }

    /// `GET {base}/models` → model ids, sorted. Blocking; call off the UI thread.
    pub fn models(&self) -> Result<Vec<String>> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(3))
            .timeout(Duration::from_secs(15))
            .build();
        let url = format!("{}/models", self.base);
        let mut req = agent.get(&url);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key.trim()));
        }
        match req.call() {
            Ok(resp) => {
                let body: Value = resp
                    .into_json()
                    .context("API: invalid /models response (پاسخ نامعتبر)")?;
                Ok(parse_models(&body))
            }
            Err(err) => Err(anyhow!(http_error_message(err))),
        }
    }

    /// Stream `POST {base}/chat/completions` (`stream: true`) on a worker
    /// thread. `reasoning` (`low|medium|high`) is sent as `reasoning_effort`;
    /// if the server rejects it with HTTP 400 the request is retried without.
    /// Emits `TextDelta`/`Thinking`, then `Done` or `Error`; a cancelled turn
    /// ends with `Done`.
    pub fn chat_stream(
        &self,
        model: &str,
        messages: Vec<OpenAiMessage>,
        reasoning: Option<String>,
        cancel: CancelToken,
    ) -> async_channel::Receiver<StreamEvent> {
        let (emitter, rx) = Emitter::new(cancel);
        emitter.watch_cancel(|| {});
        let url = format!("{}/chat/completions", self.base);
        let key = self.api_key.clone();
        let model = model.to_string();
        let reasoning = reasoning
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty());
        std::thread::spawn(move || {
            run_chat(&url, key.as_deref(), &model, &messages, reasoning, &emitter)
        });
        rx
    }
}

#[allow(clippy::result_large_err)]
fn post(
    agent: &ureq::Agent,
    url: &str,
    key: Option<&str>,
    body: &Value,
) -> Result<ureq::Response, ureq::Error> {
    let mut req = agent.post(url).set("Accept", "text/event-stream");
    if let Some(key) = key {
        req = req.set("Authorization", &format!("Bearer {}", key.trim()));
    }
    req.send_json(body.clone())
}

fn run_chat(
    url: &str,
    key: Option<&str>,
    model: &str,
    messages: &[OpenAiMessage],
    reasoning: Option<String>,
    emitter: &Emitter,
) {
    // Connect timeout only: reasoning models can be silent for a long time.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .build();
    let messages: Vec<Value> = messages
        .iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut body = json!({ "model": model, "messages": messages, "stream": true });
    let sent_reasoning = reasoning.is_some();
    if let (Some(effort), Some(obj)) = (&reasoning, body.as_object_mut()) {
        obj.insert("reasoning_effort".into(), json!(effort));
    }

    let mut result = post(&agent, url, key, &body);
    if sent_reasoning && let Err(ureq::Error::Status(400, _)) = &result {
        // Servers/models without reasoning support reject the parameter.
        if let Some(obj) = body.as_object_mut() {
            obj.remove("reasoning_effort");
        }
        if emitter.should_stop() {
            emitter.send(StreamEvent::Done);
            return;
        }
        result = post(&agent, url, key, &body);
    }
    let response = match result {
        Ok(resp) => resp,
        Err(err) => {
            emitter.send(StreamEvent::Error(http_error_message(err)));
            return;
        }
    };

    let reader = BufReader::new(response.into_reader());
    let mut parser = SseParser::new();
    for line in reader.lines() {
        if emitter.should_stop() {
            // Dropping the reader closes the connection, which stops generation.
            emitter.send(StreamEvent::Done);
            return;
        }
        let line = match line {
            Ok(line) => line,
            Err(err) => {
                emitter.send(StreamEvent::Error(format!("API: {err}")));
                return;
            }
        };
        for event in parser.feed_line(&line) {
            if !emitter.send(event) {
                return;
            }
        }
        if parser.is_finished() {
            return;
        }
    }
    for event in parser.finish() {
        if !emitter.send(event) {
            return;
        }
    }
}

/// Model ids from a `/models` body: `{"data":[{"id"}]}`, a bare array, or
/// `{"models":[{"id"|"name"}]}`; sorted and deduplicated.
fn parse_models(body: &Value) -> Vec<String> {
    let list = body
        .get("data")
        .or_else(|| body.get("models"))
        .unwrap_or(body)
        .as_array();
    let mut ids: Vec<String> = list
        .map(|items| {
            items
                .iter()
                .filter_map(|m| match m {
                    Value::String(s) => Some(s.as_str()),
                    _ => m
                        .get("id")
                        .or_else(|| m.get("name"))
                        .and_then(Value::as_str),
                })
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    ids.dedup();
    ids
}

/// Incremental parser of the SSE body of a streaming chat completion.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    splitter: ThinkSplitter,
    saw_finish: bool,
    finished: bool,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// True once a terminal event (`Done`/`Error`) was produced.
    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    /// Events for one line of the SSE stream (without line terminator).
    pub(crate) fn feed_line(&mut self, line: &str) -> Vec<StreamEvent> {
        if self.finished {
            return Vec::new();
        }
        let line = line.trim();
        // Comments (`: keep-alive`), `event:`/`id:`/`retry:` and blanks.
        let Some(data) = line.strip_prefix("data:") else {
            return Vec::new();
        };
        let data = data.trim();
        if data.is_empty() {
            return Vec::new();
        }
        if data == "[DONE]" {
            let mut events = self.splitter.finish();
            events.push(StreamEvent::Done);
            self.finished = true;
            return events;
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(err) => return self.fail(format!("API: invalid JSON: {err}")),
        };
        if let Some(error) = value.get("error").filter(|e| !e.is_null()) {
            let msg = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| error.to_string());
            return self.fail(format!("API: {msg}"));
        }
        let mut events = Vec::new();
        let choice = value.pointer("/choices/0");
        if let Some(delta) = choice.and_then(|c| c.get("delta")) {
            for key in ["reasoning_content", "reasoning"] {
                if let Some(text) = delta.get(key).and_then(Value::as_str)
                    && !text.is_empty()
                {
                    events.push(StreamEvent::Thinking(text.to_string()));
                }
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                events.extend(self.splitter.feed(text));
            }
        }
        if choice
            .and_then(|c| c.get("finish_reason"))
            .is_some_and(|f| !f.is_null())
        {
            self.saw_finish = true;
        }
        events
    }

    /// Call when the body ended: `Done` if a `finish_reason` was seen,
    /// otherwise an error. Empty when a terminal event was already produced.
    pub(crate) fn finish(&mut self) -> Vec<StreamEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = self.splitter.finish();
        events.push(if self.saw_finish {
            StreamEvent::Done
        } else {
            StreamEvent::Error("API: پاسخ ناتمام ماند (stream ended early)".into())
        });
        events
    }

    fn fail(&mut self, message: String) -> Vec<StreamEvent> {
        self.finished = true;
        vec![StreamEvent::Error(message)]
    }
}

/// A readable message for a failed request, with the server's `error.message`.
fn http_error_message(err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, resp) => {
            let text = resp.into_string().unwrap_or_default();
            let detail = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    let e = v.get("error")?;
                    e.get("message")
                        .and_then(Value::as_str)
                        .or_else(|| e.as_str())
                        .or_else(|| v.get("message").and_then(Value::as_str))
                        .map(str::to_string)
                })
                .unwrap_or(text);
            format!("API: HTTP {code}: {}", detail.trim())
        }
        ureq::Error::Transport(t) => {
            format!("سرور در دسترس نیست (API unreachable): {t}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: &[&str]) -> Vec<StreamEvent> {
        let mut p = SseParser::new();
        let mut events: Vec<_> = lines.iter().flat_map(|l| p.feed_line(l)).collect();
        events.extend(p.finish());
        // Merge adjacent text deltas (the splitter flushes held-back text late).
        let mut merged: Vec<StreamEvent> = Vec::new();
        for ev in events {
            match (merged.last_mut(), &ev) {
                (Some(StreamEvent::TextDelta(a)), StreamEvent::TextDelta(b)) => a.push_str(b),
                _ => merged.push(ev),
            }
        }
        merged
    }

    fn chunk(delta: &str, finish: &str) -> String {
        format!(r#"data: {{"choices":[{{"index":0,"delta":{delta},"finish_reason":{finish}}}]}}"#)
    }

    #[test]
    fn openai_style_stream() {
        let lines = [
            ": keep-alive".to_string(),
            "event: message".to_string(),
            chunk(r#"{"role":"assistant","content":""}"#, "null"),
            String::new(),
            chunk(r#"{"content":"سلام"}"#, "null"),
            chunk(r#"{"content":" دنیا"}"#, "null"),
            chunk("{}", r#""stop""#),
            "data: [DONE]".to_string(),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        assert_eq!(
            run(&refs),
            vec![StreamEvent::TextDelta("سلام دنیا".into()), StreamEvent::Done]
        );
    }

    #[test]
    fn inline_think_split_across_chunks() {
        let lines = [
            chunk(r#"{"content":"<thi"}"#, "null"),
            chunk(r#"{"content":"nk>hmm</th"}"#, "null"),
            chunk(r#"{"content":"ink>Hi"}"#, "null"),
            "data:[DONE]".to_string(),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        assert_eq!(
            run(&refs),
            vec![
                StreamEvent::Thinking("hmm".into()),
                StreamEvent::TextDelta("Hi".into()),
                StreamEvent::Done
            ]
        );
    }

    #[test]
    fn reasoning_fields() {
        let lines = [
            chunk(r#"{"reasoning_content":"think1","content":null}"#, "null"),
            chunk(r#"{"reasoning":"think2"}"#, "null"),
            chunk(r#"{"content":"ans"}"#, "null"),
            "data: [DONE]".to_string(),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        assert_eq!(
            run(&refs),
            vec![
                StreamEvent::Thinking("think1".into()),
                StreamEvent::Thinking("think2".into()),
                StreamEvent::TextDelta("ans".into()),
                StreamEvent::Done
            ]
        );
    }

    #[test]
    fn done_flushes_held_back_text() {
        let lines = [
            chunk(r#"{"content":"a <"}"#, "null"),
            "data: [DONE]".to_string(),
        ];
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        assert_eq!(
            run(&refs),
            vec![StreamEvent::TextDelta("a <".into()), StreamEvent::Done]
        );
    }

    #[test]
    fn end_without_done() {
        // finish_reason seen: treated as Done.
        let with = [
            chunk(r#"{"content":"x"}"#, "null"),
            chunk(r#"{"content":" <"}"#, r#""stop""#),
        ];
        let refs: Vec<&str> = with.iter().map(String::as_str).collect();
        assert_eq!(
            run(&refs),
            vec![StreamEvent::TextDelta("x <".into()), StreamEvent::Done]
        );
        // Not seen: error.
        let without = [chunk(r#"{"content":"x"}"#, "null")];
        let refs: Vec<&str> = without.iter().map(String::as_str).collect();
        let events = run(&refs);
        assert_eq!(events[0], StreamEvent::TextDelta("x".into()));
        assert!(matches!(&events[1], StreamEvent::Error(m) if m.contains("stream ended early")));
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn error_chunk_is_terminal() {
        let lines = [
            r#"data: {"error":{"message":"Rate limit exceeded","type":"x"}}"#,
            "data: [DONE]",
        ];
        assert_eq!(
            run(&lines),
            vec![StreamEvent::Error("API: Rate limit exceeded".into())]
        );
    }

    #[test]
    fn invalid_json_is_error() {
        let events = run(&["data: {nope"]);
        assert!(matches!(&events[..], [StreamEvent::Error(m)] if m.contains("invalid JSON")));
    }

    #[test]
    fn models_parsing() {
        let a = json!({"object":"list","data":[{"id":"b"},{"id":"a"},{"id":"b"}]});
        assert_eq!(parse_models(&a), vec!["a", "b"]);
        let b = json!([{"id":"z"},{"id":"y"}]);
        assert_eq!(parse_models(&b), vec!["y", "z"]);
        let c = json!({"models":[{"name":"m2"},{"id":"m1"}]});
        assert_eq!(parse_models(&c), vec!["m1", "m2"]);
        assert!(parse_models(&json!({})).is_empty());
    }

    #[test]
    fn client_normalises_inputs() {
        let c = OpenAiClient::new(" http://x/v1/ ", Some("  ".into()));
        assert_eq!(c.base, "http://x/v1");
        assert!(c.api_key.is_none());
    }

    /// Live check: `OPENAI_BASE=… OPENAI_KEY=… OPENAI_MODEL=… cargo test openai_live -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn openai_live() {
        let base = std::env::var("OPENAI_BASE").expect("OPENAI_BASE");
        let key = std::env::var("OPENAI_KEY").ok();
        let model = std::env::var("OPENAI_MODEL").expect("OPENAI_MODEL");
        let client = OpenAiClient::new(&base, key);
        match client.models() {
            Ok(models) => eprintln!("models: {models:?}"),
            Err(err) => eprintln!("models failed: {err}"),
        }
        let rx = client.chat_stream(
            &model,
            vec![OpenAiMessage {
                role: "user".into(),
                content: "Say hello in one short sentence.".into(),
            }],
            None,
            CancelToken::new(),
        );
        let mut text = String::new();
        let mut last = None;
        while let Ok(ev) = rx.recv_blocking() {
            match ev {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Thinking(t) => eprintln!("thinking: {t}"),
                other => {
                    last = Some(other);
                    break;
                }
            }
        }
        eprintln!("text: {text}\nend: {last:?}");
        assert_eq!(last, Some(StreamEvent::Done));
        assert!(!text.is_empty());
    }
}
