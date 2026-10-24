//! Blocking HTTP client for the sidecar. Every method does network I/O:
//! call it from `cx.background_spawn` or a worker thread, never from render
//! or an event handler directly.

use std::{io::Read as _, path::Path, time::Duration};

use anyhow::{Context as _, Result, anyhow};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::types::*;

#[derive(Clone)]
pub struct SidecarClient {
    base: String,
    agent: ureq::Agent,
}

impl SidecarClient {
    pub fn new(base_url: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(2))
            // Synthesis, transcription and imports can take a while on first load.
            .timeout_read(Duration::from_secs(600))
            .build();
        Self {
            base: base_url.trim_end_matches('/').to_string(),
            agent,
        }
    }

    /// The client for the sidecar URL in the current settings.
    pub fn from_settings(cx: &gpui_kit::App) -> Self {
        Self::new(&crate::store::AppSettings::get(cx).sidecar_url)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn call(&self, request: ureq::Request, body: Option<Value>) -> Result<ureq::Response> {
        let result = match body {
            Some(body) => request.send_json(body),
            None => request.call(),
        };
        match result {
            Ok(response) => Ok(response),
            Err(ureq::Error::Status(code, response)) => {
                let text = response.into_string().unwrap_or_default();
                let detail = serde_json::from_str::<Value>(&text)
                    .ok()
                    .and_then(|v| v.get("detail").cloned())
                    .map(|d| match d {
                        Value::String(s) => s,
                        other => other.to_string(),
                    })
                    .unwrap_or(text);
                Err(anyhow!("sidecar {code}: {detail}"))
            }
            Err(ureq::Error::Transport(err)) => Err(anyhow!("سایدکار در دسترس نیست: {err}")),
        }
    }

    fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let response = self.call(self.agent.get(&self.url(path)), None)?;
        response.into_json().with_context(|| format!("GET {path}"))
    }

    fn send_json<T: DeserializeOwned>(&self, method: &str, path: &str, body: Value) -> Result<T> {
        let response = self.call(self.agent.request(method, &self.url(path)), Some(body))?;
        response
            .into_json()
            .with_context(|| format!("{method} {path}"))
    }

    fn delete(&self, path: &str) -> Result<()> {
        self.call(self.agent.delete(&self.url(path)), None)?;
        Ok(())
    }

    fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let response = self.call(self.agent.get(&self.url(path)), None)?;
        let mut bytes = Vec::new();
        response.into_reader().read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// `GET /health` with a short timeout.
    pub fn health(&self) -> Result<Health> {
        let response = self.call(
            self.agent
                .get(&self.url("/health"))
                .timeout(Duration::from_secs(3)),
            None,
        )?;
        Ok(response.into_json()?)
    }

    pub fn unload(&self) -> Result<()> {
        let _: Value = self.send_json("POST", "/unload", json!({}))?;
        Ok(())
    }

    pub fn profiles(&self) -> Result<Vec<Profile>> {
        self.get_json("/profiles")
    }

    pub fn create_profile(&self, name: &str, engine: Option<&str>) -> Result<Profile> {
        let mut body = json!({ "name": name });
        if let Some(engine) = engine {
            body["engine"] = json!(engine);
        }
        self.send_json("POST", "/profiles", body)
    }

    pub fn profile(&self, pid: &str) -> Result<Profile> {
        self.get_json(&format!("/profiles/{pid}"))
    }

    pub fn update_profile(&self, pid: &str, patch: &ProfilePatch) -> Result<Profile> {
        self.send_json(
            "PATCH",
            &format!("/profiles/{pid}"),
            serde_json::to_value(patch)?,
        )
    }

    pub fn delete_profile(&self, pid: &str) -> Result<()> {
        self.delete(&format!("/profiles/{pid}"))
    }

    pub fn samples(&self, pid: &str) -> Result<Vec<Sample>> {
        self.get_json(&format!("/profiles/{pid}/samples"))
    }

    /// `POST /profiles/{pid}/samples`. `text: None` asks the sidecar to transcribe.
    pub fn add_sample(
        &self,
        pid: &str,
        wav_path: &Path,
        text: Option<&str>,
        source: &str,
    ) -> Result<Sample> {
        let mut body = json!({
            "wav_path": wav_path.to_string_lossy(),
            "source": source,
        });
        if let Some(text) = text.filter(|t| !t.trim().is_empty()) {
            body["text"] = json!(text);
        }
        self.send_json("POST", &format!("/profiles/{pid}/samples"), body)
    }

    pub fn update_sample(&self, pid: &str, sid: &str, text: &str) -> Result<Sample> {
        self.send_json(
            "PATCH",
            &format!("/profiles/{pid}/samples/{sid}"),
            json!({ "text": text }),
        )
    }

    pub fn delete_sample(&self, pid: &str, sid: &str) -> Result<()> {
        self.delete(&format!("/profiles/{pid}/samples/{sid}"))
    }

    /// WAV bytes of a sample.
    pub fn sample_audio(&self, pid: &str, sid: &str) -> Result<Vec<u8>> {
        self.get_bytes(&format!("/profiles/{pid}/samples/{sid}/audio"))
    }

    pub fn prompts(&self, n: usize) -> Result<Vec<String>> {
        self.get_json(&format!("/prompts?n={n}"))
    }

    pub fn train(&self, pid: &str, steps: Option<u32>, from_run: Option<&str>) -> Result<Run> {
        let mut body = json!({ "from_run": from_run });
        if let Some(steps) = steps {
            body["steps"] = json!(steps);
        }
        self.send_json("POST", &format!("/profiles/{pid}/train"), body)
    }

    /// Newest first.
    pub fn runs(&self, pid: &str) -> Result<Vec<Run>> {
        self.get_json(&format!("/profiles/{pid}/runs"))
    }

    pub fn run(&self, pid: &str, rid: &str) -> Result<Run> {
        self.get_json(&format!("/profiles/{pid}/runs/{rid}"))
    }

    pub fn run_log(&self, pid: &str, rid: &str) -> Result<String> {
        let bytes = self.get_bytes(&format!("/profiles/{pid}/runs/{rid}/log"))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn cancel_run(&self, pid: &str, rid: &str) -> Result<Run> {
        self.send_json(
            "POST",
            &format!("/profiles/{pid}/runs/{rid}/cancel"),
            json!({}),
        )
    }

    pub fn delete_run(&self, pid: &str, rid: &str) -> Result<()> {
        self.delete(&format!("/profiles/{pid}/runs/{rid}"))
    }

    // MARK: Models

    pub fn models(&self) -> Result<Vec<ModelInfo>> {
        self.get_json("/models")
    }

    pub fn download_model(&self, id: &str) -> Result<ModelInfo> {
        self.send_json(
            "POST",
            &format!("/models/{}/download", encode_id(id)),
            json!({}),
        )
    }

    pub fn cancel_model_download(&self, id: &str) -> Result<ModelInfo> {
        self.send_json(
            "POST",
            &format!("/models/{}/cancel", encode_id(id)),
            json!({}),
        )
    }

    /// Deletes the Hugging Face cache copy of a catalog model.
    pub fn delete_model(&self, id: &str) -> Result<()> {
        self.delete(&format!("/models/{}", encode_id(id)))
    }

    pub fn models_config(&self) -> Result<ModelsConfig> {
        self.get_json("/models/config")
    }

    pub fn update_models_config(&self, patch: &ModelsConfigPatch) -> Result<ModelsConfig> {
        self.send_json("PATCH", "/models/config", serde_json::to_value(patch)?)
    }

    /// `POST /tts` → WAV bytes (24 kHz mono 16-bit).
    pub fn tts(&self, request: &TtsRequest) -> Result<Vec<u8>> {
        let mut body = json!({
            "profile_id": request.profile_id,
            "text": request.text,
        });
        if let Some(engine) = &request.engine {
            body["engine"] = json!(engine);
        }
        match &request.run {
            TtsRun::Active => {}
            TtsRun::ZeroShot => body["run_id"] = Value::Null,
            TtsRun::Run(id) => body["run_id"] = json!(id),
        }
        if let Some(speed) = request.speed {
            body["speed"] = json!(speed);
        }
        let response = self.call(self.agent.post(&self.url("/tts")), Some(body))?;
        let mut bytes = Vec::new();
        response.into_reader().read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// `POST /stt` → Persian transcript.
    pub fn stt(&self, wav_path: &Path) -> Result<String> {
        let value: Value = self.send_json(
            "POST",
            "/stt",
            json!({ "wav_path": wav_path.to_string_lossy() }),
        )?;
        Ok(value
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .trim()
            .to_string())
    }
}

/// Percent-encode a path segment: everything but RFC 3986 unreserved
/// characters, so ids like `local:/abs/path` stay one segment.
pub fn encode_id(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for byte in id.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::encode_id;

    #[test]
    fn encodes_ids() {
        assert_eq!(encode_id("whisper-large_v3.t~"), "whisper-large_v3.t~");
        assert_eq!(encode_id("k2-fsa/OmniVoice"), "k2-fsa%2FOmniVoice");
        assert_eq!(
            encode_id("local:/home/me/my models/x"),
            "local%3A%2Fhome%2Fme%2Fmy%20models%2Fx"
        );
        assert_eq!(encode_id("مدل"), "%D9%85%D8%AF%D9%84");
        assert_eq!(encode_id("a?b#c%"), "a%3Fb%23c%25");
    }
}
