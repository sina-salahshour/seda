use std::{path::PathBuf, time::Duration};

use anyhow::anyhow;
use gpui_kit::{
    App, Context, Task,
    component::speech::{
        AudioFormat, RecognitionSession, SpeechError, SpeechRecognizer, SpeechSink, SpeechState,
    },
};

use super::{temp_wav_path, write_wav};
use crate::sidecar::SidecarClient;

/// Push-to-talk audio: 16 kHz mono, what Whisper wants.
const STT_RATE: u32 = 16_000;
/// Voice samples: 24 kHz mono, what the sidecar stores.
const CAPTURE_RATE: u32 = 24_000;
/// Push-to-talk shorter than this is ignored (an accidental tap).
const MIN_STT_SECONDS: f32 = 0.3;
/// How long `SpeechState::stop` waits for the final result.
const STOP_TIMEOUT: Duration = Duration::from_secs(120);

/// Push-to-talk recognizer: buffers 16 kHz mono PCM; on finish writes a temp
/// WAV, calls the sidecar's `POST /stt` off the UI thread, reports the text as
/// one phrase and finishes. Errors go to `SpeechSink::error`.
pub struct SidecarStt;

/// Recording recognizer for voice samples: buffers 24 kHz mono PCM; on finish
/// writes a temp WAV and reports its absolute *path* as the transcript, so
/// `SpeechEvent::Final(path)` hands the file to the caller.
pub struct WavCapture;

/// A `SpeechState` for push-to-talk using [`SidecarStt`] (long stop timeout,
/// no system fallback).
pub fn stt_speech_state(cx: &mut Context<SpeechState>) -> SpeechState {
    SpeechState::new(cx)
        .recognizer(SidecarStt)
        .input(super::microphone::SettingsMicrophone)
        .system_fallback(false)
        .stop_timeout(STOP_TIMEOUT)
}

/// A `SpeechState` that records a WAV with [`WavCapture`].
pub fn capture_speech_state(cx: &mut Context<SpeechState>) -> SpeechState {
    SpeechState::new(cx)
        .recognizer(WavCapture)
        .input(super::microphone::SettingsMicrophone)
        .system_fallback(false)
        .stop_timeout(STOP_TIMEOUT)
}

#[derive(Clone, Copy)]
enum Mode {
    Stt,
    Capture,
}

impl Mode {
    fn rate(self) -> u32 {
        match self {
            Mode::Stt => STT_RATE,
            Mode::Capture => CAPTURE_RATE,
        }
    }
}

impl SpeechRecognizer for SidecarStt {
    fn audio_format(&self) -> AudioFormat {
        AudioFormat::new(STT_RATE, 1)
    }

    fn start(
        &self,
        sink: SpeechSink,
        cx: &mut App,
    ) -> Result<Box<dyn RecognitionSession>, SpeechError> {
        Ok(Box::new(BufferedSession::start(Mode::Stt, sink, cx)))
    }
}

impl SpeechRecognizer for WavCapture {
    fn audio_format(&self) -> AudioFormat {
        AudioFormat::new(CAPTURE_RATE, 1)
    }

    fn start(
        &self,
        sink: SpeechSink,
        cx: &mut App,
    ) -> Result<Box<dyn RecognitionSession>, SpeechError> {
        Ok(Box::new(BufferedSession::start(Mode::Capture, sink, cx)))
    }
}

/// Buffers the whole utterance; all the work happens in `finish`.
/// Dropping it drops `task`, which cancels a pending transcription.
struct BufferedSession {
    mode: Mode,
    sink: SpeechSink,
    samples: Vec<i16>,
    finished: bool,
    task: Option<Task<()>>,
}

impl BufferedSession {
    fn start(mode: Mode, sink: SpeechSink, cx: &mut App) -> Self {
        // Nothing to connect to: accept audio at once.
        sink.ready(cx);
        Self {
            mode,
            sink,
            samples: Vec::new(),
            finished: false,
            task: None,
        }
    }
}

impl RecognitionSession for BufferedSession {
    fn push_audio(&mut self, samples: &[i16], _cx: &mut App) {
        if !self.finished {
            self.samples.extend_from_slice(samples);
        }
    }

    fn finish(&mut self, cx: &mut App) {
        if self.finished {
            return;
        }
        self.finished = true;
        let samples = std::mem::take(&mut self.samples);
        let rate = self.mode.rate();
        let seconds = samples.len() as f32 / rate as f32;
        let sink = self.sink.clone();

        match self.mode {
            Mode::Stt => {
                if seconds < MIN_STT_SECONDS {
                    sink.finish(cx);
                    return;
                }
                let client = SidecarClient::from_settings(cx);
                self.task = Some(cx.spawn(async move |cx| {
                    let result = cx
                        .background_executor()
                        .spawn(async move {
                            let path = temp_wav_path("ptt");
                            write_wav(&path, &samples, rate, 1)?;
                            let text = client.stt(&path);
                            let _ = std::fs::remove_file(&path);
                            text
                        })
                        .await;
                    cx.update(|cx| match result {
                        Ok(text) => {
                            let text = text.trim().to_string();
                            if !text.is_empty() {
                                sink.phrase(text, cx);
                            }
                            sink.finish(cx);
                        }
                        Err(err) => sink.error(SpeechError::recognizer(err), cx),
                    });
                }));
            }
            Mode::Capture => {
                if samples.is_empty() {
                    sink.finish(cx);
                    return;
                }
                self.task = Some(cx.spawn(async move |cx| {
                    let result = cx
                        .background_executor()
                        .spawn(async move {
                            let path = temp_wav_path("rec");
                            write_wav(&path, &samples, rate, 1)?;
                            absolute(path)
                        })
                        .await;
                    cx.update(|cx| match result {
                        Ok(path) => {
                            sink.phrase(path.to_string_lossy().into_owned(), cx);
                            sink.finish(cx);
                        }
                        Err(err) => sink.error(SpeechError::recognizer(err), cx),
                    });
                }));
            }
        }
    }
}

fn absolute(path: PathBuf) -> anyhow::Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }
    std::path::absolute(&path).map_err(|err| anyhow!("cannot resolve {}: {err}", path.display()))
}
