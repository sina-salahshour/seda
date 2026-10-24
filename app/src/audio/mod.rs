//! Audio: playback (rodio, on its own thread), the sentence-by-sentence TTS
//! queue, WAV helpers, and the two `SpeechRecognizer`s that plug the sidecar
//! into gpui-kit's `SpeechState` (so `SpeechButton`/`SpeechWaveform` give us
//! capture, levels and the waveform for free).

mod microphone;
mod player;
mod recognizer;
mod sentences;
mod tts_queue;
mod wav;

pub use microphone::input_device_names;
pub use player::Player;
pub use recognizer::{capture_speech_state, stt_speech_state};
pub use sentences::{SentenceSplitter, speakable};
pub use tts_queue::SpeechQueue;
pub use wav::{temp_wav_path, write_wav};
