use std::{
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

use anyhow::{Context as _, Result, anyhow};
use gpui_kit::Global;
use rodio::{Decoder, OutputStream, Sink, Source as _, source::EmptyCallback};

enum Command {
    /// Drop the current sink (stops it) and start a fresh one.
    Stop,
    /// Append WAV bytes, if `generation` is still current.
    Enqueue { wav: Vec<u8>, generation: u64 },
}

/// State shared between the handles and the audio thread.
struct Shared {
    /// Bumped by `stop`; work tagged with an older generation is dropped.
    generation: AtomicU64,
    /// Items queued or playing in the current generation.
    queued: AtomicUsize,
}

impl Shared {
    fn finished_one(&self, generation: u64) {
        if self.generation.load(Ordering::SeqCst) == generation {
            let _ = self
                .queued
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1));
        }
    }
}

/// Plays WAV audio on a dedicated thread that owns the rodio output stream.
/// Cheap to clone; all methods return immediately.
#[derive(Clone)]
pub struct Player {
    commands: mpsc::Sender<Command>,
    shared: Arc<Shared>,
}

impl Global for Player {}

impl Player {
    /// Open the default output device. Fails when there is none.
    pub fn new() -> Result<Self> {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            queued: AtomicUsize::new(0),
        });
        let (commands, rx) = mpsc::channel::<Command>();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<()>>(1);
        let thread_shared = shared.clone();
        thread::Builder::new()
            .name("seda-audio-out".into())
            .spawn(move || audio_thread(rx, ready_tx, thread_shared))
            .context("cannot start the audio thread")?;
        ready_rx
            .recv()
            .map_err(|_| anyhow!("the audio thread exited"))??;
        Ok(Self { commands, shared })
    }

    /// Stop whatever plays and play `wav` (complete WAV file bytes).
    pub fn play(&self, wav: Vec<u8>) {
        self.stop();
        self.enqueue(wav);
    }

    /// Append `wav` after what is queued.
    pub fn enqueue(&self, wav: Vec<u8>) {
        let generation = self.shared.generation.load(Ordering::SeqCst);
        self.shared.queued.fetch_add(1, Ordering::SeqCst);
        if self
            .commands
            .send(Command::Enqueue { wav, generation })
            .is_err()
        {
            self.shared.finished_one(generation);
        }
    }

    /// Stop playback and drop anything queued.
    pub fn stop(&self) {
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        self.shared.queued.store(0, Ordering::SeqCst);
        let _ = self.commands.send(Command::Stop);
    }

    /// Whether audio is playing or queued.
    pub fn is_playing(&self) -> bool {
        self.shared.queued.load(Ordering::SeqCst) > 0
    }
}

fn audio_thread(
    rx: mpsc::Receiver<Command>,
    ready: mpsc::SyncSender<Result<()>>,
    shared: Arc<Shared>,
) {
    let (_stream, handle) = match OutputStream::try_default() {
        Ok(output) => output,
        Err(err) => {
            let _ = ready.send(Err(anyhow!("no audio output device: {err}")));
            return;
        }
    };
    let new_sink =
        || Sink::try_new(&handle).map_err(|err| anyhow!("cannot open audio output: {err}"));
    let mut sink = match new_sink() {
        Ok(sink) => sink,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    // Exits when every `Player` handle is gone.
    while let Ok(command) = rx.recv() {
        match command {
            Command::Stop => {
                sink.stop();
                match new_sink() {
                    Ok(fresh) => sink = fresh,
                    Err(err) => eprintln!("seda: {err:#}"),
                }
            }
            Command::Enqueue { wav, generation } => {
                if shared.generation.load(Ordering::SeqCst) != generation {
                    continue;
                }
                match Decoder::new(Cursor::new(wav)) {
                    Ok(source) => {
                        let done = shared.clone();
                        sink.append(source.convert_samples::<f32>());
                        sink.append(EmptyCallback::<f32>::new(Box::new(move || {
                            done.finished_one(generation)
                        })));
                        sink.play();
                    }
                    Err(err) => {
                        eprintln!("seda: cannot decode audio: {err}");
                        shared.finished_one(generation);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn tone(seconds: f32, freq: f32, amplitude: f32) -> Vec<u8> {
        let rate = 24_000;
        let n = (rate as f32 * seconds) as usize;
        let samples: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                ((t * freq * std::f32::consts::TAU).sin() * amplitude * i16::MAX as f32) as i16
            })
            .collect();
        super::super::wav::wav_bytes(&samples, rate, 1).unwrap()
    }

    /// Plays a quiet 0.5 s 440 Hz tone on the default device:
    /// `cargo test -- --ignored player_plays_tone`.
    #[test]
    #[ignore = "needs an audio output device"]
    fn player_plays_tone() {
        let player = Player::new().expect("output device");
        let start = Instant::now();
        player.enqueue(tone(0.5, 440.0, 0.1));
        player.enqueue(tone(0.25, 660.0, 0.1));
        assert!(player.is_playing());
        while player.is_playing() && start.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(10));
        }
        let elapsed = start.elapsed();
        assert!(!player.is_playing(), "still playing after 5 s");
        // The end is seen when the mixer pulls the last sample, a buffer (~50 ms) early.
        eprintln!("played 0.75 s of audio in {elapsed:?}");
        assert!(
            elapsed >= Duration::from_millis(600),
            "finished too early: {elapsed:?}"
        );

        // stop() clears at once.
        player.play(tone(2.0, 440.0, 0.1));
        thread::sleep(Duration::from_millis(200));
        assert!(player.is_playing());
        player.stop();
        assert!(!player.is_playing());
        thread::sleep(Duration::from_millis(300));
        assert!(!player.is_playing());

        // Garbage is not counted as playing.
        player.enqueue(b"not a wav".to_vec());
        thread::sleep(Duration::from_millis(200));
        assert!(!player.is_playing());
    }

    /// Records 1 s from the default input with cpal:
    /// `cargo test -- --ignored microphone_records`.
    #[test]
    #[ignore = "needs an audio input device"]
    fn microphone_records() {
        use rodio::cpal::traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _};

        let host = rodio::cpal::default_host();
        let device = host.default_input_device().expect("input device");
        let config = device.default_input_config().expect("input config");
        eprintln!("input: {:?} {:?}", device.name().ok(), config);
        let frames = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let (f, p) = (frames.clone(), peak.clone());
        let channels = config.channels() as usize;
        let stream = device
            .build_input_stream(
                &config.config(),
                move |data: &[f32], _| {
                    f.fetch_add(data.len() / channels, Ordering::SeqCst);
                    let max = data.iter().fold(0f32, |m, s| m.max(s.abs()));
                    p.fetch_max((max * 1000.0) as usize, Ordering::SeqCst);
                },
                |err| eprintln!("input error: {err}"),
                None,
            )
            .expect("input stream (f32)");
        stream.play().unwrap();
        thread::sleep(Duration::from_secs(1));
        drop(stream);
        let frames = frames.load(Ordering::SeqCst);
        eprintln!(
            "captured {frames} frames at {} Hz, peak {:.3}",
            config.sample_rate().0,
            peak.load(Ordering::SeqCst) as f32 / 1000.0
        );
        assert!(frames as u32 > config.sample_rate().0 / 2);
    }
}
